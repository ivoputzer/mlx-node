import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

// --- Configuration Parsing ---
struct BridgeGenerateConfig: Decodable {
    var temperature: Float?
    var topP: Float?
    var repetitionPenalty: Float?
    var repetitionContextSize: Int?
    var streamChunkSize: Int? // Used only in Swift for chunking

    func toGenerateParameters() -> GenerateParameters {
        return GenerateParameters(
            temperature: temperature ?? 0.6,
            topP: topP ?? 1.0,
            repetitionPenalty: repetitionPenalty ?? 1.0,
            repetitionContextSize: repetitionContextSize ?? 20
        )
    }
}

private func parseConfig(_ jsonString: String) -> BridgeGenerateConfig {
    guard let data = jsonString.data(using: .utf8),
          let config = try? JSONDecoder().decode(BridgeGenerateConfig.self, from: data) else {
        return BridgeGenerateConfig()
    }
    return config
}


// --- Tokenizer Implementations ---
struct BridgeTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
    let tokenizer: Tokenizers.Tokenizer
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { return tokenizer.encode(text: text) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { return tokenizer.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens) }
    func convertTokenToId(_ token: String) -> Int? { return tokenizer.encode(text: token).first }
    func convertIdToToken(_ id: Int) -> String? { return tokenizer.decode(tokens: [id]) }
    var bosToken: String? { tokenizer.bosToken }
    var eosToken: String? { tokenizer.eosToken }
    var unknownToken: String? { tokenizer.unknownToken }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] { return [] }
}

struct BridgeTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return BridgeTokenizer(tokenizer: tokenizer)
    }
}


// --- Global Model Registry ---
private var modelRegistry: [Int32: ModelContext] = [:]
private var nextModelId: Int32 = 1
private let registryLock = NSLock()


// --- Bridge Logic ---

@_cdecl("mlx_swift_init_metal")
public func mlxSwiftInitMetal() {
    // 1. Create a dummy scalar array
    let a = MLXArray(0.0)

    // 2. Perform a math operation. This creates an MLX computation graph
    // that requires the Metal 'add' kernel to execute.
    let b = a + a

    // 3. Evaluate it. This forces MLX to compile the graph,
    // hit the GPU, and permanently cache the default.metallib.
    eval(b)
}

@_cdecl("mlx_swift_load_model")
public func loadModel(path: UnsafePointer<CChar>, context: UnsafeMutableRawPointer, callback: @convention(c) (UnsafeMutableRawPointer, Bool, Int32, UnsafePointer<CChar>?) -> Void) {
    let url = URL(fileURLWithPath: String(cString: path))
    Task {
        do {
            let modelCtx = try await LLMModelFactory.shared.load(from: url, using: BridgeTokenizerLoader())
            let id = registryLock.withLock {
                let currentId = nextModelId
                nextModelId += 1
                modelRegistry[currentId] = modelCtx
                return currentId
            }
            callback(context, true, id, nil)
        } catch {
            error.localizedDescription.withCString { callback(context, false, 0, $0) }
        }
    }
}

@_cdecl("mlx_swift_unload_model")
public func unloadModel(modelId: Int32) -> Int32 {
    let removed = registryLock.withLock { modelRegistry.removeValue(forKey: modelId) }
    return removed != nil ? 1 : 0
}

@_cdecl("mlx_swift_generate")
public func generate(
    modelId: Int32,
    prompt: UnsafePointer<CChar>,
    configJson: UnsafePointer<CChar>,
    context: UnsafeMutableRawPointer,
    callback: @convention(c) (UnsafeMutableRawPointer, Bool, UnsafePointer<CChar>?) -> Void
) {
    let promptStr = String(cString: prompt)
    let configObj = parseConfig(String(cString: configJson))

    Task {
        let ctx = registryLock.withLock { modelRegistry[modelId] }

        guard let modelCtx = ctx else {
            "Error: Model ID not found".withCString { callback(context, false, $0) }
            return
        }

        do {
            let model = modelCtx.model
            let tokenizer = modelCtx.tokenizer

            // Encode the raw string using the loaded Tokenizer
            let tokens = tokenizer.encode(text: promptStr)
            let input = LMInput(tokens: MLXArray(tokens))
            let parameters = configObj.toGenerateParameters()
            let kvCache = model.newCache(parameters: parameters)
            let iterator = try TokenIterator(input: input, model: model, cache: kvCache, parameters: parameters)

            let (stream, _) = MLXLMCommon.generateTask(
                promptTokenCount: tokens.count,
                modelConfiguration: modelCtx.configuration,
                tokenizer: tokenizer,
                iterator: iterator
            )

            var fullOutput = ""
            for await item in stream {
                if let chunk = item.chunk { fullOutput += chunk }
            }
            fullOutput.withCString { callback(context, true, $0) }

        } catch {
            error.localizedDescription.withCString { callback(context, false, $0) }
        }
    }
}

@_cdecl("mlx_swift_generate_stream")
public func generateStream(
    modelId: Int32,
    prompt: UnsafePointer<CChar>,
    configJson: UnsafePointer<CChar>,
    context: UnsafeMutableRawPointer,
    callback: @convention(c) (UnsafeMutableRawPointer, UnsafePointer<CChar>?, Bool, UnsafePointer<CChar>?) -> Void
) {
    let promptStr = String(cString: prompt)
    let configObj = parseConfig(String(cString: configJson))
    let chunkSize = configObj.streamChunkSize ?? 4

    Task {
        let ctx = registryLock.withLock { modelRegistry[modelId] }

        guard let modelCtx = ctx else {
            "Error: Model ID not found".withCString { callback(context, nil, true, $0) }
            return
        }

        do {
            let model = modelCtx.model
            let tokenizer = modelCtx.tokenizer

            let tokens = tokenizer.encode(text: promptStr)
            let input = LMInput(tokens: MLXArray(tokens))
            let parameters = configObj.toGenerateParameters()
            let kvCache = model.newCache(parameters: parameters)
            let iterator = try TokenIterator(input: input, model: model, cache: kvCache, parameters: parameters)

            let (stream, _) = MLXLMCommon.generateTask(
                promptTokenCount: tokens.count,
                modelConfiguration: modelCtx.configuration,
                tokenizer: tokenizer,
                iterator: iterator
            )

            var chunkBuffer = ""
            var tokenCount = 0

            for await item in stream {
                if let chunk = item.chunk {
                    chunkBuffer += chunk
                    tokenCount += 1

                    // Flush buffer when it hits chunk size
                    if tokenCount >= chunkSize {
                        chunkBuffer.withCString { cChunk in
                            callback(context, cChunk, false, nil)
                        }
                        chunkBuffer = ""
                        tokenCount = 0
                    }
                }
            }

            // Flush remaining text
            if !chunkBuffer.isEmpty {
                chunkBuffer.withCString { cChunk in
                    callback(context, cChunk, false, nil)
                }
            }

            callback(context, nil, true, nil)

        } catch {
            error.localizedDescription.withCString { callback(context, nil, true, $0) }
        }
    }
}
