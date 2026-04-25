import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// --- 1. Expanded Configuration Parsing ---
struct BridgeGenerateConfig: Decodable {
    var maxTokens: Int?
    var maxKVSize: Int?
    var kvBits: Int?
    var kvGroupSize: Int?
    var quantizedKVStart: Int?
    var temperature: Float?
    var topP: Float?
    var topK: Int?
    var minP: Float?
    var repetitionPenalty: Float?
    var repetitionContextSize: Int?
    var presencePenalty: Float?
    var presenceContextSize: Int?
    var frequencyPenalty: Float?
    var frequencyContextSize: Int?
    var prefillStepSize: Int?
    var streamChunkSize: Int?

    func toGenerateParameters() -> GenerateParameters {
        return GenerateParameters(
            maxTokens: maxTokens,
            maxKVSize: maxKVSize,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize ?? 64,
            quantizedKVStart: quantizedKVStart ?? 0,
            temperature: temperature ?? 0.6,
            topP: topP ?? 1.0,
            topK: topK ?? 0,
            minP: minP ?? 0.0,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize ?? 20,
            presencePenalty: presencePenalty,
            presenceContextSize: presenceContextSize ?? 20,
            frequencyPenalty: frequencyPenalty,
            frequencyContextSize: frequencyContextSize ?? 20,
            prefillStepSize: prefillStepSize ?? 512
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

// --- 2. The Null Tokenizer ---
struct NullTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { return [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { return "" }
    func convertTokenToId(_ token: String) -> Int? { return nil }
    func convertIdToToken(_ id: Int) -> String? { return nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] { return [] }
}

struct NullTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        return NullTokenizer()
    }
}

// --- Global Registries & Locks ---
private var modelRegistry: [Int32: ModelContext] = [:]
private var activeTasks: [Int32: Task<Void, Never>] = [:] // Track running generations
private var nextModelId: Int32 = 1
private let registryLock = NSLock()


// --- 3. Bridge Logic ---

@_cdecl("mlx_swift_init_metal")
public func mlxSwiftInitMetal() {
    let a = MLXArray(0.0)
    let b = a + a
    eval(b)
}

@_cdecl("mlx_swift_load_model")
public func loadModel(path: UnsafePointer<CChar>, context: UnsafeMutableRawPointer, callback: @convention(c) (UnsafeMutableRawPointer, Bool, Int32, UnsafePointer<CChar>?) -> Void) {
    let url = URL(fileURLWithPath: String(cString: path))
    Task {
        do {
            let modelCtx = try await LLMModelFactory.shared.load(from: url, using: NullTokenizerLoader())
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
    cancelGenerate(modelId: modelId) // Ensure we stop processing before unloading
    let removed = registryLock.withLock { modelRegistry.removeValue(forKey: modelId) }
    return removed != nil ? 1 : 0
}

@_cdecl("mlx_swift_cancel_generate")
public func cancelGenerate(modelId: Int32) {
    registryLock.withLock {
        activeTasks[modelId]?.cancel()
        activeTasks.removeValue(forKey: modelId)
    }
}

// Metrics (Synchronous)
@_cdecl("bridge_metrics")
public func bridge_metrics() -> UnsafeMutablePointer<CChar>? {

    let snapshot = Memory.snapshot() // returns a Codable struct containing activeMemory, peakMemory, and cacheMemory

    guard let jsonData = try? JSONEncoder().encode(snapshot),
          let jsonStr = String(data: jsonData, encoding: .utf8) else {
        return nil
    }

    return strdup(jsonStr)
}

@_cdecl("mlx_swift_generate_stream")
public func generateStream(
    modelId: Int32,
    promptTokens: UnsafePointer<Int32>,
    promptLength: Int32,
    configJson: UnsafePointer<CChar>,
    context: UnsafeMutableRawPointer,
    callback: @convention(c) (UnsafeMutableRawPointer, UnsafePointer<Int32>?, Int32, Bool, Bool, UnsafePointer<CChar>?) -> Void
) {
    let buffer = UnsafeBufferPointer(start: promptTokens, count: Int(promptLength))
    let tokens = Array(buffer).map { Int($0) }

    let configObj = parseConfig(String(cString: configJson))
    let chunkSize = configObj.streamChunkSize ?? 5

    let task = Task {
        guard let modelCtx = registryLock.withLock({ modelRegistry[modelId] }) else {
            "Error: Model ID not found".withCString { cStr in
                callback(context, nil, 0, true, true, cStr)
            }
            return
        }

        do {
            let input = LMInput(tokens: MLXArray(tokens))
            let parameters = configObj.toGenerateParameters()

            let (stream, _) = try MLXLMCommon.generateTokensTask(
                input: input,
                parameters: parameters,
                context: modelCtx,
                includeStopToken: true
            )

            var tokenBuffer = [Int32]()
            tokenBuffer.reserveCapacity(chunkSize)

            var finalStats: GenerateCompletionInfo? = nil
            var wasCancelled = false

            for await event in stream {
                if Task.isCancelled {
                    wasCancelled = true
                    break
                }

                switch event {
                case .token(let tokenId):
                    tokenBuffer.append(Int32(tokenId))

                    if tokenBuffer.count >= chunkSize {
                        tokenBuffer.withUnsafeBufferPointer { ptr in
                            callback(context, ptr.baseAddress, Int32(tokenBuffer.count), false, false, nil)
                        }
                        tokenBuffer.removeAll(keepingCapacity: true)
                    }

                case .info(let stats):
                    finalStats = stats
                }
            }

            // --- GUARANTEED TERMINAL BLOCK ---

            // 1. Flush any remaining tokens
            if !tokenBuffer.isEmpty {
                tokenBuffer.withUnsafeBufferPointer { ptr in
                    callback(context, ptr.baseAddress, Int32(tokenBuffer.count), false, false, nil)
                }
            }

            // 2. Fire EXACTLY ONE terminal callback
            if wasCancelled {
                "Generation Cancelled".withCString { cStr in
                    callback(context, nil, 0, true, true, cStr)
                }
            } else if let stats = finalStats {
                let statsDict: [String: Any] = [
                    "promptTokens": stats.promptTokenCount,
                    "generatedTokens": stats.generationTokenCount,
                    "promptTime": stats.promptTime,
                    "generateTime": stats.generateTime,
                    "promptTokensPerSecond": stats.promptTokensPerSecond,
                    "tokensPerSecond": stats.tokensPerSecond,
                    "stopReason": String(describing: stats.stopReason)
                ]

                if let jsonData = try? JSONSerialization.data(withJSONObject: statsDict),
                   let jsonStr = String(data: jsonData, encoding: .utf8) {
                    jsonStr.withCString { cStr in
                        callback(context, nil, 0, true, false, cStr)
                    }
                } else {
                    callback(context, nil, 0, true, false, nil)
                }
            } else {
                // Failsafe if stream ended with no stats
                callback(context, nil, 0, true, false, nil)
            }

            // Cleanup
            _ = registryLock.withLock { activeTasks.removeValue(forKey: modelId) }

        } catch {
            _ = registryLock.withLock { activeTasks.removeValue(forKey: modelId) }
            error.localizedDescription.withCString { cStr in
                callback(context, nil, 0, true, true, cStr)
            }
        }
    }

    registryLock.withLock { activeTasks[modelId] = task }
}
