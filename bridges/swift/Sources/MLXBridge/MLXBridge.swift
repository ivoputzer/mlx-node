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
    var chunkSize: Int?

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
private var activeTasks: [UnsafeMutableRawPointer: Task<Void, Never>] = [:]
private let registryLock = NSLock()


private class ModelContainer {
    let context: ModelContext
    init(_ context: ModelContext) {
        self.context = context
    }
}

// --- 3. Bridge Logic ---

@_cdecl("bridge_metal_load")
public func loadMetal() {
    let a = MLXArray(0.0)
    eval(a + a)
}

@_cdecl("bridge_model_load")
public func loadModel(path: UnsafePointer<CChar>, context: UnsafeMutableRawPointer, callback: @convention(c) (UnsafeMutableRawPointer, Bool, UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Void) {
    let url = URL(fileURLWithPath: String(cString: path))
    Task {
        do {
            let modelCtx = try await LLMModelFactory.shared.load(from: url, using: NullTokenizerLoader())

            // WRAP the struct in our class
            let container = ModelContainer(modelCtx)

            // Now we can use Unmanaged because ModelContainer is a Class
            let ptr = Unmanaged.passRetained(container).toOpaque()

            callback(context, true, ptr, nil)
        } catch {
            error.localizedDescription.withCString { callback(context, false, nil, $0) }
        }
    }
}

@_cdecl("bridge_model_free")
public func bridge_model_free(ptr: UnsafeMutableRawPointer) {
    // Release the Container class
    Unmanaged<ModelContainer>.fromOpaque(ptr).release()
}

// @_cdecl("bridge_model_unload")
// public func unloadModel(modelId: Int32) -> Int32 {
//     cancelGenerate(modelId: modelId) // Ensure we stop processing before unloading
//     let removed = registryLock.withLock { modelRegistry.removeValue(forKey: modelId) }
//     return removed != nil ? 1 : 0
// }

@_cdecl("bridge_generate_abort")
public func cancelGenerate(modelPtr: UnsafeMutableRawPointer) {
    registryLock.withLock {
        activeTasks[modelPtr]?.cancel()
        // activeTasks.removeValue(forKey: modelPtr)
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

// Idiomatic, type-safe struct for high-speed JSON serialization
private struct GenerateStats: Encodable {
    let promptTokens: Int
    let generatedTokens: Int
    let promptTime: Double
    let generateTime: Double
    let promptTokensPerSecond: Double
    let tokensPerSecond: Double
    let stopReason: String
}

@_cdecl("bridge_generate_stream")
public func generateStream(
    modelPtr: UnsafeMutableRawPointer,
    promptTokens: UnsafePointer<Int32>,
    promptLength: Int32,
    configJson: UnsafePointer<CChar>,
    context: UnsafeMutableRawPointer,
    callback: @convention(c) (UnsafeMutableRawPointer, UnsafePointer<Int32>?, Int32, Bool, Bool, UnsafePointer<CChar>?) -> Void
) {
    // Zero-copy array mapping from UnsafeBufferPointer
    let buffer = UnsafeBufferPointer(start: promptTokens, count: Int(promptLength))
    let tokens = Array(buffer).map { Int($0) }

    let configObj = parseConfig(String(cString: configJson))
    let chunkSize = Int(configObj.chunkSize ?? 5) // Ensure Int for comparisons

    let container = Unmanaged<ModelContainer>.fromOpaque(modelPtr).takeUnretainedValue()
    let modelCtx = container.context

    let task = Task {
        // DEFER: Guarantees cleanup executing immediately before the Task ends,
        // no matter how the Task exits (success, throw, or cancellation).

        defer {
            _ = registryLock.withLock { activeTasks.removeValue(forKey: modelPtr) }
        }

        // Pre-allocate buffer to prevent repeated heap allocations
        var tokenBuffer = [Int32]()
        tokenBuffer.reserveCapacity(chunkSize)

        var finalStats: GenerateCompletionInfo? = nil
        var finalErrorStr: String? = nil
        var wasCancelled = false

        do {
            let input = LMInput(tokens: MLXArray(tokens))
            let parameters = configObj.toGenerateParameters()

            let (stream, _) = try MLXLMCommon.generateTokensTask(
                input: input,
                parameters: parameters,
                context: modelCtx,
                includeStopToken: true
            )

            for await event in stream {
                if Task.isCancelled {
                    wasCancelled = true
                    break
                }

                switch event {
                case .token(let tokenId):
                    tokenBuffer.append(Int32(tokenId))

                    if tokenBuffer.count >= chunkSize {
                        // Pointer is valid ONLY inside this block, which is safe because
                        // our C code deep-copies it synchronously into the Arena block.
                        tokenBuffer.withUnsafeBufferPointer { ptr in
                            callback(context, ptr.baseAddress, Int32(tokenBuffer.count), false, false, nil)
                        }
                        // Keeps allocated heap block intact
                        tokenBuffer.removeAll(keepingCapacity: true)
                    }

                case .info(let stats):
                    finalStats = stats
                }
            }

            // Re-assert in case MLX quietly swallowed cancellation
            if Task.isCancelled { wasCancelled = true }

        } catch {
            wasCancelled = error is CancellationError
            if !wasCancelled {
                finalErrorStr = error.localizedDescription
            }
        }

        // --- 1. Flush any remaining tokens ---
        if !tokenBuffer.isEmpty {
            tokenBuffer.withUnsafeBufferPointer { ptr in
                callback(context, ptr.baseAddress, Int32(tokenBuffer.count), false, false, nil)
            }
        }

        // --- 2. FIRE GUARANTEED TERMINAL CALLBACK ---
        if wasCancelled {
            "Generation Cancelled".withCString { cStr in
                callback(context, nil, 0, true, true, cStr)
            }
        } else if let errorStr = finalErrorStr {
            errorStr.withCString { cStr in
                callback(context, nil, 0, true, true, cStr)
            }
        } else if let stats = finalStats {
            let statsStruct = GenerateStats(
                promptTokens: stats.promptTokenCount,
                generatedTokens: stats.generationTokenCount,
                promptTime: stats.promptTime,
                generateTime: stats.generateTime,
                promptTokensPerSecond: stats.promptTokensPerSecond,
                tokensPerSecond: stats.tokensPerSecond,
                stopReason: String(describing: stats.stopReason)
            )

            // Native JSON serialization using Encodable
            if let jsonData = try? JSONEncoder().encode(statsStruct),
               let jsonStr = String(data: jsonData, encoding: .utf8) {
                jsonStr.withCString { cStr in
                    callback(context, nil, 0, true, false, cStr)
                }
            } else {
                callback(context, nil, 0, true, false, nil)
            }
        } else {
            callback(context, nil, 0, true, false, nil)
        }
    }

    registryLock.withLock { activeTasks[modelPtr] = task }
}
