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

// --- Native Containers ---
private class ModelContainer {
    let context: ModelContext
    init(_ context: ModelContext) { self.context = context }
}

private class CacheContainer {
    var caches: [KVCache]
    init(_ caches: [KVCache]) { self.caches = caches }
}

private class TaskContainer {
    var task: Task<Void, Never>?
    init() {}
    deinit {
        // Ultimate safety: If V8 GCs the stream handle before it finishes, we kill the work.
        task?.cancel()
    }
}

// --- 3. Bridge Logic ---

public typealias BridgeAsyncCallback = @convention(c) (UnsafeMutableRawPointer, Bool, UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Void

@_cdecl("bridge_metal_load")
public func loadMetal() {
    let a = MLXArray(0.0)
    eval(a + a)
}

@_cdecl("bridge_metal_clear_cache")
public func bridge_metal_clear_cache() {
    // This tells the MLX GPU backend to release all unused buffers back to macOS
    // MLX.GPU.clearCache() @deprected
    Memory.clearCache()
}

@_cdecl("bridge_model_load")
public func loadModel(path: UnsafePointer<CChar>, context: UnsafeMutableRawPointer, callback: BridgeAsyncCallback) {
    let url = URL(fileURLWithPath: String(cString: path))
    Task {
        do {
            let modelCtx = try await LLMModelFactory.shared.load(from: url, using: NullTokenizerLoader())
            let container = ModelContainer(modelCtx)
            let ptr = Unmanaged.passRetained(container).toOpaque()
            callback(context, true, ptr, nil)
        } catch {
            error.localizedDescription.withCString { callback(context, false, nil, $0) }
        }
    }
}

@_cdecl("bridge_model_free")
public func bridge_model_free(ptr: UnsafeMutableRawPointer) {
    Unmanaged<ModelContainer>.fromOpaque(ptr).release()
}

// --- Cache API ---

@_cdecl("bridge_cache_create")
public func bridge_cache_create(modelPtr: UnsafeMutableRawPointer, configJson: UnsafePointer<CChar>) -> UnsafeMutableRawPointer {
    let container = Unmanaged<ModelContainer>.fromOpaque(modelPtr).takeUnretainedValue()
    let configObj = parseConfig(String(cString: configJson))
    let caches = container.context.model.newCache(parameters: configObj.toGenerateParameters())
    let cacheContainer = CacheContainer(caches)
    return Unmanaged.passRetained(cacheContainer).toOpaque()
}

@_cdecl("bridge_cache_free")
public func bridge_cache_free(ptr: UnsafeMutableRawPointer) {
    Unmanaged<CacheContainer>.fromOpaque(ptr).release()
}

@_cdecl("bridge_cache_clone")
public func bridge_cache_clone(ptr: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer {
    let container = Unmanaged<CacheContainer>.fromOpaque(ptr).takeUnretainedValue()
    let clonedCaches = container.caches.map { $0.copy() }

    // CRITICAL FIX: Force Metal to physically realize the copied arrays in VRAM.
    // This prevents "cannot generate on cloned cache" lazy-evaluation crashes across threads.
    for cache in clonedCaches {
        if !cache.state.isEmpty {
            eval(cache.state)
        }
    }

    let newContainer = CacheContainer(clonedCaches)
    return Unmanaged.passRetained(newContainer).toOpaque()
}

@_cdecl("bridge_cache_save")
public func bridge_cache_save(ptr: UnsafeMutableRawPointer, path: UnsafePointer<CChar>, context: UnsafeMutableRawPointer, callback: BridgeAsyncCallback) {
    let container = Unmanaged<CacheContainer>.fromOpaque(ptr).takeUnretainedValue()
    let url = URL(fileURLWithPath: String(cString: path))
    Task {
        do {
            try MLXLMCommon.savePromptCache(url: url, cache: container.caches)
            callback(context, true, nil, nil) // Success, no ptr needed
        } catch {
            error.localizedDescription.withCString { callback(context, false, nil, $0) }
        }
    }
}

@_cdecl("bridge_cache_load")
public func bridge_cache_load(path: UnsafePointer<CChar>, context: UnsafeMutableRawPointer, callback: BridgeAsyncCallback) {
    let url = URL(fileURLWithPath: String(cString: path))
    Task {
        do {
            let (caches, _) = try MLXLMCommon.loadPromptCache(url: url)
            let container = CacheContainer(caches)
            let ptr = Unmanaged.passRetained(container).toOpaque()
            callback(context, true, ptr, nil)
        } catch {
            error.localizedDescription.withCString { callback(context, false, nil, $0) }
        }
    }
}


@_cdecl("bridge_cache_trim")
public func bridge_cache_trim(ptr: UnsafeMutableRawPointer, numTokens: Int32) -> Int32 {
    let container = Unmanaged<CacheContainer>.fromOpaque(ptr).takeUnretainedValue()
    let trimmed = MLXLMCommon.trimPromptCache(container.caches, numTokens: Int(numTokens))
    return Int32(trimmed)
}

// --- Metrics ---
private struct BridgeMetrics: Encodable {
    let active: Int
    let cache: Int
    let peak: Int
    let memoryLimit: Int
    let cacheLimit: Int
}

@_cdecl("bridge_metal_metrics")
public func bridge_metrics() -> UnsafeMutablePointer<CChar>? {
    let snapshot = Memory.snapshot()
    let metrics = BridgeMetrics(active: snapshot.activeMemory,cache: snapshot.cacheMemory,peak: snapshot.peakMemory,memoryLimit: Memory.memoryLimit, cacheLimit: Memory.cacheLimit)

    guard let jsonData = try? JSONEncoder().encode(metrics),
          let jsonStr = String(data: jsonData, encoding: .utf8) else { return nil }
    return strdup(jsonStr)
}

private struct GenerateStats: Encodable {
    let promptTokens: Int
    let generatedTokens: Int
    let promptTime: Double
    let generateTime: Double
    let promptTokensPerSecond: Double
    let tokensPerSecond: Double
    let stopReason: String
}

private struct EvaluateStats: Encodable {
    let promptTokens: Int
    let promptTime: Double
    let promptTokensPerSecond: Double
}

@_cdecl("bridge_model_evaluate_task")
public func bridge_model_evaluate_task(
    modelPtr: UnsafeMutableRawPointer,
    cachePtr: UnsafeMutableRawPointer?,
    promptTokens: UnsafePointer<Int32>,
    promptLength: Int32,
    configJson: UnsafePointer<CChar>,
    context: UnsafeMutableRawPointer,
    callback: BridgeAsyncCallback
) -> UnsafeMutableRawPointer {

    let container = Unmanaged<ModelContainer>.fromOpaque(modelPtr).takeUnretainedValue()
    var cacheContainer: CacheContainer? = nil
    var localCache: [KVCache]? = nil

    if let ptr = cachePtr {
        cacheContainer = Unmanaged<CacheContainer>.fromOpaque(ptr).takeUnretainedValue()
        localCache = cacheContainer?.caches
    }

    let buffer = UnsafeBufferPointer(start: promptTokens, count: Int(promptLength))
    let tokens = Array(buffer).map { Int($0) }
    let configObj = parseConfig(String(cString: configJson))

    let taskContainer = TaskContainer()

    taskContainer.task = Task { [container, cacheContainer, localCache] in
        var caches = localCache

        do {
            let input = LMInput(tokens: MLXArray(tokens))
            let parameters = configObj.toGenerateParameters()

            if caches == nil {
                caches = container.context.model.newCache(parameters: parameters)
            } else {
                maybeQuantizeKVCache(
                    cache: &caches!,
                    kvBits: configObj.kvBits,
                    kvGroupSize: configObj.kvGroupSize ?? 64,
                    quantizedKVStart: configObj.quantizedKVStart ?? -1 // Force -1
                )
                if cacheContainer != nil {
                    cacheContainer!.caches = caches!
                }
            }

            let startTime = Date()

            switch try container.context.model.prepare(input, cache: caches!, windowSize: parameters.prefillStepSize) {
            case .tokens(let outTokens):
                let result = container.context.model(
                    outTokens[text: .newAxis],
                    cache: caches!.isEmpty ? nil : caches!,
                    state: nil
                )
                eval(result.logits)

            case .logits(let result):
                eval(result.logits)
            }

            if Task.isCancelled {
                "Evaluation Cancelled".withCString { callback(context, false, nil, $0) }
                return
            }

            let duration = Date().timeIntervalSince(startTime)
            let stats = EvaluateStats(
                promptTokens: tokens.count,
                promptTime: duration,
                promptTokensPerSecond: Double(tokens.count) / duration
            )

            if let jsonData = try? JSONEncoder().encode(stats),
               let jsonStr = String(data: jsonData, encoding: .utf8) {
                jsonStr.withCString { callback(context, true, nil, $0) }
            } else {
                callback(context, true, nil, nil)
            }
        } catch {
            error.localizedDescription.withCString { callback(context, false, nil, $0) }
        }
    }
    return Unmanaged.passRetained(taskContainer).toOpaque()
}

@_cdecl("bridge_model_generate_task")
public func bridge_model_generate_task(
    modelPtr: UnsafeMutableRawPointer,
    cachePtr: UnsafeMutableRawPointer?,
    promptTokens: UnsafePointer<Int32>,
    promptLength: Int32,
    configJson: UnsafePointer<CChar>,
    context: UnsafeMutableRawPointer,
    callback: @convention(c) (UnsafeMutableRawPointer, UnsafePointer<Int32>?, Int32, Bool, Bool, UnsafePointer<CChar>?) -> Void
) -> UnsafeMutableRawPointer {

    let buffer = UnsafeBufferPointer(start: promptTokens, count: Int(promptLength))
    let tokens = Array(buffer).map { Int($0) }

    let configObj = parseConfig(String(cString: configJson))
    let chunkSize = Int(configObj.chunkSize ?? 5)

    let container = Unmanaged<ModelContainer>.fromOpaque(modelPtr).takeUnretainedValue()

    var cacheContainer: CacheContainer? = nil
    var localCache: [KVCache]? = nil
    if let ptr = cachePtr {
        cacheContainer = Unmanaged<CacheContainer>.fromOpaque(ptr).takeUnretainedValue()
        localCache = cacheContainer?.caches
    }

    let taskContainer = TaskContainer()

    let task = Task { [container, cacheContainer, localCache] in
        var caches = localCache

        // CRITICAL FIX: Pre-quantize BEFORE passing to TokenIterator.
        // The TokenIterator sees they are already quantized and mutates our stable references.
        if caches != nil {
            maybeQuantizeKVCache(
                cache: &caches!,
                kvBits: configObj.kvBits,
                kvGroupSize: configObj.kvGroupSize ?? 64,
                quantizedKVStart: configObj.quantizedKVStart ?? -1 // Force -1
            )
            cacheContainer?.caches = caches!
        }

        var tokenBuffer = [Int32]()
        tokenBuffer.reserveCapacity(chunkSize)

        var finalStats: GenerateCompletionInfo? = nil
        var finalErrorStr: String? = nil
        var wasCancelled = false

        do {
            let input = LMInput(tokens: MLXArray(tokens))

            let (stream, _) = try MLXLMCommon.generateTokensTask(
                input: input,
                cache: caches,
                parameters: configObj.toGenerateParameters(),
                context: container.context,
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
                        tokenBuffer.withUnsafeBufferPointer { ptr in
                            callback(context, ptr.baseAddress, Int32(tokenBuffer.count), false, false, nil)
                        }
                        tokenBuffer.removeAll(keepingCapacity: true)
                    }

                case .info(let stats):
                    finalStats = stats
                }
            }

            if Task.isCancelled { wasCancelled = true }

        } catch {
            wasCancelled = error is CancellationError
            if !wasCancelled { finalErrorStr = error.localizedDescription }
        }

        if !tokenBuffer.isEmpty {
            tokenBuffer.withUnsafeBufferPointer { ptr in
                callback(context, ptr.baseAddress, Int32(tokenBuffer.count), false, false, nil)
            }
        }

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

    taskContainer.task = task
    return Unmanaged.passRetained(taskContainer).toOpaque()
}

@_cdecl("bridge_model_abort_task")
public func bridge_model_abort_task(ptr: UnsafeMutableRawPointer) {
    let container = Unmanaged<TaskContainer>.fromOpaque(ptr).takeUnretainedValue()
    container.task?.cancel()
}

@_cdecl("bridge_model_free_task")
public func bridge_model_free_task(ptr: UnsafeMutableRawPointer) {
    Unmanaged<TaskContainer>.fromOpaque(ptr).release()
}
