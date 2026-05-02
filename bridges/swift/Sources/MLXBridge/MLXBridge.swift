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
    var kvBits: Int?
    var kvGroupSize: Int?
    var quantizedKVStart: Int?
    init(_ caches: [KVCache], kvBits: Int? = nil, kvGroupSize: Int? = nil, quantizedKVStart: Int? = nil) {
        self.caches = caches
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
    }
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
    let cacheContainer = CacheContainer(caches, kvBits: configObj.kvBits, kvGroupSize: configObj.kvGroupSize, quantizedKVStart: configObj.quantizedKVStart)
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

    let newContainer = CacheContainer(clonedCaches, kvBits: container.kvBits, kvGroupSize: container.kvGroupSize, quantizedKVStart: container.quantizedKVStart)
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
    let trimmed = MLXLMCommon.trimPromptCache(container.caches, numTokens: Int(numTokens)) // Explicitly use Int32 for the boundary to match C's int32_t exactly
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
            let activeKvBits = configObj.kvBits ?? cacheContainer?.kvBits
            let activeGroupSize = configObj.kvGroupSize ?? cacheContainer?.kvGroupSize ?? 64
            let activeStart = configObj.quantizedKVStart ?? cacheContainer?.quantizedKVStart ?? 0

            var parameters = configObj.toGenerateParameters()
            parameters.kvBits = activeKvBits
            parameters.kvGroupSize = activeGroupSize
            parameters.quantizedKVStart = activeStart

            let input = LMInput(tokens: MLXArray(tokens))

            if caches == nil {
                caches = container.context.model.newCache(parameters: parameters)
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

            // Manually quantize safely AFTER prefill
            maybeQuantizeKVCache(
                cache: &caches!,
                kvBits: activeKvBits,
                kvGroupSize: activeGroupSize,
                quantizedKVStart: activeStart
            )

            cacheContainer?.caches = caches!

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

        let activeKvBits = configObj.kvBits ?? cacheContainer?.kvBits
        let activeGroupSize = configObj.kvGroupSize ?? cacheContainer?.kvGroupSize ?? 64
        let activeStart = configObj.quantizedKVStart ?? cacheContainer?.quantizedKVStart ?? 0

        var params = configObj.toGenerateParameters()
        params.kvBits = activeKvBits
        params.kvGroupSize = activeGroupSize
        params.quantizedKVStart = activeStart

        var tokenBuffer = [Int32]()
        tokenBuffer.reserveCapacity(chunkSize)

        var finalStats: GenerateStats? = nil
        var finalErrorStr: String? = nil
        var wasCancelled = false

        do {
            let input = LMInput(tokens: MLXArray(tokens))

            if caches == nil {
                caches = container.context.model.newCache(parameters: params)
            }

            var processor = params.processor()
            let sampler = params.sampler()
            var y: LMInput.Text
            var state: LMOutput.State? = nil

            func convertToToken(logits: MLXArray, processor: inout LogitProcessor?, sampler: LogitSampler) -> MLXArray {
                var l = logits[0..., -1, 0...]
                l = processor?.process(logits: l) ?? l
                let t = sampler.sample(logits: l)
                processor?.didSample(token: t)
                return t
            }

            let prefillStart = Date.timeIntervalSinceReferenceDate

            processor?.prompt(input.text.tokens)

            switch try container.context.model.prepare(input, cache: caches!, windowSize: params.prefillStepSize) {
              case .tokens(let outTokens):
                let result = container.context.model(outTokens[text: .newAxis], cache: caches!.isEmpty ? nil : caches!, state: state)
                state = result.state
                maybeQuantizeKVCache(cache: &caches!, kvBits: activeKvBits, kvGroupSize: activeGroupSize, quantizedKVStart: activeStart)
                let token = convertToToken(logits: result.logits, processor: &processor, sampler: sampler)
                y = .init(tokens: token)
                MLX.asyncEval(y.tokens)
              case .logits(let result):
                state = result.state
                let token = convertToToken(logits: result.logits, processor: &processor, sampler: sampler)
                y = .init(tokens: token)
                MLX.asyncEval(y.tokens)
            }

            let promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var tokenCount = 0

            var stopTokenIds = container.context.configuration.eosTokenIds

            if let tokenizerEOS = container.context.tokenizer.eosTokenId { stopTokenIds.insert(tokenizerEOS) }
            for token in container.context.configuration.extraEOSTokens {
                if let id = container.context.tokenizer.convertTokenToId(token) { stopTokenIds.insert(id) }
            }

            let unknownTokenId = container.context.tokenizer.unknownTokenId ?? -1
            var stopReason = "stop"

            while true {
                if Task.isCancelled {
                    wasCancelled = true
                    stopReason = "cancelled"
                    break
                }

                // CONTINUOUS SYNC: Ensure the JS container ALWAYS holds the exact array instance currently in VRAM
                cacheContainer?.caches = caches!

                let currentTokenId = y.tokens.item(Int.self)

                if promptTime == 0 {
                    promptTime = Date.timeIntervalSinceReferenceDate - start
                    start = Date.timeIntervalSinceReferenceDate
                }

                tokenBuffer.append(Int32(currentTokenId))
                tokenCount += 1

                if tokenBuffer.count >= chunkSize {
                    tokenBuffer.withUnsafeBufferPointer { ptr in
                        callback(context, ptr.baseAddress, Int32(tokenBuffer.count), false, false, nil)
                    }
                    tokenBuffer.removeAll(keepingCapacity: true)
                }

                if currentTokenId == unknownTokenId || stopTokenIds.contains(currentTokenId) {
                    break
                }

                if let maxT = params.maxTokens, tokenCount >= maxT {
                    stopReason = "length"
                    break
                }

                let result = container.context.model(y[text: .newAxis], cache: caches!.isEmpty ? nil : caches!, state: state)

                state = result.state
                maybeQuantizeKVCache(cache: &caches!, kvBits: activeKvBits, kvGroupSize: activeGroupSize, quantizedKVStart: activeStart)

                let nextToken = convertToToken(logits: result.logits, processor: &processor, sampler: sampler)
                y = .init(tokens: nextToken)

                MLX.asyncEval(y.tokens)
            }

            cacheContainer?.caches = caches!

            if Task.isCancelled { wasCancelled = true }

            Stream().synchronize()

            let generateTime = Date.timeIntervalSinceReferenceDate - start
            let finalPromptTime = promptTime + promptPrefillTime

            finalStats = GenerateStats(
                promptTokens: tokens.count,
                generatedTokens: tokenCount,
                promptTime: finalPromptTime,
                generateTime: generateTime,
                promptTokensPerSecond: finalPromptTime > 0 ? Double(tokens.count) / finalPromptTime : 0.0,
                tokensPerSecond: generateTime > 0 ? Double(tokenCount) / generateTime : 0.0,
                stopReason: stopReason
            )

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
            if let jsonData = try? JSONEncoder().encode(stats),
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

@_cdecl("bridge_cache_debug")
public func bridge_cache_debug(ptr: UnsafeMutableRawPointer) -> UnsafeMutablePointer<CChar>? {
    let container = Unmanaged<CacheContainer>.fromOpaque(ptr).takeUnretainedValue()
    guard let first = container.caches.first else { return strdup("{}") }

    let info = """
    {
        "layers": \(container.caches.count),
        "type": "\(type(of: first))",
        "offset": \(first.offset),
        "isTrimmable": \(first.isTrimmable)
    }
    """
    return strdup(info)
}
