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

    var batchSize: Int?
    var chunkSize: Int?
    var stopTokenIds: [Int]?

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

private func compileStopTokens(context: ModelContext, config: BridgeGenerateConfig) -> Set<Int> {
    var stopTokenIds = context.configuration.eosTokenIds
    if let tokenizerEOS = context.tokenizer.eosTokenId { stopTokenIds.insert(tokenizerEOS) }
    if let unknownTokenId = context.tokenizer.unknownTokenId { stopTokenIds.insert(unknownTokenId) }
    for token in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(token) { stopTokenIds.insert(id) }
    }
    if let customStopIds = config.stopTokenIds {
        stopTokenIds.formUnion(customStopIds)
    }
    return stopTokenIds
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
    deinit { task?.cancel() }
}

// --- 3. Shared Cache Utilities ---

private func expandCache(_ caches: inout [KVCache], batchSize: Int) {
    guard batchSize > 1 else { return }

    var prefillStates: [MLXArray] = []
    for cache in caches {
        prefillStates.append(contentsOf: cache.state)
    }
    eval(prefillStates)

    var postBroadcastStates: [MLXArray] = []
    for i in 0..<caches.count {
        // SAFETY: Only duplicate if the state actually has data
        let duplicated = caches[i].state.map { array in
            array.size > 0 ? MLX.concatenated(Array(repeating: array, count: batchSize), axis: 0) : array
        }
        caches[i].state = duplicated
        postBroadcastStates.append(contentsOf: duplicated)
    }
    eval(postBroadcastStates)
}

// --- 4. Bridge Logic ---

public typealias BridgeAsyncCallback = @convention(c) (UnsafeMutableRawPointer, Bool, UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Void

@_cdecl("bridge_metal_load")
public func loadMetal() {
    let a = MLXArray(0.0)
    eval(a + a)
}

@_cdecl("bridge_metal_clear_cache")
public func bridge_metal_clear_cache() { Memory.clearCache() }

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
    for cache in clonedCaches {
        if !cache.state.isEmpty { eval(cache.state) }
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
            callback(context, true, nil, nil)
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

@_cdecl("bridge_cache_slice")
public func bridge_cache_slice(ptr: UnsafeMutableRawPointer, start: Int32, end: Int32) -> UnsafeMutableRawPointer {
    let container = Unmanaged<CacheContainer>.fromOpaque(ptr).takeUnretainedValue()

    let slicedCaches = container.caches.map { cache -> KVCache in
        var newCache = cache.copy() // FIX: Changed 'let' to 'var'

        newCache.state = newCache.state.map { array in
            guard array.size > 0, array.shape[0] > 1 else { return array } // Protect B=1
            let actualEnd = min(Int(end), array.shape[0])

            // FIX: Use .take() instead of MLX.slice
            let indices = MLXArray((Int32(start)..<Int32(actualEnd)).map { $0 })
            return array.take(indices, axis: 0)
        }
        return newCache
    }

    let newContainer = CacheContainer(slicedCaches, kvBits: container.kvBits, kvGroupSize: container.kvGroupSize, quantizedKVStart: container.quantizedKVStart)
    return Unmanaged.passRetained(newContainer).toOpaque()
}

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

            let bSize = Int(configObj.batchSize ?? 1)
            let singleBatchTokens = MLXArray(tokens).reshaped(1, tokens.count)
            let input = LMInput(tokens: singleBatchTokens)

            if caches == nil {
                caches = container.context.model.newCache(parameters: parameters)
            }

            let startTime = Date()

            switch try container.context.model.prepare(input, cache: caches!, windowSize: parameters.prefillStepSize) {
            case .tokens(let outTokens):
                // SAFETY GUARD: Never pass an empty array to the model graph!
                if outTokens.tokens.size > 0 {
                    let result = container.context.model(
                        outTokens,
                        cache: caches!.isEmpty ? nil : caches!,
                        state: nil
                    )
                    eval(result.logits)
                } else {
                    // Fully consumed by chunking
                    eval(caches!.map { $0.state })
                }

            case .logits(let result):
                eval(result.logits)
            }

            // Centralized Quantization (Done ONCE, efficiently!)
            maybeQuantizeKVCache(cache: &caches!, kvBits: activeKvBits, kvGroupSize: activeGroupSize, quantizedKVStart: activeStart)

            expandCache(&caches!, batchSize: bSize)
            cacheContainer?.caches = caches!

            if Task.isCancelled {
                "Evaluation Cancelled".withCString { callback(context, false, nil, $0) }
                return
            }

            let duration = Date().timeIntervalSince(startTime)
            let stats = EvaluateStats(promptTokens: tokens.count, promptTime: duration, promptTokensPerSecond: Double(tokens.count) / duration)

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
    let requestedBSize = Int(configObj.batchSize ?? 1)

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

        var finalStats: GenerateStats? = nil
        var finalErrorStr: String? = nil
        var wasCancelled = false

        do {
            // 1. DETERMINE CACHE STATE & BATCH SIZES
            let isFreshCache = caches == nil || (caches!.first?.state.isEmpty ?? true)
            let currentCacheBSize = isFreshCache ? 1 : (caches!.first?.state.first?.shape[0] ?? 1)

            // If the cache is already B>1, we must lock the requested batch size to match to prevent QKV crashes.
            let activeBSize = (!isFreshCache && currentCacheBSize > 1) ? currentCacheBSize : requestedBSize

            let bufferCapacity = chunkSize * activeBSize
            var tokenBuffer = [Int32]()
            tokenBuffer.reserveCapacity(bufferCapacity)

            // 2. CACHE PRE-EXPANSION (Turn 2 Branching)
            // If the user passes a B=1 cache but wants to branch out (B>1), we must expand the cache BEFORE prefill!
            if !isFreshCache && currentCacheBSize == 1 && activeBSize > 1 {
                expandCache(&caches!, batchSize: activeBSize)
            }

            // 3. OPTIMAL SHARED PREFILL SETUP
            // If it's a fresh cache, we prefill at B=1 (fastest), then expand later.
            // Otherwise, we must match the active cache size.
            let prefillBSize = isFreshCache ? 1 : activeBSize

            var inputTokens = MLXArray(tokens).reshaped(1, tokens.count)
            if prefillBSize > 1 {
                inputTokens = MLX.concatenated(Array(repeating: inputTokens, count: prefillBSize), axis: 0)
            }
            let input = LMInput(tokens: inputTokens)

            if caches == nil {
                caches = container.context.model.newCache(parameters: params)
            }

            let sampler = params.sampler()
            var y: LMInput.Text
            var state: LMOutput.State? = nil

            var prefillProcessors = (0..<prefillBSize).map { _ in params.processor() }

            func convertToTokenBatched(logits: MLXArray, processors: inout [LogitProcessor?], bSize: Int) -> MLXArray {
                guard logits.size > 0 else { return MLXArray([0]).reshaped(1, 1) }

                var l = logits[0..., -1, 0...] // Shape: [B, V]

                // Safety Broadcast: If logits is B=1 but we expect B>1
                if l.shape[0] == 1 && bSize > 1 {
                    l = MLX.concatenated(Array(repeating: l, count: bSize), axis: 0)
                }

                var processedLogits = [MLXArray]()
                for i in 0..<bSize {
                    let indexArray = MLXArray([Int32(i)])
                    var row = l.take(indexArray, axis: 0) // Shape: [1, V]
                    if let p = processors[i] { row = p.process(logits: row) }
                    processedLogits.append(row)
                }

                l = MLX.concatenated(processedLogits, axis: 0)
                let t = sampler.sample(logits: l)

                let tArray = t.asArray(Int32.self)
                for i in 0..<bSize {
                    processors[i]?.didSample(token: MLXArray([tArray[i]]))
                }

                return t.ndim == 1 && t.size > 0 ? t.reshaped(t.shape[0], 1) : t
            }

            let prefillStart = Date.timeIntervalSinceReferenceDate
            for i in 0..<prefillBSize {
                let indexArray = MLXArray([Int32(i)])
                prefillProcessors[i]?.prompt(inputTokens.take(indexArray, axis: 0))
            }

            // 4. EXECUTE PREFILL
            switch try container.context.model.prepare(input, cache: caches!, windowSize: params.prefillStepSize) {
              case .tokens(let outTokens):
                if outTokens.tokens.size > 0 {
                    let result = container.context.model(outTokens, cache: caches!.isEmpty ? nil : caches!, state: state)
                    state = result.state
                    let token = convertToTokenBatched(logits: result.logits, processors: &prefillProcessors, bSize: prefillBSize)
                    y = .init(tokens: token)
                } else {
                    y = .init(tokens: MLXArray([tokens.last ?? 0]).reshaped(1, 1))
                    if prefillBSize > 1 {
                        y = .init(tokens: MLX.concatenated(Array(repeating: y.tokens, count: prefillBSize), axis: 0))
                    }
                }
              case .logits(let result):
                state = result.state
                let token = convertToTokenBatched(logits: result.logits, processors: &prefillProcessors, bSize: prefillBSize)
                y = .init(tokens: token)
            }

            eval(y.tokens)

            // Quantize ONCE after prefill
            maybeQuantizeKVCache(cache: &caches!, kvBits: activeKvBits, kvGroupSize: activeGroupSize, quantizedKVStart: activeStart)

            // 5. CACHE POST-EXPANSION (Turn 1 Branching)
            // If it was a fresh cache, we prefilled B=1. If the user wants B>1, we expand it now.
            if isFreshCache && activeBSize > 1 {
                expandCache(&caches!, batchSize: activeBSize)
                y = .init(tokens: MLX.concatenated(Array(repeating: y.tokens, count: activeBSize), axis: 0))
                eval(y.tokens)
            }

            cacheContainer?.caches = caches!
            MLX.asyncEval(y.tokens)

            let promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart

            // Pre-empt loop if maxTokens is 0
            if let maxT = params.maxTokens, maxT == 0 {
                finalStats = GenerateStats(promptTokens: tokens.count, generatedTokens: 0, promptTime: promptPrefillTime, generateTime: 0, promptTokensPerSecond: promptPrefillTime > 0 ? Double(tokens.count) / promptPrefillTime : 0.0, tokensPerSecond: 0, stopReason: "length")
                if let jsonData = try? JSONEncoder().encode(finalStats), let jsonStr = String(data: jsonData, encoding: .utf8) {
                    jsonStr.withCString { cStr in callback(context, nil, 0, true, false, cStr) }
                } else { callback(context, nil, 0, true, false, nil) }
                return
            }

            // 6. SETUP GENERATION PROCESSORS
            var genProcessors = (0..<activeBSize).map { _ in params.processor() }
            let yTokensArray = y.tokens.asArray(Int32.self)
            for i in 0..<activeBSize {
                let pTokens = MLXArray(tokens).reshaped(1, tokens.count)
                genProcessors[i]?.prompt(pTokens)
                genProcessors[i]?.didSample(token: MLXArray([yTokensArray[i]]))
            }

            // 7. GENERATION LOOP
            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var stepCount = 0

            let stopTokenIds = compileStopTokens(context: container.context, config: configObj)
            var stopReason = "stop"
            var isDone = [Bool](repeating: false, count: activeBSize)
            var completedCount = 0

            while true {
                if Task.isCancelled {
                    wasCancelled = true
                    stopReason = "cancelled"
                    break
                }

                let currentTokens = y.tokens.asArray(Int32.self)

                if promptTime == 0 {
                    promptTime = Date.timeIntervalSinceReferenceDate - start
                    start = Date.timeIntervalSinceReferenceDate
                }

                for i in 0..<activeBSize {
                    if !isDone[i] {
                        let tokenId = Int(currentTokens[i])
                        if stopTokenIds.contains(tokenId) {
                            isDone[i] = true
                            completedCount += 1
                            tokenBuffer.append(-1)
                        } else {
                            tokenBuffer.append(Int32(tokenId))
                        }
                    } else {
                        tokenBuffer.append(-1)
                    }
                }

                stepCount += 1

                if tokenBuffer.count >= bufferCapacity {
                    tokenBuffer.withUnsafeBufferPointer { ptr in
                        callback(context, ptr.baseAddress, Int32(tokenBuffer.count), false, false, nil)
                    }
                    tokenBuffer.removeAll(keepingCapacity: true)
                }

                if completedCount == activeBSize { break }
                if let maxT = params.maxTokens, stepCount >= maxT {
                    stopReason = "length"
                    break
                }

                let result = container.context.model(y, cache: caches!.isEmpty ? nil : caches!, state: state)
                state = result.state

                let nextToken = convertToTokenBatched(logits: result.logits, processors: &genProcessors, bSize: activeBSize)
                y = .init(tokens: nextToken)

                MLX.asyncEval(y.tokens)
            }

            cacheContainer?.caches = caches!
            if Task.isCancelled { wasCancelled = true }
            Stream().synchronize()

            let generateTime = Date.timeIntervalSinceReferenceDate - start
            let finalPromptTime = promptTime + promptPrefillTime
            let totalTokens = stepCount * activeBSize

            finalStats = GenerateStats(promptTokens: tokens.count, generatedTokens: totalTokens, promptTime: finalPromptTime, generateTime: generateTime, promptTokensPerSecond: finalPromptTime > 0 ? Double(tokens.count) / finalPromptTime : 0.0, tokensPerSecond: generateTime > 0 ? Double(totalTokens) / generateTime : 0.0, stopReason: stopReason)

            if !tokenBuffer.isEmpty {
                tokenBuffer.withUnsafeBufferPointer { ptr in callback(context, ptr.baseAddress, Int32(tokenBuffer.count), false, false, nil) }
            }
        } catch {
            wasCancelled = error is CancellationError
            if !wasCancelled { finalErrorStr = error.localizedDescription }
        }

        if wasCancelled {
            "Generation Cancelled".withCString { cStr in callback(context, nil, 0, true, true, cStr) }
        } else if let errorStr = finalErrorStr {
            errorStr.withCString { cStr in callback(context, nil, 0, true, true, cStr) }
        } else if let stats = finalStats {
            if let jsonData = try? JSONEncoder().encode(stats), let jsonStr = String(data: jsonData, encoding: .utf8) {
                jsonStr.withCString { cStr in callback(context, nil, 0, true, false, cStr) }
            } else { callback(context, nil, 0, true, false, nil) }
        } else { callback(context, nil, 0, true, false, nil) }
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

@_cdecl("bridge_model_batch_task")
public func bridge_model_batch_task(
    modelPtr: UnsafeMutableRawPointer,
    cachePtr: UnsafeMutableRawPointer?,
    flatTokens: UnsafePointer<Int32>,
    maxLen: Int32,
    batchSize: Int32,
    configJson: UnsafePointer<CChar>,
    context: UnsafeMutableRawPointer,
    callback: @convention(c) (UnsafeMutableRawPointer, UnsafePointer<Int32>?, Int32, Bool, Bool, UnsafePointer<CChar>?) -> Void
) -> UnsafeMutableRawPointer {

    let bSize = Int(batchSize)
    let mLen = Int(maxLen)
    let totalTokensCount = bSize * mLen
    let buffer = UnsafeBufferPointer(start: flatTokens, count: totalTokensCount)
    let tokensArray = Array(buffer).map { Int($0) }

    let configObj = parseConfig(String(cString: configJson))
    let chunkSize = Int(configObj.chunkSize ?? 1)
    let bufferCapacity = chunkSize * bSize

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

        var finalStats: GenerateStats? = nil
        var finalErrorStr: String? = nil
        var wasCancelled = false

        do {
            let isFreshCache = caches == nil || (caches!.first?.state.isEmpty ?? true)
            let currentCacheBSize = isFreshCache ? 1 : (caches!.first?.state.first?.shape[0] ?? 1)

            if !isFreshCache && currentCacheBSize == 1 && bSize > 1 {
                expandCache(&caches!, batchSize: bSize)
            }

            var tokenBuffer = [Int32]()
            tokenBuffer.reserveCapacity(bufferCapacity)

            // CRITICAL: We pass the dense 2D array directly to the model!
            let denseInput = MLXArray(tokensArray).reshaped(bSize, mLen)
            let input = LMInput(tokens: denseInput)

            if caches == nil {
                caches = container.context.model.newCache(parameters: params)
            }

            let sampler = params.sampler()
            var y: LMInput.Text
            var state: LMOutput.State? = nil

            var prefillProcessors = (0..<bSize).map { _ in params.processor() }

            func convertToTokenBatched(logits: MLXArray, processors: inout [LogitProcessor?], bSize: Int) -> MLXArray {
                guard logits.size > 0 else { return MLXArray([0]).reshaped(1, 1) }

                var l = logits[0..., -1, 0...] // Shape: [B, V]
                if l.shape[0] == 1 && bSize > 1 {
                    l = MLX.concatenated(Array(repeating: l, count: bSize), axis: 0)
                }

                var processedLogits = [MLXArray]()
                for i in 0..<bSize {
                    let indexArray = MLXArray([Int32(i)])
                    var row = l.take(indexArray, axis: 0) // Shape: [1, V]
                    if let p = processors[i] { row = p.process(logits: row) }
                    processedLogits.append(row)
                }

                l = MLX.concatenated(processedLogits, axis: 0)
                let t = sampler.sample(logits: l)

                let tArray = t.asArray(Int32.self)
                for i in 0..<bSize {
                    processors[i]?.didSample(token: MLXArray([tArray[i]]))
                }

                return t.ndim == 1 && t.size > 0 ? t.reshaped(t.shape[0], 1) : t
            }

            let prefillStart = Date.timeIntervalSinceReferenceDate
            for i in 0..<bSize {
                let indexArray = MLXArray([Int32(i)])
                prefillProcessors[i]?.prompt(denseInput.take(indexArray, axis: 0))
            }

            switch try container.context.model.prepare(input, cache: caches!, windowSize: params.prefillStepSize) {
              case .tokens(let outTokens):
                if outTokens.tokens.size > 0 {
                    let result = container.context.model(outTokens, cache: caches!.isEmpty ? nil : caches!, state: state)
                    state = result.state
                    let token = convertToTokenBatched(logits: result.logits, processors: &prefillProcessors, bSize: bSize)
                    y = .init(tokens: token)
                } else {
                    y = .init(tokens: MLXArray([tokensArray.last ?? 0]).reshaped(1, 1))
                    if bSize > 1 {
                        y = .init(tokens: MLX.concatenated(Array(repeating: y.tokens, count: bSize), axis: 0))
                    }
                }
              case .logits(let result):
                state = result.state
                let token = convertToTokenBatched(logits: result.logits, processors: &prefillProcessors, bSize: bSize)
                y = .init(tokens: token)
            }

            eval(y.tokens)
            maybeQuantizeKVCache(cache: &caches!, kvBits: activeKvBits, kvGroupSize: activeGroupSize, quantizedKVStart: activeStart)

            cacheContainer?.caches = caches!
            MLX.asyncEval(y.tokens)

            let promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart

            if let maxT = params.maxTokens, maxT == 0 {
                finalStats = GenerateStats(promptTokens: totalTokensCount, generatedTokens: 0, promptTime: promptPrefillTime, generateTime: 0, promptTokensPerSecond: promptPrefillTime > 0 ? Double(totalTokensCount) / promptPrefillTime : 0.0, tokensPerSecond: 0, stopReason: "length")
                if let jsonData = try? JSONEncoder().encode(finalStats), let jsonStr = String(data: jsonData, encoding: .utf8) {
                    jsonStr.withCString { cStr in callback(context, nil, 0, true, false, cStr) }
                } else { callback(context, nil, 0, true, false, nil) }
                return
            }

            var genProcessors = (0..<bSize).map { _ in params.processor() }
            let yTokensArray = y.tokens.asArray(Int32.self)
            for i in 0..<bSize {
                let indexArray = MLXArray([Int32(i)])
                let pTokens = denseInput.take(indexArray, axis: 0)
                genProcessors[i]?.prompt(pTokens)
                genProcessors[i]?.didSample(token: MLXArray([yTokensArray[i]]))
            }

            var start = Date.timeIntervalSinceReferenceDate
            var promptTime: TimeInterval = 0
            var stepCount = 0

            let stopTokenIds = compileStopTokens(context: container.context, config: configObj)
            var stopReason = "stop"
            var isDone = [Bool](repeating: false, count: bSize)
            var completedCount = 0

            while true {
                if Task.isCancelled {
                    wasCancelled = true
                    stopReason = "cancelled"
                    break
                }

                let currentTokens = y.tokens.asArray(Int32.self)

                if promptTime == 0 {
                    promptTime = Date.timeIntervalSinceReferenceDate - start
                    start = Date.timeIntervalSinceReferenceDate
                }

                for i in 0..<bSize {
                    if !isDone[i] {
                        let tokenId = Int(currentTokens[i])
                        if stopTokenIds.contains(tokenId) {
                            isDone[i] = true
                            completedCount += 1
                            tokenBuffer.append(-1)
                        } else {
                            tokenBuffer.append(Int32(tokenId))
                        }
                    } else {
                        tokenBuffer.append(-1)
                    }
                }

                stepCount += 1

                if tokenBuffer.count >= bufferCapacity {
                    tokenBuffer.withUnsafeBufferPointer { ptr in
                        callback(context, ptr.baseAddress, Int32(tokenBuffer.count), false, false, nil)
                    }
                    tokenBuffer.removeAll(keepingCapacity: true)
                }

                if completedCount == bSize { break }
                if let maxT = params.maxTokens, stepCount >= maxT {
                    stopReason = "length"
                    break
                }

                let result = container.context.model(y, cache: caches!.isEmpty ? nil : caches!, state: state)
                state = result.state

                let nextToken = convertToTokenBatched(logits: result.logits, processors: &genProcessors, bSize: bSize)
                y = .init(tokens: nextToken)

                MLX.asyncEval(y.tokens)
            }

            if !tokenBuffer.isEmpty {
                tokenBuffer.withUnsafeBufferPointer { ptr in callback(context, ptr.baseAddress, Int32(tokenBuffer.count), false, false, nil) }
            }

            cacheContainer?.caches = caches!
            if Task.isCancelled { wasCancelled = true }
            Stream().synchronize()

            let generateTime = Date.timeIntervalSinceReferenceDate - start
            let finalPromptTime = promptTime + promptPrefillTime
            let totalGenTokens = stepCount * bSize

            finalStats = GenerateStats(promptTokens: totalTokensCount, generatedTokens: totalGenTokens, promptTime: finalPromptTime, generateTime: generateTime, promptTokensPerSecond: finalPromptTime > 0 ? Double(totalTokensCount) / finalPromptTime : 0.0, tokensPerSecond: generateTime > 0 ? Double(totalGenTokens) / generateTime : 0.0, stopReason: stopReason)

        } catch {
            wasCancelled = error is CancellationError
            if !wasCancelled { finalErrorStr = error.localizedDescription }
        }

        if wasCancelled {
            "Generation Cancelled".withCString { cStr in callback(context, nil, 0, true, true, cStr) }
        } else if let errorStr = finalErrorStr {
            errorStr.withCString { cStr in callback(context, nil, 0, true, true, cStr) }
        } else if let stats = finalStats {
            if let jsonData = try? JSONEncoder().encode(stats), let jsonStr = String(data: jsonData, encoding: .utf8) {
                jsonStr.withCString { cStr in callback(context, nil, 0, true, false, cStr) }
            } else { callback(context, nil, 0, true, false, nil) }
        } else { callback(context, nil, 0, true, false, nil) }
    }

    taskContainer.task = task
    return Unmanaged.passRetained(taskContainer).toOpaque()
}
