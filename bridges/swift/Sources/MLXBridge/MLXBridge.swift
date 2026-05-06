import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// ============================================================================
// 1. TYPES & CONFIGURATIONS
// ============================================================================

public typealias BridgeAsyncCallback =
  @convention(c) (UnsafeMutableRawPointer, Bool, UnsafeMutableRawPointer?, UnsafePointer<CChar>?) ->
  Void

public typealias BridgeStreamCallback =
  @convention(c) (
    UnsafeMutableRawPointer,
    UnsafePointer<Int32>?, Int32,                // Tokens
    UnsafePointer<Int32>?, UnsafePointer<Float>?, Int32, // Top-K Tokens & Probs
    Bool, Bool, UnsafePointer<CChar>?            // Done, Error, JSON Payload
  ) -> Void

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
  var padTokenId: Int?
  var topLogits: Int? // Number of top probabilities to yield alongside tokens

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

private class ModelContainer {
  let context: ModelContext
  init(_ context: ModelContext) { self.context = context }
}

private class CacheContainer {
  var caches: [KVCache]
  var kvBits: Int?
  var kvGroupSize: Int?
  var quantizedKVStart: Int?
  var paddingCounts: [Int] // Tracks the number of injected pad tokens per sequence

  init(
    _ caches: [KVCache],
    kvBits: Int? = nil,
    kvGroupSize: Int? = nil,
    quantizedKVStart: Int? = nil,
    paddingCounts: [Int]? = nil
  ) {
    self.caches = caches
    self.kvBits = kvBits
    self.kvGroupSize = kvGroupSize
    self.quantizedKVStart = quantizedKVStart

    // Initialize paddingCounts with 0s based on the batch size, or default to 1
    if let counts = paddingCounts {
      self.paddingCounts = counts
    } else {
      let batchSize = caches.first?.state.first?.shape[0] ?? 1
      self.paddingCounts = Array(repeating: 0, count: max(1, batchSize))
    }
  }
}

private class TaskContainer {
  var task: Task<Void, Never>?
  init() {}
  deinit { task?.cancel() }
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

private struct BridgeMetrics: Encodable {
  let active: Int
  let cache: Int
  let peak: Int
  let memoryLimit: Int
  let cacheLimit: Int
}

// ============================================================================
// 2. THE NULL TOKENIZER
// ============================================================================

struct NullTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
  func encode(text: String, addSpecialTokens: Bool) -> [Int] { return [] }
  func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { return "" }
  func convertTokenToId(_ token: String) -> Int? { return nil }
  func convertIdToToken(_ id: Int) -> String? { return nil }
  var bosToken: String? { nil }
  var eosToken: String? { nil }
  var unknownToken: String? { nil }
  func applyChatTemplate(
    messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
    additionalContext: [String: any Sendable]?
  ) throws -> [Int] { return [] }
}

struct NullTokenizerLoader: TokenizerLoader {
  func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
    return NullTokenizer()
  }
}

// ============================================================================
// 3. SHARED UTILITIES
// ============================================================================

private func parseConfig(_ jsonString: String) -> BridgeGenerateConfig {
  guard let data = jsonString.data(using: .utf8),
    let config = try? JSONDecoder().decode(BridgeGenerateConfig.self, from: data)
  else {
    return BridgeGenerateConfig()
  }
  return config
}

private func encodeJson<T: Encodable>(_ payload: T) -> String? {
  guard let jsonData = try? JSONEncoder().encode(payload) else { return nil }
  return String(data: jsonData, encoding: .utf8)
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

private func expandCacheDimensions(
    _ caches: inout [KVCache],
    targetBatchSize: Int,
    paddingCounts: inout [Int]
) {
  guard targetBatchSize > 1 else { return }

  // Safeguard against multiplying a cache that's already been expanded
  if let firstShape = caches.first?.state.first?.shape, firstShape.count > 0, firstShape[0] == targetBatchSize {
      return
  }

  var prefillStates: [MLXArray] = []
  for cache in caches { prefillStates.append(contentsOf: cache.state) }
  eval(prefillStates)

  var postBroadcastStates: [MLXArray] = []
  for i in 0..<caches.count {
    let duplicated = caches[i].state.map { array in
      array.size > 0 && array.shape[0] == 1
        ? MLX.concatenated(Array(repeating: array, count: targetBatchSize), axis: 0) : array
    }
    caches[i].state = duplicated
    postBroadcastStates.append(contentsOf: duplicated)
  }
  eval(postBroadcastStates)

  // Expand padding counts to match the newly duplicated batches
  if paddingCounts.count == 1 {
    paddingCounts = Array(repeating: paddingCounts[0], count: targetBatchSize)
  }
}

/// A clean, branch-isolated token sampler that prevents hallucinations across batches
private func sampleBatchedTokens(
  logits: MLXArray,
  processors: inout [LogitProcessor?],
  sampler: LogitSampler,
  batchSize: Int
) -> MLXArray {
  guard logits.size > 0 else { return MLXArray([0]).reshaped(1, 1) }

  var latestLogits = logits[0..., -1, 0...]  // Shape: [Batch, VocabSize]

  // Safety Broadcast: If logits is B=1 but we expect B>1
  if latestLogits.shape[0] == 1 && batchSize > 1 {
    latestLogits = MLX.concatenated(Array(repeating: latestLogits, count: batchSize), axis: 0)
  }

  var processedLogits = [MLXArray]()

  for index in 0..<batchSize {
    let batchIndex = MLXArray([Int32(index)])
    var rowLogits = latestLogits.take(batchIndex, axis: 0)  // Shape: [1, V]

    if let processor = processors[index] {
      rowLogits = processor.process(logits: rowLogits)
    }
    processedLogits.append(rowLogits)
  }

  let combinedLogits = MLX.concatenated(processedLogits, axis: 0)
  let sampledTokens = sampler.sample(logits: combinedLogits)

  // Update history states independently to prevent cross-contamination
  let tokensArray = sampledTokens.asArray(Int32.self)
  for index in 0..<batchSize {
    processors[index]?.didSample(token: MLXArray([tokensArray[index]]))
  }

  return sampledTokens.ndim == 1 && sampledTokens.size > 0
    ? sampledTokens.reshaped(sampledTokens.shape[0], 1) : sampledTokens
}

// ============================================================================
// 4. THE CORE GENERATION ENGINE
// ============================================================================

/// A unified generation engine that powers both Homogeneous and Heterogeneous batching.
private func executeGenerationTask(
  modelContainer: ModelContainer,
  cacheContainer: CacheContainer?,
  localCaches: [KVCache]?,
  inputMatrix: MLXArray,  // Shape: [BatchSize, SequenceLength]
  config: BridgeGenerateConfig,
  isHomogeneous: Bool,
  context: UnsafeMutableRawPointer,
  callback: @escaping BridgeStreamCallback
) {
  let batchSize = inputMatrix.shape[0]
  let sequenceLength = inputMatrix.shape[1]
  let totalTokensCount = batchSize * sequenceLength
  let chunkSize = Int(config.chunkSize ?? 1)
  let bufferCapacity = chunkSize * batchSize

  var caches = localCaches
  let activeKvBits = config.kvBits ?? cacheContainer?.kvBits
  let activeGroupSize = config.kvGroupSize ?? cacheContainer?.kvGroupSize ?? 64
  let activeStart = config.quantizedKVStart ?? cacheContainer?.quantizedKVStart ?? 0

  var params = config.toGenerateParameters()
  params.kvBits = activeKvBits
  params.kvGroupSize = activeGroupSize
  params.quantizedKVStart = activeStart

  let padTokenId = config.padTokenId ?? 0
  let requestedTopK = config.topLogits ?? 0

  var finalStats: GenerateStats? = nil
  var finalErrorStr: String? = nil
  var wasCancelled = false

  do {
    // 1. DETERMINE CACHE STATE
    let isFreshCache = caches == nil || (caches!.first?.state.isEmpty ?? true)
    let currentCacheBatchSize = isFreshCache ? 1 : (caches!.first?.state.first?.shape[0] ?? 1)

    var paddingCounts = cacheContainer?.paddingCounts ?? Array(repeating: 0, count: currentCacheBatchSize)

    // Expand existing B=1 caches to B=N before prefill if necessary
    if !isFreshCache && currentCacheBatchSize == 1 && batchSize > 1 {
      expandCacheDimensions(&caches!, targetBatchSize: batchSize, paddingCounts: &paddingCounts)
    } else if isFreshCache && paddingCounts.count == 1 && batchSize > 1 && !isHomogeneous {
      // If heterogeneous and fresh, the cache will be created as B=N directly during prefill
      paddingCounts = Array(repeating: paddingCounts[0], count: batchSize)
    }

    // Prepare Buffers
    var tokenBuffer = [Int32]()
    tokenBuffer.reserveCapacity(bufferCapacity)

    var topTokensBuffer = [Int32]()
    var topProbsBuffer = [Float]()

    if requestedTopK > 0 {
        topTokensBuffer.reserveCapacity(bufferCapacity * requestedTopK)
        topProbsBuffer.reserveCapacity(bufferCapacity * requestedTopK)
    }

    // Helper: Safely flush buffers over the C-Bridge boundary
    let flushBuffers = {
        if !tokenBuffer.isEmpty {
            tokenBuffer.withUnsafeBufferPointer { tokPtr in
                topTokensBuffer.withUnsafeBufferPointer { topTokPtr in
                    topProbsBuffer.withUnsafeBufferPointer { topProbPtr in
                        callback(
                            context,
                            tokPtr.baseAddress,
                            Int32(tokenBuffer.count),
                            requestedTopK > 0 ? topTokPtr.baseAddress : nil,
                            requestedTopK > 0 ? topProbPtr.baseAddress : nil,
                            Int32(requestedTopK),
                            false, false, nil
                        )
                    }
                }
            }
            tokenBuffer.removeAll(keepingCapacity: true)
            topTokensBuffer.removeAll(keepingCapacity: true)
            topProbsBuffer.removeAll(keepingCapacity: true)
        }
    }

    // Helper: Evaluate Logits to probabilities using Unified Memory
    let extractTopK: (MLXArray, Int, [Bool]) -> Void = { logits, activeBatchSize, isDoneFlag in
        guard requestedTopK > 0 else { return }

        // 1. Get the latest step logits [Batch, VocabSize]
        let latestLogits = logits[0..., -1, 0...]

        // 2. Compute probabilities on GPU
        let probs = MLX.softmax(latestLogits, axis: -1)

        // 3. Evaluate to sync GPU with CPU unified memory
        eval(probs)

        let probsArray = probs.asArray(Float.self)
        let vocabSize = probs.shape.last!

        // 4. Swift CPU partial sorting (insanely fast for < 150k arrays)
        for b in 0..<activeBatchSize {
            if isDoneFlag[b] {
                // Keep Arrays rectangular if sequence is finished
                topTokensBuffer.append(Int32(padTokenId))
                topProbsBuffer.append(1.0)
                for _ in 1..<requestedTopK {
                    topTokensBuffer.append(0)
                    topProbsBuffer.append(0.0)
                }
            } else {
                let start = b * vocabSize
                let slice = probsArray[start..<(start + vocabSize)]
                let topK = slice.enumerated()
                    .sorted { $0.element > $1.element }
                    .prefix(requestedTopK)

                topTokensBuffer.append(contentsOf: topK.map { Int32($0.offset) })
                topProbsBuffer.append(contentsOf: topK.map { $0.element })
            }
        }
    }

    let input = LMInput(tokens: inputMatrix)

    if caches == nil {
      caches = modelContainer.context.model.newCache(parameters: params)
    }

    let sampler = params.sampler()
    var currentTokenInput: LMInput.Text
    var state: LMOutput.State? = nil

    // Homogeneous prompts can optimize by prefilling at B=1. Heterogeneous MUST prefill at B=N.
    let prefillBatchSize = (isHomogeneous && isFreshCache) ? 1 : batchSize
    var prefillProcessors = (0..<prefillBatchSize).map { _ in params.processor() }
    let prefillStart = Date.timeIntervalSinceReferenceDate

    for index in 0..<prefillBatchSize {
      let batchIndex = MLXArray([Int32(index)])
      prefillProcessors[index]?.prompt(inputMatrix.take(batchIndex, axis: 0))
    }

    // 2. EXECUTE PREFILL
    let activeInput =
      isHomogeneous ? LMInput(tokens: inputMatrix.take(MLXArray([Int32(0)]), axis: 0)) : input

    let prefillIsDone = Array(repeating: false, count: prefillBatchSize)

    switch try modelContainer.context.model.prepare(
      activeInput, cache: caches!, windowSize: params.prefillStepSize)
    {
    case .tokens(let outTokens):
      if outTokens.tokens.size > 0 {
        let result = modelContainer.context.model(
          outTokens, cache: caches!.isEmpty ? nil : caches!, state: state)
        state = result.state

        extractTopK(result.logits, prefillBatchSize, prefillIsDone)

        let token = sampleBatchedTokens(
          logits: result.logits, processors: &prefillProcessors, sampler: sampler,
          batchSize: prefillBatchSize)
        currentTokenInput = .init(tokens: token)
      } else {
        // Fallback for edge cases where the chunk absorbs all tokens
        let lastTokenArray = inputMatrix[0..., -1].reshaped(batchSize, 1)
        currentTokenInput = .init(tokens: lastTokenArray)

        if requestedTopK > 0 {
            for _ in 0..<prefillBatchSize {
                topTokensBuffer.append(Int32(padTokenId))
                topProbsBuffer.append(1.0)
                for _ in 1..<requestedTopK {
                    topTokensBuffer.append(0)
                    topProbsBuffer.append(0.0)
                }
            }
        }
      }
    case .logits(let result):
      state = result.state

      extractTopK(result.logits, prefillBatchSize, prefillIsDone)

      let token = sampleBatchedTokens(
        logits: result.logits, processors: &prefillProcessors, sampler: sampler,
        batchSize: prefillBatchSize)
      currentTokenInput = .init(tokens: token)
    }

    eval(currentTokenInput.tokens)

    // Quantize ONCE after prefill
    maybeQuantizeKVCache(
      cache: &caches!, kvBits: activeKvBits, kvGroupSize: activeGroupSize,
      quantizedKVStart: activeStart)

    // 3. POST-EXPANSION (Turn 1 Homogeneous Branching)
    if isHomogeneous && isFreshCache && batchSize > 1 {
      expandCacheDimensions(&caches!, targetBatchSize: batchSize, paddingCounts: &paddingCounts)
      currentTokenInput = .init(
        tokens: MLX.concatenated(
          Array(repeating: currentTokenInput.tokens, count: batchSize), axis: 0))
      eval(currentTokenInput.tokens)

      // Expand TopK buffers to match the new homogeneous batch size
      if requestedTopK > 0 {
          let singleTopTokens = topTokensBuffer
          let singleTopProbs = topProbsBuffer
          topTokensBuffer = Array(repeating: singleTopTokens, count: batchSize).flatMap { $0 }
          topProbsBuffer = Array(repeating: singleTopProbs, count: batchSize).flatMap { $0 }
      }
    }

    cacheContainer?.caches = caches!
    cacheContainer?.paddingCounts = paddingCounts
    MLX.asyncEval(currentTokenInput.tokens)

    let promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart

    // 4. EARLY EXIT CHECK
    if let maxTokens = params.maxTokens, maxTokens == 0 {
      finalStats = GenerateStats(
        promptTokens: totalTokensCount, generatedTokens: 0, promptTime: promptPrefillTime,
        generateTime: 0,
        promptTokensPerSecond: promptPrefillTime > 0
          ? Double(totalTokensCount) / promptPrefillTime : 0.0, tokensPerSecond: 0,
        stopReason: "length")
      if let jsonStr = encodeJson(finalStats) {
        jsonStr.withCString { cStr in callback(context, nil, 0, nil, nil, 0, true, false, cStr) }
      } else {
        callback(context, nil, 0, nil, nil, 0, true, false, nil)
      }
      return
    }

    // 5. SETUP GENERATION PROCESSORS
    var generationProcessors = (0..<batchSize).map { _ in params.processor() }
    let currentTokensArray = currentTokenInput.tokens.asArray(Int32.self)

    for index in 0..<batchSize {
      let batchIndex = MLXArray([Int32(index)])
      let promptTokens = inputMatrix.take(batchIndex, axis: 0)
      generationProcessors[index]?.prompt(promptTokens)
      generationProcessors[index]?.didSample(token: MLXArray([currentTokensArray[index]]))
    }

    // 6. GENERATION LOOP
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0
    var stepCount = 0

    let stopTokenIds = compileStopTokens(context: modelContainer.context, config: config)
    var stopReason = "stop"
    var isDone = [Bool](repeating: false, count: batchSize)
    var completedCount = 0

    while true {
      if Task.isCancelled {
        wasCancelled = true
        stopReason = "cancelled"
        break
      }

      let latestTokens = currentTokenInput.tokens.asArray(Int32.self)
      var nextInputTokens = latestTokens

      if promptTime == 0 {
        promptTime = Date.timeIntervalSinceReferenceDate - start
        start = Date.timeIntervalSinceReferenceDate
      }

      for index in 0..<batchSize {
        if !isDone[index] {
          let tokenId = Int(latestTokens[index])
          if stopTokenIds.contains(tokenId) {
            isDone[index] = true
            completedCount += 1
            tokenBuffer.append(Int32(padTokenId))
            nextInputTokens[index] = Int32(padTokenId)
          } else {
            tokenBuffer.append(Int32(tokenId))
          }
        } else {
          tokenBuffer.append(Int32(padTokenId))
          nextInputTokens[index] = Int32(padTokenId)
        }
      }

      stepCount += 1

      if tokenBuffer.count >= bufferCapacity {
        flushBuffers()
      }

      if completedCount == batchSize { break }
      if let maxT = params.maxTokens, stepCount >= maxT {
        stopReason = "length"
        break
      }

      for index in 0..<batchSize {
        if isDone[index] { paddingCounts[index] += 1 }
      }

      currentTokenInput = .init(tokens: MLXArray(nextInputTokens).reshaped(batchSize, 1))

      let result = modelContainer.context.model(
        currentTokenInput, cache: caches!.isEmpty ? nil : caches!, state: state)
      state = result.state

      extractTopK(result.logits, batchSize, isDone)

      let nextToken = sampleBatchedTokens(
        logits: result.logits, processors: &generationProcessors, sampler: sampler,
        batchSize: batchSize)
      currentTokenInput = .init(tokens: nextToken)

      MLX.asyncEval(currentTokenInput.tokens)
    }

    flushBuffers()

    cacheContainer?.caches = caches!
    cacheContainer?.paddingCounts = paddingCounts

    if Task.isCancelled { wasCancelled = true }
    Stream().synchronize()

    let generateTime = Date.timeIntervalSinceReferenceDate - start
    let finalPromptTime = promptTime + promptPrefillTime
    let totalGeneratedTokens = stepCount * batchSize

    finalStats = GenerateStats(
      promptTokens: totalTokensCount, generatedTokens: totalGeneratedTokens,
      promptTime: finalPromptTime, generateTime: generateTime,
      promptTokensPerSecond: finalPromptTime > 0 ? Double(totalTokensCount) / finalPromptTime : 0.0,
      tokensPerSecond: generateTime > 0 ? Double(totalGeneratedTokens) / generateTime : 0.0,
      stopReason: stopReason)

  } catch {
    wasCancelled = error is CancellationError
    if !wasCancelled { finalErrorStr = error.localizedDescription }
  }

  if wasCancelled {
    "Generation Cancelled".withCString { cStr in callback(context, nil, 0, nil, nil, 0, true, true, cStr) }
  } else if let errorStr = finalErrorStr {
    errorStr.withCString { cStr in callback(context, nil, 0, nil, nil, 0, true, true, cStr) }
  } else if let stats = finalStats {
    if let jsonStr = encodeJson(stats) {
      jsonStr.withCString { cStr in callback(context, nil, 0, nil, nil, 0, true, false, cStr) }
    } else {
      callback(context, nil, 0, nil, nil, 0, true, false, nil)
    }
  } else {
    callback(context, nil, 0, nil, nil, 0, true, false, nil)
  }
}

// ============================================================================
// 5. C-EXPORTS (NATIVE BOUNDARY)
// ============================================================================

@_cdecl("bridge_metal_load")
public func bridge_metal_load() {
  let a = MLXArray(0.0)
  eval(a + a)
}

@_cdecl("bridge_metal_clear_cache")
public func bridge_metal_clear_cache() { Memory.clearCache() }

@_cdecl("bridge_metal_metrics")
public func bridge_metrics() -> UnsafeMutablePointer<CChar>? {
  let snapshot = Memory.snapshot()
  let metrics = BridgeMetrics(
    active: snapshot.activeMemory, cache: snapshot.cacheMemory, peak: snapshot.peakMemory,
    memoryLimit: Memory.memoryLimit, cacheLimit: Memory.cacheLimit)
  guard let jsonStr = encodeJson(metrics) else { return nil }
  return strdup(jsonStr)
}

@_cdecl("bridge_model_load")
public func bridge_model_load(
  path: UnsafePointer<CChar>, context: UnsafeMutableRawPointer, callback: BridgeAsyncCallback
) {
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
public func bridge_cache_create(modelPtr: UnsafeMutableRawPointer, configJson: UnsafePointer<CChar>)
  -> UnsafeMutableRawPointer
{
  let container = Unmanaged<ModelContainer>.fromOpaque(modelPtr).takeUnretainedValue()
  let configObj = parseConfig(String(cString: configJson))
  let caches = container.context.model.newCache(parameters: configObj.toGenerateParameters())
  let cacheContainer = CacheContainer(
    caches, kvBits: configObj.kvBits, kvGroupSize: configObj.kvGroupSize,
    quantizedKVStart: configObj.quantizedKVStart, paddingCounts: nil)
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
  let newContainer = CacheContainer(
    clonedCaches, kvBits: container.kvBits, kvGroupSize: container.kvGroupSize,
    quantizedKVStart: container.quantizedKVStart,
    paddingCounts: container.paddingCounts)
  return Unmanaged.passRetained(newContainer).toOpaque()
}

@_cdecl("bridge_cache_slice")
public func bridge_cache_slice(ptr: UnsafeMutableRawPointer, start: Int32, end: Int32)
  -> UnsafeMutableRawPointer
{
  let container = Unmanaged<CacheContainer>.fromOpaque(ptr).takeUnretainedValue()

  let slicedCaches = container.caches.map { cache -> KVCache in
    var newCache = cache.copy()
    newCache.state = newCache.state.map { array in
      guard array.size > 0, array.shape[0] > 1 else { return array }
      let actualEnd = min(Int(end), array.shape[0])
      let actualStart = min(Int(start), actualEnd)
      let indices = MLXArray((Int32(actualStart)..<Int32(actualEnd)).map { $0 })
      return array.take(indices, axis: 0)
    }
    return newCache
  }

  // Extract the sliced padding counts corresponding to the batch range
  let actualEnd = min(Int(end), container.paddingCounts.count)
  let actualStart = min(Int(start), actualEnd)
  var slicedPaddingCounts = Array(container.paddingCounts[actualStart..<actualEnd])

  // Trim the lowest common denominator of pad tokens among the extracted sequences
  let minPadding = slicedPaddingCounts.min() ?? 0
  if minPadding > 0 && MLXLMCommon.canTrimPromptCache(slicedCaches) {
    MLXLMCommon.trimPromptCache(slicedCaches, numTokens: minPadding)
    slicedPaddingCounts = slicedPaddingCounts.map { $0 - minPadding }
  }

  let newContainer = CacheContainer(
    slicedCaches, kvBits: container.kvBits, kvGroupSize: container.kvGroupSize,
    quantizedKVStart: container.quantizedKVStart,
    paddingCounts: slicedPaddingCounts
  )
  return Unmanaged.passRetained(newContainer).toOpaque()
}

@_cdecl("bridge_cache_save")
public func bridge_cache_save(
  ptr: UnsafeMutableRawPointer, path: UnsafePointer<CChar>, context: UnsafeMutableRawPointer,
  callback: BridgeAsyncCallback
) {
  let container = Unmanaged<CacheContainer>.fromOpaque(ptr).takeUnretainedValue()
  let url = URL(fileURLWithPath: String(cString: path))
  Task {
    do {
      let paddingCountsStr = container.paddingCounts.map { String($0) }.joined(separator: ",")
      let metadata = ["paddingCounts": paddingCountsStr]
      try MLXLMCommon.savePromptCache(url: url, cache: container.caches, metadata: metadata)
      callback(context, true, nil, nil)
    } catch {
      error.localizedDescription.withCString { callback(context, false, nil, $0) }
    }
  }
}

@_cdecl("bridge_cache_load")
public func bridge_cache_load(
  path: UnsafePointer<CChar>, context: UnsafeMutableRawPointer, callback: BridgeAsyncCallback
) {
  let url = URL(fileURLWithPath: String(cString: path))
  Task {
    do {
      let (caches, metadata) = try MLXLMCommon.loadPromptCache(url: url)

      var paddingCounts: [Int]? = nil
      if let countsStr = metadata["paddingCounts"] {
          paddingCounts = countsStr.split(separator: ",").compactMap { Int($0) }
      }

      let container = CacheContainer(caches, paddingCounts: paddingCounts)
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

  // If user trims manually from JS, deduct from the known padded hallucination tracker
  container.paddingCounts = container.paddingCounts.map { max(0, $0 - Int(trimmed)) }

  return Int32(trimmed)
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

// ============================================================================
// 6. EXECUTION TASKS (ROUTERS)
// ============================================================================

@_cdecl("bridge_model_generate_task")
public func bridge_model_generate_task(
  modelPtr: UnsafeMutableRawPointer,
  cachePtr: UnsafeMutableRawPointer?,
  promptTokens: UnsafePointer<Int32>,
  promptLength: Int32,
  configJson: UnsafePointer<CChar>,
  context: UnsafeMutableRawPointer,
  callback: @escaping BridgeStreamCallback
) -> UnsafeMutableRawPointer {

  let container = Unmanaged<ModelContainer>.fromOpaque(modelPtr).takeUnretainedValue()
  let cacheContainer = cachePtr.map {
    Unmanaged<CacheContainer>.fromOpaque($0).takeUnretainedValue()
  }

  let configObj = parseConfig(String(cString: configJson))
  let batchSize = Int(configObj.batchSize ?? 1)

  let buffer = UnsafeBufferPointer(start: promptTokens, count: Int(promptLength))
  let tokensArray = Array(buffer).map { Int($0) }

  // Homogeneous: Build the matrix by repeating the 1D array
  var inputMatrix = MLXArray(tokensArray).reshaped(1, tokensArray.count)
  if batchSize > 1 {
    inputMatrix = MLX.concatenated(Array(repeating: inputMatrix, count: batchSize), axis: 0)
  }

  let taskContainer = TaskContainer()
  taskContainer.task = Task {
    executeGenerationTask(
      modelContainer: container, cacheContainer: cacheContainer,
      localCaches: cacheContainer?.caches, inputMatrix: inputMatrix, config: configObj,
      isHomogeneous: true, context: context, callback: callback)
  }

  return Unmanaged.passRetained(taskContainer).toOpaque()
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
  callback: @escaping BridgeStreamCallback
) -> UnsafeMutableRawPointer {

  let container = Unmanaged<ModelContainer>.fromOpaque(modelPtr).takeUnretainedValue()
  let cacheContainer = cachePtr.map {
    Unmanaged<CacheContainer>.fromOpaque($0).takeUnretainedValue()
  }

  let configObj = parseConfig(String(cString: configJson))
  let activeBatchSize = Int(batchSize)
  let maxLength = Int(maxLen)
  let totalTokensCount = activeBatchSize * maxLength

  let buffer = UnsafeBufferPointer(start: flatTokens, count: totalTokensCount)
  let tokensArray = Array(buffer).map { Int($0) }

  // Heterogeneous: Reshape the dense 1D flat array directly into 2D
  let inputMatrix = MLXArray(tokensArray).reshaped(activeBatchSize, maxLength)

  let taskContainer = TaskContainer()
  taskContainer.task = Task {
    executeGenerationTask(
      modelContainer: container, cacheContainer: cacheContainer,
      localCaches: cacheContainer?.caches, inputMatrix: inputMatrix, config: configObj,
      isHomogeneous: false, context: context, callback: callback)
  }

  return Unmanaged.passRetained(taskContainer).toOpaque()
}

@_cdecl("bridge_model_evaluate_task")
public func bridge_model_evaluate_task(
  modelPtr: UnsafeMutableRawPointer,
  cachePtr: UnsafeMutableRawPointer?,
  promptTokens: UnsafePointer<Int32>,
  promptLength: Int32,
  configJson: UnsafePointer<CChar>,
  context: UnsafeMutableRawPointer,
  callback: @escaping BridgeAsyncCallback
) -> UnsafeMutableRawPointer {

  let container = Unmanaged<ModelContainer>.fromOpaque(modelPtr).takeUnretainedValue()
  let cacheContainer = cachePtr.map {
    Unmanaged<CacheContainer>.fromOpaque($0).takeUnretainedValue()
  }

  let buffer = UnsafeBufferPointer(start: promptTokens, count: Int(promptLength))
  let tokensArray = Array(buffer).map { Int($0) }
  let configObj = parseConfig(String(cString: configJson))

  let taskContainer = TaskContainer()
  taskContainer.task = Task {
    var caches = cacheContainer?.caches
    do {
      let activeKvBits = configObj.kvBits ?? cacheContainer?.kvBits
      let activeGroupSize = configObj.kvGroupSize ?? cacheContainer?.kvGroupSize ?? 64
      let activeStart = configObj.quantizedKVStart ?? cacheContainer?.quantizedKVStart ?? 0

      var params = configObj.toGenerateParameters()
      params.kvBits = activeKvBits
      params.kvGroupSize = activeGroupSize
      params.quantizedKVStart = activeStart

      let batchSize = Int(configObj.batchSize ?? 1)
      let inputMatrix = MLXArray(tokensArray).reshaped(1, tokensArray.count)
      let input = LMInput(tokens: inputMatrix)

      if caches == nil {
        caches = container.context.model.newCache(parameters: params)
      }

      let startTime = Date()

      switch try container.context.model.prepare(
        input, cache: caches!, windowSize: params.prefillStepSize)
      {
      case .tokens(let outTokens):
        if outTokens.tokens.size > 0 {
          let result = container.context.model(
            outTokens, cache: caches!.isEmpty ? nil : caches!, state: nil)
          eval(result.logits)
        } else {
          eval(caches!.map { $0.state })
        }
      case .logits(let result):
        eval(result.logits)
      }

      let isFreshCache = caches == nil || (caches!.first?.state.isEmpty ?? true)
      let currentCacheBatchSize = isFreshCache ? 1 : (caches!.first?.state.first?.shape[0] ?? 1)
      var paddingCounts = cacheContainer?.paddingCounts ?? Array(repeating: 0, count: currentCacheBatchSize)

      maybeQuantizeKVCache(
        cache: &caches!, kvBits: activeKvBits, kvGroupSize: activeGroupSize,
        quantizedKVStart: activeStart)

      expandCacheDimensions(&caches!, targetBatchSize: batchSize, paddingCounts: &paddingCounts)

      cacheContainer?.caches = caches!
      cacheContainer?.paddingCounts = paddingCounts

      if Task.isCancelled {
        "Evaluation Cancelled".withCString { callback(context, false, nil, $0) }
        return
      }

      let duration = Date().timeIntervalSince(startTime)
      let stats = EvaluateStats(
        promptTokens: tokensArray.count, promptTime: duration,
        promptTokensPerSecond: Double(tokensArray.count) / duration)

      if let jsonStr = encodeJson(stats) {
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

@_cdecl("bridge_model_abort_task")
public func bridge_model_abort_task(ptr: UnsafeMutableRawPointer) {
  Unmanaged<TaskContainer>.fromOpaque(ptr).takeUnretainedValue().task?.cancel()
}

@_cdecl("bridge_model_free_task")
public func bridge_model_free_task(ptr: UnsafeMutableRawPointer) {
  Unmanaged<TaskContainer>.fromOpaque(ptr).release()
}
