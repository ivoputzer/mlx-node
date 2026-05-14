import mlx from 'mlx-swift'

export class MLXResource {
  #ref = null

  constructor (ref) {
    if (!ref) throw new Error('Invalid resource pointer')
    this.#ref = ref
  }

  get ref () { return this.#ref }
  get available () { return this.#ref !== null }

  dispose () {
    if (!this.#ref) return false // Prevent double-free logic from running
    mlx.freeResource(this.#ref)
    this.#ref = null
    return true
  }

  [Symbol.dispose] () {
    this.dispose()
  }
}

export class MLXTask extends MLXResource {
  abort () {
    if (this.available) mlx.abortTask(this.ref)
  }
}

export class MLXStream extends MLXTask {
  #queue = []
  #resolvers = null
  #batchSize = 1

  constructor (taskPointer, batchSize) {
    super(taskPointer)
    this.#batchSize = batchSize
  }

  push (event) {
    this.#queue.push(event)
    if (this.#resolvers) {
      this.#resolvers.resolve()
      this.#resolvers = null
    }
  }

  * #unpack (buffer, topTokens, topProbs, topK) {
    const ticks = buffer.length / this.#batchSize
    for (let i = 0; i < ticks; i++) {
      const start = i * this.#batchSize

      const tokens = Array.from(buffer.subarray(start, start + this.#batchSize)) // Zero-copy read from the C-pointer
      let logits = null

      if (topK > 0 && topTokens && topProbs) {
        logits = new Array(this.#batchSize)
        for (let s = 0; s < this.#batchSize; s++) {
          const tokenIdx = start + s
          // calculate the flat memory offsets for this specific token
          const startK = tokenIdx * topK
          const endK = startK + topK
          // zero-copy views. No loops! No object creation per logit!
          logits[s] = {
            ids: topTokens.subarray(startK, endK),
            probs: topProbs.subarray(startK, endK)
          }
        }
      }
      yield { tokens, logits } // yield standard object to prevent V8 Dictionary Mode deopt
    }
  }

  async * [Symbol.asyncIterator] () {
    try {
      while (true) {
        if (this.#queue.length === 0) {
          this.#resolvers = Promise.withResolvers()
          await this.#resolvers.promise // sleep until push wakes us up
        }

        const events = this.#queue
        this.#queue = [] // O(1) queue swap, no splice gc overhead

        for (const { error, tokens, topTokens, topProbs, topK, done, json } of events) {
          if (error) throw error
          if (done) return parseJson(json)
          if (tokens) yield * this.#unpack(tokens, topTokens, topProbs, topK)
        }
      }
    } finally {
      this.abort()
      this.dispose()
    }
  }
}

export class MLXEvaluate extends MLXTask {
  #promise

  constructor (model, cache, tokens, options = {}) {
    const { promise, resolve, reject } = Promise.withResolvers()
    super(
      mlx.evaluateTask(model?.ref, cache?.ref, tokens, configFrom(options), (error, json) => {
        this.dispose()
        if (error) {
          reject(error)
        } else {
          resolve({ ...parseJson(json), stopReason: 'prefill' })
        }
      })
    )

    this.#promise = promise
  }

  then (onFullfilled, onRejected) {
    return this.#promise.then(onFullfilled, onRejected)
  }

  catch (onRejected) {
    return this.#promise.catch(onRejected)
  }

  finally (onFinally) {
    return this.#promise.finally(onFinally)
  }
}

export class MLXGenerate extends MLXStream {
  // #queue = []
  // #wakeup = () => {} // noop

  // #batchSize = 1
  // #chunkSize = 5

  constructor (model, cache, tokens, options = {}) {
    super(
      mlx.generateTask(model?.ref, cache?.ref, tokens, configFrom(options), (err, tok, topTok, topProb, k, done, json) => {
        this.push({ error: err, tokens: tok, topTokens: topTok, topProbs: topProb, topK: k, done, json })
      }),
      options.batchSize ?? 1
    )
  }

  // _constructor (model, cache, tokens, options = { chunkSize: 5, batchSize: 1 }) {
  //   Function.prototype(
  //     mlx.generateTask(model?.ref, cache?.ref, tokens, configFrom(options), (error, tokens, topTokens, topProbs, topK, done, json) => {
  //       let formattedTopLogits = null
  //       if (topK > 0 && topTokens && topProbs) {
  //         formattedTopLogits = []
  //         // Group the flat arrays by batch size and topK
  //         for (let b = 0; b < tokens.length; b++) {
  //           const sequenceTops = []
  //           for (let k = 0; k < topK; k++) {
  //             const idx = (b * topK) + k
  //             sequenceTops.push({ id: topTokens[idx], prob: topProbs[idx] })
  //           }
  //           formattedTopLogits.push(sequenceTops)
  //         }
  //       }

  //       console.log('formattedTopLogits:', done, formattedTopLogits)

  //       this.#push({ error, tokens, topLogits: formattedTopLogits, done, json })
  //     })
  //   )
  //   this.#batchSize = options?.batchSize ?? 1
  //   this.#chunkSize = options?.chunkSize ?? 5
  // }

  // #push (event) {
  //   this.#queue.push(event)
  //   this.#wakeup() // promise is immutable once settled, should not require to set to null | noop after first call
  // }

  // async * [Symbol.asyncIterator] () {
  //   try {
  //     while (true) {
  //       if (this.#queue.length === 0) {
  //         const { promise, resolve } = Promise.withResolvers()
  //         this.#wakeup = resolve
  //         await promise // sleep
  //       }
  //       for (const { error, tokens, done, json } of this.#queue.splice(0, this.#queue.length)) {
  //         if (error) throw error
  //         if (done) return parseSafe(json)
  //         if (tokens) {
  //           // Replicate the 'yield *' behavior:
  //           // Unpack the C flat buffer into discrete ticks (time-steps)
  //           // Each yield represents ONE tick containing an array of tokens (one per sequence).
  //           yield * (function * (buffer, batchSize) {
  //             const ticks = buffer.length / batchSize
  //             for (let i = 0; i < ticks; i++) {
  //               // Slice exactly 1 time-step across all sequences
  //               yield Array.from(buffer.slice(i * batchSize, (i + 1) * batchSize))
  //             }
  //           })(tokens, this.#batchSize)
  //         }
  //       }
  //     }
  //   } finally {
  //     this.abort()
  //     this.dispose()
  //   }
  // }
}

export class MLXTarget extends MLXResource {
  get model () {
    throw new Error('Not implemented')
  }

  get cache () {
    throw new Error('Not implemented')
  }

  get config () {
    return {}
  }

  generate (tokens, options = {}) {
    if (!this.available) throw new Error('Target unavailable')
    // todo: merge configurations and do checks
    return new MLXGenerate(this.model, this.cache, tokens, options)
  }

  async evaluate (tokens, options = {}) {
    if (!this.available) throw new Error('Target unavailable')
    // todo: what if the user calls this on a model 🤔
    // should we create a cache on the fly MLXCache.fromModel(this)
    // or should we have swift accept no cache just to extract prompt stats?
    return new MLXEvaluate(this.model, this.cache, tokens, options)
  }

  async * batch (promptTokensArrays, options = {}) {
    if (!this.available) throw new Error('Target unavailable')
    if (!Array.isArray(promptTokensArrays) || promptTokensArrays.length === 0) {
      throw new Error('batch() requires an array of prompt token arrays')
    }

    const activeBatchSize = options.batchSize || 1
    const isolatedCaches = promptTokensArrays.map(() => this.cache ? this.cache.clone() : null)

    const tasks = promptTokensArrays.map((tokens, i) =>
      new MLXGenerate(this.model, isolatedCaches[i], tokens, { ...options, chunkSize: 1, batchSize: activeBatchSize })
    )

    const iterators = tasks.map(task => task[Symbol.asyncIterator]())
    const isDone = new Array(tasks.length).fill(false)
    let active = tasks.length
    const finalStats = new Array(tasks.length).fill(null)

    const padTokens = new Array(activeBatchSize).fill(-1) // Pre-allocate the padding array ONCE outside the loop. Pushing this exact reference saves thousands of array allocations per second.

    try {
      while (active > 0) {
        const tickResults = await Promise.all(iterators.map(async (it, i) => {
          if (isDone[i]) return { done: true }
          return it.next()
        }))

        const tickTokens = []
        let allFinishedThisTick = true

        for (let i = 0; i < tickResults.length; i++) {
          const res = tickResults[i]
          if (res.done) {
            if (!isDone[i]) {
              isDone[i] = true
              active--
              finalStats[i] = res.value
            }
            // OPTIMIZED: Push the shared reference
            tickTokens.push(padTokens)
          } else {
            allFinishedThisTick = false
            // OPTIMIZED: Support the new `{ tokens, topLogits }` object structure
            tickTokens.push(res.value.tokens ?? res.value)
          }
        }

        if (allFinishedThisTick) break
        yield tickTokens
      }

      return finalStats.map((stats, i) => ({ stats, cache: isolatedCaches[i] }))
    } finally {
      if (active > 0) {
        tasks.forEach(task => task.abort())
        isolatedCaches.forEach(c => c?.dispose())
      }
    }
  }

  async * _batch (promptTokensArrays, options = {}) {
    if (!this.available) throw new Error('Target unavailable')
    if (!Array.isArray(promptTokensArrays) || promptTokensArrays.length === 0) {
      throw new Error('batch() requires an array of prompt token arrays')
    }

    // Usually pad_token_id is in tokenizer.config.pad_token_id,
    // otherwise fallback to 0 (which is safe for testing)
    const padTokenId = options.padTokenId ?? 0

    // Convert jagged arrays into a dense flat tensor
    const { flatTokens, maxLen, batchSize } = createPaddedBatch(promptTokensArrays, padTokenId)

    // Start the single unified C-Task!
    const generate = new MLXBatch(this.model, this.cache, flatTokens, maxLen, batchSize, options)

    try {
      for await (const batchTokens of generate) {
        yield batchTokens
      }
    } finally {
      generate.abort()
    }
  }
}

export class MLXCache extends MLXTarget {
  static CACHE_CONFIG_KEYS = ['maxKVSize', 'kvBits', 'kvGroupSize', 'quantizedKVStart']

  static async fromPath (path, model) { // Load from Disk
    const { promise, resolve, reject } = Promise.withResolvers()
    mlx.loadCache(path, (err, ref) => err ? reject(err) : resolve(new MLXCache(ref, model))) // c should throw: Error {message: '<from swift>' code: 'mlx_load_error'}
    return promise
  }

  static fromModel (model, options = {}) { // Create Empty Cache
    if (!model.available) throw new Error('Model unavailable')
    const ref = mlx.createCache(model.ref, configFrom(options))
    return new MLXCache(ref, model)
  }

  #model

  constructor (ref, model) {
    super(ref)
    this.#model = model

    const { layers, type, offset, isTrimmable } = this.debug()

    this.layers = layers
    this.type = type
    this.offset = offset
    this.isTrimmable = isTrimmable
  }

  get model () {
    return this.#model
  }

  get cache () {
    return this
  }

  async save (path) { // Save to Disk
    if (!this.available) throw new Error('Cache unavailable')
    const { promise, resolve, reject } = Promise.withResolvers()
    mlx.saveCache(this.ref, path, (err) => err ? reject(err) : resolve()) // c should throws: Error {message: '<from swift>' code: 'mlx_save_error'}
    return promise
  }

  trim (numTokens) {
    if (!this.ref) throw new Error('Cache unavailable')
    return mlx.trimCache(this.ref, numTokens)
  }

  clone () {
    if (!this.ref) throw new Error('Cache unavailable')
    return new MLXCache(mlx.cloneCache(this.ref), this.#model)
  }

  slice (start, end) {
    if (!this.available) throw new Error('Cache unavailable')
    return new MLXCache(mlx.sliceCache(this.ref, start, end), this.#model)
  }

  debug () {
    if (!this.available) throw new Error('Cache unavailable')
    return JSON.parse(mlx.debugCache(this.ref))
  }
}

export class MLXModel extends MLXTarget {
  static MODEL_CONFIG_KEYS = ['stopTokenIds', 'maxTokens', 'temperature', 'topP', 'topK', 'minP', 'repetitionPenalty', 'repetitionContextSize', 'presencePenalty', 'presenceContextSize', 'frequencyPenalty', 'frequencyContextSize', 'prefillStepSize']

  static async fromPath (path) {
    const { promise, resolve, reject } = Promise.withResolvers()
    mlx.loadModel(path, (err, ref) => err ? reject(err) : resolve(new MLXModel(ref)))
    return promise
  }

  get model () {
    return this
  }

  get cache () {
    return null
  }
}

export class MLXMetrics {
  static fromSnapshot () {
    return new MLXMetrics(parseJson(mlx.systemMetrics()))
  }

  constructor ({ active = 0, cache = 0, peak = 0, memoryLimit = 0, cacheLimit = 0 }) {
    this.active = active
    this.cache = cache
    this.peak = peak
    this.memoryLimit = memoryLimit
    this.cacheLimit = cacheLimit
  }

  get total () {
    return this.active + this.cache
  }

  get usage () {
    return this.total / this.memoryLimit
  }

  get usagePercent () {
    return this.usage * 100
  }

  toJSON () {
    return { active: this.active, cache: this.cache, peak: this.peak, total: this.total, limit: this.memoryLimit, percent: this.usagePercent }
  }

  toString () {
    return `MLX Memory:\n  Active: ${format(this.active)} (Weights & Active Tensors)\n  Cached: ${format(this.cache)} (Recyclable Pool)\n  Total:  ${format(this.total)} / ${format(this.memoryLimit)} (${100 * this.usage.toFixed(1)}%)\n  Peak:   ${format(this.peak)} (High Water Mark)`
    function format (bytes, sizes = ['B', 'KB', 'MB', 'GB', 'TB'], k = 1024) {
      if (bytes === 0) return '0 B'
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    }
  }
}

export class MLXBatch extends MLXTask {
  #queue = []
  #resolvers = null
  #batchSize = 1

  constructor (model, cache, flatTokens, maxLen, batchSize, options = {}) {
    super(
      mlx.batchTask(model?.ref, cache?.ref, flatTokens, maxLen, batchSize, configFrom(options), (error, tokens, done, json) => {
        this.#push({ error, tokens, done, json })
      })
    )
    this.#batchSize = batchSize
  }

  #push (event) {
    this.#queue.push(event)
    if (this.#resolvers) {
      this.#resolvers.resolve()
      this.#resolvers = null
    }
  }

  * #unpack (buffer) {
    const ticks = buffer.length / this.#batchSize
    for (let i = 0; i < ticks; i++) {
      const start = i * this.#batchSize
      yield Array.from(buffer.subarray(start, start + this.#batchSize)) // zero-copy
    }
  }

  async * [Symbol.asyncIterator] () {
    try {
      while (true) {
        if (this.#queue.length === 0) {
          this.#resolvers = Promise.withResolvers()
          await this.#resolvers.promise
        }

        const events = this.#queue
        this.#queue = []

        for (const { error, tokens, done, json } of events) {
          if (error) throw error
          if (done) return parseJson(json)
          if (tokens) yield * this.unpack(tokens)
        }
      }
    } finally {
      this.abort()
      this.dispose()
    }
  }
}

// HELPERS

export function configFrom (config) {
  if (config.chunkSize > 2147483647) throw new Error('ChunkSize exceeds INT32_MAX')
  return JSON.stringify(Object.fromEntries(Object.entries(config).filter(([key]) => ['topLogits', 'stopTokenIds', 'padTokenId', 'batchSize', 'chunkSize', 'maxTokens', 'maxKVSize', 'kvBits', 'kvGroupSize', 'quantizedKVStart', 'temperature', 'topP', 'topK', 'minP', 'repetitionPenalty', 'repetitionContextSize', 'presencePenalty', 'presenceContextSize', 'frequencyPenalty', 'frequencyContextSize', 'prefillStepSize'].includes(key))))
}

export function parseJson (json, fallback = null) {
  if (!json) return fallback
  try {
    return JSON.parse(json)
  } catch {
    return fallback
  }
}

export function createPaddedBatch (promptTokensArrays, padTokenId = 0) {
  const batchSize = promptTokensArrays.length
  const maxLen = Math.max(...promptTokensArrays.map(arr => arr.length))

  if (maxLen === 0) throw new Error('Cannot evaluate empty prompts')

  const flatTokens = new Int32Array(batchSize * maxLen)

  for (let i = 0; i < batchSize; i++) {
    const tokens = promptTokensArrays[i]
    const padCount = maxLen - tokens.length

    // 1. Left-pad with padTokenId
    for (let p = 0; p < padCount; p++) {
      flatTokens[i * maxLen + p] = padTokenId
    }
    // 2. Insert actual tokens at the end
    for (let t = 0; t < tokens.length; t++) {
      flatTokens[i * maxLen + padCount + t] = tokens[t]
    }
  }

  return { flatTokens, maxLen, batchSize }
}
