import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const mlx = require('mlx-swift')

// HELPERS

function parseSafe (json, fallback = null) {
  if (!json) return fallback
  try {
    return JSON.parse(json)
  } catch {
    return fallback
  }
}

function configFrom (config) {
  if (config.chunkSize > 2147483647) throw new Error('ChunkSize exceeds INT32_MAX')
  return JSON.stringify(Object.fromEntries(Object.entries(config).filter(([key]) => ['stopTokenIds', 'batchSize', 'chunkSize', 'maxTokens', 'maxKVSize', 'kvBits', 'kvGroupSize', 'quantizedKVStart', 'temperature', 'topP', 'topK', 'minP', 'repetitionPenalty', 'repetitionContextSize', 'presencePenalty', 'presenceContextSize', 'frequencyPenalty', 'frequencyContextSize', 'prefillStepSize'].includes(key))))
}

// CLASSES (this will be moved to mlx-node later, so that we can share them with mlx-cpp)

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
          resolve({ ...parseSafe(json), stopReason: 'prefill' })
        }
      })
    )

    this.#promise = promise
  }

  then (onFullfilled, onRejected) {
    return this.#promise.then(onFullfilled, onRejected)
  }
}

export class MLXGenerate extends MLXTask {
  #queue = []
  #wakeup = () => {} // noop

  #batchSize = 1
  #chunkSize = 5

  constructor (model, cache, tokens, options = { chunkSize: 5, batchSize: 1 }) {
    super(
      mlx.generateTask(model?.ref, cache?.ref, tokens, configFrom(options), (error, tokens, done, json) => {
        this.#push({ error, tokens, done, json })
      })
    )
    this.#batchSize = options?.batchSize ?? 1
    this.#chunkSize = options?.chunkSize ?? 5
  }

  #push (event) {
    this.#queue.push(event)
    this.#wakeup() // promise is immutable once settled, should not require to set to null | noop after first call
  }

  async * [Symbol.asyncIterator] () {
    try {
      while (true) {
        if (this.#queue.length === 0) {
          const { promise, resolve } = Promise.withResolvers()
          this.#wakeup = resolve
          await promise // sleep
        }
        for (const { error, tokens, done, json } of this.#queue.splice(0, this.#queue.length)) {
          if (error) throw error
          if (done) return parseSafe(json)
          if (tokens) {
            // Replicate the 'yield *' behavior:
            // Unpack the C flat buffer into discrete ticks (time-steps)
            // Each yield represents ONE tick containing an array of tokens (one per sequence).
            yield * (function * (buffer, batchSize) {
              const ticks = buffer.length / batchSize
              for (let i = 0; i < ticks; i++) {
                // Slice exactly 1 time-step across all sequences
                yield Array.from(buffer.slice(i * batchSize, (i + 1) * batchSize))
              }
            })(tokens, this.#batchSize)
          }
        }
      }
    } finally {
      this.abort()
      this.dispose()
    }
  }
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

  // async * batch (promptTokensArray, options = {}) {
  //   const tasks = promptTokensArray.map(tokens => this.generate(tokens, { ...options, chunkSize: 1, batchSize: 1 }))
  //   const iterators = tasks.map(task => task[Symbol.asyncIterator]())
  //   const active = iterators.length
  //   try {
  //     while (active > 0) {
  //       // Wait for 1 tick from all active agents concurrently
  //       const tickResults = await Promise.all(iterators.map(it => it.next()))
  //       const tickTokens = []
  //       for (let i = 0; i < tickResults.length; i++) {
  //         const res = tickResults[i]
  //         if (res.done) {
  //           tickTokens.push(null) // Pad finished tasks
  //           if (res.value) { /* Handle final stats if needed */ }
  //         } else {
  //           // Assume single token yielded because chunkSize is 1
  //           tickTokens.push(res.value[0])
  //         }
  //       }
  //       // If all returned -1, we are done
  //       if (tickTokens.every(t => t === -1)) break
  //       yield tickTokens
  //     }
  //   } finally {
  //     // Ensure all tasks abort if the user breaks the loop
  //     tasks.forEach(task => task.abort())
  //   }
  // }
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
    const ref = mlx.createCache(model.ref, JSON.stringify(options))
    return new MLXCache(ref, model)
  }

  #model = null

  constructor (ref, model) {
    super(ref)
    this.#model = model
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

  get isTrimmable () {
    const { isTrimmable } = this.debug()
    return isTrimmable // This value should be cached
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
    return new MLXMetrics(parseSafe(mlx.systemMetrics()))
  }

  constructor (data) {
    this.active = data?.active || 0
    this.cache = data?.cache || 0
    this.peak = data?.peak || 0
    this.memoryLimit = data?.memoryLimit || 0
    this.cacheLimit = data?.cacheLimit || 0
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

  toString () {
    return `MLX Memory:\n  Active: ${format(this.active)} (Weights & Active Tensors)\n  Cached: ${format(this.cache)} (Recyclable Pool)\n  Total:  ${format(this.total)} / ${format(this.memoryLimit)} (${100 * this.usage.toFixed(1)}%)\n  Peak:   ${format(this.peak)} (High Water Mark)`
    function format (bytes, sizes = ['B', 'KB', 'MB', 'GB', 'TB'], k = 1024) {
      if (bytes === 0) return '0 B'
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    }
  }

  toJSON () {
    return {
      active: this.active,
      cache: this.cache,
      peak: this.peak,
      total: this.total,
      limit: this.memoryLimit,
      percent: this.usagePercent
    }
  }
}

export default mlx
