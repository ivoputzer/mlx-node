import { createRequire } from 'node:module'
import { parse } from 'node:path'

const require = createRequire(import.meta.url)
const addon = require('mlx-swift')

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
  return JSON.stringify(Object.fromEntries(Object.entries(config).filter(([key]) => ['chunkSize', 'maxTokens', 'maxKVSize', 'kvBits', 'kvGroupSize', 'quantizedKVStart', 'temperature', 'topP', 'topK', 'minP', 'repetitionPenalty', 'repetitionContextSize', 'presencePenalty', 'presenceContextSize', 'frequencyPenalty', 'frequencyContextSize', 'prefillStepSize'].includes(key))))
}

// OVERRIDES (eventually we will refactor the addon to conform to the js api)

const mlx = {
  // MLXResource
  freeResource: addon.resourceFree,

  // MLXCache
  createCache: addon.cacheCreate,
  loadCache: addon.cacheLoad,
  saveCache: addon.cacheSave,
  cloneCache: addon.cacheClone,
  trimCache: addon.cacheTrim,

  // MLXModel
  loadModel: addon.modelLoad,

  // MLXTarget|MLXGenerate|MLXEvaluate
  generateTask: addon.modelGenerate,
  evaluateTask: addon.modelEvaluate,
  abortTask: addon.streamAbort,

  // Functions?
  systemMetrics: addon.systemMetrics,
  systemClearCache: addon.systemClearCache
}

// CLASSES (this will be moved to mlx-node later, so that we can share them with mlx-cpp)

export class MLXResource {
  #ref = null

  constructor (ref) {
    this.#ref = ref // verify it is a pointer not just != null because we already do that in available
  }

  get ref () { return this.#ref }
  get available () { return this.#ref !== null }

  dispose () {
    console.log('MLXResource.dispose', this.#ref)
    if (!this.#ref) return
    mlx.freeResource(this.#ref) // C tombstones this (if called multiple times it still returns true 🤔)
    this.#ref = null
  }

  [Symbol.dispose] () {
    this.dispose()
  }
}

export class MLXTask extends MLXResource {
  abort () {
    if (this.ref) {
      mlx.abort(this.ref)
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
        return error
          ? reject(error)
          : resolve({ ...parseSafe(json), stopReason: 'prefill' })
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

  constructor (model, cache, tokens, options = {}) {
    super(
      mlx.generateTask(model?.ref, cache?.ref, tokens, configFrom(options), (error, tokens, done, json) => {
        this.#push({ error, tokens, done, json })
      })
    )
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
          if (tokens) yield * tokens
        }
      }
    } finally {
      this.abort()
      this.dispose()
    }
  }

  abort () {
    if (this.ref) {
      mlx.abort(this.ref)
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
    if (!this.valid) throw new Error('Target unavailable')
    // todo: what if the user calls this on a model 🤔
    // should we create a cache on the fly MLXCache.fromModel(this)
    // or should we have swift accept no cache just to extract prompt stats?
    return new MLXEvaluate(this.model, this.chat, tokens, options)
  }
}

export class MLXCache extends MLXTarget {
  static CACHE_CONFIG_KEYS = ['maxKVSize', 'kvBits', 'kvGroupSize', 'quantizedKVStart']

  static async fromPath (path) { // Load from Disk
    const { promise, resolve, reject } = Promise.withResolvers()
    mlx.loadCache(path, (err, ref) => err ? reject(err) : resolve(new MLXCache(ref))) // c should throw: Error {message: '<from swift>' code: 'mlx_load_error'}
    return promise
  }

  static fromModel (model, options = {}) { // Create Empty Cache
    if (!model.available) throw new Error('Model unavailable')
    const ref = mlx.createCache(model.ref, JSON.stringify(options))
    return new MLXCache(ref, model)
  }

  #model = null

  get model () {
    return this.#model
  }

  get cache () {
    return this
  }

  constructor (ref, model) {
    super(ref)
    this.#model = model
  }

  async save (path) { // Save to Disk
    if (!this.available) throw new Error('Cache unavailable')
    const { promise, resolve, reject } = Promise.withResolvers()
    mlx.saveCache(this.ref, path, (err) => err ? reject(err) : resolve()) // c should throws: Error {message: '<from swift>' code: 'mlx_save_error'}
    return promise
  }

  trim (numTokens) {
    if (!this.ref) throw new Error('Cache unavailable')
    return mlx.trimCache(this.id, numTokens) // i assume this is instant otherwise we need to revert to async
  }

  clone () {
    if (!this.ref) throw new Error('Cache unavailable')
    return new MLXCache(mlx.cloneCache(this.ref)) // i assume this is instant otherwise we need to revert to async
  }
}

export class MLXModel extends MLXTarget {
  static MODEL_CONFIG_KEYS = ['maxTokens', 'temperature', 'topP', 'topK', 'minP', 'repetitionPenalty', 'repetitionContextSize', 'presencePenalty', 'presenceContextSize', 'frequencyPenalty', 'frequencyContextSize', 'prefillStepSize']

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
