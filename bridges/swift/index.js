import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const mlx = require('mlx-swift') // package.json|exports.require=./mlx.node

// API

export const load = mlx.load
export const unload = mlx.unload
export const abort = mlx.abort

export function metrics ({ metrics } = mlx) {
  try {
    return JSON.parse(metrics())
  } catch (_) {
    return {}
  }
}

export const generate = async (modelId, promptTokens, config, { stream } = mlx) => {
  return new Promise((resolve, reject) => {
    const streamChunkSize = 2147483647 // int32_t
    const configJson = JSON.stringify({ ...config, streamChunkSize })
    stream(modelId, promptTokens, configJson, (error, tokens, done, stats) => {
      if (error) {
        return reject(error)
      } else if (done) {
        try {
          resolve({ tokens, stats: JSON.parse(stats) })
        } catch (_) {
          resolve({ tokens })
        }
      }
    })
  })
}

export async function * stream (modelId, prompt, config, { stream } = mlx) {
  const queue = []
  let resolveNext = null
  let rejectNext = null
  let isFinished = false

  const configJson = JSON.stringify(
    Object.fromEntries(
      Object.entries(config).filter(([key]) => ['streamChunkSize', 'maxTokens', 'maxKVSize', 'kvBits', 'kvGroupSize', 'quantizedKVStart', 'temperature', 'topP', 'topK', 'minP', 'repetitionPenalty', 'repetitionContextSize', 'presencePenalty', 'presenceContextSize', 'frequencyPenalty', 'frequencyContextSize', 'prefillStepSize'].includes(key))
    )
  )

  // config should be converted into JSON inside here
  stream(modelId, prompt, configJson, (cause, tokens, done, json) => {
    if (cause) {
      const error = cause instanceof Error ? cause : new Error(cause?.message || 'Unknown error')
      if (resolveNext) { rejectNext(error); resolveNext = null; rejectNext = null } else queue.push({ err: error })
      isFinished = true
    } else if (done) {
      let stats = null
      if (json) try { stats = JSON.parse(json) } catch (e) {}
      if (resolveNext) { resolveNext({ done: true, stats }); resolveNext = null; rejectNext = null } else queue.push({ done: true, stats })
      isFinished = true
    } else if (tokens) {
      if (resolveNext) { resolveNext({ tokens }); resolveNext = null; rejectNext = null } else queue.push({ tokens })
    }
  })

  while (true) {
    let item
    if (queue.length > 0) {
      item = queue.shift()
    } else if (isFinished) {
      break
    } else {
      item = await new Promise((resolve, reject) => {
        resolveNext = resolve
        rejectNext = reject
      })
    }

    if (item.err) throw item.err
    if (item.done) return item.stats // RETURN THE STATS!
    yield item.tokens // YIELD RAW ARRAYS!
  }
}

export default mlx
