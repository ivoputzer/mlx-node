import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const bridge = require('mlx-swift') // package.json|exports.require=./mlx.node

// API

export const load = bridge.load
export const unload = bridge.unload
export const abort = bridge.abort

export function metrics ({ metrics } = bridge) {
  try {
    return JSON.parse(metrics())
  } catch (_) {
    return {}
  }
}

function configJson (config) {
  if (config.streamChunkSize > 2147483647) throw new Error('StreamChunkSize INT32_MAX=2147483647')
  return JSON.stringify(
    Object.fromEntries(
      Object.entries(config).filter(([key]) => ['streamChunkSize', 'maxTokens', 'maxKVSize', 'kvBits', 'kvGroupSize', 'quantizedKVStart', 'temperature', 'topP', 'topK', 'minP', 'repetitionPenalty', 'repetitionContextSize', 'presencePenalty', 'presenceContextSize', 'frequencyPenalty', 'frequencyContextSize', 'prefillStepSize'].includes(key))
    )
  )
}

export const generate = async (modelId, promptTokens, config, { stream } = bridge) => {
  // we cannot resolve right away
  // bridge will fire callback twice (once with tokens, once with done & stats)
  return new Promise((resolve, reject) => {
    let bufferedTokens = []
    stream(modelId, promptTokens, configJson({ ...config, streamChunkSize: 2147483647 }), (error, tokens, done, stats) => {
      if (error) {
        return reject(error)
      } else if (tokens) { // done === false
        bufferedTokens = tokens
      } else if (done) { // tokens === null
        try {
          resolve({ tokens: bufferedTokens, stats: JSON.parse(stats) })
        } catch (_) {
          resolve({ tokens: bufferedTokens })
        }
      }
    })
  })
}

export async function * stream (modelId, prompt, config, { stream } = bridge) {
  const queue = []
  let resolveNext = null
  let rejectNext = null
  let isFinished = false

  stream(modelId, prompt, configJson(config), (cause, tokens, done, json) => {
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

export default bridge
