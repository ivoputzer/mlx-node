import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const bridge = require('mlx-swift') // package.json (exports.require=mlx.node)

// API

export const load = bridge.load
export const unload = bridge.unload
export const abort = bridge.abort

export function metrics ({ metrics } = bridge) {
  return parseSafe(metrics())
}

export async function * generate (modelId, promptTokens, config = {}, { stream } = bridge) {
  const queue = []
  let resume = null

  stream(modelId, promptTokens, configFrom(config), (error, tokens, done, json) => {
    queue.push({ error, tokens, done, json })
    if (resume) {
      resume()
      resume = null
    }
  })

  while (true) {
    if (queue.length === 0) {
      const { promise, resolve } = Promise.withResolvers() // better than closure for gc
      resume = resolve
      await promise
    }

    const unqueue = queue.splice(0, queue.length)

    for (const { error, tokens, done, json } of unqueue) {
      if (error) throw error
      if (done) return parseSafe(json)
      if (tokens) yield * tokens
    }
  }
}

export default bridge

// HELPERS

function parseSafe (json, fallback = null) {
  if (!json) return fallback // fast path for empty responses
  try {
    return JSON.parse(json)
  } catch {
    return fallback
  }
}

function configFrom (config) {
  if (config.chunkSize > 2147483647) throw new Error('ChunkSize exceeds INT32_MAX (2147483647)')

  const allowedKeys = [
    'chunkSize',
    'maxTokens',
    'maxKVSize',
    'kvBits',
    'kvGroupSize',
    'quantizedKVStart',
    'temperature',
    'topP',
    'topK',
    'minP',
    'repetitionPenalty',
    'repetitionContextSize',
    'presencePenalty',
    'presenceContextSize',
    'frequencyPenalty',
    'frequencyContextSize',
    'prefillStepSize'
  ]

  return JSON.stringify(
    Object.fromEntries(Object.entries(config).filter(([key]) => allowedKeys.includes(key)))
  )
}
