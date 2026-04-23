import { swift as native } from 'mlx-node'

export async function load (path) {
  return native.loadModel(path)
}

export function unload (modelId) {
  return native.unloadModel(modelId)
}

/**
 * Generates a complete output (non-blocking).
 * @param {number} modelId
 * @param {string} prompt - The raw string (pre-formatted with templates if needed)
 * @param {Object} [config] - e.g., { temperature: 0.7, topP: 0.9, maxTokens: 1000 }
 * @returns {Promise<string>}
 */
export async function generate (modelId, prompt, config = {}) {
  const configJson = JSON.stringify(config)
  return native.generate(modelId, prompt, configJson)
}

/**
 * Streams generation chunks back to JS, accumulated by streamChunkSize.
 * @param {number} modelId
 * @param {string} prompt - The raw string
 * @param {Object} [config] - e.g., { temperature: 0.7, streamChunkSize: 5 }
 * @returns {AsyncGenerator<string, void, unknown>}
 */
export async function * stream (modelId, prompt, config = {}) {
  const configJson = JSON.stringify(config)

  const queue = []
  let resolveNext = null
  let rejectNext = null
  let isFinished = false

  const streamCallback = (err, chunk, isDone) => {
    if (err) {
      const error = new Error(err)
      if (resolveNext) {
        rejectNext(error)
        resolveNext = null
        rejectNext = null
      } else {
        queue.push({ err: error })
      }
      isFinished = true
    } else if (isDone) {
      if (resolveNext) {
        resolveNext({ done: true })
        resolveNext = null
        rejectNext = null
      } else {
        queue.push({ done: true })
      }
      isFinished = true
    } else {
      if (resolveNext) {
        resolveNext({ value: chunk })
        resolveNext = null
        rejectNext = null
      } else {
        queue.push({ value: chunk })
      }
    }
  }

  native.generateStream(modelId, prompt, configJson, streamCallback)

  while (true) {
    if (queue.length > 0) {
      const item = queue.shift()
      if (item.err) throw item.err
      if (item.done) break
      yield item.value
    } else if (isFinished) {
      break
    } else {
      const item = await new Promise((resolve, reject) => {
        resolveNext = resolve
        rejectNext = reject
      })
      if (item.err) throw item.err
      if (item.done) break
      yield item.value
    }
  }
}
