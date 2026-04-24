import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const native = require('mlx-swift')

export const loadModel = native.loadModel // fixme: load
export const unloadModel = native.unloadModel // fixme: unload|dispose

// export const generateStream = native.generateStream // fixme: stream|generate
export const cancelGenerate = native.cancelGenerate // fixme: cancel|abort

export async function * generateStream (modelId, promptTokens, configJson, { generateStream } = native) {
  const queue = []
  let resolveNext = null
  let rejectNext = null
  let isFinished = false

  const callback = (cause, chunkInt32Array, isDone, payloadStr) => {
    if (cause) {
      const error = new Error(cause.message, { cause })
      if (resolveNext) { rejectNext(error); resolveNext = null; rejectNext = null } else queue.push({ err: error })
      isFinished = true
    } else if (isDone) {
      let stats = null
      if (payloadStr) try { stats = JSON.parse(payloadStr) } catch (e) {}
      if (resolveNext) { resolveNext({ done: true, stats }); resolveNext = null; rejectNext = null } else queue.push({ done: true, stats })
      isFinished = true
    } else if (chunkInt32Array) {
      if (resolveNext) { resolveNext({ tokens: chunkInt32Array }); resolveNext = null; rejectNext = null } else queue.push({ tokens: chunkInt32Array })
    }
  }

  generateStream(modelId, promptTokens, configJson, callback)

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

export default native
