import { test, describe } from 'node:test'
import { ok, fail, strictEqual } from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { createRequire } from 'node:module'

import mlx, { loadModel, unloadModel, generate, generateStream } from '../index.js'

describe('mlx-node (native bridge)', () => {
  test('supports both require and import', () => {
    const require = createRequire(import.meta.url)
    strictEqual(mlx, require('mlx-swift'))
  })

  test('default.metallib is bundled and locatable', () => {
    const metallibPath = join(import.meta.dirname, '..', 'default.metallib')
    ok(existsSync(metallibPath), 'default.metallib must exist next to the binary')
  })

  test('native module exports all expected properties', () => {
    ok(mlx, 'Module should exist')

    strictEqual(typeof loadModel, 'function', 'loadModel should be exported')
    strictEqual(typeof unloadModel, 'function', 'unloadModel should be exported')
    strictEqual(typeof generate, 'function', 'generate should be exported')
    strictEqual(typeof generateStream, 'function', 'generateStream should be exported')
  })

  test('loadModel handles invalid paths gracefully via Promise rejection', async () => {
    // In your C-code, success=false triggers napi_reject_deferred.
    // We want to ensure this actually throws a JS error instead of segfaulting.
    try { // fixme: use assert.rejects instead
      await loadModel('/path/to/absolute/nowhere/fake_model')
      fail('Should have thrown an error for invalid model path')
    } catch (error) {
      strictEqual(error.code, 'MLX_ERR', 'Error should have custom code MLX_ERR')
      ok(error.message.length > 0, 'Error should have a message from Swift')
    }
  })

  test('unloadModel returns false for non-existent model IDs', () => {
    // Unloading a random integer should return boolean false, not crash.
    const result = unloadModel(99999)
    strictEqual(result, false, 'Unloading invalid model ID should return false')
  })

  test('generateStream handles JS callbacks properly', () => {
    return new Promise((resolve) => {
      // Testing N-API ThreadSafeFunctions is crucial.
      // We pass an invalid model, we expect the callback to be fired with an error.
      generateStream(99999, 'Hello', '{}', (errorMsg, chunk, isDone) => {
        ok(errorMsg, 'Should receive an error message')
        strictEqual(chunk, null, 'Chunk should be null on error')
        strictEqual(isDone, true, 'Stream should close on error')
        resolve()
      })
    })
  })
})

// import { createRequire } from 'module'

// const require = createRequire(import.meta.url)
// const binary = require('../mlx_swift.node')

// // import { load, stream } from '../index.js'
// // import { Readable } from 'node:stream'

// // const modelId = await load('/Users/ivoputzer/github/models/Jackrong/MLX-Qwopus3.5-9B-v3-8bit')
// // // const text = await applyChatTemplate(modelId, [{ role: 'system', content: 'You are a helpful assistant' }]) // array token ma la stringa compilata

// // // const response = await generate(modelId, text)
// // // console.log(response)

// // // for await (const chunk of ) {
// // //   console.log(chunk)
// // // }

// // const s = stream(modelId, 'this is the raw template', { streamChunkSize: 0 })

// // Readable.from(s).pipe(process.stdout)

// // // const loaded = native.load()

// // // const native = createFactory()

// // // console.log(process.env)

// // // console.log(native.load)
// // // console.log(native.generate)

// // // console.log(loaded)
// // // await loaded

// // // if (loaded) {
// // //   console.log('Generating...')
// // //   for await (const element of native.stream('Why is Swift better than Rust for MLX on MacOS?')) {
// // //     console.log(element)
// // //   }
// // //   // console.log(p)
// // //   // const result = await p
// // //   // console.log('Result:', result)
// // // } else {
// // //   console.log('Failed to load model.')
// // // }
