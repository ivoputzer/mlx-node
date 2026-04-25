import { it, describe } from 'node:test'
import { ok, equal, rejects, deepEqual, strictEqual, doesNotThrow } from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { createRequire } from 'node:module'

import mlx, { load as loadModel, unload as unloadModel, stream as generateStream, generate } from 'mlx-swift'

describe('mlx-node (native bridge)', () => {
  it('Supports both require and import', () => {
    const require = createRequire(import.meta.url)
    strictEqual(mlx, require('mlx-swift'))
  })

  it('Exports all expected properties', () => {
    ok(mlx, 'Module should exist')
    strictEqual(typeof loadModel, 'function', 'loadModel should be exported')
    strictEqual(typeof unloadModel, 'function', 'unloadModel should be exported')
    strictEqual(typeof generateStream, 'function', 'generateStream should be exported')
  })

  it('Has default.metallib bundled and locatable', () => {
    const metallibPath = join(import.meta.dirname, 'default.metallib')
    ok(existsSync(metallibPath), 'default.metallib must exist next to the binary')
  })

  describe('.loadModel', () => {
    it('Handles invalid paths gracefully via promise rejection', async () => {
      await rejects(
        loadModel('/path/to/absolute/nowhere/fake_model'),
        // In C-bridge, success=false triggers napi_reject_deferred.
        // Ensure this actually throws a JS error instead of segfaulting.
        (error) => {
          strictEqual(error.code, 'MLX_ERR', 'Error should have custom code MLX_ERR')
          ok(error.message.length > 0, 'Error should have a message from Swift')
          return true
        },
        'Should have thrown an error for invalid model path'
      )
    })
  })

  describe('.unloadModel', () => {
    it('Returns false for non-existent model IDs', () => {
      const result = unloadModel(99999)
      strictEqual(result, false, 'Unloading invalid model ID should return false')
    })
  })

  describe('.generateStream', () => {
    it('Handles gpu being faster than js (queue fills up)', async ({ mock }) => {
      const generateStreamMock = mock.fn((id, tokens, config, cb) => {
        cb(null, new Int32Array([1, 2]), false, null)
        cb(null, new Int32Array([3, 4]), false, null)
        cb(null, null, true, '{"stopReason":"stop"}') // GPU is super fast: fires everything synchronously before JS can pull
      })

      const stream = generateStream(1, new Int32Array([99]), '{"temp":0.7}', { generateStream: generateStreamMock })

      // JS is slow, pulls from the queue after it's already full
      const result1 = await stream.next()
      deepEqual(result1.value, new Int32Array([1, 2]))
      equal(result1.done, false)

      const result2 = await stream.next()
      deepEqual(result2.value, new Int32Array([3, 4]))
      equal(result2.done, false)

      const result3 = await stream.next()
      deepEqual(result3.value.stopReason, 'stop') // stats
      equal(result3.done, true)

      // Verify the C-Bridge was called correctly
      equal(generateStreamMock.mock.calls.length, 1)
      equal(generateStreamMock.mock.calls[0].arguments[0], 1) // modelId
      equal(generateStreamMock.mock.calls[0].arguments[2], '{"temp":0.7}') // config:GenerationProperties
    })

    it('Handles JS being faster than GPU (JS awaits Promises)', async ({ mock }) => {
      let storedCallback
      const generateStreamMock = mock.fn((id, tokens, config, cb) => {
        storedCallback = cb // Keep the callback to trigger manually
      })

      const stream = generateStream(1, new Int32Array([]), '{}', { generateStream: generateStreamMock })

      // JS asks for next token before C has provided it. Promise is created internally.
      const promise1 = stream.next()

      // Simulate GPU taking 5ms to generate tokens
      setTimeout(() => storedCallback(null, new Int32Array([99]), false, null), 5)

      const result1 = await promise1
      deepEqual(result1.value, new Int32Array([99]))

      // Ask for stats
      const promise2 = stream.next()
      setTimeout(() => storedCallback(null, null, true, '{"stopReason":"length"}'), 5)

      const result2 = await promise2

      equal(result2.done, true)
      equal(result2.value.stopReason, 'length')
    })

    it('Handles c-bridge errors correctly', async ({ mock }) => {
      const generateStreamMock = mock.fn((id, tokens, config, cb) => {
        cb(null, new Int32Array([10]), false, null)
        cb(new Error('Metal out of memory'), null, true, null) // GPU outputs one chunk, then crashes
      })

      const stream = generateStream(1, new Int32Array([]), '{}', { generateStream: generateStreamMock })
      const { value, done } = await stream.next()

      deepEqual(value, new Int32Array([10]))
      deepEqual(done, false)

      await rejects(async () => await stream.next(), { message: 'Metal out of memory' })
    })

    it('Handles malformed payload/stats gracefully', async ({ mock }) => {
      const generateStreamMock = mock.fn((id, tokens, config, cb) => {
        cb(null, null, true, '{"broken_json: oops')
      })

      const stream = generateStream(1, new Int32Array([]), '{}', { generateStream: generateStreamMock })
      const { done, value } = await stream.next()

      equal(done, true)
      equal(value, null) // Gracefully falls back to null instead of throwing
    })

    it('Handles multiple parallel streams without crossing wires', async ({ mock }) => {
      const generateStreamMock = mock.fn((id, tokens, config, cb) => {
        setTimeout(() => {
          cb(null, new Int32Array([id]), false, null) // Each "model" returns its own ID as a token
          cb(null, null, true, JSON.stringify({ id }))
        }, Math.random() * 10) // Randomize timing to stress test concurrency
      })

      // Launch 50 streams in parallel
      const streams = Array.from({ length: 50 }, async (_, i) => {
        const stream = generateStream(i, new Int32Array([]), '{}', { generateStream: generateStreamMock })
        const result = await stream.next() // Get first token
        const stats = await stream.next() // Get stats (done)
        return { token: result.value[0], statsId: stats.value.id }
      })

      const results = await Promise.all(streams)

      // Assert that every stream got its own correct ID back
      results.forEach((res, i) => {
        strictEqual(res.token, i, `Stream ${i} got wrong token`)
        strictEqual(res.statsId, i, `Stream ${i} got wrong stats`)
      })
    })

    it('Maintains order and data integrity when V8 is busy', async ({ mock }) => {
      let storedCallback

      const generateStreamMock = mock.fn((id, tokens, config, cb) => {
        storedCallback = cb
      })

      const stream = generateStream(1, new Int32Array([]), '{}', { generateStream: generateStreamMock })

      // 1. Ask for the first token (JS is now awaiting)
      const firstPromise = stream.next()

      // 2. Flood the queue while JS is "away"
      storedCallback(null, new Int32Array([1]), false, null)
      storedCallback(null, new Int32Array([2]), false, null)
      storedCallback(null, new Int32Array([3]), false, null)
      storedCallback(null, null, true, '{"done":true}')

      // 3. Block the event loop for a moment
      const start = Date.now()
      while (Date.now() - start < 50) { /* spinning... */ }

      // 4. Check results
      const r1 = await firstPromise
      const r2 = await stream.next()
      const r3 = await stream.next()
      const r4 = await stream.next()

      deepEqual(r1.value, new Int32Array([1]))
      deepEqual(r2.value, new Int32Array([2]))
      deepEqual(r3.value, new Int32Array([3]))
      strictEqual(r4.done, true)
    })

    it('Handles late-arriving callbacks after the stream is closed', async (t) => {
      let storedCallback
      const generateStreamMock = t.mock.fn((id, tokens, config, cb) => {
        storedCallback = cb // This only runs AFTER stream.next() is called
      })

      const mockNative = { generateStream: generateStreamMock }
      const stream = generateStream(1, new Int32Array([]), '{}', mockNative)

      // 1. PRIME THE GENERATOR
      // We call .next() but we don't await it yet because it's waiting for a callback
      const firstRequest = stream.next()

      // Now, the generator body has executed up to the native call,
      // so storedCallback IS a function.
      strictEqual(typeof storedCallback, 'function')

      // 2. Force a crash
      storedCallback(new Error('First Error'), null, true, null)

      // 3. Now await the request and it should reject
      await rejects(firstRequest, { message: 'First Error' })

      // 4. Test the "Late" callback
      // This simulates the Swift thread firing again even though JS is done.
      // It should not throw because our C-callback logic handles TSFN release.
      doesNotThrow(() => {
        storedCallback(null, new Int32Array([99]), false, null)
      })
    })
  })
})
