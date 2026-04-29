import { it, describe } from 'node:test'
import { ok, equal, rejects, deepEqual, strictEqual, doesNotThrow } from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { createRequire } from 'node:module'

import mlx, { metrics, load, unload, abort, generate } from 'mlx-swift'

describe('mlx-swift (native bridge)', () => {
  it('Supports both require and import', () => {
    const require = createRequire(import.meta.url)
    strictEqual(mlx, require('mlx-swift'))
  })

  it('Exports all expected properties', () => {
    ok(mlx, 'Module should exist')
    strictEqual(typeof metrics, 'function', 'metrics should be exported')
    strictEqual(typeof load, 'function', 'load should be exported')
    strictEqual(typeof unload, 'function', 'unload should be exported')
    strictEqual(typeof generate, 'function', 'generate should be exported')
    strictEqual(typeof abort, 'function', 'abort should be exported')
  })

  it('Has default.metallib bundled and locatable', () => {
    const metallibPath = join(import.meta.dirname, 'default.metallib')
    ok(existsSync(metallibPath), 'default.metallib must exist next to the binary')
  })

  describe('.metrics', () => {
  })

  describe('.load', () => {
    it('Handles invalid paths gracefully via promise rejection', async () => {
      await rejects(
        load('/path/to/absolute/nowhere/fake_model'),
        // In C-bridge, success=false triggers napi_reject_deferred.
        // Ensure this actually throws a JS error instead of segfaulting.
        (error) => {
          strictEqual(error.code, 'MLX_LOAD_ERR', 'Error should have custom code MLX_LOAD_ERR')
          ok(error.message.length > 0, 'Error should have a message from Swift')
          return true
        },
        'Should have thrown an error for invalid model path'
      )
    })
  })

  describe('.unload', () => {
    it('Returns false for non-existent model IDs', () => {
      const result = unload(99999)
      strictEqual(result, false, 'Unloading invalid model ID should return false')
    })
  })

  describe('.generate', () => {
    it.todo('serializes valid configuration into json')
    it.todo('skips invalid configuration parameters json')

    it('Handles gpu being faster than js (queue fills up)', async ({ mock }) => {
      const streamMock = mock.fn((id, tokens, config, cb) => {
        cb(null, new Int32Array([1, 2]), false, null)
        cb(null, new Int32Array([3, 4]), false, null)
        cb(null, null, true, '{"stopReason":"stop"}') // GPU is super fast
      })

      const chunks = generate(1, new Int32Array([99]), { temperature: 0.7 }, { stream: streamMock })

      const result1 = await chunks.next()
      strictEqual(result1.value, 1)
      strictEqual(result1.done, false)

      const result2 = await chunks.next()
      strictEqual(result2.value, 2)
      strictEqual(result2.done, false)

      const result3 = await chunks.next()
      strictEqual(result3.value, 3)
      strictEqual(result3.done, false)

      const result4 = await chunks.next()
      strictEqual(result4.value, 4)
      strictEqual(result4.done, false)

      const result5 = await chunks.next()
      strictEqual(result5.value.stopReason, 'stop') // stats
      strictEqual(result5.done, true)

      // Verify the C-Bridge was called correctly
      strictEqual(streamMock.mock.calls.length, 1)
      strictEqual(streamMock.mock.calls[0].arguments[0], 1) // modelId
      strictEqual(streamMock.mock.calls[0].arguments[2], JSON.stringify({ temperature: 0.7 }))
    })

    it('Handles JS being faster than GPU (JS awaits Promises)', async ({ mock }) => {
      let storedCallback
      const streamMock = mock.fn((id, tokens, config, cb) => {
        storedCallback = cb // Keep the callback to trigger manually
      })

      const chunks = generate(1, new Int32Array([]), {}, { stream: streamMock })

      // JS asks for next token before C has provided it. Promise is created internally.
      const promise1 = chunks.next()

      // Simulate GPU taking 5ms to generate tokens
      setTimeout(() => storedCallback(null, new Int32Array([99]), false, null), 5)

      const result1 = await promise1
      strictEqual(result1.value, 99) // Unrolled from array

      // Ask for stats
      const promise2 = chunks.next()
      setTimeout(() => storedCallback(null, null, true, '{"stopReason":"length"}'), 5)

      const result2 = await promise2

      strictEqual(result2.done, true)
      strictEqual(result2.value.stopReason, 'length')
    })

    it('Handles c-bridge errors correctly', async ({ mock }) => {
      const streamFn = mock.fn((id, tokens, config, cb) => {
        cb(null, new Int32Array([10]), false, null)
        cb(new Error('Metal out of memory'), null, true, null) // GPU outputs one chunk, then crashes
      })

      const chunks = generate(1, new Int32Array([]), '{}', { stream: streamFn })
      const { value, done } = await chunks.next()

      strictEqual(value, 10) // Unrolled from array
      strictEqual(done, false)

      await rejects(async () => await chunks.next(), { message: 'Metal out of memory' })
    })

    it('Handles malformed payload/stats gracefully', async ({ mock }) => {
      const streamFn = mock.fn((id, tokens, config, cb) => {
        cb(null, null, true, '{"broken_json: oops')
      })

      const chunks = generate(1, new Int32Array([]), '{}', { stream: streamFn })
      const { done, value } = await chunks.next()

      strictEqual(done, true)
      strictEqual(value, null) // Gracefully falls back to null instead of throwing
    })

    it('Handles multiple parallel streams without crossing wires', async ({ mock }) => {
      const streamFm = mock.fn((id, tokens, config, cb) => {
        setTimeout(() => {
          cb(null, new Int32Array([id]), false, null) // Each "model" returns its own ID as a token
          cb(null, null, true, JSON.stringify({ id }))
        }, Math.random() * 10) // Randomize timing to stress test concurrency
      })

      // Launch 50 streams in parallel
      const streams = Array.from({ length: 50 }, async (_, i) => {
        const chunks = generate(i, new Int32Array([]), '{}', { stream: streamFm })
        const result = await chunks.next() // Get first token
        const stats = await chunks.next() // Get stats (done)

        // Notice result.value is now directly the number, not result.value[0]
        return { token: result.value, statsId: stats.value.id }
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

      const streamFn = mock.fn((id, tokens, config, cb) => {
        storedCallback = cb
      })

      const chunks = generate(1, new Int32Array([]), '{}', { stream: streamFn })

      // 1. Ask for the first token (JS is now awaiting)
      const firstPromise = chunks.next()

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
      const r2 = await chunks.next()
      const r3 = await chunks.next()
      const r4 = await chunks.next()

      strictEqual(r1.value, 1)
      strictEqual(r2.value, 2)
      strictEqual(r3.value, 3)
      strictEqual(r4.done, true)
    })

    it('Handles late-arriving callbacks after the stream is closed', async (t) => {
      let storedCallback
      const streamFn = t.mock.fn((id, tokens, config, cb) => {
        storedCallback = cb
      })

      const chunks = generate(1, new Int32Array([]), '{}', { stream: streamFn })

      // 1. PRIME THE GENERATOR
      const firstRequest = chunks.next()
      strictEqual(typeof storedCallback, 'function')

      // 2. Force a crash
      storedCallback(new Error('First Error'), null, true, null)

      // 3. Now await the request and it should reject
      await rejects(firstRequest, { message: 'First Error' })

      // 4. Test the "Late" callback
      doesNotThrow(() => {
        storedCallback(null, new Int32Array([99]), false, null)
      })
    })
  })
})
