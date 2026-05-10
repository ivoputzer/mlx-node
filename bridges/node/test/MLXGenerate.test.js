import { describe, it } from 'node:test'
import { deepStrictEqual, ok, rejects, strictEqual } from 'node:assert'

import mlx from 'mlx-swift'
import { MLXModel, MLXGenerate } from '../index.js'

describe('MLXGenerate', () => {
  it('yields tokens and returns final stats', async ({ mock }) => {
    let nativeCallback

    mock.method(mlx, 'generateTask', (model, cache, tokens, config, callback) => {
      nativeCallback = callback // Capture the callback to push simulated events
      return { taskPtr: 1 }
    })

    const stream = new MLXGenerate(new MLXModel({ ptr: 1 }), null, [1, 2], { batchSize: 1 })

    // Simulate native stream events asynchronously
    setTimeout(() => {
      // Push 1 tick (token 99)
      nativeCallback(null, new Int32Array([99]), null, null, 0, false, null)
      // Push 1 tick (token 100)
      nativeCallback(null, new Int32Array([100]), null, null, 0, false, null)
      // Push done signal with JSON stats
      nativeCallback(null, null, null, null, 0, true, '{"generatedTokens": 2}')
    }, 10)

    const receivedTokens = []
    for await (const tick of stream) {
      receivedTokens.push(tick[0]) // tick is an array across the batch size
    }

    deepStrictEqual(receivedTokens, [99, 100])

    // Test that the generator return value (stats) can't be fetched inside 'for await'
    // but the task should be properly disposed by the 'finally' block.
    ok(!stream.available)
  })

  it('unpacks topLogits correctly', async ({ mock }) => {
    let nativeCallback
    mock.method(mlx, 'generateTask', (m, c, t, cfg, callback) => {
      nativeCallback = callback
      return { taskPtr: 1 }
    })

    const stream = new MLXGenerate(new MLXModel({ ptr: 1 }), null, [1], { batchSize: 1 })

    setTimeout(() => {
      // Send 1 token, with TopK=2
      nativeCallback(
        null,
        new Int32Array([42]), // tokens
        new Int32Array([42, 43]), // topTokens
        new Float32Array([0.9, 0.1]), // topProbs
        2, // topK
        false,
        null
      )
      nativeCallback(null, null, null, null, 0, true, '{"done": true}')
    }, 5)

    let firstTick = null
    for await (const tick of stream) {
      firstTick = tick
    }

    strictEqual(firstTick[0], 42)
    deepStrictEqual(firstTick.topLogits, [
      [
        { id: 42, prob: Math.fround(0.9) },
        { id: 43, prob: Math.fround(0.1) }
      ]
    ])
  })

  it('throws and aborts immediately on native stream error', async ({ mock }) => {
    let nativeCallback
    const abortMock = mock.method(mlx, 'abortTask', () => {})
    mock.method(mlx, 'generateTask', (m, c, t, cfg, callback) => {
      nativeCallback = callback
      return { taskPtr: 1 }
    })

    const stream = new MLXGenerate(new MLXModel({ ptr: 1 }), null, [1])

    setTimeout(() => {
      nativeCallback(new Error('CUDA out of memory? Oh wait, this is Metal!'))
    }, 5)

    await rejects(
      async () => {
        await Array.fromAsync(stream)
      },
      /Metal/
    )

    strictEqual(abortMock.mock.callCount(), 1, 'abortTask must be called in finally block')
    ok(!stream.available, 'Task must be disposed')
  })
})
