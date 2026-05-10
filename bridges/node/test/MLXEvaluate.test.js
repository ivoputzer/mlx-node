import { describe, it } from 'node:test'
import { ok, rejects, deepStrictEqual } from 'node:assert'

import mlx from 'mlx-swift'
import { MLXModel, MLXEvaluate } from '../index.js'

describe('MLXEvaluate', () => {
  it('acts as a Promise and resolves stats on success', async ({ mock }) => {
    const mockModel = new MLXModel({ modelPtr: 1 })

    mock.method(mlx, 'evaluateTask', (m, c, t, cfg, callback) => {
      // Simulate C executing in background, then resolving
      setImmediate(() => callback(null, '{"promptTokens": 10}'))
      return { taskPtr: 1 }
    })

    const evaluate = new MLXEvaluate(mockModel, null, [1, 2, 3])
    const stats = await evaluate

    deepStrictEqual(stats, { promptTokens: 10, stopReason: 'prefill' })
    ok(!evaluate.available, 'Evaluate task should auto-dispose on completion')
  })

  it('rejects and cleans up on native error', async ({ mock }) => {
    mock.method(mlx, 'evaluateTask', (m, c, t, cfg, callback) => {
      setImmediate(() => callback(new Error('eval_crash')))
      return { taskPtr: 1 }
    })

    const evaluate = new MLXEvaluate(new MLXModel({ ptr: 1 }), null, [1])

    await rejects(() => evaluate, /eval_crash/)
    ok(!evaluate.available)
  })
})
