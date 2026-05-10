import { describe, it } from 'node:test'
import { strictEqual } from 'node:assert'

import mlx from 'mlx-swift'
import { MLXTask } from '../index.js'

describe('MLXTask', () => {
  it('aborts using native driver', ({ mock }) => {
    const abortMock = mock.method(mlx, 'abortTask', Function.prototype)
    const task = new MLXTask({ pointer: 1 })

    task.abort()
    strictEqual(abortMock.mock.callCount(), 1)
  })

  it('does not abort if already disposed', ({ mock }) => {
    const abortMock = mock.method(mlx, 'abortTask', Function.prototype)
    mock.method(mlx, 'freeResource', Function.prototype)

    const task = new MLXTask({ pointer: 1 })

    task.dispose()
    task.abort() // Should do nothing

    strictEqual(abortMock.mock.callCount(), 0)
  })
})
