import { describe, it } from 'node:test'
import { ok, strictEqual, throws } from 'node:assert'

import mlx from 'mlx-swift'
import { MLXResource } from '../index.js'

describe('MLXResource', () => {
  it('throws on instantiation without a native pointer', () => {
    throws(() => new MLXResource(null), /Invalid resource pointer/)
  })

  it('reports availability correctly', () => {
    const res = new MLXResource({ dummy: true })
    ok(res.available)
    strictEqual(res.ref.dummy, true)
  })

  it('disposes only once and calls mlx.freeResource', ({ mock }) => {
    const freeMock = mock.method(mlx, 'freeResource', Function.prototype)
    const res = new MLXResource({ dummy: true })

    ok(res.dispose(), 'First dispose should return true')
    ok(!res.available, 'Resource should no longer be available')
    ok(!res.dispose(), 'Second dispose should return false')

    strictEqual(freeMock.mock.callCount(), 1, 'Native freeResource called exactly once')
  })
})
