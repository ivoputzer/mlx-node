import { describe, it } from 'node:test'
import { ok, strictEqual, throws } from 'node:assert'

import mlx from 'mlx-swift'
import { MLXResource } from '../index.js'
import { deepStrictEqual } from 'node:assert/strict'

describe('MLXResource', () => {
  it('throws on instantiation without a native pointer', () => {
    throws(() => new MLXResource(null), /Invalid resource pointer/)
  })

  it('initializes correctly with a valid ref and reports availability correctly', () => {
    const dummyRef = {} // N-API Externals behave like empty JS objects
    const res = new MLXResource(dummyRef)
    ok(res.available)
    strictEqual(res.ref, dummyRef)
  })

  it('disposes only once and calls mlx.freeResource', ({ mock }) => {
    const freeMock = mock.method(mlx, 'freeResource', Function.prototype)
    const dummyRef = {}
    const res = new MLXResource(dummyRef)

    ok(res.dispose(), 'First dispose should return true')
    ok(!res.available, 'Resource should no longer be available')
    ok(!res.dispose(), 'Second dispose should return false')

    strictEqual(freeMock.mock.callCount(), 1, 'Native freeResource called exactly once')
    strictEqual(freeMock.mock.calls[0].arguments[0], dummyRef, 'Native freeResource should receive the correct ref')
  })
})
