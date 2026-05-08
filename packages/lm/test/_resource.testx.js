import { describe, it } from 'node:test'
import { strictEqual, throws } from 'node:assert/strict'

import mlx, { free } from 'mlx-node/swift'

describe('MLXResource', () => {
  it('named exports have the same function reference', () => {
    strictEqual(mlx.free, free)
  })

  it('Throws if instantiated without a native ref', () => {
    throws(() => new MLXResource(), { message: 'Native reference is required' })
  })

  it('Initializes correctly with a valid ref', () => {
    const dummyRef = {} // N-API Externals behave like empty JS objects
    const resource = new MLXResource(dummyRef)

    strictEqual(resource.ref, dummyRef)
    strictEqual(resource.available, true)
  })

  it('Calls mlx.free() when dispose is invoked', ({ mock }) => {
    const free = mock.method(mlx, 'free', () => true)
    const ref = {}
    const resource = new MLXResource(ref)

    const result = resource.dispose() // 2. Trigger the private #mlx.free via dispose()

    strictEqual(result, true, 'Dispose should return the result of mlx.free')
    strictEqual(resource.available, false, 'Resource should no longer be available')
    strictEqual(resource.ref, null, 'Reference should be nullified')

    strictEqual(free.mock.calls.length, 1, 'mlx.free should be called exactly once')
    strictEqual(free.mock.calls[0].arguments[0], ref, 'mlx.free should receive the correct ref')
  })

  it('Safely handles double dispose without throwing or double-freeing', ({ mock }) => {
    const free = mock.method(mlx, 'free', () => true)
    const resource = new MLXResource({})

    resource.dispose() // First time: success
    const secondDispose = resource.dispose() // Second time: skips safely

    strictEqual(secondDispose, false, 'Second dispose should return false')
    strictEqual(free.mock.calls.length, 1, 'mlx.free should STILL only be called once')
  })
})

class MLXResource {
  #mlx = mlx
  #ref = null

  constructor (ref) {
    if (!ref) throw new Error('Native reference is required')
    this.#ref = ref
  }

  get ref () {
    return this.#ref
  }

  get available () {
    return this.#ref !== null
  }

  dispose () {
    if (!this.#ref) return false
    const success = this.#mlx.free(this.#ref)
    this.#ref = null
    return success
  }

  [Symbol.dispose] () {
    this.dispose()
  }
}
