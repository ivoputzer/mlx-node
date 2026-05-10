import { describe, it } from 'node:test'
import { ok, deepEqual, strictEqual } from 'node:assert/strict'
import { createRequire } from 'node:module'

import mlx from 'mlx-swift' // Adjust path to your file

const require = createRequire(import.meta.url)
const native = require('mlx-swift')

describe('mlx-swift', () => {
  it('should contain all keys present in the native module', () => {
    for (const key of Object.getOwnPropertyNames(native)) {
      ok(key in mlx, `Driver is missing key: ${key}`)
      strictEqual(typeof mlx[key], typeof native[key], `Type mismatch for key: ${key}`)
    }
  })

  it('should maintain functional parity with native methods', () => {
    const { version } = require('./package.json')
    strictEqual(mlx.version, version, 'Native method call failed through driver')
  })

  describe('Mocking Capabilities', () => {
    it('should allow mocking methods', ({ mock }) => {
      mock.method(mlx, 'systemMetrics', Function.prototype)
      strictEqual(mlx.systemMetrics(), undefined, 'The method was not successfully mocked')
      strictEqual(mlx.systemMetrics.mock.calls.length, 1)
    })

    it('should have configurable property descriptors', () => {
      // N-API default properties are usually non-configurable.
      // Our wrapper should have made them configurable.
      const firstKey = Object.keys(mlx)[0]
      const descriptor = Object.getOwnPropertyDescriptor(mlx, firstKey)

      strictEqual(descriptor.configurable, true, `Property ${firstKey} should be configurable`)
      strictEqual(descriptor.writable, true, `Property ${firstKey} should be writable`)
    })
  })
})
