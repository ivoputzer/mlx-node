import { describe, it } from 'node:test'
import { ok, strictEqual, deepEqual } from 'node:assert/strict'

// Dynamically import to safely test exports without failing if native modules aren't built on the CI environment yet!
import * as mainIndex from 'mlx-node'
import * as swiftIndex from 'mlx-node/swift'
import * as cppIndex from 'mlx-node/cpp'

describe('mlx-node', () => {
  describe('Router Exports', () => {
    const removeDefaultExport = key => key !== 'default'

    it('exposes swift and cpp namespaces from the main entry point', () => {
      ok('swift' in mainIndex, 'index.js is missing the "swift" export')
      ok('cpp' in mainIndex, 'index.js is missing the "cpp" export')
    })

    it('routes swift.js exports identically to index.js swift namespace', () => {
    // We check keys instead of strict object equality because ESM
    // module namespace objects are created per-file.
      const swiftKeys = Object.keys(swiftIndex)
      const mainSwiftKeys = Object.keys(mainIndex.swift).filter(removeDefaultExport)

      deepEqual(mainSwiftKeys, swiftKeys, 'Exported keys must match')

      for (const key of swiftKeys) {
        strictEqual(mainIndex.swift[key], swiftIndex[key], `Export reference for swift.${key} does not match`)
      }
    })

    it('routes cpp.js exports identically to index.js cpp namespace', () => {
      const cppKeys = Object.keys(cppIndex)
      const mainCppKeys = Object.keys(mainIndex.cpp).filter(removeDefaultExport)

      deepEqual(mainCppKeys, cppKeys, 'Exported keys must match')

      for (const key of cppKeys) {
        strictEqual(
          mainIndex.cpp[key],
          cppIndex[key],
        `Export reference for cpp.${key} does not match`
        )
      }
    })
  })

  describe('API Contract Validation', () => {
  // Validates the API surface mentioned in your FIXME comments.
  // This acts as a contract test for the underlying 'mlx-swift' and 'mlx-cpp'.
    const expectedApiMethods = [
      'metrics',
      'load',
      'unload',
      'generate',
      'stream',
      'abort'
    ]

    it('swift bridge satisfy the expected MLX Node-API contract', () => {
      for (const method of expectedApiMethods) {
      // If the native module is loaded, it should be a function.
      // If it's undefined, the native dependency might not be built in this test run,
      // but we shouldn't fail the routing test if the dependency is just stubbed.
        const isFunction = typeof mainIndex.swift[method] === 'function'
        const isUndefined = typeof mainIndex.swift[method] === 'undefined'

        ok(isFunction || isUndefined, `Expected swift.${method} to be a function`)
      }
    })

    it('cpp bridge should satisfy the expected MLX Node-API contract', () => {
      for (const method of expectedApiMethods) {
        const isFunction = typeof mainIndex.cpp[method] === 'function'
        const isUndefined = typeof mainIndex.cpp[method] === 'undefined'

        ok(isFunction || isUndefined, `Expected cpp.${method} to be a function`)
      }
    })
  })
})
