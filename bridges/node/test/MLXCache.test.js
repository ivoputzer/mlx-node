import { describe, it } from 'node:test'
import { strictEqual, ok, deepStrictEqual } from 'node:assert'

import mlx from 'mlx-swift'
import { MLXCache, MLXModel } from '../index.js'

describe('MLXCache', () => {
  it('creates cache from an existing model', ({ mock }) => {
    mock.method(mlx, 'createCache', (modelRef, configJson) => {
      strictEqual(modelRef.pointer, 1)
      deepStrictEqual(JSON.parse(configJson), { kvBits: 8 })
      return { pointer: 2 }
    })

    const model = new MLXModel({ pointer: 1 })
    const cache = MLXCache.fromModel(model, { kvBits: 8, ignoreMe: true })

    ok(cache instanceof MLXCache)
    strictEqual(cache.model, model)
    strictEqual(cache.cache, cache)
  })

  it('loads cache from path', async ({ mock }) => {
    mock.method(mlx, 'loadCache', (path, callback) => {
      strictEqual(path, '/cache/path')
      callback(null, { pointer: 2 })
    })

    const mockModel = new MLXModel({ modelPtr: 1 })
    const cache = await MLXCache.fromPath('/cache/path', mockModel)
    ok(cache.available)
  })

  it('saves cache to path', async ({ mock }) => {
    mock.method(mlx, 'saveCache', (ref, path, callback) => {
      strictEqual(path, '/save/path')
      callback(null)
    })

    const cache = new MLXCache({ cachePtr: 1 }, null)
    await cache.save('/save/path') // Should resolve seamlessly
  })

  it('exposes debug info and trimmability', ({ mock }) => {
    mock.method(mlx, 'debugCache', () => '{"layers": 32, "isTrimmable": true}')

    const cache = new MLXCache({ cachePtr: 1 }, null)
    ok(cache.isTrimmable)
    deepStrictEqual(cache.debug(), { layers: 32, isTrimmable: true })
  })
})
