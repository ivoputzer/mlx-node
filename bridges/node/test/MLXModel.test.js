import { describe, it } from 'node:test'
import { strictEqual, ok, rejects } from 'node:assert'

import mlx from 'mlx-swift'
import { MLXModel } from '../index.js'

describe('MLXModel', () => {
  it('loads a model from path', async ({ mock }) => {
    mock.method(mlx, 'loadModel', (path, callback) => {
      strictEqual(path, '/fake/path')
      callback(null, { modelPtr: 1 }) // Simulate C success callback
    })

    const model = await MLXModel.fromPath('/fake/path')
    ok(model instanceof MLXModel)
    ok(model.available)
    strictEqual(model.model, model)
    strictEqual(model.cache, null)
  })

  it('rejects if native loadModel fails', async ({ mock }) => {
    mock.method(mlx, 'loadModel', (path, callback) => {
      callback(new Error('an_error_message'))
    })

    await rejects(() => MLXModel.fromPath('/bad'), /an_error_message/)
  })
})
