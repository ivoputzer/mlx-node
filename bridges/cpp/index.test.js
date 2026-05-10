import { describe, it } from 'node:test'
import { deepEqual } from 'node:assert/strict'

import mlx from 'mlx-cpp'

describe('mlx-cpp', () => {
  it('returns an empty object', () => {
    deepEqual(mlx, {})
  })
})
