import { describe, it } from 'node:test'
import { strictEqual, deepStrictEqual } from 'node:assert/strict'

import * as lm from 'mlx-lm'
import * as vlm from '../index.js'

describe('mlx-vlm', () => {
  it('should perfectly mirror the named exports of mlx-lm', () => {
    // The ESM Quirk: `export *` ignores default exports, dropping the `default` key from our comparison.
    const { default: _, ...lmNamedExports } = lm
    const { default: vlmDefault, ...vlmNamedExports } = vlm

    strictEqual(vlmDefault, undefined, 'export * should not generate a default export')
    deepStrictEqual(vlmNamedExports, lmNamedExports)
  })
})
