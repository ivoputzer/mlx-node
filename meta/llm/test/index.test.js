import { describe, it } from 'node:test'
import { strictEqual, deepStrictEqual } from 'node:assert/strict'

import * as lm from 'mlx-lm'
import * as llm from '../index.js'

describe('mlx-llm', () => {
  it('should perfectly mirror the named exports of mlx-lm', () => {
    // The ESM Quirk: `export *` ignores default exports, dropping the `default` key from our comparison.
    const { default: _, ...lmNamedExports } = lm
    const { default: llmDefault, ...llmNamedExports } = llm

    strictEqual(llmDefault, undefined, 'export * should not generate a default export')
    deepStrictEqual(llmNamedExports, lmNamedExports)
  })
})
