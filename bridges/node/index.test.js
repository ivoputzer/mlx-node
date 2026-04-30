import { describe, it } from 'node:test'
import { ok } from 'node:assert/strict'

import * as mainIndex from 'mlx-node'

describe('mlx-node', () => {
  describe('exports', () => {
    it('exposes swift and cpp namespaces from the main entry point', () => {
      ok('swift' in mainIndex, 'index.js is missing the "swift" export')
      ok('cpp' in mainIndex, 'index.js is missing the "cpp" export')
    })
    it.todo('exports swift individually')
    it.todo('exports cpp individually')
  })

  describe('API Contract Validation', () => {
    //
  })
})
