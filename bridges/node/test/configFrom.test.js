import { describe, it } from 'node:test'
import { strictEqual } from 'node:assert'
import { configFrom } from 'mlx-node'

describe('.configFrom', () => {
  it('filters only allowed keys', (t) => {
    const raw = { temperature: 0.7, unknownKey: 'ignoreMe', maxTokens: 100 }
    const json = configFrom(raw)
    const parsed = JSON.parse(json)

    strictEqual(parsed.temperature, 0.7)
    strictEqual(parsed.maxTokens, 100)
    strictEqual(parsed.unknownKey, undefined)
  })
})
