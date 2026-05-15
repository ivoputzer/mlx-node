import { describe, it } from 'node:test'
import { strictEqual } from 'node:assert'
import { createPaddedBatch } from 'mlx-node'

describe('.createPaddedBatch', () => {
  it('correctly left-pads jagged arrays', (t) => {
    const prompts = [
      [1, 2], // Length 2
      [1, 2, 3, 4] // Length 4
    ]

    const { flatTokens, maxLen, batchSize } = createPaddedBatch(prompts, 0)

    strictEqual(maxLen, 4)
    strictEqual(batchSize, 2)
    strictEqual(flatTokens.length, 8)

    // Row 0 should be [0, 0, 1, 2]
    strictEqual(flatTokens[0], 0)
    strictEqual(flatTokens[1], 0)
    strictEqual(flatTokens[2], 1)
    strictEqual(flatTokens[3], 2)

    // Row 1 should be [1, 2, 3, 4]
    strictEqual(flatTokens[4], 1)
    strictEqual(flatTokens[7], 4)
  })
})
