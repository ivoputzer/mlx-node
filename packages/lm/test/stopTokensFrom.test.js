import { describe, it } from 'node:test'
import { equal, deepEqual } from 'node:assert/strict'
import { stopTokensFrom } from '../index.js'

describe('stopTokensFrom', () => {
  it('finds and deduplicates valid eos, eot, and pad tokens', ({ mock }) => {
    const tokenizer = {
      config: {
        // These 4 keys guarantee every logical path of the `||` chain is tested:
        eos_token: '<eos>', // True (short-circuits the rest)
        eot_token: '<eot>', // False || True
        pad_token: '<pad>', // False || False || True
        bos_token: '<bos>' // False || False || False (Fails the if statement entirely)
      },
      model: {
        tokens_to_ids: {
          has: mock.fn(() => true) // Evaluates inner `if` to true
        }
      },
      token_to_id: mock.fn((val) => {
        if (val === '<eos>') return 100
        if (val === '<eot>') return 101
        if (val === '<pad>') return 100 // Duplicate ID ensures Set() deduplication works
      })
    }

    const result = stopTokensFrom(tokenizer)

    // Returns deduplicated array
    deepEqual(result, [100, 101])

    // Proves <bos> was skipped and didn't reach the `has` check
    equal(tokenizer.model.tokens_to_ids.has.mock.calls.length, 3)
  })

  it('ignores matched keys if the token is missing from tokens_to_ids', ({ mock }) => {
    const tokenizer = {
      config: {
        eos_token: '<missing>'
      },
      model: {
        tokens_to_ids: {
          has: mock.fn(() => false) // Evaluates inner `if` to false
        }
      },
      token_to_id: mock.fn()
    }

    const result = stopTokensFrom(tokenizer)

    deepEqual(result, [])

    // Proves `token_to_id` was never executed
    equal(tokenizer.token_to_id.mock.calls.length, 0)
  })
})
