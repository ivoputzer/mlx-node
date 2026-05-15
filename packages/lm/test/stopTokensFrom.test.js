import { describe, it } from 'node:test'
import { equal, deepEqual } from 'node:assert/strict'
import { stopTokensFrom } from '../lib/tokenizer.js'

describe('stopTokensFrom', () => {
  it('finds and deduplicates valid eos, eot, and pad string tokens', ({ mock }) => {
    const tokenizer = {
      config: {
        eos_token: '<eos>',
        eot_token: '<eot>',
        pad_token: '<pad>',
        bos_token: '<bos>' // Should be completely ignored
      },
      model: {
        tokens_to_ids: { has: mock.fn(() => true) }
      },
      token_to_id: mock.fn((val) => {
        if (val === '<eos>') return 100
        if (val === '<eot>') return 101
        if (val === '<pad>') return 100 // Duplicate ID ensures Map() deduplication works
      }),
      id_to_token: mock.fn()
    }

    const result = stopTokensFrom(tokenizer)

    // Returns deduplicated array of objects
    deepEqual(result, [
      { id: 100, text: '<eos>' },
      { id: 101, text: '<eot>' }
    ])

    // Proves <bos> was skipped
    equal(tokenizer.model.tokens_to_ids.has.mock.calls.length, 2)
  })

  it('ignores matched keys if the token string is missing from tokens_to_ids', ({ mock }) => {
    const tokenizer = {
      config: {
        eos_token: '<missing>'
      },
      model: {
        tokens_to_ids: { has: mock.fn(() => false) }
      },
      token_to_id: mock.fn(),
      id_to_token: mock.fn()
    }

    const result = stopTokensFrom(tokenizer)

    deepEqual(result, [])
    equal(tokenizer.token_to_id.mock.calls.length, 0)
  })

  it('correctly handles numeric *_token_id configs and arrays (e.g. Llama 3)', ({ mock }) => {
    const tokenizer = {
      config: {
        eos_token_id: [128001, 128009], // Array of numbers
        pad_token_id: 128001, // Duplicate numeric ID to test deduplication
        eot_token_id: null // Should be safely ignored
      },
      model: {
        tokens_to_ids: { has: mock.fn() }
      },
      token_to_id: mock.fn(),
      id_to_token: mock.fn((id) => {
        if (id === 128001) return '<|end_of_text|>'
        if (id === 128009) return '<|eot_id|>'
      })
    }

    const result = stopTokensFrom(tokenizer)

    deepEqual(result, [
      { id: 128001, text: '<|end_of_text|>' },
      { id: 128009, text: '<|eot_id|>' }
    ])
  })

  it('provides a fallback text if id_to_token fails for a numeric id', ({ mock }) => {
    const tokenizer = {
      config: { eos_token_id: 999 },
      model: { tokens_to_ids: { has: mock.fn() } },
      token_to_id: mock.fn(),
      id_to_token: mock.fn(() => undefined) // Tokenizer doesn't know the text
    }

    const result = stopTokensFrom(tokenizer)

    deepEqual(result, [
      { id: 999, text: '<unknown>' }
    ])
  })
})
