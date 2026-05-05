import { describe, it } from 'node:test'
import { equal } from 'node:assert/strict'
import { padTokenFrom } from '../index.js'

describe('padTokenFrom', () => {
  it('returns pad_token_id when strictly defined', ({ mock }) => {
    const tokenizer = {
      config: { pad_token_id: 42 },
      model: { tokens_to_ids: { has: mock.fn() } },
      token_to_id: mock.fn()
    }

    equal(padTokenFrom(tokenizer), 42)
  })

  it('returns token_to_id(pad_token) when pad_token_id is null', ({ mock }) => {
    const tokenizer = {
      // Using `null` guarantees V8 tests the `!== null` boolean branch fully
      config: { pad_token_id: null, pad_token: '<pad>' },
      model: { tokens_to_ids: { has: mock.fn(() => true) } },
      token_to_id: mock.fn(() => 101)
    }

    equal(padTokenFrom(tokenizer), 101)

    // Optional: verify the exact arguments passed to the mocks
    equal(tokenizer.model.tokens_to_ids.has.mock.calls[0].arguments[0], '<pad>')
    equal(tokenizer.token_to_id.mock.calls[0].arguments[0], '<pad>')
  })

  it('returns unk_token_id when pad_token exists but is not in tokens_to_ids', ({ mock }) => {
    const tokenizer = {
      // has() returning false tests the short-circuit failure of the && condition
      config: { pad_token: '<pad>', unk_token_id: 99 },
      model: { tokens_to_ids: { has: mock.fn(() => false) } },
      token_to_id: mock.fn()
    }

    equal(padTokenFrom(tokenizer), 99)
  })

  it('returns token_to_id(unk_token) when unk_token_id is null', ({ mock }) => {
    const tokenizer = {
      config: { unk_token_id: null, unk_token: '<unk>' },
      model: { tokens_to_ids: { has: mock.fn(() => true) } },
      token_to_id: mock.fn(() => 102)
    }

    equal(padTokenFrom(tokenizer), 102)
  })

  it('returns 0 as fallback when all conditions fail', ({ mock }) => {
    const tokenizer = {
      config: { unk_token: '<unk>' },
      model: { tokens_to_ids: { has: mock.fn(() => false) } },
      token_to_id: mock.fn()
    }

    equal(padTokenFrom(tokenizer), 0)
  })
})
