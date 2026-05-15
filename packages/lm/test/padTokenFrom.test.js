import { describe, it } from 'node:test'
import { equal, deepEqual } from 'node:assert/strict'
import { padTokenFrom } from '../lib/tokenizer.js'
import { loadTokenizer } from '../index.js'
import { join } from 'node:path'
import { env } from 'node:process'

const skip = env?.GITHUB_ACTIONS?.includes('true')

describe('padTokenFrom', () => {
  it('returns pad_token_id and its resolved text when strictly defined', ({ mock }) => {
    const tokenizer = {
      config: { pad_token_id: 42 },
      model: { tokens_to_ids: { has: mock.fn() } },
      token_to_id: mock.fn(),
      id_to_token: mock.fn((id) => id === 42 ? '[PAD]' : undefined)
    }

    deepEqual(padTokenFrom(tokenizer), { id: 42, text: '[PAD]' })
  })

  it('returns token_to_id(pad_token) when pad_token_id is null', ({ mock }) => {
    const tokenizer = {
      // Using `null` guarantees V8 tests the `!== null` boolean branch fully
      config: { pad_token_id: null, pad_token: '<pad>' },
      model: { tokens_to_ids: { has: mock.fn(() => true) } },
      token_to_id: mock.fn(() => 101),
      id_to_token: mock.fn()
    }

    deepEqual(padTokenFrom(tokenizer), { id: 101, text: '<pad>' })

    // Verify the exact arguments passed to the mocks
    equal(tokenizer.model.tokens_to_ids.has.mock.calls[0].arguments[0], '<pad>')
    equal(tokenizer.token_to_id.mock.calls[0].arguments[0], '<pad>')
  })

  it('returns unk_token_id when pad_token exists but is not in tokens_to_ids', ({ mock }) => {
    const tokenizer = {
      // has() returning false tests the short-circuit failure of the && condition
      config: { pad_token: '<pad>', unk_token_id: 99 },
      model: { tokens_to_ids: { has: mock.fn(() => false) } },
      token_to_id: mock.fn(),
      id_to_token: mock.fn((id) => id === 99 ? '<unk>' : undefined)
    }

    deepEqual(padTokenFrom(tokenizer), { id: 99, text: '<unk>' })
  })

  it('returns token_to_id(unk_token) when unk_token_id is null', ({ mock }) => {
    const tokenizer = {
      config: { unk_token_id: null, unk_token: '<unk>' },
      model: { tokens_to_ids: { has: mock.fn(() => true) } },
      token_to_id: mock.fn(() => 102),
      id_to_token: mock.fn()
    }

    deepEqual(padTokenFrom(tokenizer), { id: 102, text: '<unk>' })
  })

  it('returns 0 and standard <unk> string as an absolute last resort if 0 has no text', ({ mock }) => {
    const tokenizer = {
      config: { }, // Totally empty config
      model: { tokens_to_ids: { has: mock.fn(() => false) } },
      token_to_id: mock.fn(),
      id_to_token: mock.fn(() => undefined) // Simulate token not existing in vocab
    }

    deepEqual(padTokenFrom(tokenizer), null)
  })

  describe('Integration', { skip }, () => {
    // check what models actually support pad token
    async function tokenizerFor (model) {
      const path = join(import.meta.dirname, '..', '..', '..', 'models', model)
      return loadTokenizer(path)
    }

    describe('Gemma', () => {
      it('mlx-community/gemma-4-e4b-6bit', async () => {
        const padToken = padTokenFrom(await tokenizerFor('gemma-4-e4b-6bit'))

        equal(padToken?.text, '<pad>')
        equal(padToken?.id, 0)
      })

      it('mlx-community/gemma-3-4b-it-4bit', async () => {
        const padToken = padTokenFrom(await tokenizerFor('gemma-3-4b-it-4bit'))

        equal(padToken?.text, '<pad>')
        equal(padToken?.id, 0)
      })

      it('mlx-community/gemma-2-2b-it-4bit', async () => {
        const padToken = padTokenFrom(await tokenizerFor('gemma-2-2b-it-4bit'))

        equal(padToken?.text, '<pad>')
        equal(padToken?.id, 0)
      })
    })

    describe('Ministral', () => {
      it('mistralai/ministral-8b-instruct-2410', async () => {
        const padToken = padTokenFrom(await tokenizerFor('ministral-8b-instruct-2410'))

        equal(padToken?.text, '<unk>')
        equal(padToken?.id, 0)
      })
      it('mistralai/mistral-7b-instruct-v0.3-4bit', async () => {
        const padToken = padTokenFrom(await tokenizerFor('mistral-7b-instruct-v0.3-4bit'))

        equal(padToken?.text, '<unk>')
        equal(padToken?.id, 0)
      })
    })

    describe('Deepseek', () => {
      it('deepseek-ai/deepseek-v3', async () => {
        const padToken = padTokenFrom(await tokenizerFor('deepseek-v3'))

        equal(padToken, null)
      })

      it('deepseek-ai/deepseek-v4-flash', async () => {
        const padToken = padTokenFrom(await tokenizerFor('deepseek-v4-flash'))

        equal(padToken, null)
      })

      it('mlx-community/deepseek-r1-distill-llama-8b-4bit', async () => {
        const padToken = padTokenFrom(await tokenizerFor('deepseek-r1-distill-llama-8b-4bit'))

        equal(padToken?.text, '<｜end▁of▁sentence｜>')
        equal(padToken?.id, 128001)
      })

      it('mlx-community/deepseek-r1-distill-qwen-7b-4bit', async () => {
        const padToken = padTokenFrom(await tokenizerFor('deepseek-r1-distill-qwen-7b-4bit'))

        equal(padToken?.text, '<｜end▁of▁sentence｜>')
        equal(padToken?.id, 151643)
      })
    })

    it('returns padToken for qwen/qwen3.6-27b', async () => {
      const padToken = padTokenFrom(await tokenizerFor('qwen3.6-27b'))

      equal(padToken?.text, '<|endoftext|>')
      equal(padToken?.id, 248044)
    })

    it('returns padToken for openai/gpt-oss-20b', async () => {
      const padToken = padTokenFrom(await tokenizerFor('gpt-oss-20b'))

      equal(padToken?.text, '<|endoftext|>')
      equal(padToken?.id, 199999)
    })

    it('returns padToken for jackrong/mlx-qwen3.5-9b-claude-4.6-opus-reasoning-distilled-8bit', async () => {
      const padToken = padTokenFrom(await tokenizerFor('mlx-qwen3.5-9b-claude-4.6-opus-reasoning-distilled-8bit'))

      equal(padToken?.text, '<|endoftext|>')
      equal(padToken?.id, 248044)
    })

    it('returns padToken for mlx-community/gemma-4-e4b-6bit', async () => {
      const padToken = padTokenFrom(await tokenizerFor('gemma-4-e4b-6bit'))

      equal(padToken?.text, '<pad>')
      equal(padToken?.id, 0)
    })

    it('returns padToken for mlx-community/qwen3.5-9b-6bit', async () => {
      const padToken = padTokenFrom(await tokenizerFor('qwen3.5-9b-6bit'))

      equal(padToken?.text, '<|endoftext|>')
      equal(padToken?.id, 248044)
    })

    it('returns padToken for mlx-community/granite-4.1-8b-8bit', async () => {
      const padToken = padTokenFrom(await tokenizerFor('granite-4.1-8b-8bit'))

      equal(padToken?.text, '<|pad|>')
      equal(padToken?.id, 100256)
    })

    it('returns padToken for mlx-community/meta-llama-3.1-8b-instruct-8bit', async () => {
      const padToken = padTokenFrom(await tokenizerFor('meta-llama-3.1-8b-instruct-8bit'))

      equal(padToken, null)
      equal(padToken?.text, undefined)
      equal(padToken?.id, undefined)
    })

    it('returns padToken for mlx-community/deepseek-r1-distill-llama-8b-4bit', async () => {
      const padToken = padTokenFrom(await tokenizerFor('deepseek-r1-distill-llama-8b-4bit'))

      equal(padToken?.text, '<｜end▁of▁sentence｜>')
      equal(padToken?.id, 128001)
    })

    it('returns padToken for huggingfacetb/smollm2-135m-instruct', async () => {
      const padToken = padTokenFrom(await tokenizerFor('smollm2-135m-instruct'))

      equal(padToken?.text, '<|im_end|>') // pad_token is set to eos_token, unk_token
      equal(padToken?.id, 2)
    })

    it('returns padToken for qwen/qwen2.5-coder-7b-instruct', async () => {
      const padToken = padTokenFrom(await tokenizerFor('qwen2.5-coder-7b-instruct'))

      equal(padToken?.text, '<|endoftext|>')
      equal(padToken?.id, 151643)
    })
  })
})
