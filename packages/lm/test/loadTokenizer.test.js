import { describe, it } from 'node:test'
import { deepEqual, ok } from 'node:assert/strict'
import { loadTokenizer } from '../index.js'

describe('loadTokenizer', () => {
  class Tokenizer {
    constructor (tokenizerJson, configJson) {
      this.tokenizer = tokenizerJson
      this.config = configJson
    }
  }

  const aTokenizer = { vocab: { a: 1 } }
  const aTokenizerConfig = { pad_token: '<pad>' }

  it('successfully loads and parses both json files', async () => {
    const readJson = async (path) => {
      if (path.endsWith('tokenizer.json')) return aTokenizer
      if (path.endsWith('tokenizer_config.json')) return aTokenizerConfig
      throw new Error('Not found')
    }

    const tokenizer = await loadTokenizer('path/to/model', { Tokenizer }, undefined, { readJson })

    ok(tokenizer instanceof Tokenizer)
    deepEqual(tokenizer.tokenizer, aTokenizer)
    deepEqual(tokenizer.config, aTokenizerConfig)
  })
})
