import { describe, it } from 'node:test'
import { strictEqual, ok, fail, rejects } from 'node:assert'
import fs from 'node:fs/promises'

import { loadTemplate } from '../index.js'

describe('loadTemplate()', () => {
  it('loads chat_template.jinja override if it exists', async ({ mock }) => {
    mock.method(fs, 'readFile', async () => 'custom-template-string')

    const tokenizer = { config: { chat_template: null } }
    const template = await loadTemplate('/mock/path', tokenizer, { Template })

    ok(template instanceof Template)
    strictEqual(template.parsed, 'custom-template-string')
  })

  it('loads chat_template from tokenizer config', async ({ mock }) => {
    mock.method(fs, 'readFile', async () => { throw new Error() })

    const tokenizer = { config: { chat_template: 'tokenizer-chat-template' } }
    const template = await loadTemplate('/mock/path', tokenizer, { Template })

    ok(template instanceof Template)
    strictEqual(template.parsed, 'tokenizer-chat-template')
  })

  describe('loads chat_template array from tokenizer config', () => {
    const tokenizer = {
      config: {
        chat_template: [
          { name: 'default', template: 'tokenizer-chat-template-default' },
          { name: 'tool_use', template: 'tokenizer-chat-template-tool-use' }
        ]
      }
    }

    it('routes to default when no tools are provided', async ({ mock }) => {
      mock.method(fs, 'readFile', async () => { throw new Error() })

      const template = await loadTemplate('/mock/path', tokenizer, { Template })
      const rendered = template.render({ messages: [] })

      ok(template instanceof Object)
      strictEqual(rendered, 'tokenizer-chat-template-default')
    })

    it('routes to tool_use when tools are provided', async ({ mock }) => {
      mock.method(fs, 'readFile', async () => { throw new Error() })

      const template = await loadTemplate('/mock/path', tokenizer, { Template })
      const rendered = template.render({ tools: [{ type: 'function' }] })

      ok(template instanceof Object)
      strictEqual(rendered, 'tokenizer-chat-template-tool-use')
    })

    it('falls back to default if tools are provided but no tool template exists', async ({ mock }) => {
      mock.method(fs, 'readFile', async () => { throw new Error('ENOENT') })

      const noToolTokenizer = {
        config: {
          chat_template: [
            { name: 'default', template: 'tokenizer-chat-template-default' },
            { name: 'tool_use_not_available', template: 'tokenizer-chat-template-tool-use-not-available' }
          ]
        }
      }

      const template = await loadTemplate('/mock/path', noToolTokenizer, { Template })
      const rendered = template.render({ tools: [{ type: 'function' }] })

      strictEqual(rendered, 'tokenizer-chat-template-default')
    })

    it('handles malformed context objects safely (null/undefined tools)', async ({ mock }) => {
      mock.method(fs, 'readFile', async () => { throw new Error('ENOENT') })
      const template = await loadTemplate('/mock/path', tokenizer, { Template })

      strictEqual(template.render(null), 'tokenizer-chat-template-default')
      strictEqual(template.render(undefined), 'tokenizer-chat-template-default')
      strictEqual(template.render({ tools: [] }), 'tokenizer-chat-template-default')
    })
  })

  it('throws if config.chat_template is undefined', async ({ mock }) => {
    mock.method(fs, 'readFile', async () => { throw new Error() })
    await rejects(
      loadTemplate('/mock/path', { config: {} }, { Template }),
      /No valid template found in model path/
    )
  })

  it('throws if config.chat_template is an empty array', async ({ mock }) => {
    mock.method(fs, 'readFile', async () => { throw new Error() })
    await rejects(
      loadTemplate('/mock/path', { config: { chat_template: [] } }, { Template }),
      /No valid template found in model path/
    )
  })

  class Template {
    constructor (parsed) {
      this.parsed = parsed
    }

    render (context) {
      return this.parsed
    }

    format () {
      return this.parsed
    }
  }
})
