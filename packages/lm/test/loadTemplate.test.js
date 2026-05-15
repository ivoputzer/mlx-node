import { describe, it } from 'node:test'
import { strictEqual, ok, rejects } from 'node:assert'

import { loadTemplate } from '../index.js'

describe('loadTemplate()', () => {
  const readFile = async () => { throw new Error('ENOENT') }
  const readJson = async () => { throw new Error('Should not be called') }

  it('loads chat_template.jinja override if it exists', async () => {
    const readFile = async () => { return 'custom-template-string' }

    const template = await loadTemplate('path/to/model', { Template }, { readFile }, undefined, { readJson })

    ok(template instanceof Template)
    strictEqual(template.format(), 'custom-template-string')
  })

  it('loads chat_template from tokenizer config', async () => {
    const readJson = async () => { return { chat_template: 'tokenizer-chat-template' } }

    const template = await loadTemplate('path/to/model', { Template }, { readFile }, undefined, { readJson })

    ok(template instanceof Template)
    strictEqual(template.parsed, 'tokenizer-chat-template')
  })

  describe('loads chat_template array from tokenizer config', () => {
    const readJson = async () => ({
      chat_template: [
        { name: 'default', template: 'tokenizer-chat-template-default' },
        { name: 'tool_use', template: 'tokenizer-chat-template-tool-use' },
      ]
    })

    it('routes to default when no tools are provided', async ({ mock }) => {
      const template = await loadTemplate('path/to/model', { Template }, { readFile }, undefined, { readJson })
      const rendered = template.render({ messages: [] })

      ok('entries' in template)
      strictEqual(rendered, 'tokenizer-chat-template-default')
    })

    it('routes to tool_use when tools are provided', async ({ mock }) => {
      const template = await loadTemplate('path/to/model', { Template }, { readFile }, undefined, { readJson })
      const rendered = template.render({ tools: [{ type: 'function' }] })

      ok(template instanceof Object)
      strictEqual(rendered, 'tokenizer-chat-template-tool-use')
    })

    it('falls back to default if tools are provided but no tool template exists', async () => {
      const readJson = async () => ({
        chat_template: [
          { name: 'default', template: 'tokenizer-chat-template-default' },
          { name: 'tool_use_not_available', template: 'tokenizer-chat-template-tool-use-not-available' }
        ]
      })

      const template = await loadTemplate('path/to/model', { Template }, { readFile }, undefined, { readJson })
      const rendered = template.render({ tools: [{ type: 'function' }] })

      strictEqual(rendered, 'tokenizer-chat-template-default')
    })

    it('handles malformed context objects safely (null/undefined tools)', async () => {
      const template = await loadTemplate('path/to/model', { Template }, { readFile }, undefined, { readJson })

      strictEqual(template.render(null), 'tokenizer-chat-template-default')
      strictEqual(template.render(undefined), 'tokenizer-chat-template-default')
      strictEqual(template.render({ tools: [] }), 'tokenizer-chat-template-default')
    })
  })

  it('throws if config.chat_template is undefined', async () => {
    const readJson = async () => ({})

    await rejects(
      loadTemplate('path/to/model', undefined, { readFile }, undefined, { readJson }),
      /No valid template found in model path/
    )
  })

  it('throws if config.chat_template is an empty array', async ({ mock }) => {
    const readJson = async () => ([])
    await rejects(
      loadTemplate('path/to/model', undefined, { readFile }, undefined, { readJson }),
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
