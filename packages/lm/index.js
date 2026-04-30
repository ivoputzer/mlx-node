import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

import * as mlx from 'mlx-node/swift'
import * as tokenizers from '@huggingface/tokenizers'
import * as jinja from '@huggingface/jinja'

import { text } from 'node:stream/consumers'

// API

const registry = new FinalizationRegistry(({ id, dispose }) => {
  console.log('FinalizationRegistry', id)
  if (id) dispose(id)
})

export async function evaluate (target, prompt, options = {}) {
  if (!prompt) throw new Error('Prompt cannot be empty')
  if (!target.ready) throw new Error('Target not loaded or disposed already')

  const signal = options?.signal ?? new AbortController().signal
  if (signal?.aborted) throw new AbortError()

  const onAbort = target.abort.bind(target)
  signal?.addEventListener('abort', onAbort, { once: true })

  try {
    const { ids } = target.encode(prompt, { template: options?.template })
    const generate = target.generate(new Int32Array(ids), { ...options, maxTokens: 0 })

    while (true) {
      const { value, done } = await generate.next() // Drain but prevent it from generating new tokens by setting maxTokens: 0
      if (done) {
        if (signal?.aborted) throw new AbortError(signal?.reason)
        return value
      }
    }
  } catch (error) {
    if (error.message?.includes('Generation Cancelled')) throw new AbortError(signal?.reason) // [Error: Generation Cancelled] { code: 'MLX_STREAM_ERR' }
    throw error
  } finally {
    signal?.removeEventListener('abort', onAbort)
  }
}

export async function generate (target, prompt, options = {}) {
  if (!prompt) throw new Error('Prompt cannot be empty')
  if (!target.ready) throw new Error('Target not loaded or disposed already')

  const signal = options?.signal ?? new AbortController().signal
  if (signal?.aborted) throw new AbortError()

  const onAbort = target.abort.bind(target)
  signal?.addEventListener('abort', onAbort, { once: true })

  try {
    const { ids } = target.encode(prompt, { template: options?.template })

    const tokens = []
    const generate = target.generate(new Int32Array(ids), { ...options })

    while (true) {
      const { value, done } = await generate.next()
      if (done) {
        if (signal?.aborted) throw new AbortError(signal?.reason)
        return {
          text: target.decode(tokens, { skip_special_tokens: options?.skipSpecialTokens ?? true, clean_up_tokenization_spaces: options?.cleanUpTokenizationSpaces ?? true }),
          finish: value?.stopReason,
          tokens,
          stats: value
        }
      } else {
        tokens.push(value)
      }
    }
  } catch (error) {
    if (error.message?.includes('Generation Cancelled')) throw new AbortError(signal?.reason)
    throw error
  } finally {
    signal?.removeEventListener('abort', onAbort)
  }
}

export async function * stream (target, prompt, options = {}) {
  if (!prompt) throw new Error('Prompt cannot be empty')
  if (!target.ready) throw new Error('Target not loaded or disposed already')

  const signal = options?.signal ?? new AbortController().signal
  if (signal?.aborted) throw new AbortError()

  const onAbort = target.abort.bind(target)
  signal?.addEventListener('abort', onAbort, { once: true })

  try {
    const { ids } = target.encode(prompt, { template: options?.template })
    const generate = target.generate(new Int32Array(ids), { ...options })

    const tokens = []
    const buffer = []

    while (true) {
      const { value, done } = await generate.next()

      if (done) {
        // flush anything left in the boundary queue
        if (buffer.length > 0) {
          const text = target.decode(buffer, { skip_special_tokens: options?.skipSpecialTokens ?? true })
          yield { text, done: false }
        }
        yield { // for the absolute perfect final string, we decode the entire array once more
          done: true,
          text: target.decode(tokens, { skip_special_tokens: options?.skipSpecialTokens ?? true, clean_up_tokenization_spaces: options?.cleanUpTokenizationSpaces ?? true }),
          finish: value?.stopReason || 'stop',
          tokens,
          stats: value
        }
        break
      } else {
        tokens.push(value)
        buffer.push(value)

        const text = target.decode(buffer, {
          skip_special_tokens: options?.skipSpecialTokens ?? true,
          clean_up_tokenization_spaces: false
        })

        if (text.endsWith('\uFFFD')) {
          continue // Keep it in the buffer and wait for the next iteration!
        } else {
          if (text.length > 0) { // is this condition necessary? can there be a text one or more tokens and have zero length 🤔
            yield { text, done: false }
            buffer.length = 0
          }
        }
      }
    }
  } catch (error) {
    if (error.message?.includes('Cancelled') || error.message?.includes('Abort')) {
      yield { done: true, finish: 'abort', stats: null }
      return
    }
    throw error
  } finally {
    signal?.removeEventListener('abort', onAbort)
  }
}

// HELPERS

class AbortError extends Error {
  constructor (message = 'The operation was aborted') {
    super(message)
    this.name = 'AbortError'
    Error.captureStackTrace(this, this.constructor)
  }
}

export async function loadTokenizer (path, { Tokenizer } = tokenizers) {
  const tokenizerPath = join(path, 'tokenizer.json')
  const tokenizerConfigPath = join(path, 'tokenizer_config.json')

  const [tokenizerFile, tokenizerConfigFile] = await Promise.all([
    readFile(tokenizerPath, 'utf8'),
    readFile(tokenizerConfigPath, 'utf8')
  ])

  const tokenizerJson = JSON.parse(tokenizerFile)
  const tokenizerConfigJson = JSON.parse(tokenizerConfigFile)

  return new Tokenizer(tokenizerJson, tokenizerConfigJson)
}

export async function loadTemplate (path, { Template } = jinja) {
  const tokenizerConfigPath = join(path, 'tokenizer_config.json')
  const templatePath = join(path, 'chat_template.jinja')

  const [tokenizerFile, tokenizerConfigFile] = await Promise.all([
    readFile(templatePath, 'utf8'),
    readFile(tokenizerConfigPath, 'utf8')
  ])

  const tokenizerConfigJson = JSON.parse(tokenizerConfigFile)

  return new Template(tokenizerConfigJson.chat_template ?? tokenizerFile)
}

export async function loadOptions (path) {
  const configPath = join(path, 'config.json')
  const generationConfigPath = join(path, 'generation_config.json')

  const [configFile, generationConfigFile] = await Promise.all([
    readFile(configPath, 'utf8'),
    readFile(generationConfigPath, 'utf8').catch(() => '{}')
  ])

  const config = JSON.parse(configFile)
  const generation = JSON.parse(generationConfigFile)

  return { config, generation }
}

// CLASSES

export class Target {
  #bridge = mlx // this is easier to mock via node:test
  #ready = true
  #id = -1

  constructor (id, bridge) {
    this.#id = id
    // this.#bridge = bridge
    // this.#dispose = bridge.unload.bind(bridge, id)
    // registry.register(this, { id, dispose: this.dispose }, this)
  }

  get id () {
    return this.#id
  }

  get ready () {
    return this.#ready
  }

  get bridge () {
    return this.#bridge
  }

  dispose () {
    console.log('Target.dispose', this.#id, this.#ready)
    if (!this.#ready) return
    this.#bridge.unload(this.#id)
    this.#ready = false
    this.#id = -1
    // console.log('Target.dispose', this.#id, this.#loaded)
    // registry.unregister(this)
  }

  [Symbol.dispose] () {
    this.dispose()
  }
}

export class Model extends Target {
  #tokenizer
  #template
  #options
  #bridge

  static async load (path, bridge = mlx) {
    const tokenizer = await loadTokenizer(path)
    const template = await loadTemplate(path)

    const id = await bridge.load(path)
    return new Model(id, bridge, tokenizer, template, { config: {}, generate: {} })
  }

  constructor (id, bridge, tokenizer, template, options) {
    super(id, bridge)

    this.#tokenizer = tokenizer
    this.#template = template
    this.#options = options
  }

  encode (prompt, options = {}, { Template } = jinja) {
    if (typeof prompt === 'string') {
      return this.#tokenizer.encode(prompt, { add_special_tokens: true, ...options })
    } else {
      return options?.template?.length
        ? this.#tokenizer.encode(new Template(options.template).render(prompt), { add_special_tokens: false, ...options })
        : this.#tokenizer.encode(this.#template.render({ add_generation_prompt: true, ...prompt }), { add_special_tokens: false, ...options })
    }
  }

  decode (tokens, options = {}) {
    return this.#tokenizer.decode(tokens, options)
  }

  generate (tokens, options) {
    return this.bridge.generate(this.id, tokens, options)
  }

  abort () {
    console.log('Model.abort id:%s loaded:%s', this.id, this.loaded)
    if (this.ready) {
      this.bridge.abort(this.id)
      console.log('Model.abort', this.bridge.abort(this.id))
    }
  }
}
