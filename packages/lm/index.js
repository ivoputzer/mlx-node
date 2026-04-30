import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

import mlx from 'mlx-node/swift'

import * as tokenizers from '@huggingface/tokenizers'
import * as jinja from '@huggingface/jinja'

// API

export async function evaluate (target, prompt, options = {}) {
  if (!prompt) throw new Error('Prompt cannot be empty')
  if (!target.available) throw new Error('Target not loaded or disposed already')

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
  if (!target.available) throw new Error('Target not loaded or disposed already')

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
  if (!target.available) throw new Error('Target not loaded or disposed already')

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
        const text = target.decode(buffer, { skip_special_tokens: options?.skipSpecialTokens ?? true, clean_up_tokenization_spaces: false })
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

class MLXResource {
  #mlx = mlx
  #ref = null

  constructor (ref) {
    if (!ref) throw new Error('Native reference is required')
    this.#ref = ref
  }

  get ref () {
    return this.#ref
  }

  get available () {
    return this.#ref !== null
  }

  dispose () {
    if (!this.#ref) return false
    const success = this.#mlx.free(this.#ref)
    this.#ref = null
    return success
  }

  [Symbol.dispose] () {
    this.dispose()
  }
}

export class MLXTarget extends MLXResource {
  // todo: Implement shared interface evaluate, generate, stream and other top level functions will be using
}

export class MLXCache extends MLXTarget {
  // todo: base implementation is not necessairly for chat_template
}

export class MLXArray extends MLXTarget {
  // todo:
}

export class MLXModel extends MLXTarget {
  #mlx = mlx

  #tokenizer
  #template

  static async load (path) {
    const tokenizer = await loadTokenizer(path)
    const template = await loadTemplate(path)
    const ref = await mlx.load(path) // C Pointer
    return await new MLXModel(ref, tokenizer, template)
  }

  constructor (ref, tokenizer, template) {
    super(ref)

    this.#tokenizer = tokenizer
    this.#template = template
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
    return this.#mlx.generate(this.ref, tokens, options)
  }

  abort () {
    if (!this.available) return
    return this.#mlx.abort(this.ref)
  }
}

export class SpeculativeCache extends MLXCache {
}

export class Session extends MLXCache {
}

export class SlidingWindowSession extends Session {
}
