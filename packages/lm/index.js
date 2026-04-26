import { readFile } from 'node:fs/promises'
import { inspect } from 'node:util'
import { join } from 'node:path'

import * as mlx from 'mlx-node/swift'
import * as tokenizers from '@huggingface/tokenizers'
import * as jinja from '@huggingface/jinja'

class AbortError extends Error {
  constructor (message = 'The operation was aborted') {
    super(message)
    this.name = 'AbortError'
    Error.captureStackTrace(this, this.constructor)
  }
}

export async function load (path, options = {}, { Tokenizer } = tokenizers, { Template } = jinja, native = mlx) {
  try {
    const tokenizerPath = join(path, 'tokenizer.json')
    const tokenizerConfigPath = join(path, 'tokenizer_config.json')
    const templatePath = join(path, 'chat_template.jinja')

    // const generationConfigPath = join(path, 'generation_config.json') // -> this usually includes stuff that needs to be used during generate/stream (bos_token_id, do_sample, eos_token_id, pad_token_id, temperature)
    // const specialTokensMapPath = join(path, 'special_tokens_map.json') // this is usually what you will find in generationConfig
    // const configPath = join(path, 'config.json') // this is usually what you will find in generationConfig

    const [tokenizerFile, tokenizerConfigFile] = await Promise.all([
      readFile(tokenizerPath, 'utf8'),
      readFile(tokenizerConfigPath, 'utf8')
    ])

    const tokenizerJson = JSON.parse(tokenizerFile)
    const tokenizerConfigJson = JSON.parse(tokenizerConfigFile)

    const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfigJson)
    const template = new Template(tokenizerConfigJson.chat_template ?? await readFile(templatePath, 'utf8'))

    const modelId = await native.load(path)
    let isLoaded = true

    const unload = () => {
      if (isLoaded && native.unload(modelId)) isLoaded = false
    }

    // Return a standardized "Target" object
    return {
      type: 'Model',
      id: modelId,
      tokenizer,
      template,
      native,
      get loaded () { return isLoaded },
      unload,
      [Symbol.dispose]: unload,
      [Symbol.toStringTag]: 'MLXModel',
      [inspect.custom] () {
        return `MLXModel { id: ${modelId}, status: "${isLoaded ? 'loaded' : 'unloaded'}" }`
      }
    }
  } catch (err) {
    throw new Error(`Failed to load model configurations: ${err.message}`)
  }
}

export async function generate (target, prompt, config = {}) {
  const { id: modelId, tokenizer, template, loaded, native } = target // In Phase 2: Target could be a Cache object, we'll route appropriately

  if (!prompt) throw new Error('Prompt must be a string or an array of messages')
  if (!loaded) throw new Error('Target is not loaded or has been destroyed')
  if (config.signal?.aborted) throw new AbortError()

  const abortHandler = native.abort.bind(native, modelId)
  const promptTokens = new Int32Array(
    tokenizer.encode(
      Array.isArray(prompt)
        ? template.render({ messages: prompt, add_generation_prompt: true })
        : prompt
    ).ids
  )

  config?.signal?.addEventListener('abort', abortHandler, { once: true })

  try {
    const { tokens, stats } = await native.generate(modelId, promptTokens, config)
    if (config.signal?.aborted) throw new AbortError()
    return { stats, text: tokenizer.decode(Array.from(tokens)) }
  } catch (error) {
    if (error.message?.includes('Cancelled') || error.message?.includes('Abort')) {
      throw new AbortError()
    }
    throw error
  } finally {
    config?.signal?.removeEventListener('abort', abortHandler)
  }
}

export async function * stream (target, prompt, config = {}) {
  const { id: modelId, tokenizer, template, loaded, native } = target

  if (!prompt) throw new Error('Prompt must be a string or an array of messages')
  if (!loaded) throw new Error('Target is not loaded or has been destroyed')
  if (config.signal?.aborted) throw new AbortError()

  const abortHandler = native.abort.bind(native, modelId)
  const promptTokens = new Int32Array(
    tokenizer.encode(
      Array.isArray(prompt)
        ? template.render({ messages: prompt, add_generation_prompt: true })
        : prompt
    ).ids
  )

  try {
    config?.signal?.addEventListener('abort', abortHandler, { once: true })

    const tokenBuffer = []
    const nativeStream = native.stream(modelId, promptTokens, config)

    while (true) {
      const { value, done } = await nativeStream.next()

      if (done) {
        if (tokenBuffer.length > 0) {
          yield { text: tokenizer.decode(tokenBuffer), done: false }
        }
        yield { done: true, finish: value?.stopReason || 'stop', stats: value }
        break
      }

      for (const tokenId of value) tokenBuffer.push(tokenId)
      const chunk = tokenizer.decode(tokenBuffer)

      if (chunk.endsWith('\uFFFD')) {
        continue // wait for next chunk to resolve unicode boundary
      }

      if (chunk.length > 0) {
        yield { text: chunk, done: false }
        tokenBuffer.length = 0
      }
    }
  } catch (error) {
    if (error.message?.includes('Cancelled') || error.message?.includes('Abort')) {
      yield { done: true, finish: 'abort', stats: null }
      return
    }
    throw error
  } finally {
    config?.signal?.removeEventListener('abort', abortHandler)
  }
}
