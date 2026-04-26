import fs, { readFile } from 'node:fs/promises'

import { inspect } from 'node:util'
import { join } from 'node:path'

import * as mlx from 'mlx-node/swift'
import * as tokenizers from '@huggingface/tokenizers'
import * as jinja from '@huggingface/jinja'

export async function load (path, options = {}, { Tokenizer } = tokenizers, { Template } = jinja, { load, unload, stream, abort, metrics } = mlx) {
  // const jsonFiles = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json']
  // const [config, tokenizer, tokenizerConfig, specialTokenMap, generationConfig] = await Promise.all(jsonFiles.map(file => readFile(join(path, file), 'utf8').then(JSON.parse).catch(() => ({}))))

  // TOKENIZER IS MANDATORY
  try {
    const [tokenizerFile, tokenizerConfigFile] = await Promise.all([
      readFile(join(path, 'tokenizer.json'), 'utf8'),
      readFile(join(path, 'tokenizer_config.json'), 'utf8')
    ])

    const tokenizerJson = JSON.parse(tokenizerFile)
    const tokenizerConfigJson = JSON.parse(tokenizerConfigFile)

    // rationale is, tokenizerConfigFile is more likely to be there already, and if there's a chat_template in there we're golden
    // this could be an option.chat_template = 'chat_template.jinja' in future
    const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfigJson)
    const template = new Template(tokenizerConfigFile.chat_template ?? await readFile(join(path, 'chat_template.jinja'), 'utf8'))

    const modelId = await load(path)
    console.log()

    let loaded = true
    const unload = () => {
      if (loaded && unload(modelId)) loaded = false
    }

    return {
      tokenizer,
      async generate (prompt, config = {/* tools, */}, native = mlx) {
        if (!prompt) throw new Error('Prompt must be a string or an array of messages')
        if (!loaded) throw new Error('Model is not loaded')

        if (config.signal?.aborted) throw new AbortError()

        const abortHandler = native.abort.bind(native, modelId)
        const promptTokens = new Int32Array(
          tokenizer.encode(
            Array.isArray(prompt)
              ? template.render({ messages: prompt, add_generation_prompt: true /*, enable_thinking, special_tokens and other variables for the template */ })
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
      },
      async * stream (prompt, config = {/* tools, */}, native = mlx) {
        if (!prompt) throw new Error('Prompt must be a string or an array of messages')
        if (!loaded) throw new Error('Model is not loaded')

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

            if (done) { // value is stats now
              if (tokenBuffer.length > 0) { // If there's anything left in the buffer if it's an incomplete unicode we flush before closing the stream
                yield { text: tokenizer.decode(tokenBuffer), done: false }
              }
              yield { done: true, finish: value?.stopReason || 'stop', stats: value }
              break
            }

            // value is the Int32Array!
            for (const tokenId of value) tokenBuffer.push(tokenId)

            const chunk = tokenizer.decode(tokenBuffer)

            if (chunk.endsWith('\uFFFD')) {
              continue // this means chunk finishes in the middle of unicode token, wait for next tick (needs some refinement still though)
            }

            if (chunk.length > 0) {
              yield { text: chunk, done: false }
              tokenBuffer.length = 0
            }
          }
        } catch (error) {
          if (error.message?.includes('Cancelled') || error.message?.includes('Abort')) {
            yield { done: true, finish: 'abort', stats: null } // Yielding an abort frame avoids unhandled rejections during UI updates
            return
          }
          throw error
        } finally {
          config?.signal?.removeEventListener('abort', abortHandler)
        }
      },
      unload,
      [Symbol.dispose]: unload, // Symbol.asyncDispose also falls back to Symbol.dispose (but unloadModel is sync by design to prevent OOM)
      [Symbol.toStringTag]: 'MLXModel',
      [inspect.custom] () {
        return `MLXModel { id: ${modelId}, status: "${loaded ? 'loaded' : 'unloaded'}" }`
      }
    }
  } catch (err) {
    throw new Error('Tokenizer configuration missing') // fixme: this error is misleading
  }
}

async function fileExists (path, { access, constants: F_OK } = fs) {
  try {
    await access(path, F_OK)
    return true
  } catch (err) {
    return false
  }
}

class AbortError extends Error {
  constructor (message = 'The operation was aborted') {
    super(message)
    this.name = 'AbortError'
    Error.captureStackTrace(this, this.constructor)
  }
}
