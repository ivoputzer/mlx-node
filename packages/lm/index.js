import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

import mlx from 'mlx-node/swift'

import * as tokenizers from '@huggingface/tokenizers'
import * as jinja from '@huggingface/jinja'

// API

export async function prefill (target, prompt, options = {}) {
  const { ids } = target.encode(prompt)
  const signal = options.signal

  // Create the evaluate task
  const task = target.evaluate(new Int32Array(ids), options)

  // Link signal if provided
  const onAbort = () => task.abort()
  if (signal) {
    if (signal.aborted) {
      task.abort()
    } else {
      signal.addEventListener('abort', onAbort, { once: true })
    }
  }

  try {
    return await task // Await it directly!
  } finally {
    if (signal) signal.removeEventListener('abort', onAbort)
  }
}

export async function evaluate (target, prompt, options = {}) {
  if (!prompt) throw new Error('Prompt cannot be empty')
  if (!target.available) throw new Error('Target not loaded or disposed already')

  const signal = options?.signal ?? new AbortController().signal
  if (signal?.aborted) throw new AbortError(signal?.reason)

  const { ids } = target.encode(prompt, { template: options?.template })
  const generate = target.generate(new Int32Array(ids), { ...options, maxTokens: 0 })

  // Directly bind signal to the underlying MLX C Task Cancellation
  const onAbort = () => generate.abort()
  signal?.addEventListener('abort', onAbort, { once: true })

  try {
    while (true) {
      const { value, done } = await generate.next()
      if (done) {
        if (signal?.aborted) throw new AbortError(signal?.reason)
        return value
      }
    }
  } catch (error) {
    if (error.message?.includes('Generation Cancelled')) throw new AbortError(signal?.reason)
    throw error
  } finally {
    signal?.removeEventListener('abort', onAbort)
  }
}

export async function generate (target, prompt, options = {}) {
  if (!prompt) throw new Error('Prompt cannot be empty')
  if (!target.available) throw new Error('Target not loaded or disposed already')

  const signal = options?.signal ?? new AbortController().signal
  if (signal?.aborted) throw new AbortError(signal?.reason)

  const { ids } = target.encode(prompt, { template: options?.template })
  const tokens = []

  const generate = target.generate(new Int32Array(ids), { ...options })

  const onAbort = () => generate.abort()
  signal?.addEventListener('abort', onAbort, { once: true })

  try {
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
  if (signal?.aborted) throw new AbortError(signal?.reason)

  const { ids } = target.encode(prompt, { template: options?.template })
  const generate = target.generate(new Int32Array(ids), { ...options })

  const onAbort = () => generate.abort()
  signal?.addEventListener('abort', onAbort, { once: true })

  const tokens = []
  const buffer = []

  try {
    while (true) {
      const { value, done } = await generate.next()
      if (done) {
        if (buffer.length > 0) {
          const text = target.decode(buffer, { skip_special_tokens: options?.skipSpecialTokens ?? true })
          yield { text, done: false }
        }
        yield {
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
          continue
        } else {
          if (text.length > 0) {
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

export async function * streamBatch (target, prompts, options = {}) {
  if (!Array.isArray(prompts) || prompts.length === 0) throw new Error('Prompts must be a non-empty array')
  if (!target.available) throw new Error('Target not loaded or disposed already')

  const signal = options?.signal ?? new AbortController().signal
  if (signal?.aborted) throw new AbortError(signal?.reason)

  // Encode all prompts independently
  const encodedPrompts = prompts.map(prompt => {
    const { ids } = target.encode(prompt, { template: options?.template })
    return new Int32Array(ids)
  })

  const generate = target.batch(encodedPrompts, options)

  const onAbort = () => generate.abort() // Wait, `batch` doesn't return a task, it returns the generator.
  // We need to attach abort signal handling
  signal?.addEventListener('abort', onAbort, { once: true })

  const buffers = prompts.map(() => [])
  const tokensAccumulator = prompts.map(() => [])

  try {
    while (true) {
      const { value, done } = await generate.next()

      if (done) {
        // Yield the final stats and full text payload
        yield {
          done: true,
          results: value.map((stats, i) => ({
            text: target.decode(tokensAccumulator[i], { skip_special_tokens: options?.skipSpecialTokens ?? true, clean_up_tokenization_spaces: options?.cleanUpTokenizationSpaces ?? true }),
            finish: stats?.stopReason || 'stop',
            tokens: tokensAccumulator[i],
            stats
          }))
        }
        break
      } else {
        // value is an array of tokens, e.g., [token1, token2]
        const textYields = new Array(prompts.length).fill('')

        for (let i = 0; i < value.length; i++) {
          const token = value[i]
          if (token !== -1) {
            tokensAccumulator[i].push(token)
            buffers[i].push(token)

            const text = target.decode(buffers[i], { skip_special_tokens: options?.skipSpecialTokens ?? true, clean_up_tokenization_spaces: false })
            if (!text.endsWith('\uFFFD')) { // Wait for valid unicode characters
              textYields[i] = text
              buffers[i].length = 0
            }
          }
        }

        // Only yield if at least one stream produced valid text
        if (textYields.some(t => t.length > 0)) {
          yield { text: textYields, done: false }
        }
      }
    }
  } catch (error) {
    if (error.message?.includes('Cancelled') || error.message?.includes('Abort')) {
      yield { done: true, finish: 'abort', results: null }
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

export function padTokenFrom (tokenizer) {
  const config = tokenizer.config
  const hasTokenId = tokenizer.model.tokens_to_ids.has.bind(tokenizer.model.tokens_to_ids) // tokenizer.model.tokens_to_ids is a Map()
  const getTokenId = tokenizer.token_to_id

  // Actual pad token ID
  if (config.pad_token_id !== undefined && config.pad_token_id !== null) {
    return config.pad_token_id
  }

  // Sometimes it's passed as a string in tokenizer_config.json
  if (config.pad_token && hasTokenId(config.pad_token)) {
    return getTokenId(config.pad_token)
  }

  // Fallback: Unknown Token (The safest "white noise" for Mamba KVCache)
  if (config.unk_token_id !== undefined && config.unk_token_id !== null) {
    return config.unk_token_id
  }

  if (config.unk_token && hasTokenId(config.unk_token)) {
    return getTokenId(config.unk_token)
  }

  return 0 // Absolute Last Resort: 0 (Usually maps to <unk>, <s>, or <pad> anyway)
}

export const stopTokenIdsFrom = stopTokensFrom // @deprecated

export function stopTokensFrom (tokenizer) {
  const stopTokens = new Set()

  for (const [key, value] of Object.entries(tokenizer.config)) {
    if (key.includes('eos_token') || key.includes('eot_token') || key.includes('pad_token')) {
      if (tokenizer.model.tokens_to_ids.has(value)) {
        stopTokens.add(tokenizer.token_to_id(value))
      }
    }
  }

  return Array.from(stopTokens)
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
  // todo: maybe check if path is a directory or a file, if it's a file we might load the file directly?
  const tokenizerConfigPath = join(path, 'tokenizer_config.json')
  const templatePath = join(path, 'chat_template.jinja')
  try {
    return new Template(await readFile(templatePath, 'utf8'))
  } catch {
    // this is just a lazy fallback if there's a template it should be there and we shouldn't even have to read this file (again)
    return new Template(JSON.parse(await readFile(tokenizerConfigPath, 'utf8'))?.chat_template)
  }
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

export async function loadModel (path) {
  const tokenizer = await loadTokenizer(path)
  const template = await loadTemplate(path)
  return MLXModel.loadFrom(path)
}

export async function loadCache (path) {
}

export async function createCache (model) {

}

// CLASSES

export class Model {
  #tokenizer
  #template
  // static async load (path, tokenizer, template) {
  //   const ref = await mlx.loadModel(path)
  //   return new MLXModel(ref, tokenizer, template)
  // }

  // constructor (ref, tokenizer, template) {
  //   super(ref)

  //   this.#tokenizer = tokenizer
  //   this.#template = template
  // }

  // // this has nothing to do with mlx
  // encode (prompt, options = {}, { Template } = jinja) {
  //   if (typeof prompt === 'string') {
  //     return this.#tokenizer.encode(prompt, { add_special_tokens: true, ...options })
  //   } else {
  //     return options?.template?.length
  //       ? this.#tokenizer.encode(new Template(options.template).render(prompt), { add_special_tokens: false, ...options })
  //       : this.#tokenizer.encode(this.#template.render({ add_generation_prompt: true, ...prompt }), { add_special_tokens: false, ...options })
  //   }
  // }

  // decode (tokens, options = {}) {
  //   return this.#tokenizer.decode(tokens, options)
  // }

  // generate (tokens, options) {
  //   /* returns a MLXStream */
  //   return mlx.generate(this.ref, null, tokens, options)
  // }

  // abort () {
  //   if (!this.available) return
  //   return mlx.abort(this.ref)
  // }
}

export class Cache {
  #model

  // async save (path) {
  //   if (!this.available) throw new Error('Cannot save disposed cache')
  //   try {
  //     await mlx.cacheSave(this.ref, path)
  //     return true
  //   } catch (error) {
  //     // error.code will be 'MLX_SAVE_ERROR' from the C layer
  //     throw error
  //   }
  // }

  // static async load (path) {
  //   try {
  //     const ref = await mlx.cacheLoad(path)
  //     return new MLXCache(ref)
  //   } catch (error) {
  //     // error.code will be 'MLX_LOAD_ERROR' from the C layer
  //     throw error
  //   }
  // }

  // async static load (path) {
  //   const id = await mlx.cacheLoad(path)
  //   return new MLXCache(id)
  // }

  // static from (model, options = {}) {
  //   const id = mlx.cacheCreate(model.id, JSON.stringify(options))
  //   return new MLXCache(id)
  // }

  // save (path) {
  //   if (!this.available) throw new Error('Cache unavailable')
  //   return mlx.cacheSave(this.id, path) // throws message:{from swift} code: MLX_SAVE_ERROR
  // }

  // trim (numTokens) {
  //   if (!this.available) throw new Error('Cache unavailable')
  //   return mlx.cacheTrim(this.id, numTokens)
  // }

  // clone () {
  //   if (!this.available) throw new Error('Cache unavailable')
  //   const id = mlx.cacheClone(this.id)
  //   return new MLXCache(id)
  // }
}

// class ChatSession {
//   #cache
//   #template

//   #stopToken

//   constructor (cache, tokenizer, template) {
//     this.#cache = cache
//     this.#template = template
//     this.stopToken = tokenizer.config.c

//     // Create the C-pointer cache
//     this.cache = this.model.createCache()

//     // State trackers
//     this.isCacheHot = false
//   }

//   // Helper: Renders ONLY the delta (the new stuff)
//   _renderDelta (messages, addGenPrompt) {
//     return this.templateRenderer.render({
//       messages,
//       add_generation_prompt: addGenPrompt
//     })
//   }

//   async generate (newMessages) {
//     let promptString = ''

//     if (!this.isCacheHot) {
//       // SCENARIO 1: COLD CACHE (Turn 1)
//       // Render the full history including System Prompt
//       promptString = this._renderDelta(newMessages, true)
//     } else {
//       // SCENARIO 2: HOT CACHE (Turn 2+)
//       // 1. Close the model's mouth from the previous turn
//       // 2. Render ONLY the new user message
//       const deltaText = this._renderDelta(newMessages, true)
//       promptString = this.stopTokenString + deltaText
//     }

//     // Tokenize ONLY the promptString
//     const inputTokens = this.model.tokenizer.encode(promptString)

//     // Run the C-Bridge generate task
//     const response = await this.model.generateTask(inputTokens, this.cache)

//     // Mark cache as hot so the next turn knows what to do!
//     this.isCacheHot = true

//     return response.text
//   }

//   async continue (injectedText) {
//     // SCENARIO 3: ASSISTANT PREFILLING / CONTINUATION
//     if (!this.isCacheHot) throw new Error('Cannot continue an empty cache!')

//     // Notice we DO NOT prepend the stopTokenString.
//     // We just feed the raw injected text straight into the open mouth!
//     const inputTokens = this.model.tokenizer.encode(injectedText)

//     // The model will seamlessly continue from injectedText
//     const response = await this.model.generateTask(inputTokens, this.cache)
//     return response.text
//   }
// }
