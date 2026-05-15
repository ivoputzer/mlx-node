import { Readable } from 'node:stream'
import path from 'node:path'
import os from 'node:os'
import fs from 'node:fs/promises'

import { MLXModel, MLXCache } from 'mlx-node'

import * as tokenizers from '@huggingface/tokenizers'
import * as jinja from '@huggingface/jinja'

import { padTokenFrom, stopTokensFrom } from './lib/tokenizer.js'

/// INTERNAL FUNCTIONS

export const internal = {
  readJson (path, { readFile } = fs) {
    return readFile(path, 'utf8').then(JSON.parse)
  },
  expandTilde (reltive, { homedir } = os, { join, resolve } = path) {
    if (reltive[0] === '~') {
      return join(homedir(), reltive.slice(1))
    }
    return resolve(reltive)
  }
}

/// LM FUNCTIONS

function resolveSignal (options) {
  const controller = new AbortController()
  const signal = options?.signal ? AbortSignal.any([options.signal, controller.signal]) : controller.signal
  return { signal, controller }
}

function flattenPayload (payload, batchSize) {
  if (batchSize > 1) return payload
  const flat = {}
  for (const [key, val] of Object.entries(payload)) {
    flat[key] = Array.isArray(val) ? val[0] : val
  }
  return flat
}

function formatLogits (logitsObj) {
  if (!logitsObj) return []

  const { ids, probs } = logitsObj
  return Array.from(ids).map((id, i) => ({ id, prob: probs[i] })) // The user only calls this exactly when they need it, preventing 99% of unnecessary allocations in the background stream.
}

export async function prefill (target, prompt, options = {}) {
  const { signal } = resolveSignal(options)
  if (signal.aborted) throw new AbortError(signal.reason)

  const promptString = target.render(prompt)
  const promptTokens = target.encode(promptString)

  const task = target.prefill(promptTokens, options) // fixme: this needs to be implemented this.mlx.evaluate
  const onAbort = () => task.abort() // this is broken

  try {
    signal.addEventListener('abort', onAbort, { once: true })
    return await task
  } finally {
    if (signal) signal.removeEventListener('abort', onAbort)
  }
}

/*
  This function is kinda duplicated but it reduces the amount of complexity we in streaming responses

  -> Less GC means fewer memory leaks and better performance
  -> Less parser overhead as they will only be called on setup(), flush(), skipping push() entirely
     Note: If not implemented correctly this can lead to flakey controller.abort() behaviours

  target:
    - the target object to render the prompt against (Model, Cache, Session, ...)
  prompt:
    - When string: The prompt to render against the target's chat template
    - When object: The object is passed directly to the Template.render() function (ie. {messages, tools, add_generation_prompt:false})
      -> add_generation_prompt: true is added automatically but can be overwritten
      -> Note: look at the chat_tempalte.jinja, or tokenizer_config.json .chat_tempalte for the available variables
      -> Note: if you dont know what template is being used, loadTemplate(pathToModelDir).then(t => t.format(options))
  options:
    - tempalte<string> can be an inline template to override the target's chat template
    - signal<AbortSignal> can be used to abort the request
    generation options (loadOptions() -> reads generation_config.json to load default values for a given model):
    - batchSize<number> can be used to control the batch size (default: 1)
    - chunkSize<number> can be used to control the chunk size (default: 2147483647 – MAX_SAFE_INTEGER to prevent TSFN boundary crossings)
    - maxTokens<number> can be used to control the maximum number of tokens to generate (default: Infinity)

    - topLogits<number> can be used to control the number of top logits to consider (default: 0)
    - stopTokenIds
    - padTokenId

    - temperature
    - topP
    - topK
    - minP
    - repetitionPenalty
    - repetitionContextSize
    - presencePenalty
    - presenceContextSize
    - frequencyPenalty
    - frequencyContextSize
    - prefillStepSize

    Only for Caches and Cache-Extensions:
    - maxKVSize
    - kvBits
    - kvGroupSize
    - quantizedKVStart

    tokenizer options:
    - skipSpecialTokens<boolean> can be used to skip special tokens in the prompt (default: true)
    - cleanUpTokenizationSpaces<boolean> can be used to clean up tokenization spaces in the prompt (default: true)
*/
export async function _generate (target, prompt, options = {}, parsers = [/* new ReasoningParser(), new ToolParser(), new MarkdownBlockParser() */]) {
  if (!prompt) throw new Error('Prompt cannot be empty')

  const { controller, signal } = resolveSignal(options) // we should be able to resuse this
  if (signal.aborted) throw new AbortError(signal.reason)

  const batchSize = options?.batchSize ?? 1
  const promptString = target.render(prompt, options)
  const promptTokens = target.encode(promptString, options)

  for (const parser of parsers) {
    parser?.target(target)
    parser?.controller(controller)
    parser.setup(promptString, promptTokens, batchSize)
  }

  const task = target.generate(promptTokens, { ...options, chunkSize: 2147483647 })
  const onAbort = task.abort.bind(task)

  try {
    signal.addEventListener('abort', onAbort, { once: true })
    const iterator = task[Symbol.asyncIterator]()

    const batchedTokens = Array.from({ length: batchSize }, () => [])
    const batchedLogits = Array.from({ length: batchSize }, () => []) // this dependes on options?.topLogits

    while (true) {
      const { value, done } = await iterator.next()
      if (done) {
        const batchedText = batchedTokens.map(tokens => target.decode(tokens, options))
        // we can do all text manipulation independently from stats!
        // done:value -> (stopReason: 'length', tokensPerSecond: 45.03159658694281, generateTime: 45.479178071022034, promptTokensPerSecond: 41.50265147433228, promptTokens: 32, promptTime: 0.7710350751876831, generatedTokens: 2048)
        //     const text = batches.map(batch => target.decode(batch, options)) // should this be a parser also? 🤔
        //     return {
        //       finish: value?.stopReason,
        //       stats: value,
        //       text: length > 1 ? text : text[0],
        //       ...parsers.reduce((props, parser) => ({ ...props, ...parser.flush(batches, text) }), {})
        //     }
        //
        let parserFields = {}

        for (const parser of parsers) {
          parserFields = {
            ...parserFields,
            ...parser.flush(batchedText, batchedTokens, batchedLogits, batchSize)
          }
        }
        return {
          stats: value,
          ...flattenPayload({
            text: batchedText,
            // tokens: batchedTokens,
            // logits: batchedLogits,
            parserFields
          }, batchSize)
        }
      } else {
        const { tokens, logits } = value
        for (let i = 0; i < batchSize; i++) {
          batchedTokens[i].push(tokens[i])
          if (logits !== null) {
            batchedLogits[i].push(logits[i])
          }
        }
      }
    }
  } catch (error) {
    if (error.message?.includes('Generation Cancelled')) throw new AbortError(signal.reason)
    throw error
  } finally {
    signal.removeEventListener('abort', onAbort)
  }
}

export async function generate (target, prompt, options = {}, plugins = []) {
  if (!prompt) throw new Error('Prompt cannot be empty')

  const { controller, signal } = resolveSignal(options)
  if (signal.aborted) throw new AbortError(signal.reason)

  const batchSize = options?.batchSize ?? 1
  const promptTokens = target.encode(target.render(prompt, options), options)

  const task = target.generate(promptTokens, { ...options, chunkSize: 2147483647 })
  const onAbort = task.abort.bind(task)

  try {
    signal.addEventListener('abort', onAbort, { once: true })

    async function * stream () {
      const iterator = task[Symbol.asyncIterator]()
      const allTokens = Array.from({ length: batchSize }, () => [])
      const allLogits = Array.from({ length: batchSize }, () => [])

      while (true) {
        const { value, done } = await iterator.next()

        if (done) {
          yield {
            done: true,
            finish: value?.stopReason || 'stop',
            stats: value,
            text: allTokens.map(tokens => target.decode(tokens, options)),
            tokens: allTokens,
            logits: allLogits
          }
          break
        }

        for (let i = 0; i < batchSize; i++) {
          allTokens[i].push(value.tokens[i])
          if (value.logits) allLogits[i].push(value.logits[i])
        }

        // Headless tick
        yield {
          done: false,
          text: null,
          // tokens: value.tokens,
          // topN: value.logits
        }
      }
    }

    const context = { target, options, batchSize, controller }
    const pipeline = plugins.reduce((pipeline, plugin) => plugin(pipeline, context), stream())

    // Silently consume the pipeline, returning the final mutated chunk
    let final
    for await (const chunk of pipeline) final = chunk
    return final // return Readable.from(pipeline).reduce((_, chunk) => chunk)
  } finally {
    signal.removeEventListener('abort', onAbort)
  }
}

export async function * stream (target, prompt, options = {}, parsers = []) {
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
    signal.removeEventListener('abort', onAbort)
  }
}

export async function * batch (target, prompts, options = {}, parsers = []) {}

/// HELPERS FUNCTIONS

export async function loadTokenizer (base, { Tokenizer } = tokenizers, { join } = path, { readJson } = internal) {
  const tokenizerPath = join(base, 'tokenizer.json')
  const tokenizerConfigPath = join(base, 'tokenizer_config.json')

  return new Tokenizer(
    ...await Promise.all([
      readJson(tokenizerPath),
      readJson(tokenizerConfigPath) // Note: as of right now we dont care about older models without tokenizer_config.json
    ])
  )
}

export async function loadTemplate (base, { Template } = jinja, { readFile } = fs, { join } = path, { readJson } = internal) {
  try {
    const path = join(base, 'chat_template.jinja')
    return new Template(await readFile(path, 'utf8'))
  } catch {
    try {
      const path = join(base, 'tokenizer_config.json')
      const { chat_template: template } = await readJson(path)
      if (Array.isArray(template)) {
        const templates = Object.fromEntries(
          template.map(({ name, template }) => [name, new Template(template)])
        )
        return {
          entries: templates,
          render (items) {
            return items?.tools?.length && templates?.tool_use // lfg mistral 😅
              ? templates?.tool_use.render(items)
              : templates?.default.render(items)
          },
          format (options) {
            return Object.fromEntries(
              Object.entries(templates).map(([name, template]) => [name, template.format(options)])
            )
          }
        }
      } else {
        return new Template(template) // let it throw if there's no template so we have a cause
      }
    } catch (cause) {
      throw new Error('No valid template found in model path.', { cause })
    }
  }
}

export async function loadModel (path, { expandTilde } = internal) {
  return Model.load(expandTilde(path))
}

export async function loadCache (path) { }

export async function loadOptions (base, { join } = path, { readJson } = internal) {
  try {
    const path = join(base, 'generation_config.json')
    return await readJson(path)
  } catch {
    return {}
  }
}

/// CLASSES

export class AbortError extends Error {
  constructor (message = 'The operation was aborted') {
    super(message)
    this.name = 'AbortError'
    Error.captureStackTrace(this, this.constructor)
  }
}

export class Target {
  #tokenizer
  #template

  get tokenizer () {
    return this.#tokenizer
  }

  get template () {
    return this.#template
  }

  constructor (tokenizer, template) {
    this.#tokenizer = tokenizer
    this.#template = template

    this.stopTokens = stopTokensFrom(this.#tokenizer)
    this.padToken = padTokenFrom(this.#tokenizer)

    this.stopTokenIds = this.stopTokens.map(({ id }) => id)
    this.padTokenId = this.padToken?.id
  }

  render (prompt, options = {}, { Template } = jinja) {
    if (typeof prompt === 'string') return prompt // this does not need rendering
    return options?.template?.length
      ? new Template(options.template).render(prompt)
      : this.#template.render({ add_generation_prompt: true, ...prompt })
  }

  encode (prompt, options = {}) {
    const { ids } = this.#tokenizer.encode(prompt, { add_special_tokens: true, ...options })
    return new Int32Array(ids)
  }

  decode (tokens, { skipSpecialTokens, cleanUpTokenizationSpaces }) {
    return this.#tokenizer.decode(tokens, {
      skip_special_tokens: skipSpecialTokens ?? true,
      clean_up_tokenization_spaces: cleanUpTokenizationSpaces ?? true
    })
  }

  prefill (tokens, options = {}) {
    return this.mlx.evaluate(tokens, {
      ...options
    })
  }

  generate (tokens, options = {}) {
    return this.mlx.generate(tokens, {
      stopTokenIds: this.stopTokenIds,
      padTokenId: this.padTokenId,
      ...options
    })
  }

  // get tokenizer () { return this.#tokenizer }
  // get template () { return this.#template }
  // get model () { return this }
  // get mlx () { throw new Error('Not implemented') }
  // get available () { return this.mlx?.available }

  // encode (prompt, options = {}) {
  //   const text = typeof prompt === 'string'
  //     ? prompt
  //     : this.#template.render({ messages: prompt, add_generation_prompt: true, ...options })

  //   return this.#tokenizer.encode(text, { add_special_tokens: true, ...options }).ids
  // }

  // decode (tokens, options = { skip_special_tokens: true }) {
  //   return this.#tokenizer.decode(Array.from(tokens), options)
  // }

  // formatTopLogits (rawTopLogits) {
  //   if (!rawTopLogits) return null
  //   return rawTopLogits.map(sequenceTops =>
  //     sequenceTops.map(top => ({
  //       id: top.id,
  //       text: this.#tokenizer.model.id_to_token(top.id),
  //       prob: top.prob
  //     }))
  //   )
  // }

  dispose () {
    this.mlx?.dispose()
  }

  [Symbol.dispose] () {
    this.mlx?.dispose()
  }
}

export class Model extends Target {
  #model

  get mlx () {
    return this.#model
  }

  static async load (path) {
    return new Model(
      await MLXModel.fromPath(path),
      await loadTokenizer(path),
      await loadTemplate(path),
      await loadOptions(path)
    )
  }

  constructor (model, tokenizer, template, options = {}) {
    super(tokenizer, template, options)
    this.#model = model
  }
}

export class Cache extends Target {
  #cache

  get mlx () {
    return this.#cache
  }

  constructor (model, options = {}) {
    super(model.tokenizer, model.template, options)
    this.#cache = MLXCache.fromModel(model.mlx, options)
  }

  // clone () { return new Cache(this, this.#mlxCache.clone()) }
  // trim (tokens) { return this.#mlxCache.trim(tokens) }
}

export class Session extends Cache {
  // #history = []
  // #cachedTokens = 0

  // get history () { return this.#history }

  // getDeltaTokens (newMessage) {
  //   if (newMessage) this.#history.push(newMessage)
  //   const fullTokens = this.encode(this.#history, { add_generation_prompt: true })
  //   return {
  //     deltaTokens: fullTokens.slice(this.#cachedTokens),
  //     newTotal: fullTokens.length
  //   }
  // }

  // commit (newTotal, generatedCount) {
  //   this.#cachedTokens = newTotal + generatedCount
  // }
}

/// PLUGINS

export function ExamplePlugin (options) {
  return async function * (pipeline, context) {
    // const capturedToolTokens = []
    for await (const event of pipeline) {
      if (event.done) {
        // Because we mutate the object directly, the user gets it automatically!
        // event.toolCalls = [{ name: 'weather', args: {} }]
        event.example = 'ExamplePluginFinalResult'
        yield event
        continue
      } else {
        yield event
      }
      // // We can rely on `chunk.tokens` to do logic, regardless of if we are
      // // in generate() (where text is null) or stream() (where text is decoded).
      // if (event.tokens && event.tokens[0] === 1500) {
      //   event.isExamplePlugin = true // Attach boolean flag to intermediate chunk
      //   // chunk.text = '' // We could swallow the text if we wanted
      // }
    }
  }
}

export function ReasoningPlugin (options) {
  return async function * (pipeline, context) {
    // const capturedToolTokens = []
    for await (const event of pipeline) {
      if (event.done) {
        // Because we mutate the object directly, the user gets it automatically!
        // event.toolCalls = [{ name: 'weather', args: {} }]
        event.reasoningConent = 'The reasoning was this and that'
        yield event
        continue
      } else {
        yield event
      }
      // // We can rely on `chunk.tokens` to do logic, regardless of if we are
      // // in generate() (where text is null) or stream() (where text is decoded).
      // if (event.tokens && event.tokens[0] === 1500) {
      //   event.isExamplePlugin = true // Attach boolean flag to intermediate chunk
      //   // chunk.text = '' // We could swallow the text if we wanted
      // }
    }
  }
}
