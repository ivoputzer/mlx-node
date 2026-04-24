import fs, { readFile } from 'node:fs/promises'

import { inspect } from 'node:util'
import { join } from 'node:path'

import { swift as native } from 'mlx-node'

import * as tokenizers from '@huggingface/tokenizers'
import * as jinja from '@huggingface/jinja'

async function fileExists (path, { access, constants: F_OK } = fs) {
  try {
    await access(path, F_OK)
    return true
  } catch (err) {
    return false
  }
}

// mlx-lm/index.js (Inside your runStream function)
// const nativeStream = bindings.generateStream(modelId, promptTokens, JSON.stringify(mlxConfig), native);

// while (true) {
//   const { value, done } = await nativeStream.next();

//   if (done) {
//     // value is now the returned stats object!
//     yield { done: true, finish: value?.stopReason || 'stop', stats: value };
//     break;
//   }

//   // value is the Int32Array!
//   for (const tokenId of value) tokenBuffer.push(tokenId);
//   const chunkStr = tokenizer.decode(tokenBuffer);

//   if (chunkStr.length > 0) {
//     yield { text: chunkStr, done: false };
//     tokenBuffer = [];
//   }
// }

// async function * runStream (modelId, prompt, tokenizer, { signal, ...parameters }, { generateStream, cancelGenerate } = native) {
//   if (signal?.aborted) throw new Error('AbortError') // request was intentionally aborted

//   const abortHandler = cancelGenerate(modelId)
//   const queue = []

//   let resolveNext = null
//   let rejectNext = null

//   let isFinished = false

//   if (signal) {
//     signal.addEventListener('abort', abortHandler, { once: true })
//   }

//   // 2. Queue State

//   // 3. Abort Handling

//   // 4. Native Callback
//   const streamCallback = (cause, chunkInt32Array, isDone, payloadStr) => {
//     if (cause) {
//       const error = new Error(cause.message, { cause })
//       if (resolveNext) {
//         rejectNext(error)
//         resolveNext = null
//         rejectNext = null
//       } else {
//         queue.push({ err: error })
//       }
//       isFinished = true
//     } else if (isDone) {
//       let stats = null
//       if (payloadStr) {
//         try { stats = JSON.parse(payloadStr) } catch (e) {}
//       }
//       if (resolveNext) { resolveNext({ done: true, stats }); resolveNext = null; rejectNext = null } else queue.push({ done: true, stats })
//       isFinished = true
//     } else if (chunkInt32Array) {
//       if (resolveNext) { resolveNext({ tokens: chunkInt32Array }); resolveNext = null; rejectNext = null } else queue.push({ tokens: chunkInt32Array })
//     }
//   }

//   // Start native generation
//   generateStream(modelId, new Int32Array(tokenizer.encode(prompt).ids), JSON.stringify(parameters), streamCallback)

//   let tokenBuffer = []

//   try {
//     while (true) {
//       let item
//       if (queue.length > 0) {
//         item = queue.shift()
//       } else if (isFinished) {
//         break
//       } else {
//         item = await new Promise((resolve, reject) => {
//           resolveNext = resolve
//           rejectNext = reject
//         })
//       }

//       // Handle Errors & Aborts safely
//       if (item.err) {
//         if (item.err.message.includes('Cancelled') || item.err.message.includes('Abort')) break
//         throw item.err
//       }

//       // Handle Completion
//       if (item.done) {
//         yield { done: true, finish: 'stop', stats: item.stats }
//         break
//       }

//       // Handle decoding
//       for (const tokenId of item.tokens) {
//         tokenBuffer.push(tokenId)
//       }

//       const chunkStr = tokenizer.decode(tokenBuffer)
//       if (chunkStr.length > 0) {
//         yield { text: chunkStr, done: false, reasoning: false }
//         tokenBuffer = [] // Clear buffer once successfully decoded
//       }
//     }
//   } finally {
//     if (signal) {
//       signal.removeEventListener('abort', abortHandler)
//     }
//   }
// }

export async function load (path, options = {}, { Tokenizer } = tokenizers, { Template } = jinja, { loadModel, unloadModel, cancelGenerate, generateStream } = native) {
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

    const modelId = await loadModel(path)
    console.log()

    let isLoaded = true
    const unload = () => {
      if (isLoaded && unloadModel(modelId)) {
        isLoaded = false
      }
    }

    return {
      tokenizer,
      async * stream (prompt, config = {}) { // Returns AsyncIterator
        if (!isLoaded) throw new Error('Model is not loaded')
        if (!prompt) throw new Error('Prompt must be a string or an array of messages')
        if (config.signal?.aborted) throw new Error('AbortError') // request was intentionally aborted

        // tools: config.tools || [], add_generation_prompt: config.add_generation_prompt || true,  ...tokenizer.config
        const promptTokens = new Int32Array(tokenizer.encode(Array.isArray(prompt) ? template.render({ messages: prompt, add_generation_prompt: true }) : prompt).ids)
        const abortHandler = () => cancelGenerate(modelId)

        try {
          if (config.signal) {
            config.signal.addEventListener('abort', abortHandler, { once: true })
          }

          let tokenBuffer = []

          // Get the async iterator from mlx-node
          const nativeStream = generateStream(modelId, promptTokens, JSON.stringify(Object.fromEntries(Object.entries(config).filter(([key]) => ['streamChunkSize', 'maxTokens', 'maxKVSize', 'kvBits', 'kvGroupSize', 'quantizedKVStart', 'temperature', 'topP', 'topK', 'minP', 'repetitionPenalty', 'repetitionContextSize', 'presencePenalty', 'presenceContextSize', 'frequencyPenalty', 'frequencyContextSize', 'prefillStepSize'].includes(key)))))

          // Pass it to our pure formatting function
          while (true) {
            const { value, done } = await nativeStream.next()

            if (done) { // value is now the returned stats object!
              yield { done: true, finish: value?.stopReason || 'stop', stats: value }
              break
            }

            // value is the Int32Array!
            for (const tokenId of value) tokenBuffer.push(tokenId)
            const chunk = tokenizer.decode(tokenBuffer)

            if (chunk.length > 0) {
              yield { text: chunk, done: false }
              tokenBuffer = [] // tokenBuffer.length = 0 so we can use a const instead of let 🤔
            }
          }
        } catch (err) {
          if (err.message.includes('Cancelled') || err.message.includes('Abort')) return
          throw err
        } finally {
          if (config.signal) {
            config.signal.removeEventListener('abort', abortHandler)
          }
        }
      },
      async generate (messages, config) {
        // return runText(deps, messages, config)
      },
      // [Symbol.toPrimitive] (hint) {
      //   return hint === 'number' ? modelId : `MLXModel:${modelId}`
      // },
      // [Symbol.toStringTag]: 'MLXModel',
      // [inspect.custom] () {
      //   return `MLXModel { id: ${modelId}, status: "${isUnloaded ? 'unloaded' : 'active'}" }`
      // },
      unload,
      [Symbol.dispose]: unload // Symbol.asyncDispose also falls back to Symbol.dispose (but unloadModel is sync by design to prevent OOM)
    }
  } catch (err) {
    throw new Error('Tokenizer configuration missing')
  }

  // const x = new jinja.Template({}?.surelymissing)
  // console.log(x)

  // readFile(join(path, 'special_tokens_map.json'), 'utf8') // this needs to be passed down while rendering the template
  // const tokenizer = await loadTokenizer(path)
  // const template = await loadTemplate(path)
  //

  // fixme: tokenizer.chat_template is not always defined, if not we have to do something about it

  // loadModel(path),

  // // // Initialize instances
  //
  // // const chatTemplateStr = configJson.chat_template ||
  // //   "{% for message in messages %}{{ message.role + ': ' + message.content + '\n' }}{% endfor %}"

  // // Bundle dependencies for the isolated stream function
  // const deps = {
  //   modelId,
  //   tokenizer,
  //   template,
  //   bindings,
  //   bosToken: configJson.bos_token || '',
  //   eosToken: configJson.eos_token || ''
  // }

  // return {
  //   tokenizer,
  //   // template,
  //   get loaded () {
  //     return !isUnloaded
  //   },
  //   dispose () {
  //     console.log('----- model.dispose')
  //     if (isUnloaded) return true
  //     const success = unloadModel(modelId)
  //     isUnloaded = true
  //     return success
  //   },
  //   [Symbol.asyncDispose]: async () => {
  //     console.log('----- Symbol.asyncDispose')
  //     if (isUnloaded) return true
  //     const success = unloadModel(modelId)
  //     isUnloaded = true
  //     return success
  //   }

  //   // async stream (messages, config = {}) {
  //   //   if (isUnloaded) throw new Error('Cannot generate: Model is unloaded.')
  //   //   return runStream(deps, messages, config)
  //   // },
  //   // async text (messages, config) {
  //   //   let text = ''
  //   //   let stats = null

  //   //   for await (const chunk of stream(messages, config)) {
  //   //     if (chunk.text) text += chunk.text
  //   //     if (chunk.stats) stats = chunk.stats
  //   //   }

  //   //   return { text, stats }
  //   // }

  // }
}
