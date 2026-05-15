import { describe, it } from 'node:test'
import { strictEqual, deepEqual, rejects, ok } from 'node:assert/strict'
import { generate, stream, prefill } from '../index.js'

describe('mlx-lm', () => {
  const createMockTarget = (overrides = {}) => ({
    available: true,
    encode: () => ({ ids: [1, 2, 3] }),
    decode: (tokens) => Array.from(tokens).map(t => `[${t}]`).join(''),
    generate: async function * () {
      yield 10
      yield 20
      return { stopReason: 'stop', generatedTokens: 2 }
    },
    abort: () => {},
    ...overrides
  })

  describe('.generate', () => {
    // Input Handling
    it.todo('should return a result object when given a string prompt')
    it.todo('should render and process prompt objects via chat templates')
    it.todo('should execute prompts concurrently when given an array of inputs')
    it.todo('should fork the target (clone cache) for each sequence in concurrent mode')

    // Performance & Optimization
    it.todo('should skip intermediate text decoding during generation (headless mode)')
    it.todo('should respect the batchSize option for homogeneous processing')
    it.todo('should execute plugins in the specified pipeline order')

    // Configuration & Logic
    it.todo('should use inline chat template overrides when provided in options')
    it.todo('should immediately abort and clean up if the AbortSignal is triggered')
    it.todo('should correctly reconstruct multi-byte (UTF-8) strings in the final output')

    // Schema Validation (Final Chunk)
    it.todo('should include the stop reason in the "finish" property')
    it.todo('should include the complete token history in the "tokens" property')
    it.todo('should include the accumulated "logits" when topK is enabled')
    it.todo('should include a "stats" object containing native performance metrics')
    it.todo('should include plugin-mutated fields in the final response')

    // Deprecated
    it.skip('should return complete text, tokens, and stats on success', async () => {
      const target = createMockTarget()
      const result = await generate(target, 'test prompt')

      strictEqual(result.finish, 'stop')
      deepEqual(result.tokens, [10, 20])
      strictEqual(result.text, '[10][20]')
      strictEqual(result.stats.generatedTokens, 2)
    })
    it.skip('should throw AbortError if cancelled during generation', async () => {
      const controller = new AbortController()
      const target = createMockTarget({
        generate: async function * () {
          yield 10
          controller.abort('User stopped')
          yield 20
          return {}
        }
      })

      await rejects(
        generate(target, 'test', { signal: controller.signal }),
        { name: 'AbortError', message: 'User stopped' }
      )
    })
    it.skip('should correctly pass tokenizer options down', async () => {
      let capturedOptions = null
      const target = createMockTarget({
        decode: (tokens, options) => {
          capturedOptions = options
          return 'mocked'
        }
      })

      await generate(target, 'test', { skipSpecialTokens: false, cleanUpTokenizationSpaces: false })

      strictEqual(capturedOptions.skip_special_tokens, false)
      strictEqual(capturedOptions.clean_up_tokenization_spaces, false)
    })
  })

  describe('.stream', () => {
    // Yielding Behavior
    it.todo('should yield intermediate text chunks as they are generated')
    it.todo('should yield a final chunk with done: true and full results')
    it.todo('should yield an array of results when batchSize > 1')
    it.todo('should zip multiple streams together when given an array of prompts')

    // Streaming Integrity
    it.todo('should buffer and hold tokens that would result in a partial UTF-8 sequence')
    it.todo('should yield multiple tokens in a single tick if they complete a UTF-8 character')
    it.todo('should yield empty text strings for ticks that only contain special/swallowed tokens')

    // Plugin Interaction (The Middleware)
    it.todo('should allow plugins to attach boolean flags to intermediate chunks')
    it.todo('should allow plugins to swallow text by modifying the chunk before it reaches the user')
    it.todo('should provide plugins with access to raw token IDs even if text is not yet decoded')
    it.todo('should merge plugin-calculated final data into the done: true chunk')

    // Resource Management
    it.todo('should stop generation and release native resources when the loop is broken')
    it.todo('should stop generation and release native resources when the AbortSignal is triggered')

    // Configuration Accuracy
    it.todo('should respect skipSpecialTokens for every yielded intermediate chunk')
    it.todo('should respect cleanUpTokenizationSpaces for the final chunk but not intermediate ticks')
    it.todo('should provide zero-copy logit views in every tick if topK is enabled')

    // Deprecated
    it.skip('should yield chunks and a final done object via incremental decoding', async () => {
      const target = createMockTarget({
        decode: (tokens) => {
          const str = Array.from(tokens).join('')
          // Logic for the streaming loop (buffer is cleared each time)
          if (str === '10') return 'Hello'
          if (str === '20') return ' World'

          // Logic for the final 'done' event (receives allTokens [10, 20])
          if (str === '1020') return 'Hello World'

          return ''
        }
      })

      const iterator = stream(target, 'test prompt')

      const event1 = await iterator.next()
      strictEqual(event1.value.done, false)
      strictEqual(event1.value.text, 'Hello')

      const event2 = await iterator.next()
      strictEqual(event2.value.done, false)
      strictEqual(event2.value.text, ' World')

      const event3 = await iterator.next()
      strictEqual(event3.value.done, true)
      strictEqual(event3.value.text, 'Hello World')
      strictEqual(event3.value.finish, 'stop')
      deepEqual(event3.value.tokens, [10, 20])
      ok(event3.value.stats)
    })
    it.skip('should handle unicode \\uFFFD boundaries by buffering', async () => {
      const target = createMockTarget({
        generate: async function * () {
          yield 1 // Incomplete char
          yield 2 // Complete char
          return { stopReason: 'stop' }
        },
        decode: (tokens) => {
          if (tokens.length === 1) return '\uFFFD'
          if (tokens.length === 2) return '🚀'
          return ''
        }
      })

      const events = []
      for await (const event of stream(target, 'test')) {
        events.push(event)
      }

      strictEqual(events.length, 2)
      strictEqual(events[0].done, false)
      strictEqual(events[0].text, '🚀')
      strictEqual(events[1].done, true)
      strictEqual(events[1].text, '🚀')
    })
    it.skip('should swallow native Cancelled errors and yield a graceful abort event', async () => {
      const target = createMockTarget({
        generate: async function * () {
          yield 10
          throw new Error('Native Generation Cancelled')
        }
      })

      const events = []
      for await (const event of stream(target, 'test')) {
        events.push(event)
      }

      strictEqual(events.length, 2)
      strictEqual(events[1].done, true)
      strictEqual(events[1].finish, 'abort')
      strictEqual(events[1].stats, null)
    })
  })

  describe('.batch', () => {
    // Tensor Construction (The "Padded" Logic)
    it.todo('should correctly left-pad jagged prompt arrays into a dense tensor')
    it.todo('should use the user-provided "padTokenId" for padding the tensor')
    it.todo('should throw an error if all prompts in the batch are empty')
    it.todo('should handle extreme length differences (e.g., 1 token vs 1000 tokens) gracefully')

    // Execution Behavior
    it.todo('should yield the same number of results as the input batch size')
    it.todo('should continue generating until all sequences in the batch hit a stop condition')
    it.todo('should yield padding tokens (-1 or padId) for sequences that finish early')
    it.todo('should execute as a single native task (verify only one FFI call to bridge_model_batch_task)')

    // Plugin Integration
    it.todo('should initialize a separate plugin pipeline instance for each sequence in the batch')
    it.todo('should correctly route batch-indexed tokens to the corresponding plugin pipeline')

    // Final Output
    it.todo('should return an array of stats objects, one for each sequence in the batch')
    it.todo('should correctly map the final text outputs back to the original prompt indices')
  })

  describe('.prefill', () => {
    // Core Functional Tests
    it.todo('should process prompt tokens and update the KV cache state')
    it.todo('should return a stats object containing prefill speed (tokens/sec) and latency')
    it.todo('should not generate any new tokens (output text should be empty)')

    // Cache State Integrity
    it.todo('should increase the "cached tokens" count in the target cache after execution')
    it.todo('should allow additive prefills (calling prefill multiple times for a growing conversation)')
    it.todo('should result in identical KV state compared to a generate() call with the same prompt')

    // Resource & Performance
    it.todo('should respect the "prefillStepSize" option to control processing chunking')
    it.todo('should throw an error if the prompt exceeds the maximum KV cache size')
    it.todo('should be significantly faster than a generate() call for the same number of tokens')

    // Signal Handling
    it.todo('should allow aborting a long prefill task mid-way and releasing the task pointer')

    // Deprecated
    it.skip('should return stats when evaluation completes successfully', async () => {
      const target = createMockTarget({
        generate: async function * () {
          yield [1]
          return { promptTokens: 3, generatedTokens: 0 }
        }
      })
      const result = await prefill(target, 'test prompt')

      strictEqual(result.promptTokens, 3)
      strictEqual(result.generatedTokens, 0)
    })
    it.skip('should throw an error if the prompt is empty', async () => {
      const target = createMockTarget()
      await rejects(prefill(target, ''), { message: 'Prompt cannot be empty' })
    })
    it.skip('should throw an error if the target is not ready', async () => {
      const target = createMockTarget({ available: false })
      await rejects(prefill(target, 'test'), { message: 'Target not loaded or disposed already' })
    })
    it.skip('should throw AbortError if signal is already aborted', async () => {
      const target = createMockTarget()
      const controller = new AbortController()
      controller.abort('User cancelled early')

      await rejects(
        prefill(target, 'test', { signal: controller.signal }),
        (err) => {
          strictEqual(err.name, 'AbortError')
          strictEqual(err.message, 'The operation was aborted')
          return true
        }
      )
    })
    it.skip('should call target.abort() and throw when signal is aborted during execution', async () => {
      let abortCalled = false
      const controller = new AbortController()

      const target = createMockTarget({
        abort: () => { abortCalled = true },
        generate: async function * () {
          await new Promise(resolve => setTimeout(resolve, 10))
          controller.abort('Timeout')
          yield [1]
          return { stats: true }
        }
      })

      await rejects(
        prefill(target, 'test', { signal: controller.signal }),
        (err) => {
          strictEqual(err.name, 'AbortError')
          strictEqual(err.message, 'Timeout')
          strictEqual(abortCalled, true)
          return true
        }
      )
    })
    it.skip('should clean up the abort listener after completion', async () => {
      const controller = new AbortController()
      let abortCalledAfterReturning = false

      const target = createMockTarget({
        generate: async function * () { return {} },
        abort: () => { abortCalledAfterReturning = true }
      })

      await prefill(target, 'test', { signal: controller.signal })

      controller.abort() // Triggered after completion
      strictEqual(abortCalledAfterReturning, false)
    })
    it.skip('should translate native "Generation Cancelled" errors into AbortErrors', async () => {
      const target = createMockTarget({
        generate: async function * () {
          throw new Error('Native Generation Cancelled')
        }
      })

      await rejects(prefill(target, 'test'), { name: 'AbortError' })
    })
    it.skip('should pass template options to the encoder', async () => {
      let capturedConfig = null
      const target = createMockTarget({
        encode: (prompt, config) => {
          capturedConfig = config
          return { ids: [1] }
        },
        generate: async function * () { return {} }
      })

      await prefill(target, { messages: [] }, { template: { some: 'config' } })

      ok(capturedConfig.template)
      strictEqual(capturedConfig.template.some, 'config')
    })
  })

  describe('Integration (End-to-End)', { skip: false }, () => {
    // Prefill -> Generate
    it.todo('should significantly reduce "time-to-first-token" in generate() if prefill() was called first')

    // Cache Forking + Prefill
    it.todo('should allow prefilling a parent cache, forking it, and then generating two different continuations')

    // Batch + Plugins
    it.todo('should correctly extract reasoning/tool calls from a batch where only some sequences triggered them')

    // Global Memory Cleanup
    it.todo('should ensure all native task pointers are freed even if an error occurs inside a plugin')
  })
})
