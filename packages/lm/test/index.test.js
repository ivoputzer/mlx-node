import { describe, it } from 'node:test'
import { strictEqual, deepEqual, rejects, ok } from 'node:assert/strict'
import { evaluate, generate, stream } from '../index.js'

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
    it('should return complete text, tokens, and stats on success', async () => {
      const target = createMockTarget()
      const result = await generate(target, 'test prompt')

      strictEqual(result.finish, 'stop')
      deepEqual(result.tokens, [10, 20])
      strictEqual(result.text, '[10][20]')
      strictEqual(result.stats.generatedTokens, 2)
    })

    it('should throw AbortError if cancelled during generation', async () => {
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

    it('should correctly pass tokenizer options down', async () => {
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
    it('should yield chunks and a final done object via incremental decoding', async () => {
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

    it('should handle unicode \\uFFFD boundaries by buffering', async () => {
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

    it('should swallow native Cancelled errors and yield a graceful abort event', async () => {
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

  describe('.evaluate', () => {
    it('should return stats when evaluation completes successfully', async () => {
      const target = createMockTarget({
        generate: async function * () {
          yield [1]
          return { promptTokens: 3, generatedTokens: 0 }
        }
      })
      const result = await evaluate(target, 'test prompt')

      strictEqual(result.promptTokens, 3)
      strictEqual(result.generatedTokens, 0)
    })

    it('should throw an error if the prompt is empty', async () => {
      const target = createMockTarget()
      await rejects(evaluate(target, ''), { message: 'Prompt cannot be empty' })
    })

    it('should throw an error if the target is not ready', async () => {
      const target = createMockTarget({ available: false })
      await rejects(evaluate(target, 'test'), { message: 'Target not loaded or disposed already' })
    })

    it('should throw AbortError if signal is already aborted', async () => {
      const target = createMockTarget()
      const controller = new AbortController()
      controller.abort('User cancelled early')

      await rejects(
        evaluate(target, 'test', { signal: controller.signal }),
        (err) => {
          strictEqual(err.name, 'AbortError')
          strictEqual(err.message, 'The operation was aborted')
          return true
        }
      )
    })

    it('should call target.abort() and throw when signal is aborted during execution', async () => {
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
        evaluate(target, 'test', { signal: controller.signal }),
        (err) => {
          strictEqual(err.name, 'AbortError')
          strictEqual(err.message, 'Timeout')
          strictEqual(abortCalled, true)
          return true
        }
      )
    })

    it('should clean up the abort listener after completion', async () => {
      const controller = new AbortController()
      let abortCalledAfterReturning = false

      const target = createMockTarget({
        generate: async function * () { return {} },
        abort: () => { abortCalledAfterReturning = true }
      })

      await evaluate(target, 'test', { signal: controller.signal })

      controller.abort() // Triggered after completion
      strictEqual(abortCalledAfterReturning, false)
    })

    it('should translate native "Generation Cancelled" errors into AbortErrors', async () => {
      const target = createMockTarget({
        generate: async function * () {
          throw new Error('Native Generation Cancelled')
        }
      })

      await rejects(evaluate(target, 'test'), { name: 'AbortError' })
    })

    it('should pass template options to the encoder', async () => {
      let capturedConfig = null
      const target = createMockTarget({
        encode: (prompt, config) => {
          capturedConfig = config
          return { ids: [1] }
        },
        generate: async function * () { return {} }
      })

      await evaluate(target, { messages: [] }, { template: { some: 'config' } })

      ok(capturedConfig.template)
      strictEqual(capturedConfig.template.some, 'config')
    })
  })
})
