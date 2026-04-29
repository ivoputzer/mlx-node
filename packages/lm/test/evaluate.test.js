import { describe, it } from 'node:test'
import { strictEqual, rejects, ok } from 'node:assert'
import { evaluate } from '../index.js'

describe('.evaluate', () => {
  // 1. Setup a standard mock target
  const createMockTarget = (overrides = {}) => ({
    ready: true,
    encode: () => new Int32Array([1, 2, 3]),
    generate: async function * () {
      yield [1] // simulate prefill chunks
      return { promptTokens: 3, generatedTokens: 0 }
    },
    abort: () => {},
    assertReady () { if (!this.ready) throw new Error('Target not ready') },
    ...overrides
  })

  it('should return stats when evaluation completes successfully', async () => {
    const target = createMockTarget()
    const result = await evaluate(target, 'test prompt')

    strictEqual(result.promptTokens, 3)
    strictEqual(result.generatedTokens, 0)
  })

  it('should throw an error if the prompt is empty', async () => {
    const target = createMockTarget()
    await rejects(
      evaluate(target, ''),
      { message: 'Prompt cannot be empty' }
    )
  })

  it('should throw an error if the target is not ready', async () => {
    const target = createMockTarget({ ready: false })
    await rejects(
      evaluate(target, 'test'),
      { message: 'Target not loaded or disposed already' }
    )
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
        // Simulate a long prefill that gives us time to abort
        await new Promise(resolve => setTimeout(resolve, 50))
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
    const target = createMockTarget()

    // We can't easily check private listeners, but we can verify
    // the code doesn't crash and the signal doesn't trigger abort
    // on the target AFTER the function returns.
    let abortCalledAfterReturning = false
    target.abort = () => { abortCalledAfterReturning = true }

    await evaluate(target, 'test', { signal: controller.signal })

    controller.abort()
    strictEqual(abortCalledAfterReturning, false, 'Abort should not be called after function finished')
  })

  it('should translate native "Generation Cancelled" errors into AbortErrors', async () => {
    const target = createMockTarget({
      generate: async function * () {
        throw new Error('Native Generation Cancelled')
      }
    })

    await rejects(
      evaluate(target, 'test'),
      { name: 'AbortError' }
    )
  })

  it('should pass template options to the encoder', async () => {
    let capturedConfig = null
    const target = createMockTarget({
      encode: (prompt, config) => {
        capturedConfig = config
        return new Int32Array([1])
      }
    })

    await evaluate(target, { messages: [] }, { template: { some: 'config' } })

    ok(capturedConfig.template)
    strictEqual(capturedConfig.template.some, 'config')
  })
})
