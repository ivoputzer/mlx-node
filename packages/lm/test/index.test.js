import { describe, it, mock } from 'node:test'
import assert from 'node:assert/strict'
import { load, unload, generate, stream } from '../index.js'

describe('LLM Module', () => {
  describe('load()', () => {
    it('should successfully load a model by path', async () => {
      const mockLoadModel = mock.fn(async () => 42) // Returns modelId 42

      const result = await load('/models/mlx-llama', { loadModel: mockLoadModel })

      assert.strictEqual(result, 42)
      assert.strictEqual(mockLoadModel.mock.calls.length, 1)
      assert.deepStrictEqual(mockLoadModel.mock.calls[0].arguments, ['/models/mlx-llama'])
    })

    it('should bubble up errors if native loading fails', async () => {
      const mockLoadModel = mock.fn(async () => { throw new Error('File not found') })

      await assert.rejects(
        () => load('/bad/path', { loadModel: mockLoadModel }),
        { name: 'Error', message: 'File not found' }
      )
    })
  })

  describe('unload()', () => {
    it('should unload a model by ID', () => {
      const mockUnloadModel = mock.fn(() => true)

      const result = unload(42, { unloadModel: mockUnloadModel })

      assert.strictEqual(result, true)
      assert.strictEqual(mockUnloadModel.mock.calls.length, 1)
      assert.deepStrictEqual(mockUnloadModel.mock.calls[0].arguments, [42])
    })
  })

  describe('generate()', () => {
    it('should stringify config and generate text', async () => {
      const mockGenerate = mock.fn(async () => 'Generated output')
      const config = { temperature: 0.8, topP: 0.9 }

      const result = await generate(42, 'Hello World', config, { generate: mockGenerate })

      assert.strictEqual(result, 'Generated output')
      assert.strictEqual(mockGenerate.mock.calls.length, 1)

      // Asserts that the config object was successfully serialized to JSON
      assert.deepStrictEqual(mockGenerate.mock.calls[0].arguments, [
        42,
        'Hello World',
        '{"temperature":0.8,"topP":0.9}'
      ])
    })

    it('should handle empty config defaulting to {}', async () => {
      const mockGenerate = mock.fn(async () => 'Fallback')

      await generate(42, 'Hello World', undefined, { generate: mockGenerate })

      assert.strictEqual(mockGenerate.mock.calls[0].arguments[2], '{}')
    })
  })

  describe('stream()', () => {
    it('should stream data correctly when callbacks fire asynchronously (Slow Emitter)', async () => {
      const mockGenerateStream = mock.fn((id, prompt, conf, cb) => {
        // Simulate real-world async native thread firing callbacks over time
        setTimeout(() => cb(null, 'chunk 1 ', false), 5)
        setTimeout(() => cb(null, 'chunk 2', false), 10)
        setTimeout(() => cb(null, null, true), 15)
      })

      const chunks = []
      const iterator = stream(42, 'prompt', { chunkSize: 5 }, { generateStream: mockGenerateStream })

      for await (const chunk of iterator) {
        chunks.push(chunk)
      }

      assert.deepStrictEqual(chunks, ['chunk 1 ', 'chunk 2'])
      assert.strictEqual(mockGenerateStream.mock.calls[0].arguments[2], '{"chunkSize":5}')
    })

    it('should drain correctly when callbacks fire synchronously (Fast Emitter / Queue logic)', async () => {
      // If the native module fires callbacks faster than JS can await them,
      // the queue mechanism handles it. This tests the queue logic.
      const mockGenerateStream = mock.fn((id, prompt, conf, cb) => {
        cb(null, 'A', false)
        cb(null, 'B', false)
        cb(null, 'C', false)
        cb(null, null, true)
      })

      const chunks = []
      for await (const chunk of stream(42, 'fast', {}, { generateStream: mockGenerateStream })) {
        chunks.push(chunk)
      }

      assert.deepStrictEqual(chunks, ['A', 'B', 'C'])
    })

    it('should throw and halt if an error is emitted asynchronously', async () => {
      const mockGenerateStream = mock.fn((id, prompt, conf, cb) => {
        setTimeout(() => cb(null, 'chunk 1', false), 5)
        setTimeout(() => cb('Native inference failure', null, false), 10) // eslint-disable-line n/no-callback-literal
      })

      const iterator = stream(42, 'error test', {}, { generateStream: mockGenerateStream })

      // First chunk is fine
      const first = await iterator.next()
      assert.strictEqual(first.value, 'chunk 1')

      // Second chunk throws
      await assert.rejects(
        () => iterator.next(),
        { name: 'Error', message: 'Native inference failure' }
      )
    })

    it('should throw if an error is queued synchronously before consumption', async () => {
      const mockGenerateStream = mock.fn((id, prompt, conf, cb) => {
        cb(null, 'good chunk', false)
        cb('Immediate queue failure', null, false) // eslint-disable-line n/no-callback-literal
      })

      const iterator = stream(42, 'sync error test', {}, { generateStream: mockGenerateStream })

      const first = await iterator.next()
      assert.strictEqual(first.value, 'good chunk')

      await assert.rejects(
        () => iterator.next(),
        { name: 'Error', message: 'Immediate queue failure' }
      )
    })

    it('should gracefully handle empty configurations', async () => {
      const mockGenerateStream = mock.fn((id, prompt, conf, cb) => {
        cb(null, null, true) // Immediately finish
      })

      const iterator = stream(42, 'prompt', undefined, { generateStream: mockGenerateStream })

      const result = await iterator.next()

      assert.strictEqual(result.done, true)
      assert.strictEqual(mockGenerateStream.mock.calls[0].arguments[2], '{}')
    })
  })
})
