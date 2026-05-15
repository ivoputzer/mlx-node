import { describe, it } from 'node:test'
import { deepStrictEqual, strictEqual } from 'node:assert/strict'
import { parseJson } from 'mlx-node'

describe('.parseJson', () => {
  describe('Valid JSON', () => {
    it('should parse a valid JSON object', () => {
      const input = '{"name":"MLX","version":1}'
      const expected = { name: 'MLX', version: 1 }
      deepStrictEqual(parseJson(input), expected)
    })

    it('should parse a valid JSON array', () => {
      const input = '[1, 2, 3]'
      const expected = [1, 2, 3]
      deepStrictEqual(parseJson(input), expected)
    })

    it('should parse a valid JSON string literal', () => {
      strictEqual(parseJson('"hello"'), 'hello')
    })
  })

  describe('Invalid JSON & Error Handling', () => {
    it('should return null (default fallback) for malformed JSON', () => {
      const input = '{ invalid: json }'
      strictEqual(parseJson(input), null)
    })

    it('should return a custom fallback for malformed JSON', () => {
      const input = '!!!'
      const fallback = { error: true }
      deepStrictEqual(parseJson(input, fallback), fallback)
    })
  })

  describe('Falsy & Empty Inputs', () => {
    it('should return fallback when input is an empty string', () => {
      strictEqual(parseJson('', 'fallback'), 'fallback')
    })

    it('should return fallback when input is null', () => {
      strictEqual(parseJson(null, 'fallback'), 'fallback')
    })

    it('should return fallback when input is undefined', () => {
      strictEqual(parseJson(undefined, 'fallback'), 'fallback')
    })
  })

  describe('Edge Cases', () => {
    it('should return false if the fallback is specifically false', () => {
      strictEqual(parseJson(null, false), false)
    })

    it('should handle numeric JSON correctly (valid but unusual)', () => {
      strictEqual(parseJson('123'), 123)
    })
  })
})
