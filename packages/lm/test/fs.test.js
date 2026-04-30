import { describe, it } from 'node:test'
import { strictEqual, deepStrictEqual } from 'node:assert/strict'
import path from 'node:path'
import { expand } from '../lib/fs.js'

describe('resolve function suite', async (t) => {
  it('handles tilde (~) by calling os.homedir and join', ({ mock }) => {
    const homedir = mock.fn(() => '/Users/test')
    const join = mock.fn((...args) => path.join(...args))
    const resolve = mock.fn()

    const result = expand('~/documents/file.txt', { homedir }, { join, resolve })

    strictEqual(result, '/Users/test/documents/file.txt')
    strictEqual(join.mock.callCount(), 1)
    strictEqual(resolve.mock.callCount(), 0)
    deepStrictEqual(join.mock.calls[0].arguments, ['/Users/test', '/documents/file.txt'])
  })

  it('handles standard paths by calling resolve', ({ mock }) => {
    const homedir = mock.fn(() => '/Users/test')
    const resolve = mock.fn((...args) => path.join('/Users/test', ...args))
    const join = mock.fn()

    const result = expand('./src/index.js', { homedir }, { join, resolve })

    strictEqual(result, '/Users/test/src/index.js')
    strictEqual(resolve.mock.callCount(), 1)
    strictEqual(join.mock.callCount(), 0)
    strictEqual(resolve.mock.calls[0].arguments[0], './src/index.js')
  })
})
