import { describe, it } from 'node:test'
import { ok, equal } from 'node:assert/strict'
import { Readable } from 'node:stream'
import { text } from 'node:stream/consumers'
import { toWebStream, toStream } from '../lib/util.js' // Adjust path as needed

describe('.util', () => {
  it('.toWebStream correctly converts and filters an async iterable', async () => {
    const webStream = toWebStream(anAsyncIterator())

    ok(webStream instanceof ReadableStream)
    equal((await text(webStream)), 'Hello Node 25!')
  })

  it('.toStream correctly converts and filters an async iterable', async () => {
    const nodeStream = toStream(anAsyncIterator())

    ok(nodeStream instanceof Readable)
    equal(await text(nodeStream), 'Hello Node 25!')
  })

  async function * anAsyncIterator () {
    yield { text: 'Hello' }
    yield { ignored: 'This should be filtered out' }
    yield { text: ' ' }
    yield { text: 'Node 25!' }
    yield { text: '' }
    return { stats: {}, text: 'Hello Node 25!' }
  }
})
