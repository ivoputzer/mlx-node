import { Readable } from 'node:stream'

export function toStream (asyncIterable) {
  return Readable.from(asyncIterable)
    .filter(({ text }) => text)
    .map(({ text }) => text)
}

export function toWebStream (asyncIterable) {
  return Readable.toWeb(toStream(asyncIterable))
  return new ReadableStream({
    async start (controller) {
      for await (const { text } of asyncIterable) {
        if (text) controller.enqueue(new TextEncoder().encode(text))
      }
      controller.close()
    }
  })
}
