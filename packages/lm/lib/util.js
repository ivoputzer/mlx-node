import { Readable } from 'node:stream'
import { ReadableStream } from 'node:stream/web'

export function toStream (asyncIterable) {
  return Readable.from(asyncIterable)
    .filter(({ text }) => text)
    .map(({ text }) => text)
}

export function toWebStream (asyncIterable) {
  return new ReadableStream({
    async start (controller) {
      const encoder = new TextEncoder()
      try {
        for await (const { text } of asyncIterable) {
          if (text) {
            controller.enqueue(encoder.encode(text))
          }
        }
        controller.close()
      } catch (error) {
        controller.error(error)
      }
    }
  })
}
