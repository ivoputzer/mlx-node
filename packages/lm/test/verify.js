import { load, generate, stream } from 'mlx-lm'

console.time('model loaded')
const model = await load('/Users/ivoputzer/github/models/Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit')
console.timeEnd('model loaded')

const controller = new AbortController()
const messages = [{ role: 'system', content: 'You are a helpful assistant' }, { role: 'user', content: 'Tell me an interesting fact about Node.js' }]

console.log('------------------- GENERATE -------------------')

console.time('generate time')
// Using the new functional API
const response = await generate(model, messages, { signal: controller.signal, streamChunkSize: 1 })
console.timeEnd('generate time')

console.dir(response)

// console.log('-------------------- STREAM --------------------')

// // setTimeout(() => {
// //   console.log('---- aborting (10s)')
// //   controller.abort()
// // }, 5000)

// console.time('stream time')
// // Using the new functional API
// for await (const { text, done, ...rest } of stream(model, messages, { signal: controller.signal, streamChunkSize: 5 })) {
//   if (done) {
//     console.log(rest)
//   } else {
//     process.stdout.write(text)
//   }
// }
// console.timeEnd('stream time')

// model.unload()
// console.log('---- unload')

process.stdin.resume()
