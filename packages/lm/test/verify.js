import { load } from 'mlx-lm'

const controller = new AbortController()

console.time('model loaded')
const model = await load('/Users/ivoputzer/github/models/gemma-4-e2b')
console.timeEnd('model loaded')

const messages = [{
  role: 'system',
  content: 'You are a helpful assistant'
}, {
  role: 'user',
  content: 'Tell me something about italy'
}]

// const prompt = new Template(chatTemplate).render({
//   messages
//   /*
//   bos_token: configJson.bos_token,
//   eos_token: configJson.eos_token
//   add_generation_prompt: true
//   tools
//   */
// })
// const { ids, tokens, attention_mask } = tokenizer.encode(prompt)

// console.log({ ids, tokens, attention_mask })

// new Int32Array(ids)

setTimeout(() => {
  console.log('---- aborting')
  controller.abort()
}, 10000)

for await (const { text, done, ...rest } of model.stream(messages, { signal: controller.signal, streamChunkSize: 1 })) {
  if (done) {
    console.log(rest)
  } else {
    process.stdout.write(text)
  }
}

model.unload()
