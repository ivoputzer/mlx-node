import { MLXMetrics, MLXModel } from 'mlx-swift'
import { loadTokenizer, loadTemplate, stopTokensFrom, padTokenFrom } from 'mlx-lm'

import { styleText, parseArgs } from 'node:util'
import { stdin, stdout } from 'node:process'

const { values } = parseArgs({
  options: {
    model: { short: 'm', type: 'string', default: '~/github/models/Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit' },
    prompt: { short: 'p', type: 'string', default: 'Write a detailed story about a space dog. Make it exciting.' },
    temperature: { short: 't', type: 'string', default: '0.8' }
  }
})

const tokenizer = await loadTokenizer(values.model)
const template = await loadTemplate(values.model)

const stopTokens = stopTokensFrom(tokenizer)
const padTokens = padTokenFrom(tokenizer)

const promptString = template.render({
  messages: [
    { role: 'system', content: 'You are a creative AI storyteller.' },
    { role: 'user', content: values.prompt }
  ],
  add_generation_prompt: true
})

const promptTokens = new Int32Array(tokenizer.encode(promptString).ids)


stdout.write(styleText('cyan', '\n[System] Loading model...\n'))
using model = await MLXModel.fromPath(values.model)


stdout.write(styleText('blue', '\n[System] Starting generation... \n'))
const task = model.generate(promptTokens, { topLogits:5, maxTokens:10, chunkSize: 1, batchSize: 1, padTokenId: padTokens, stopTokenIds: stopTokens})

for await (const [firstBatchTokenId] of task) {
  const text = tokenizer.decode([firstBatchTokenId], { skip_special_tokens: false, clean_up_tokenization_spaces: false })
  stdout.write(text)
}
