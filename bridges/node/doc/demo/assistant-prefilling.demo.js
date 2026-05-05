import { MLXCache, MLXMetrics, MLXModel } from 'mlx-swift'
import { loadTokenizer, loadTemplate, stopTokenIdsFrom } from 'mlx-lm'
import { styleText, parseArgs } from 'node:util'
import { createInterface } from 'node:readline'
import { stdin, stdout } from 'node:process'

const { values } = parseArgs({
  options: {
    model: { short: 'm', type: 'string', default: '/Users/ivoputzer/github/models/Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit' },
    prompt: { short: 'p', type: 'string', default: 'Write a detailed story about a space dog. Make it exciting.' },
    temperature: { short: 't', type: 'string', default: '0.8' }
  }
})

const tokenizer = await loadTokenizer(values.model)
const template = await loadTemplate(values.model)
const stopTokenIds = stopTokenIdsFrom(tokenizer)
const readline = createInterface({ input: stdin, output: stdout })

console.log(styleText('cyan', '\n[System] Loading model and creating Cache...'))
const model = await MLXModel.fromPath(values.model)
const cache = MLXCache.fromModel(model)

// ============================================================================
// PART 1: THE INITIAL GENERATION (INTENTIONALLY SHORT)
// ============================================================================

const promptString = template.render({
  messages: [
    { role: 'system', content: 'You are a creative AI storyteller.' },
    { role: 'user', content: values.prompt }
  ],
  add_generation_prompt: true
})

const promptTokens = new Int32Array(tokenizer.encode(promptString).ids)

console.log(styleText('blue', '\n[System] Starting generation... (Stopping early at 25 tokens)\n'))

// We intentionally set maxTokens very low to cut the model off mid-sentence
const turn1 = cache.generate(promptTokens, {
  batchSize: 1,
  stopTokenIds,
  temperature: Number(values.temperature),
  maxTokens: 25
})

let history = ''

for await (const batches of turn1) {
  const token = batches[0]
  if (token !== -1) {
    const text = tokenizer.decode([token], { skip_special_tokens: false })
    history += text
    process.stdout.write(styleText('white', text))
  }
}

console.log(styleText('red', '\n\n[System] Model hit maxTokens and stopped! (Stop Reason: length)'))
console.log(styleText('gray', 'The KV Cache does NOT contain an EOS token. The model\'s mouth is OPEN.'))

// ============================================================================
// PART 2: ASSISTANT INJECTION / CONTINUATION
// ============================================================================

const injection = await new Promise(resolve => {
  readline.question(styleText('magenta', '\n[System] Inject text into the model\'s mouth (e.g. " Suddenly, "): '), resolve)
})

// We format the injection cleanly. Notice WE DO NOT ADD AN EOS TOKEN.
const safeInjection = injection ? ` ${injection.trim()} ` : ' '
history += safeInjection

console.log(styleText('blue', '\n[System] Forcing injection into cache and resuming generation...\n'))

// We just print what we forced so it looks seamless in the console
process.stdout.write(styleText('green', safeInjection))

const injectionTokens = new Int32Array(tokenizer.encode(safeInjection).ids)

// We pass the injected tokens. The model will evaluate them, update the cache,
// and immediately continue generating from that exact state!
const turn2 = cache.generate(injectionTokens, {
  batchSize: 1,
  stopTokenIds,
  temperature: Number(values.temperature),
  maxTokens: 512 // Let it run freely now
})

for await (const batches of turn2) {
  const token = batches[0]
  if (token !== -1) {
    const text = tokenizer.decode([token], { skip_special_tokens: false })
    history += text
    process.stdout.write(styleText('yellow', text))
  }
}

console.log(styleText('blue', '\n\n[System] Generation Complete! Notice how it adapted to your injection.'))
console.log('\n' + MLXMetrics.fromSnapshot().toString())

cache.dispose()
model.dispose()
readline.close()
