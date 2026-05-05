# The "Open Mouth" Paradigm: Assistant Prefilling & Generation Continuation

### What is it?
In standard inference engines, when a model finishes generating a response, the engine permanently writes a "Stop Token" (EOS/EOT) into the KV Cache. This tells the model, "Your thought process is completely finished."

In the **Open Mouth Paradigm**, the engine deliberately **omits the Stop Token from the KV Cache**. When generation stops—whether because it hit `maxTokens`, was manually aborted, or naturally finished—the model is left "hanging" mid-sentence.

### Why is this powerful?
Because the model's mouth is still open, you have **God-Mode control** over what it says next.
1. **Output Forcing (JSON/Code):** If the model stops, you can inject ``\n```json\n{`` into the cache. The model is now mathematically forced to output valid JSON, skipping the usual conversational yapping.
2. **Mid-Flight Correction:** If the model starts going down the wrong path, you can interrupt it, inject `" Wait, I meant to say:"`, and let it resume.
3. **Seamless Resumption:** If generation hits a length limit, you can simply press "continue" without re-evaluating the history.

### How is this different from `MLXTarget.evaluate`?
*   **`evaluate`** is a *silent* operation. It ingests tokens into the KV Cache to update the state, but it **does not sample or yield any new tokens**. It is used purely for prefilling (e.g., "Read this 10,000 word document into memory so I can ask questions about it later").
*   **Generation Continuation** (via `generate`) *starts* by silently ingesting your injected tokens (acting exactly like `evaluate`), but then immediately transitions into the generation loop to produce new text based on what you just forced into its mouth.

---

## 🛠️ Driver Guide (`bridge/swift`)
**For Backend Maintainers:**
The Swift engine is entirely agnostic to conversational turns. It is a mathematically pure, append-only tensor processor.
*   **No Stop Tokens in Cache:** When the model emits an `EOS|EOT` token, Swift catches it, stops generation, and replaces the token with a `PAD` token before the final evaluation. The `EOS|EOT` token *never* enters the `KVCache`.
*   **Automatic Trimming:** If sequences are batched, Swift tracks how many padding tokens were used to keep the tensors rectangular. When the user extracts a sequence via `bridge_cache_slice`, Swift automatically trims the trailing pad tokens if KVCache allows it. If multiple sequences are being returned via `bridge_cache_slice` the greatest common denominator is searched before trimming.
*   **Result:** The resulting cache is always left perfectly on the very last real word generated.

## 📦 User Guide (`packages/lm`)
**For Frontend / JS Developers:**
Because the backend leaves the cache "Open Mouthed," the JS wrapper acts as the state machine. You must handle the flow logically:

**1. Continuing a Generation (Assistant Prefilling)**
To force the model to say something, or to resume a cut-off sequence, just pass the raw string directly into the cache.
```javascript
// The model got cut off. Force it to write python!
const forcedTokens = tokenizer.encode("\n```python\nimport")
const stream = cache.generate(forcedTokens.ids)
```

**2. Starting a New Turn (Closing the Mouth)**
Because the model is still "open," if you just send a new user prompt, the model will hallucinate. You must **manually close the mouth** from the previous turn by prepending the stop token.
```javascript
const newTurn = "<|im_end|>\n<|im_start|>user\nWhat is step 2?<|im_end|>\n<|im_start|>assistant\n"
const stream = cache.generate(tokenizer.encode(newTurn).ids)
```

---

## 🚀 Interactive CLI Example

Here is a script you can run. It demonstrates the Open Mouth paradigm by starting a generation, cutting it off early, prompting you to inject text into the model's mouth (like "However,"), and seamlessly continuing the generation.

[Assistant Prefilling](demo/assistant-prefilling.demo.js)

```javascript
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

console.log(styleText('cyan', `\n[System] Loading model and creating Cache...`))
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

console.log(styleText('blue', `\n[System] Starting generation... (Stopping early at 25 tokens)\n`))

// We intentionally set maxTokens very low to cut the model off mid-sentence
const turn1 = cache.generate(promptTokens, {
  batchSize: 1,
  stopTokenIds,
  temperature: Number(values.temperature),
  maxTokens: 25
})

let history = ""

for await (const batches of turn1) {
  const token = batches[0]
  if (token !== -1) {
    const text = tokenizer.decode([token], { skip_special_tokens: false })
    history += text
    process.stdout.write(styleText('white', text))
  }
}

console.log(styleText('red', `\n\n[System] Model hit maxTokens and stopped! (Stop Reason: length)`))
console.log(styleText('gray', `The KV Cache does NOT contain an EOS token. The model's mouth is OPEN.`))

// ============================================================================
// PART 2: ASSISTANT INJECTION / CONTINUATION
// ============================================================================

const injection = await new Promise(resolve => {
  readline.question(styleText('magenta', `\n[System] Inject text into the model's mouth (e.g. " Suddenly, "): `), resolve)
})

// We format the injection cleanly. Notice WE DO NOT ADD AN EOS TOKEN.
const safeInjection = injection ? ` ${injection.trim()} ` : " "
history += safeInjection

console.log(styleText('blue', `\n[System] Forcing injection into cache and resuming generation...\n`))

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
```
