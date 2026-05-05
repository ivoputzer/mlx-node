# Cache Management (Save, Load, Trim, Clone)

### What is it?
KV Caches are literal files of floating-point numbers. Cache management gives you complete control over the model's memory state:
*   **Save/Load:** Serialize the KV state to disk (`.safetensors` format).
*   **Clone:** Duplicate a cache in RAM.
*   **Trim:** Delete the last N tokens from the model's memory (Time Travel).

### Why is this powerful?
1. **Cold Starts:** If you have a complex System Prompt with RAG data (e.g., a massive codebase), you evaluate it once, `save()` to disk, and for every future user session, you instantly `load()` it—zero prefill latency!
2. **Undo/Regenerate:** If the model outputs a bad answer, you don't need to re-evaluate the history. You `trim()` the cache back to the end of the user's prompt, and `generate()` again with a higher temperature.

### 🛠️ Driver Guide (`bridge/swift`)
*   **Disk I/O:** Uses `MLXLMCommon.savePromptCache` and `loadPromptCache`.
*   **Trim:** Modifies the internal array shapes by dropping elements from the sequence axis (`axis: 1` usually, depending on cache type), and safely decrements the `paddingCounts` metadata to keep the state tracker accurate.

### 📦 User Guide (`packages/lm`)
*   **`MLXCache.fromPath(path, model)` / `cache.save(path)`**
*   **`cache.trim(numTokens)`**: Returns the actual number of tokens trimmed (useful if you ask to trim 100 but only 50 exist).
*   **`cache.clone()`**: Creates an exact duplicate reference in VRAM.

### 🚀 CLI Example: Save, Load, and Time Travel
```javascript
import { MLXCache, MLXModel } from 'mlx-swift'
import { loadTokenizer, stopTokenIdsFrom } from 'mlx-lm'
import { createInterface } from 'node:readline'
import { stdin, stdout } from 'node:process'
import { styleText } from 'node:util'

const readline = createInterface({ input: stdin, output: stdout })
// ... [Init model, tokenizer as usual] ...

// 1. SAVE & LOAD
console.log(styleText('blue', `[System] Evaluating System Prompt and Saving to Disk...`))
const systemCache = MLXCache.fromModel(model)
await systemCache.evaluate(new Int32Array(tokenizer.encode("System: You are a helpful bot.\nUser: Tell me a joke.\nAssistant:").ids))
await systemCache.save('./system_prompt.safetensors')
systemCache.dispose()

console.log(styleText('green', `[System] Loading Cache from Disk instantly...`))
const activeCache = await MLXCache.fromPath('./system_prompt.safetensors', model)

// 2. GENERATE
let jokeTokens = []
const turn1 = activeCache.generate(new Int32Array(0), { batchSize: 1, stopTokenIds, temperature: 0.5 })

process.stdout.write("Joke 1: ")
for await (const batches of turn1) {
  if (batches[0] !== -1) {
    jokeTokens.push(batches[0])
    process.stdout.write(tokenizer.decode([batches[0]]))
  }
}

// 3. TRIM & REGENERATE (Time Travel)
await new Promise(res => readline.question(styleText('magenta', `\n\n[System] That joke was bad. Press Enter to time-travel and try again!`), res))

// We trim EXACTLY the number of tokens the model just generated!
activeCache.trim(jokeTokens.length)
console.log(styleText('red', `[System] Trimmed ${jokeTokens.length} tokens. Memory is back to the original prompt.`))

const turn2 = activeCache.generate(new Int32Array(0), { batchSize: 1, stopTokenIds, temperature: 1.0 }) // Higher temp for a different joke

process.stdout.write("Joke 2: ")
for await (const batches of turn2) {
  if (batches[0] !== -1) process.stdout.write(tokenizer.decode([batches[0]]))
}

activeCache.dispose()
model.dispose()
readline.close()
```
