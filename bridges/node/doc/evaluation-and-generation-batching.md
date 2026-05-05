# Batch Evaluation & Batch Generation

### What is it?
*   **Batch Evaluation (`evaluate`):** The process of purely ingesting tokens into the KV Cache to update its internal state *without* sampling new tokens. Also known as "Prefilling".
*   **Batch Generation (`generate` with `batchSize > 1`):** Sampling multiple independent streams of text simultaneously from the same shared cache state.

### Why is this powerful?
By separating `evaluate` and `generate`, you unlock massive performance gains. If you want 3 different summaries of a 10,000-word document, you don't evaluate the document 3 times. You `evaluate` it once at `batchSize = 1`, duplicate the cache in memory, and then `generate` at `batchSize = 3`.

### 🛠️ Driver Guide (`bridge/swift`)
*   **Evaluate Task:** Swift evaluates the `LMInput` tensor using `model.prepare()`. If `batchSize > 1` is passed to the config, Swift will intelligently process the prefill at B=1 (fastest), and *then* use `MLX.concatenated` to broadcast the KV Cache dimensions to B=N in VRAM instantly.
*   **Generate Task (Homogeneous):** Swift accepts an N-dimensional tensor, samples B=N logits concurrently via Metal, and safely tracks `paddingCounts` as sequences finish at different times.

### 📦 User Guide (`packages/lm`)
*   **`evaluate(tokens)`:** Returns a promise that resolves when the cache has absorbed the tokens.
*   **`generate(tokens, { batchSize: N })`:** Yields an array of length N containing the next tokens for each branch.

### 🚀 CLI Example: Evaluate Context & Batch Generate
```javascript
import { MLXCache, MLXModel } from 'mlx-swift'
import { loadTokenizer, stopTokenIdsFrom } from 'mlx-lm'
import { styleText } from 'node:util'

const modelPath = '/path/to/model'
const tokenizer = await loadTokenizer(modelPath)
const stopTokenIds = stopTokenIdsFrom(tokenizer)

const model = await MLXModel.fromPath(modelPath)
const cache = MLXCache.fromModel(model)

// 1. EVALUATE: Read a massive document into memory ONCE
const document = "System: You are an expert analyst.\nUser: Here is a long document... [Pretend this is 5,000 words]... What are the key takeaways?\nAssistant:"
const evalTokens = new Int32Array(tokenizer.encode(document).ids)

console.log(styleText('blue', `[System] Evaluating document (Prefilling cache)...`))
const evalStats = await cache.evaluate(evalTokens, { batchSize: 3 })
console.log(styleText('green', `[System] Prefill complete! ${evalStats.promptTokensPerSecond.toFixed(0)} t/s`))

// 2. BATCH GENERATE: Branch into 3 simultaneous streams
console.log(styleText('cyan', `[System] Generating 3 different analyses concurrently...\n`))
const stream = cache.generate(new Int32Array(0), {
  batchSize: 3,
  temperature: 0.9,
  stopTokenIds
})

const buffers = ["", "", ""]
for await (const batches of stream) {
  batches.forEach((token, i) => {
    if (token !== -1) {
      buffers[i] += tokenizer.decode([token], { skip_special_tokens: false })
    }
  })

  // Clear console and print current state
  console.clear()
  buffers.forEach((text, i) => console.log(styleText('yellow', `Branch ${i + 1}:\n${text}\n`)))
}

cache.dispose()
model.dispose()
```
