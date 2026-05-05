# 2. A/B Testing & Tree Search (Slicing)

### What is it?
After performing a Batch Generation, you have a cache with dimension B=N. **Slicing** allows you to extract a subset of those dimensions (e.g., extracting Branch 2 to create a pristine B=1 cache), allowing you to discard the failed branches and continue generating on the successful one.

### Why is this powerful?
This enables **Monte Carlo Tree Search (MCTS)** and **Best-of-N Sampling**—the exact mechanisms used to train and run reasoning models like DeepSeek-R1. It is also perfect for interactive storytelling apps where users pick the path they like best.

### 🛠️ Driver Guide (`bridge/swift`)
*   **The Slice Function (`bridge_cache_slice`):** Swift iterates through every tensor in the `KVCache` array and performs an `array.take(indices, axis: 0)` to extract the requested batch subset.
*   **Auto-Trimming:** Swift checks the `paddingCounts` of the newly sliced subset, finds the smallest padding amount, and automatically calls `MLXLMCommon.trimPromptCache`. This ensures the extracted cache is returned to a pristine, hallucination-free "Open Mouth" state.

### 📦 User Guide (`packages/lm`)
*   **`cache.slice(start, end)`:** Returns a *new* `MLXCache` instance containing only the selected branches. **Important:** You must manually `dispose()` the original heavy cache to free up VRAM.

### 🚀 CLI Example: Branching, Slicing, and Continuing
*(Note: A highly polished version of your A/B testing script)*
```javascript
import { MLXCache, MLXModel } from 'mlx-swift'
import { loadTokenizer, loadTemplate, stopTokenIdsFrom } from 'mlx-lm'
import { createInterface } from 'node:readline'
import { stdin, stdout } from 'node:process'
import { styleText } from 'node:util'

const readline = createInterface({ input: stdin, output: stdout })
// ... [Init model, tokenizer, template as usual] ...

console.log(styleText('cyan', `[System] Generating 3 story ideas...`))
const prompt = tokenizer.encode(template.render({ messages: [{role: 'user', content: 'Give me a 1-sentence story idea.'}], add_generation_prompt: true}))

const cache = MLXCache.fromModel(model)
const turn1 = cache.generate(new Int32Array(prompt.ids), { batchSize: 3, temperature: 0.9, stopTokenIds })

const branches = ["", "", ""]
for await (const batches of turn1) {
  batches.forEach((token, i) => { if (token !== -1) branches[i] += tokenizer.decode([token]) })
}

branches.forEach((b, i) => console.log(`[${i + 1}] ${b.trim()}`))

const choiceStr = await new Promise(res => readline.question(styleText('magenta', `\nWhich branch do you want to continue? (1/2/3): `), res))
const choice = parseInt(choiceStr) - 1

// SLICE THE CACHE!
console.log(styleText('blue', `[System] Slicing branch ${choice + 1}...`))
const winningCache = cache.slice(choice, choice + 1)

// Free the heavy B=3 cache immediately
cache.dispose()

// CONTINUE THE TURN (Closing the mouth, opening Turn 2)
const stopToken = tokenizer.config.eos_token
const turn2Prompt = tokenizer.encode(`${stopToken}\n<|im_start|>user\nExpand this into a paragraph.<|im_end|>\n<|im_start|>assistant\n`)

console.log(styleText('cyan', `[System] Continuing story...`))
const turn2 = winningCache.generate(new Int32Array(turn2Prompt.ids), { batchSize: 1, stopTokenIds })

for await (const batches of turn2) {
  if (batches[0] !== -1) process.stdout.write(tokenizer.decode([batches[0]]))
}

winningCache.dispose()
model.dispose()
readline.close()
```

---
