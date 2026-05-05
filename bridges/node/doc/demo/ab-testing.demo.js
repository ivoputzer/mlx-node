import { MLXCache, MLXMetrics, MLXModel } from 'mlx-swift'
import { loadTokenizer, loadTemplate, stopTokenIdsFrom } from 'mlx-lm'
import { styleText, parseArgs, stripVTControlCharacters } from 'node:util'
import { createInterface } from 'node:readline'
import { stdin, stdout } from 'node:process'

// Import our utilities (assuming they are in the same directory)
import { createBatchRenderer, printBatchHeaders } from '../../../../packages/cli/lib/grid-formatter.js'
import { inlineDiff } from '../../../../packages/cli/lib/string-diff.js'

const { values } = parseArgs({
  options: {
    model: { short: 'm', type: 'string', default: '/Users/ivoputzer/github/models/Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit' },
    prompt: { short: 'p', type: 'string', default: 'Give me a cool name for a space dog.' },
    size: { short: 's', type: 'string', default: '2' },
    temperature: { short: 't', type: 'string', default: '0.8' },
    maxTokens: { type: 'string', default: '512' }
  }
})

const tokenizer = await loadTokenizer(values.model)
const template = await loadTemplate(values.model)
const stopTokenIds = stopTokenIdsFrom(tokenizer)
const readline = createInterface({ input: stdin, output: stdout })

// ============================================================================
// TURN 1: PREFILL AND A/B TEST (Homogeneous Batching)
// ============================================================================

const model = await MLXModel.fromPath(values.model)
const batchSize = Number(values.size)
const batchBuffer = Array(batchSize).fill('')

console.log(styleText('cyan', `\n[System] Generating ${batchSize} variations simultaneously...\n`))

printBatchHeaders(batchSize, {
  titles: Array.from({ length: batchSize }, (_, i) => `Branch ${1 + i}`)
})

const sharedPrompt = template.render({
  messages: [
    { role: 'system', content: 'You are a helpful, very brief AI.' },
    { role: 'user', content: values.prompt }
  ],
  add_generation_prompt: true
})

const render = createBatchRenderer(batchSize, { readline, lineFormatter })
const cache = MLXCache.fromModel(model)

const sharedPromptTokens = new Int32Array(tokenizer.encode(sharedPrompt).ids)
const turn1 = cache.generate(sharedPromptTokens, {
  batchSize,
  stopTokenIds,
  temperature: Number(values.temperature),
  maxTokens: Number(values.maxTokens)
})

for await (const batches of turn1) {
  batches.forEach((tokenId, i) => {
    if (tokenId !== -1) {
      batchBuffer[i] += tokenizer.decode([tokenId], { skip_special_tokens: false, clean_up_tokenization_spaces: false })
    }
  })
  render(batchBuffer)
}
render(batchBuffer, true)

// ============================================================================
// BRANCH SELECTION (Interactive)
// ============================================================================

console.log(styleText('blue', `\n\n[System] Turn 1 Complete. We have a B=${1 + batchSize} Cache in memory.`))
console.log('\n', MLXMetrics.fromSnapshot().toString())
console.log('\n', cache.debug())

const choice = await new Promise(resolve => {
  const options = Array.from({ length: batchSize }, (_, i) => i + 1).join('/')
  readline.question(styleText('magenta', `\n[System] Which branch was better? (${options}): `), resolve)
})

console.log(styleText('blue', `\n[System] Slicing cache to keep ${choice} [cache.slice(${choice - 1}, ${choice})]. Disposing the rest.`))

// Slice cache based on user choice
const winningCache = cache.slice(Number(choice - 1), Number(choice)) // Returns a B={choice} cache!
cache.dispose() // Free the heavy B=N cache

// ============================================================================
// TURN 2: CONTINUING THE CONVERSATION (Single Branch)
// ============================================================================

/*

Without caches we had to append the model's actual answer to the message history, then ask our next question

const turn2Prompt = template.render({
  messages: [
    { role: 'system', content: 'You are a helpful, very brief AI.' },
    { role: 'user', content: 'Give me a cool name for a space dog.' },
    { role: 'assistant', content: batchBuffer[choice - 1] },
    { role: 'user', content: 'Why is that name good?' }
  ],
  add_generation_prompt: true
})

But since the KVCache already contains the history up to `branchTexts[0]`,
we ONLY need to pass the *new* tokens to the model!

Assuming your JS LM layer or Tokenizer can diff the prompt, but for this raw test,
let's encode the whole thing and rely on the bridge to evaluate it.

Note: If you pass the whole prompt into an already-populated cache, your Swift code
currently might evaluate it from scratch if offset logic isn't perfectly mapped.
But assuming standard MLX cache offset logic applies, you just pass the new tokens).

*/

const turn2Prompt = template.render({
  messages: [
    { role: 'system', content: 'You are a helpful, very brief AI.' },
    { role: 'user', content: 'Give me a cool name for a space dog.' },
    { role: 'assistant', content: batchBuffer[choice - 1] },
    { role: 'user', content: 'Why is that name good?' }
  ],
  add_generation_prompt: true
})

const userPromptOnly = template.render({
  messages: [{ role: 'user', content: 'Why is that name good?' }],
  add_generation_prompt: true
})

/*
  When in multi-turn conversation, kvcache is left without eos|eot token on purpose,
  for assistant prefilling, thus we need to add the token manually before initiating
  next turn if chat_template does not do it already.

  Other than that kvcache will have thinking tokens still, not all models like that though...
*/

console.log(styleText('blue', '\n[System] What the KVCache contains compared to a full history re-render.'))

const stopToken = tokenizer.config?.eos_token ?? tokenizer.config?.eot_token
console.log(inlineDiff(turn2Prompt, `${sharedPrompt}${batchBuffer[choice - 1]}${stopToken}\n${userPromptOnly}`))

const tokens2 = new Int32Array(tokenizer.encode(`${stopToken}${userPromptOnly}`).ids)

console.log(styleText('cyan', `\n[System] Continuing conversation from Branch ${choice} history...\n`))

// // New renderer for a single column
const renderWinner = createBatchRenderer(1, { readline, cellFormatter: (t) => styleText('green', t) })

const finalResponse = ['']
const turn2 = winningCache.generate(tokens2, { batchSize: 1, stopTokenIds, maxTokens: Number(values.maxTokens) })

for await (const batches of turn2) {
  const token = batches[0]
  if (token !== -1) {
    finalResponse[0] += tokenizer.decode([token], { skip_special_tokens: false })
  }
  renderWinner(finalResponse)
}
renderWinner(finalResponse, true)

console.log(styleText('blue', '\n[System] Session Complete! Metrics:'))
console.log(MLXMetrics.fromSnapshot().toString())

winningCache.dispose()
model.dispose()
readline.close()

// ============================================================================
// UTILITIES
// ============================================================================

function lineFormatter (lineText, lineIdx, colIndex, allBuffers, currentView) {
  if (allBuffers.length < 2) return lineText

  const thisBuffer = allBuffers[colIndex]
  const otherBuffer = allBuffers[colIndex === 0 ? 1 : 0]

  // 1. Find the global divergence point across the ENTIRE buffer
  let globalSplitIndex = 0
  while (
    globalSplitIndex < thisBuffer.length &&
    globalSplitIndex < otherBuffer.length &&
    thisBuffer[globalSplitIndex] === otherBuffer[globalSplitIndex]
  ) {
    globalSplitIndex++
  }

  // 2. Map the global divergence point to this specific wrapped line
  // Calculate how many characters came before this line
  let charsBeforeThisLine = 0
  for (let i = 0; i < lineIdx; i++) {
    // Add 1 for the spaces/newlines that got consumed during wrap
    charsBeforeThisLine += currentView[i].length + 1
  }

  // If the divergence happened BEFORE this line started, the whole line is Normal
  if (charsBeforeThisLine >= globalSplitIndex) {
    return lineText
  }

  // If the divergence happens AFTER this line ends, the whole line is Yellow
  if (charsBeforeThisLine + lineText.length <= globalSplitIndex) {
    return styleText('green', lineText)
  }

  // If the divergence happens IN THE MIDDLE of this line, split it!
  const localSplitPoint = globalSplitIndex - charsBeforeThisLine
  const identicalPart = styleText('green', lineText.slice(0, localSplitPoint))
  const divergedPart = lineText.slice(localSplitPoint)

  return identicalPart + divergedPart
}
