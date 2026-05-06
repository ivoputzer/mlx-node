/*

Why this is the Ultimate CLI Tool:
1. The Heatmap Phase: It starts by smoothly printing the generation in real-time. If it hits an unexpected red word, you know immediately the model hallucinated or guessed wildly.
2. The "Scrubbing" TUI: Once generation finishes, it clears the screen and creates an interactive UI. A highlighted cursor wraps around the text.
3. The Token Inspector: As you press Left and Right, the dashboard at the bottom instantly updates. It decodes the top 5 alternative tokens the model considered at that exact moment.
3. BPE Visualization: Because BPE strings are tricky, the visualizeSpaces() function replaces invisible spaces with · and newlines with ↵. Now you can see exactly why the model chose "·dog" instead of "dog".

This tool completely changes how you understand LLMs. When you scrub over a word and see the top 2 probabilities were 49.9% and 50.1%, you suddenly realize how fragile and fascinating the generation process is.

*/

import { MLXModel } from 'mlx-swift'
import { loadTokenizer, loadTemplate, padTokenFrom, stopTokensFrom } from 'mlx-lm'
import { styleText } from 'node:util'
import readline from 'node:readline'

// --- CONFIGURATION ---
const MODEL_PATH = '/Users/ivoputzer/github/models/Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit'
const PROMPT = 'Explain quantum gravity using a weird metaphor.'
const TOP_K = 5

// --- UTILITIES ---
function colorizeByConfidence (text, probability) {
  if (probability >= 0.90) return styleText('green', text)
  if (probability >= 0.50) return styleText('yellow', text)
  if (probability >= 0.10) return styleText('red', text)
  return styleText(['bgRed', 'white'], text)
}

function drawBar (probability, width = 30) {
  const filled = Math.round(probability * width)
  const empty = width - filled
  return styleText('cyan', '█'.repeat(filled)) + styleText('gray', '░'.repeat(empty))
}

// Replaces invisible BPE spaces with a visible character so you can see token boundaries
function visualizeSpaces (text) {
  return text.replace(/ /g, '·').replace(/\n/g, '↵')
}

// --- MAIN SCRIPT ---
async function main () {
  console.log(styleText('cyan', '\n[System] Loading Model & Tokenizer...'))
  const tokenizer = await loadTokenizer(MODEL_PATH)
  const template = await loadTemplate(MODEL_PATH)
  const stopTokenIds = stopTokensFrom(tokenizer)
  const padTokenId = padTokenFrom(tokenizer)
  const model = await MLXModel.fromPath(MODEL_PATH)

  const promptStr = template.render({
    messages: [{ role: 'user', content: PROMPT }],
    add_generation_prompt: true
  })

  const promptTokens = new Int32Array(tokenizer.encode(promptStr).ids)

  console.log(styleText('cyan', '\n[System] Generating Response (Heatmap Mode)...\n'))

  const generateTask = model.generate(promptTokens, {
    batchSize: 1,
    stopTokenIds,
    padTokenId,
    temperature: 0.7,
    maxTokens: 512,
    topLogits: TOP_K
  })

  // We will store the entire history of the generation to scrub through it later
  const history = []

  // 1. LIVE GENERATION PHASE
  for await (const batches of generateTask) {
    const token = batches[0]
    if (token === padTokenId || token === -1) continue

    // Decode just this token (clean_up_tokenization_spaces must be false to preserve boundaries)
    const textChunk = tokenizer.decode([token], { skip_special_tokens: true, clean_up_tokenization_spaces: false })
    const topLogs = batches.topLogits?.[0]

    // Skip empty chunks (like BOS tokens) to keep our interactive cursor clean
    if (textChunk.length === 0) continue

    const prob = topLogs ? topLogs[0].prob : 1.0

    // Save to history for the interactive explorer
    history.push({ token, text: textChunk, topLogs })

    // Print live heatmap
    process.stdout.write(colorizeByConfidence(textChunk, prob))
  }

  console.log('\n')

  // 2. INTERACTIVE EXPLORER PHASE
  let cursorIndex = 0

  function renderTUI () {
    // Clear screen and reset cursor
    console.clear()
    console.log(styleText(['bgBlue', 'white', 'bold'], ' MLX INTERACTIVE TOKEN EXPLORER ') + styleText('dim', ' (Use Left/Right Arrows to scrub. Press Q to quit)\n'))

    // Render the text with the active token inverted
    let reconstructedText = ''
    for (let i = 0; i < history.length; i++) {
      const chunk = history[i].text
      if (i === cursorIndex) {
        reconstructedText += styleText(['bgWhite', 'black'], chunk)
      } else {
        reconstructedText += colorizeByConfidence(chunk, history[i].topLogs[0].prob)
      }
    }
    console.log(reconstructedText)
    console.log('\n' + '─'.repeat(process.stdout.columns || 80) + '\n')

    // Render the Inspector Dashboard for the currently selected token
    const activeData = history[cursorIndex]
    const titleText = ` TOKEN INSPECTOR: "${visualizeSpaces(activeData.text)}" `
    console.log(styleText(['bold', 'cyan'], titleText))

    if (activeData.topLogs) {
      for (let k = 0; k < activeData.topLogs.length; k++) {
        const { id, prob } = activeData.topLogs[k]
        const decodedAlt = tokenizer.decode([id], { skip_special_tokens: false, clean_up_tokenization_spaces: false })

        const percent = (prob * 100).toFixed(1).padStart(5, ' ')
        const rank = (k + 1).toString()
        const tokenDisplay = `"${visualizeSpaces(decodedAlt)}"`.padEnd(15, ' ')

        const isChosen = k === 0 ? '▶' : ' '

        console.log(` ${styleText('yellow', isChosen)} ${rank}. [${percent}%] ${tokenDisplay} ${drawBar(prob)}`)
      }
    } else {
      console.log(styleText('dim', ' No Top-K data available for this token.'))
    }
  }

  // Setup Raw Mode for immediate keypress handling
  readline.emitKeypressEvents(process.stdin)
  process.stdin.setRawMode(true)

  process.stdin.on('keypress', (str, key) => {
    if (key.ctrl && key.name === 'c' || key.name === 'q') {
      process.stdin.setRawMode(false)
      console.clear()
      console.log(styleText('green', 'Exited Token Explorer.'))
      process.exit()
    }

    if (key.name === 'left') {
      cursorIndex = Math.max(0, cursorIndex - 1)
      renderTUI()
    }

    if (key.name === 'right') {
      cursorIndex = Math.min(history.length - 1, cursorIndex + 1)
      renderTUI()
    }
  })

  // Initial Render
  renderTUI()
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
