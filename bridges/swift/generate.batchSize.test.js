import mlx, { MLXCache, MLXModel, MLXMetrics } from 'mlx-swift' // Adjust to your bridge
import { loadTokenizer, loadTemplate, stopTokenIdsFrom } from 'mlx-lm'       // Adjust to your package

import { styleText } from 'node:util'
import { cwd } from 'node:process'
import { readFile } from 'node:fs/promises'
import { parseArgs } from 'node:util'

const { values: { model }} = parseArgs({
  options: {
    model: { type: 'string', default: '/Users/ivoputzer/github/models/Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit' }
  }
})

// --- Multi-Column UI Helpers (No Dependencies!) ---

/**
 * Wraps text to a specific column width.
 */
function wrapLines(text, width) {
  if (!text) return ['']
  const lines =[]
  for (const paragraph of text.split('\n')) {
    let currentLine = ''
    const words = paragraph.split(/(\s+)/) // Preserve spaces
    for (const word of words) {
      if ((currentLine.length + word.length) > width && currentLine.length > 0) {
        lines.push(currentLine)
        currentLine = word.trimStart()
      } else {
        currentLine += word
      }
    }
    lines.push(currentLine)
  }
  return lines
}

/**
 * Renders an array of text buffers side-by-side using ANSI cursor resets.
 */
function renderColumns(buffers, batchSize) {
  const terminalWidth = process.stdout.columns || 120
  const gap = 3 // Width of the ' │ ' separator
  const colWidth = Math.floor((terminalWidth - (batchSize - 1) * gap) / batchSize)

  // Move cursor to home (0,0) to redraw without flickering
  process.stdout.write('\x1b[H')

  // Print Header Row
  const headers = Array.from({ length: batchSize }, (_, i) =>
    styleText(['bold', 'cyan'], ` Stream ${i + 1} `.padEnd(colWidth, '─'))
  ).join(styleText('dim', ' ┬ '))

  process.stdout.write('\x1b[K' + headers + '\n')

  // Process text columns into rows
  const columnLines = buffers.map(text => wrapLines(text, colWidth))
  const maxLines = Math.max(...columnLines.map(lines => lines.length), 1)

  for (let i = 0; i < maxLines; i++) {
    const row = columnLines.map(lines => {
      const line = lines[i] || ''
      return line.padEnd(colWidth, ' ') // Pad to maintain column alignment
    }).join(styleText('dim', ' │ '))

    process.stdout.write('\x1b[K' + row + '\n') // \x1b[K clears line to the right
  }

  // Clear any leftover lines from previous taller renders below the current cursor
  process.stdout.write('\x1b[J')
}

// --- Main Testing Script ---

async function runTestsUsing(modelPath) {
  console.log(styleText('yellow', 'Loading model and tokenizer...'))

  const tokenizer = await loadTokenizer(modelPath)
  const template = await loadTemplate(modelPath)

  // Use a prompt that encourages creative divergence (Temperature > 0)
  const promptString = template.render({
    messages:[
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'What is the difference between an animal and a human?' }
      // { role: 'user', content: 'What is the meaning of life, and why do people say 42?' }
      // { role: 'user', content: 'In exactly 3 sentences, describe what it feels like to be an artificial intelligence waking up.' }
    ],
    add_generation_prompt: true
  })

  // const promptString = template.render({
  //   messages:[
  //     { role: 'user', content: generateMassivePrompt(20) }
  //   ],
  //   add_generation_prompt: true
  // })
  // const promptString = template.render({
  //   messages:[
  //     { role: 'assistant', content: 'waiting for more input...' },
  //     { role: 'user', content: 'you can work with what you have ;)' }
  //   ],
  //   add_generation_prompt: true
  // })
  // console.log('-----')
  // console.log(promptString)
  // console.log('-----')

  const batchSize = 3
  const promptTokens = new Int32Array(tokenizer.encode(promptString).ids)

  using model = await MLXModel.fromPath(modelPath)
  // using cache = await MLXCache.fromPath('./evaluate.500.safetensors', model)

  const task = model.generate(promptTokens, {
    batchSize,
    chunkSize: 1,
    temperature: 0.8, // High temperature ensures streams diverge!
    maxTokens: 512,
    stopTokenIds: stopTokenIdsFrom(tokenizer)
  })

  const iterator = task[Symbol.asyncIterator]()
  const buffers = Array(batchSize).fill('')
  let finalStats = null
  let ttft = 0

  // Setup UI: Hide Cursor and Clear Screen completely once before streaming starts
  process.stdout.write('\x1b[?25l\x1b[2J')

  // Ensure cursor restores if the user CTRL+C's mid-stream
  process.on('SIGINT', () => {
    process.stdout.write('\x1b[?25h\n\n')
    process.exit(0)
  })

  try {
    const prefillStart = performance.now()
    let isFirstToken = true

    // Lockstep evaluation loop
    while (true) {
      const { done, value } = await iterator.next()

      if (isFirstToken && !done) {
        ttft = performance.now() - prefillStart
        isFirstToken = false
      }

      if (done) {
        finalStats = value // Async generator returns the JSON stats upon completion
        break
      }

      // value is our 2D batched array: e.g., [[token1], [token2],[]]
      const tick = value

      for (let b = 0; b < batchSize; b++) {
        const token = tick[b]
        if (token !== -1) { // Ignore padded sequences that already finished
          buffers[b] += tokenizer.decode([token], { skip_special_tokens: false, clean_up_tokenization_spaces: false })
        }
      }

      renderColumns(buffers, batchSize)
    }
  } finally {
    // Restore Cursor
    process.stdout.write('\x1b[?25h\n\n')
  }

  // Print MLX Engine Statistics
  console.log(styleText(['bold', 'green'], '✨ Generation Complete!'))

  if (finalStats) {
    console.log(`\n${styleText('underline', 'Generation Stats')}:`)
    console.log(`  Prompt tokens:  ${finalStats.promptTokens} (${finalStats.promptTokensPerSecond.toFixed(1)} t/s)`)

    // Highlight the TTFT!
    const ttftSeconds = (ttft / 1000).toFixed(2)
    console.log(styleText(['bold', 'magenta'], `  TTFT (Time To First Token): ${ttftSeconds} seconds`))

    console.log(`  Output tokens:  ${finalStats.generatedTokens} (${finalStats.tokensPerSecond.toFixed(1)} t/s combined)`)
    console.log(`  Stop reason:    ${finalStats.stopReason}`)
  }

  console.log(`\n${styleText('underline', 'Memory Metrics')}:`)
  console.log(MLXMetrics.fromSnapshot().toString())
}

// --- Execution ---

try {
  // Use a local path or whichever identifier your wrapper accepts
  // await runTestsUsing('/Users/ivoputzer/github/models/Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit')
  // await runTestsUsing('/Volumes/SanDisk\ 4TB/MODELS/mlx-community/gpt-oss-20b-MXFP4-Q8')
  // await runTestsUsing('/Volumes/SanDisk\ 4TB/MODELS/mlx-community/granite-4.1-3b-8bit')
  // await runTestsUsing('/Volumes/SanDisk\ 4TB/MODELS/Jackrong/Qwopus3.5-4B-v3')
  // await runTestsUsing('/Volumes/SanDisk\ 4TB/MODELS/gemma-4-e2b')
  // await runTestsUsing('/Volumes/SanDisk\ 4TB/MODELS/qwen3-2b')
  await runTestsUsing(model)
} catch (error) {
  process.stdout.write('\x1b[?25h') // ensure cursor is shown on crash
  console.error(styleText(['bold', 'red'], '\nTest Failed:'), error)
}


function generateMassivePrompt(numLogs = 200) {
  let logs = ''
  for (let i = 0; i < numLogs; i++) {
    const timestamp = new Date(Date.now() - i * 10000).toISOString()
    const ip = `192.168.1.${(i % 255)}`
    const status = i % 17 === 0 ? 'CRITICAL_ERROR [Code 0x0A9]' : 'INFO [User heartbeat]'
    logs += `[${timestamp}] [${ip}] SYSTEM_DAEMON: ${status} - Memory address localized to sector ${i * 42}.\n`
  }

  return `You are a highly creative AI narrative designer. Review the following server logs:
<logs>
${logs}
</logs>
Based on the logs above, specifically the critical errors, write a dramatic, creative 3-paragraph fictional story about a sysadmin who realizes the server has become sentient. Make it suspenseful.`
}
