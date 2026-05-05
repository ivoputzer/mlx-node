import { styleText } from 'node:util'
import { createInterface } from 'node:readline'
import { setTimeout } from 'node:timers/promises'

import { createBatchRenderer } from '../lib/grid-formatter.js'
import { emojis } from '../data/emojis.js'
import { words } from '../data/words.js'

// Example 1: The Tail Formatter (TPS & Spinners)
const tailFormatter = (colIndex, buffers) => {
  const currentLen = buffers[colIndex].length
  // If the stream is still generating, show a spinner and stat block!
  if (currentLen > 0 && currentLen < 1000) { // Assume 1000 is "done"
    return styleText(['dim', 'cyan'], ' ⠧ [32 t/s]')
  }
  return styleText(['green'], ' ✓ Done')
}

// Example 2: The Homogeneous Diff Formatter
// This compares this buffer against Buffer 0. Where it diverges, it turns yellow!
const diffFormatter = (text, colIndex, allBuffers) => {
  if (colIndex === 0) return text // Stream 0 is the baseline

  const baseline = allBuffers[0]
  let divergencePoint = 0

  // Find where this string breaks away from the baseline
  while (divergencePoint < text.length && text[divergencePoint] === baseline[divergencePoint]) {
    divergencePoint++
  }

  // Return the string: Normal text up to the split, yellow text afterwards
  const identicalPart = text.slice(0, divergencePoint)
  const divergedPart = styleText('yellow', text.slice(divergencePoint))

  return identicalPart + divergedPart
}

// Pass them into your renderer!
const tokens = [...words, ...emojis]

const rl = createInterface({ input: process.stdin, output: process.stdout })
rl.setPrompt(styleText('magenta', 'READLINE PROMPT ❯ '))
rl.prompt()

// Pass the rl instance into our renderer!
const render = createBatchRenderer(3, { readline: rl, tailFormatter, cellFormatter: diffFormatter })
const buffers = ['', '', '']

// Simulate stream
for (let i = 0; i < 2000; i++) {
  if (i < 100) {
    const sameBuffer = tokens[Math.floor(Math.random() * tokens.length)]
    buffers[0] += ' ' + sameBuffer
    buffers[1] += ' ' + sameBuffer
    buffers[2] += ' ' + sameBuffer
  } else {
    buffers[0] += (Math.random() > 0.90 ? '\n' : ' ') + tokens[Math.floor(Math.random() * tokens.length)]
    buffers[1] += (Math.random() > 0.90 ? '\n' : ' ') + tokens[Math.floor(Math.random() * tokens.length)]
    buffers[2] += (Math.random() > 0.90 ? '\n' : ' ') + tokens[Math.floor(Math.random() * tokens.length)]
  }

  render(buffers)

  await setTimeout(10)
}
