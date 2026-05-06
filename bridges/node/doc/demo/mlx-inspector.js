import { MLXModel, MLXMetrics } from 'mlx-swift'
import { loadTokenizer, loadTemplate, padTokenFrom, stopTokensFrom } from 'mlx-lm'
import { styleText, stripVTControlCharacters } from 'node:util'
import readline from 'node:readline'

// --- CONFIGURATION ---
const MODEL_PATH = '/Users/ivoputzer/github/models/mlx-community/granite-4.1-8b-8bit'
const PROMPT = 'Explain quantum gravity'
const TOP_K = 5
const MAX_FPS = 60

// --- TUI ENGINE UTILITIES ---
const TUI = {
  enter: () => {
    process.stdout.write('\x1b[?1049h\x1b[?25l') // Alt screen, hide cursor
    process.stdout.write('\x1b[?1000l') // Ensure mouse tracking is off so clicks don't send garbage
  },
  exit: () => process.stdout.write('\x1b[?1049l\x1b[?25h'), // Main screen, show cursor
  color: (prob, text) => {
    if (prob >= 0.85) return styleText('green', text)
    if (prob >= 0.50) return styleText('yellow', text)
    if (prob >= 0.10) return styleText('red', text)
    return styleText(['bgRed', 'white'], text)
  },
  bar: (prob, width = 10) => {
    const filled = Math.max(0, Math.min(width, Math.round(prob * width)))
    return styleText('cyan', '█'.repeat(filled)) + styleText('gray', '░'.repeat(width - filled))
  },
  sparkline: (probs, width = 25) => {
    const chars = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    const line = probs.map(p => chars[Math.min(7, Math.floor(p * 8))]).join('')
    return line.padStart(width, ' ')
  },
  entropy: (logs) => {
    if (!logs || logs.length === 0) return 0
    return logs.reduce((acc, { prob }) => acc - (prob > 0 ? prob * Math.log2(prob) : 0), 0)
  },
  // Safely escapes control characters before applying ANSI to avoid truncation bleeding
  cleanToken: (str) => {
    if (!str || str.length === 0) return '∅'
    return str.replace(/\n/g, '↵').replace(/ /g, '·').replace(/\r/g, '←').replace(/\t/g, '⇥')
  },
  dimSpecials: (str) => {
    return str.replace(/[·↵←⇥∅…]/g, match => styleText('dim', match))
  }
}

// Ensure clean exit
process.on('SIGINT', () => { TUI.exit(); process.exit(0) })
process.on('uncaughtException', (err) => { TUI.exit(); console.error(err); process.exit(1) })

// --- MAIN APPLICATION ---
async function main () {
  console.log(styleText('cyan', '[System] Booting MLX Inspector Engine...'))
  const tokenizer = await loadTokenizer(MODEL_PATH)
  const template = await loadTemplate(MODEL_PATH)
  const stopTokenIds = stopTokensFrom(tokenizer)
  const padTokenId = padTokenFrom(tokenizer)
  const model = await MLXModel.fromPath(MODEL_PATH)

  const promptStr = template.render({ messages: [{ role: 'user', content: PROMPT }], add_generation_prompt: true })
  const promptTokens = new Int32Array(tokenizer.encode(promptStr).ids)

  // --- STATE ---
  const history = []
  let isGenerating = true
  let isAutoScroll = true
  let cursorIdx = -1
  const startTime = Date.now()
  let lastRender = 0

  // Session Analytics State
  let statsHigh = 0
  let statsMed = 0
  let statsLow = 0
  let totalEntropy = 0

  // We map history to wrapped lines exactly once per token to save CPU
  const wrappedLines = [[]]
  let currentLineWidth = 0

  TUI.enter()

  let metrics = MLXMetrics.fromSnapshot()

  const updateMetrics = () => {
    metrics = MLXMetrics.fromSnapshot()
    setTimeout(updateMetrics, 1000)
  }

  setImmediate(updateMetrics)

  // --- RENDER FUNCTION ---
  function render () {
    const cols = process.stdout.columns || 80
    const rows = process.stdout.rows || 24

    // Layout boundaries
    const LEFT_W = Math.floor(cols * 0.55)
    const RIGHT_W = cols - LEFT_W - 3
    const CONTENT_H = rows - 4 // Header(1), Div(1), Content(N), Div(1), Footer(1) -> 4 non-content lines

    // Determine the active token to display in the inspector
    const activeIdx = isAutoScroll ? Math.max(0, history.length - 1) : cursorIdx
    const activeToken = history[activeIdx]

    // Determine scroll position (find which line contains the active token)
    let activeLineIdx = wrappedLines.length - 1
    if (!isAutoScroll) {
      activeLineIdx = wrappedLines.findIndex(line => line.some(t => t.idx === activeIdx))
      if (activeLineIdx === -1) activeLineIdx = 0
    }

    // Keep active line vertically centered if possible
    let viewStart = Math.max(0, activeLineIdx - Math.floor(CONTENT_H / 2))
    if (wrappedLines.length <= CONTENT_H) viewStart = 0
    else if (viewStart + CONTENT_H > wrappedLines.length) viewStart = wrappedLines.length - CONTENT_H

    // Build the Right Pane (Inspector)
    const rightPane = []
    if (activeToken) {
      const logs = activeToken.topLogs || []
      const p1 = logs[0]?.prob || 0
      const ent = TUI.entropy(logs)

      // Section 1: Active Token Details
      rightPane.push(styleText(['bold', 'cyan'], ' TOKEN DETAILS '))
      rightPane.push(` String:  "${TUI.dimSpecials(TUI.cleanToken(activeToken.text))}"`)
      rightPane.push(` Conf:    ${TUI.bar(p1, 10)} ${(p1 * 100).toFixed(1).padStart(5)}%`)
      rightPane.push(` Entropy: ${ent.toFixed(2)} bits ${ent > 1.0 ? '⚠️' : ' '}`)

      rightPane.push('')

      // Section 2: Top-K Distribution (Aligned!)
      rightPane.push(styleText('dim', ' TOP K DISTRIBUTION '))
      for (let k = 0; k < TOP_K; k++) {
        if (!logs[k]) break
        const { id, prob } = logs[k]

        const pct = (prob * 100).toFixed(1).padStart(5, ' ')
        const bar = TUI.bar(prob, 8)
        const tokId = styleText('yellow', id.toString().padStart(6, ' '))
        const indicator = k === 0 ? styleText('green', '▶') : ' '

        // Clean and Truncate the string BEFORE applying ANSI colors to prevent bleed
        let rawStr = TUI.cleanToken(tokenizer.decode([id], { skip_special_tokens: false }))
        const maxStrLen = Math.max(4, RIGHT_W - 32)
        if (rawStr.length > maxStrLen) rawStr = rawStr.substring(0, maxStrLen - 1) + '…'

        const finalStr = TUI.dimSpecials(rawStr)
        rightPane.push(` ${indicator} ${pct}% ${bar} ${tokId} ${finalStr}`)
      }

      rightPane.push('')

      // Section 3: Session Analytics
      rightPane.push(styleText('dim', ' SESSION ANALYTICS '))
      const totalT = Math.max(1, history.length)
      const avgEnt = (totalEntropy / totalT).toFixed(2)

      rightPane.push(` Avg Entropy: ${avgEnt} bits`)
      rightPane.push(` 🟩 High (>=85%): ${((statsHigh / totalT) * 100).toFixed(1).padStart(5)}%`)
      rightPane.push(` 🟨 Med  (>=50%): ${((statsMed / totalT) * 100).toFixed(1).padStart(5)}%`)
      rightPane.push(` 🟥 Low  (< 50%): ${((statsLow / totalT) * 100).toFixed(1).padStart(5)}%`)

      rightPane.push('')

      // Section 4: Trajectory
      rightPane.push(styleText('dim', ' RECENT TRAJECTORY '))
      const trailing = history.slice(Math.max(0, activeIdx - 30), activeIdx + 1).map(h => h.topLogs?.[0]?.prob || 0)
      rightPane.push(` ${styleText('magenta', TUI.sparkline(trailing, 30))}`)
    }

    // --- DOUBLE BUFFERING COMPOSITING ---
    let frame = '\x1b[H' // Move cursor home (do not clear screen)

    // Header
    const modeStr = isAutoScroll ? styleText(['bgGreen', 'black'], ' FOLLOW ') : styleText(['bgYellow', 'black'], ' SCRUB ')
    const headerTitle = ' ⚡ MLX LOGIT INSPECTOR '.padEnd(LEFT_W, ' ')
    const headerControls = '[Arrows: Scrub | Space: Auto | Q: Quit] '.padStart(RIGHT_W + 2, ' ')
    frame += styleText(['bgBlue', 'white', 'bold'], headerTitle) + modeStr + styleText(['bgBlue', 'white'], headerControls.substring(8)) + '\x1b[K\n'

    // Content Body
    for (let i = 0; i < CONTENT_H; i++) {
      const lineData = wrappedLines[viewStart + i]
      let leftStr = ''
      let leftVisLen = 0

      if (lineData) {
        for (const token of lineData) {
          const isTarget = token.idx === activeIdx
          const displayTxt = token.text.replace(/\n/g, ' ') // Flat display

          if (isTarget) {
            leftStr += styleText(['bgWhite', 'black'], displayTxt || ' ')
          } else {
            leftStr += TUI.color(token.prob, displayTxt)
          }
          leftVisLen += displayTxt.length
        }
      }

      // Pad the left pane with physical spaces
      const padLeft = Math.max(0, LEFT_W - leftVisLen)
      leftStr += ' '.repeat(padLeft)

      const rightStr = rightPane[i] || ''

      // Combine with divider and clear-to-end-of-line
      frame += `${leftStr} ${styleText('dim', '│')} ${rightStr}\x1b[K\n`
    }

    // Footer

    const memGb = (metrics.active / 1024 / 1024 / 1024).toFixed(1)
    const tps = history.length / ((Date.now() - startTime) / 1000)
    const status = isGenerating ? styleText('green', '● GENERATING') : styleText('dim', '○ FINISHED')

    frame += styleText('dim', '─'.repeat(cols)) + '\x1b[K\n'

    // CRITICAL FIX: The very last line must NOT end with a newline `\n`.
    // This prevents the terminal from scrolling the alternate buffer up by one row!
    frame += ` ${status}  │  Speed: ${tps.toFixed(1)} t/s  │  Mem: ${memGb}GB  │  Tokens: ${history.length}\x1b[K`

    // Blast the entire frame to the terminal at once
    process.stdout.write(frame)
    lastRender = Date.now()
  }

  // --- KEYBOARD HANDLING ---
  readline.emitKeypressEvents(process.stdin)
  process.stdin.setRawMode(true)

  process.stdin.on('keypress', (str, key) => {
    if ((key.ctrl && key.name === 'c') || key.name === 'q') {
      TUI.exit()
      process.exit(0)
    }
    if (key.name === 'space') {
      isAutoScroll = !isAutoScroll
      if (isAutoScroll) cursorIdx = history.length - 1
    }
    if (key.name === 'left' && history.length > 0) {
      isAutoScroll = false
      cursorIdx = Math.max(0, cursorIdx - 1)
    }
    if (key.name === 'right' && history.length > 0) {
      isAutoScroll = false
      cursorIdx = Math.min(history.length - 1, cursorIdx + 1)
    }

    // Immediate feedback on keypress
    render()
  })

  // --- GENERATION LOOP ---
  const generateTask = model.generate(promptTokens, {
    batchSize: 1, stopTokenIds, padTokenId, maxTokens: 2048, topLogits: TOP_K
  })

  try {
    for await (const batches of generateTask) {
      const token = batches[0]
      if (token === padTokenId || token === -1) continue

      const textChunk = tokenizer.decode([token], { skip_special_tokens: true, clean_up_tokenization_spaces: false })
      const topLogs = batches.topLogits?.[0]
      const prob = topLogs ? topLogs[0].prob : 1.0
      const hIndex = history.length

      history.push({ idx: hIndex, id: token, text: textChunk, prob, topLogs })

      // Update Analytics
      if (prob >= 0.85) statsHigh++
      else if (prob >= 0.50) statsMed++
      else statsLow++
      totalEntropy += TUI.entropy(topLogs)

      // Fast Text Wrapper
      const parts = textChunk.split('\n')
      for (let p = 0; p < parts.length; p++) {
        if (p > 0) {
          wrappedLines.push([])
          currentLineWidth = 0
        }

        let remaining = parts[p]
        while (remaining.length > 0) {
          const space = Math.floor(process.stdout.columns * 0.55) - currentLineWidth
          if (space <= 0) {
            wrappedLines.push([])
            currentLineWidth = 0
            continue
          }

          const take = remaining.substring(0, space)
          wrappedLines[wrappedLines.length - 1].push({ idx: hIndex, text: take, prob })
          currentLineWidth += take.length
          remaining = remaining.substring(space)
        }
      }

      if (isAutoScroll) cursorIdx = hIndex

      // Throttle rendering
      if (Date.now() - lastRender > (1000 / MAX_FPS)) {
        render()
      }
    }
  } catch (err) {
    if (!err.message.includes('Cancel')) throw err
  } finally {
    isGenerating = false
    render() // Final draw
  }
}

main().catch(err => {
  TUI.exit()
  console.error(err)
})
