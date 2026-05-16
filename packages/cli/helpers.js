import readline from 'node:readline'

import { IncrementalGridFormatter } from './lib/grid-formatter.js'

export const ANSI = {
  HIDE_CURSOR: '\x1b[?25l',
  SHOW_CURSOR: '\x1b[?25h',
  DISABLE_WRAP: '\x1b[?7l',
  ENABLE_WRAP: '\x1b[?7h',
  CLEAR_LINE: '\x1b[K',
  CLEAR_DOWN: '\x1b[J',
  moveUp: (n) => `\x1b[${n}A`
}

export function createBatchHeaders (batchSize, options = {}) {
  const titles = options.titles || Array.from({ length: batchSize }, (_, i) => `Stream ${i + 1}`)
  const terminalWidth = process.stdout.columns || 120
  const gap = 3
  const colWidth = Math.max(2, Math.floor((terminalWidth - (batchSize - 1) * gap) / batchSize))

  const positions = []
  let currentPos = 1
  for (let i = 0; i < batchSize; i++) {
    positions.push({ type: 'col', pos: currentPos, index: i })
    currentPos += colWidth
    if (i < batchSize - 1) {
      positions.push({ type: 'gap', pos: currentPos })
      currentPos += gap
    }
  }

  let headers = ''
  for (const p of positions) {
    if (p.type === 'col') {
      const title = ` ${titles[p.index]} `
      const padLen = Math.max(0, colWidth - (title.length))
      headers += `\x1b[${p.pos}G\x1b[1m\x1b[36m${title}${'─'.repeat(padLen)}\x1b[0m`
    } else {
      headers += `\x1b[${p.pos}G\x1b[2m │ \x1b[0m`
    }
  }
  return '\r\x1b[K' + headers + '\n'
}

export function createBatchRenderer (batchSize, options = {}) {
  const formatter = new IncrementalGridFormatter(batchSize, options)
  const rl = options.readline
  let previousRows = []
  let needsFullRedraw = true

  process.stdout.on('resize', () => { needsFullRedraw = true })

  return function renderTick (columnBuffers, isFinished = false) {
    const termWidth = process.stdout.columns || 120

    // Ensure buffers exist to prevent 'undefined'
    const safeBuffers = Array.from({ length: batchSize }, (_, i) => columnBuffers[i] || '')
    const { rows } = formatter.formatGrid(safeBuffers, termWidth)

    if (rl) {
      readline.cursorTo(process.stdout, 0)
      readline.clearLine(process.stdout, 0)
    }

    process.stdout.write(ANSI.HIDE_CURSOR + ANSI.DISABLE_WRAP)

    if (needsFullRedraw || previousRows.length === 0) {
      rows.forEach(r => process.stdout.write('\r' + ANSI.CLEAR_LINE + r + '\n'))
      needsFullRedraw = false
    } else {
      let diff = 0
      while (diff < previousRows.length && diff < rows.length && previousRows[diff] === rows[diff]) diff++

      if (diff < rows.length) {
        const moveUp = previousRows.length - diff
        const maxSafeMoveUp = (process.stdout.rows || 24) - 2

        if (moveUp > maxSafeMoveUp) {
          process.stdout.write(ANSI.moveUp(maxSafeMoveUp) + '\r' + ANSI.CLEAR_DOWN)
          const safeDiff = previousRows.length - maxSafeMoveUp
          rows.slice(safeDiff).forEach(r => process.stdout.write('\r' + ANSI.CLEAR_LINE + r + '\n'))
        } else {
          if (moveUp > 0) process.stdout.write(ANSI.moveUp(moveUp))
          process.stdout.write('\r' + ANSI.CLEAR_DOWN)
          rows.slice(diff).forEach(r => process.stdout.write('\r' + ANSI.CLEAR_LINE + r + '\n'))
        }
      }
    }
    previousRows = rows

    process.stdout.write(ANSI.SHOW_CURSOR + ANSI.ENABLE_WRAP)

    if (rl) {
      if (isFinished) {
        process.stdout.write('\n')
        rl.resume()
      }
      rl.prompt(true)
    }
  }
}
