import { styleText, stripVTControlCharacters } from 'node:util'
import readline from 'node:readline'

// @deprecated
export class GridFormatter {
  constructor (batchSize, options = {}) {
    this.batchSize = batchSize
    this.tabSize = options.tabSize || 2
    this.segmenter = new Intl.Segmenter('en', { granularity: 'grapheme' })
  }

  // A simplified East Asian Width heuristic for word wrapping
  getWidth (str) {
    let w = 0
    for (const { segment } of this.segmenter.segment(stripVTControlCharacters(str))) {
      const code = segment.codePointAt(0)
      if (
        code >= 0x1F000 || // Emojis
        (code >= 0x2600 && code <= 0x27BF) || // Misc Symbols
        (code >= 0x2B00 && code <= 0x2BFF) || // Arrows
        (code >= 0x4E00 && code <= 0x9FFF) || // CJK
        (code >= 0x3040 && code <= 0x30FF) || // Kana
        (code >= 0xFF00 && code <= 0xFFEF) // Fullwidth
      ) w += 2
      else w += 1
    }
    return w
  }

  wrapLines (text, colWidth) {
    text = text.replace(/\t/g, ' '.repeat(this.tabSize))
    const lines = []

    for (const paragraph of text.split('\n')) {
      let currentLine = ''
      let currentW = 0

      const words = paragraph.match(/\S+|\s+/g) || []
      for (const word of words) {
        if (currentW === 0 && word.trim() === '') continue // Drop leading spaces on wrap
        const wordW = this.getWidth(word)

        if (wordW > colWidth) {
          if (currentW > 0) lines.push(currentLine)
          currentLine = ''; currentW = 0
          let temp = ''; let tempW = 0

          for (const { segment } of this.segmenter.segment(word)) {
            const sw = this.getWidth(segment)
            if (tempW + sw > colWidth) {
              lines.push(temp)
              temp = segment; tempW = sw
            } else {
              temp += segment; tempW += sw
            }
          }
          currentLine = temp; currentW = tempW
        } else if (currentW + wordW > colWidth) {
          lines.push(currentLine)
          currentLine = word.trim() === '' ? '' : word
          currentW = this.getWidth(currentLine)
        } else {
          currentLine += word
          currentW += wordW
        }
      }
      lines.push(currentLine)
    }
    return lines
  }

  // PURE FUNCTION: Turns buffers into perfectly aligned ANSI string arrays
  formatGrid (buffers, terminalWidth) {
    const gap = 3
    const colWidth = Math.max(2, Math.floor((terminalWidth - (this.batchSize - 1) * gap) / this.batchSize))
    const wrappedCols = buffers.map(b => this.wrapLines(b, colWidth))
    const maxLines = Math.max(...wrappedCols.map(c => c.length), 1)

    // Calculate Absolute 1-Based Column Positions
    const positions = []
    let currentPos = 1
    for (let i = 0; i < this.batchSize; i++) {
      positions.push({ type: 'col', pos: currentPos, index: i })
      currentPos += colWidth
      if (i < this.batchSize - 1) {
        positions.push({ type: 'gap', pos: currentPos })
        currentPos += gap
      }
    }

    const rows = []
    for (let i = 0; i < maxLines; i++) {
      let rowStr = ''
      for (const p of positions) {
        if (p.type === 'col') {
          const text = wrappedCols[p.index][i] || ''
          rowStr += `\x1b[${p.pos}G${text}` // Teleport cursor to exact column
        } else {
          rowStr += `\x1b[${p.pos}G${styleText('dim', ' │ ')}` // Teleport separator
        }
      }
      rows.push(rowStr)
    }

    let headerStr = ''
    for (const p of positions) {
      if (p.type === 'col') {
        const title = ` Stream ${p.index + 1} `
        const padLength = Math.max(0, colWidth - this.getWidth(title))
        const padded = title + '─'.repeat(padLength)
        headerStr += `\x1b[${p.pos}G${styleText(['bold', 'cyan'], padded)}`
      } else {
        headerStr += `\x1b[${p.pos}G${styleText('dim', ' ┬ ')}`
      }
    }

    return { headers: headerStr, rows, colWidth }
  }
}

export class IncrementalGridFormatter {
  #segmenter
  #batchSize
  #tabSize
  #lastColWidth
  #colStates

  #lineFormatter
  #tailFormatter

  constructor (batchSize, options = {}) {
    this.#batchSize = batchSize
    this.#tabSize = options.tabSize || 2
    // Format individual wrapped lines to prevent ANSI bleed
    this.#lineFormatter = options.lineFormatter || ((line) => line)
    this.#tailFormatter = options.tailFormatter || (() => '')

    this.#segmenter = new Intl.Segmenter('en', { granularity: 'grapheme' })
    this.#lastColWidth = 0
    this.#colStates = this.#createEmptyState()
  }

  #createEmptyState () {
    return Array.from({ length: this.#batchSize }, () => ({
      lockedLines: [],
      lockedLen: 0,
      currentView: []
    }))
  }

  getWidth (str) {
    let asciiOnly = true
    for (let i = 0; i < str.length; i++) {
      if (str.charCodeAt(i) > 127) { asciiOnly = false; break }
    }
    if (asciiOnly) return str.length

    let w = 0
    for (const { segment } of this.#segmenter.segment(stripVTControlCharacters(str))) {
      const code = segment.codePointAt(0)
      if (
        code >= 0x1F000 || (code >= 0x2600 && code <= 0x27BF) ||
        (code >= 0x2B00 && code <= 0x2BFF) || (code >= 0x4E00 && code <= 0x9FFF) ||
        (code >= 0x3040 && code <= 0x30FF) || (code >= 0xFF00 && code <= 0xFFEF)
      ) w += 2
      else w += 1
    }
    return w
  }

  wrapParagraph (text, colWidth) {
    if (!text) return ['']
    text = text.replace(/\t/g, ' '.repeat(this.#tabSize))
    const lines = []
    let currentLine = ''
    let currentW = 0

    const words = text.match(/\S+|\s+/g) || []
    for (const word of words) {
      if (currentW === 0 && word.trim() === '') continue
      const wordW = this.getWidth(word)

      if (wordW > colWidth) {
        if (currentW > 0) lines.push(currentLine)
        currentLine = ''; currentW = 0
        let temp = ''; let tempW = 0
        for (const { segment } of this.#segmenter.segment(word)) {
          const sw = this.getWidth(segment)
          if (tempW + sw > colWidth) {
            lines.push(temp)
            temp = segment; tempW = sw
          } else {
            temp += segment; tempW += sw
          }
        }
        currentLine = temp; currentW = tempW
      } else if (currentW + wordW > colWidth) {
        lines.push(currentLine)
        currentLine = word.trim() === '' ? '' : word
        currentW = this.getWidth(currentLine)
      } else {
        currentLine += word
        currentW += wordW
      }
    }
    lines.push(currentLine)
    return lines
  }

  formatGrid (buffers, terminalWidth) {
    const gap = 3
    const colWidth = Math.max(2, Math.floor((terminalWidth - (this.#batchSize - 1) * gap) / this.#batchSize))

    if (colWidth !== this.#lastColWidth) {
      this.#lastColWidth = colWidth
      this.#colStates = this.#createEmptyState()
    }

    const wrappedCols = []

    for (let i = 0; i < this.#batchSize; i++) {
      const state = this.#colStates[i]
      const buffer = buffers[i] || '' // Prevent undefined

      const unprocessed = buffer.slice(state.lockedLen)

      if (unprocessed) {
        const parts = unprocessed.split('\n')
        for (let p = 0; p < parts.length - 1; p++) {
          state.lockedLines.push(...this.wrapParagraph(parts[p], colWidth))
          state.lockedLen += parts[p].length + 1
        }

        const tail = this.#tailFormatter(i, buffers)
        const activeWrapped = this.wrapParagraph(parts[parts.length - 1] + tail, colWidth)
        state.currentView = [...state.lockedLines, ...activeWrapped]
      } else if (state.currentView.length === 0) {
        const tail = this.#tailFormatter(i, buffers)
        state.currentView = this.wrapParagraph(tail, colWidth)
      }

      // Apply the Line Formatter to prevent color bleed!
      const formattedView = state.currentView.map((line, lineIdx) =>
        this.#lineFormatter(line, lineIdx, i, buffers, state.currentView)
      )

      wrappedCols.push(formattedView)
    }

    const maxLines = Math.max(...wrappedCols.map(c => c.length), 1)
    const positions = []
    let currentPos = 1

    for (let i = 0; i < this.#batchSize; i++) {
      positions.push({ type: 'col', pos: currentPos, index: i })
      currentPos += colWidth
      if (i < this.#batchSize - 1) {
        positions.push({ type: 'gap', pos: currentPos })
        currentPos += gap
      }
    }

    const rows = []
    for (let i = 0; i < maxLines; i++) {
      let rowStr = ''
      for (const p of positions) {
        if (p.type === 'col') {
          // Wrap everything in \x1b[0m reset just to be absolutely certain it doesn't bleed
          rowStr += `\x1b[${p.pos}G${wrappedCols[p.index][i] || ''}\x1b[0m`
        } else {
          rowStr += `\x1b[${p.pos}G${styleText('dim', ' │ ')}`
        }
      }
      rows.push(rowStr)
    }

    return { rows, colWidth }
  }
}

function getVisualWidth (str) {
  // Simplistic width for headers
  return stripVTControlCharacters(str).length // (Can reuse the robust segmenter here)
}

const ANSI = {
  HIDE_CURSOR: '\x1b[?25l',
  SHOW_CURSOR: '\x1b[?25h',
  DISABLE_WRAP: '\x1b[?7l',
  ENABLE_WRAP: '\x1b[?7h',
  CLEAR_LINE: '\x1b[K',
  CLEAR_DOWN: '\x1b[J',
  moveUp: (n) => `\x1b[${n}A`
}

export function printBatchHeaders (batchSize, options = {}) {
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
  process.stdout.write('\r\x1b[K' + headers + '\n')
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
