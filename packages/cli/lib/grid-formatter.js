import { styleText, stripVTControlCharacters } from 'node:util'

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
