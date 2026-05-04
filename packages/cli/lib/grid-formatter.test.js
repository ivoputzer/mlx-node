import { describe, it } from 'node:test'
import { equal, ok } from 'node:assert/strict'
import { GridFormatter } from './grid-formatter.js'

describe('lib/GridFormatter', () => {
  it('Grid Layout calculates absolute cursor positions correctly', () => {
    const formatter = new GridFormatter(2) // 2 columns
    const buffers = ['hello', 'world']
    const terminalWidth = 83

    const result = formatter.formatGrid(buffers, terminalWidth)

    // (83 - 3 gap) / 2 cols = 40 colWidth
    // Col 1 = pos 1, Gap = pos 41, Col 2 = pos 44
    equal(result.colWidth, 40)

    ok(result.headers.includes('\x1b[1G'), 'Header 1 starts at col 1')
    ok(result.headers.includes('\x1b[41G'), 'Gap starts at col 41')
    ok(result.headers.includes('\x1b[44G'), 'Header 2 starts at col 44')

    // The rows should use the same absolute positioning
    ok(result.rows[0].includes('\x1b[1Ghello'))
    ok(result.rows[0].includes('\x1b[44Gworld'))
  })

  it('Word wrapping correctly handles tab replacement', () => {
    const formatter = new GridFormatter(1, { tabSize: 4 })
    const lines = formatter.wrapLines('a\tb', 10)
    equal(lines[0], 'a    b')
  })
})
