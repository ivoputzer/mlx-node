import { describe, it } from 'node:test'
import { equal, ok } from 'node:assert/strict'
import { GridFormatter, IncrementalGridFormatter } from './grid-formatter.js'

describe('lib/grid-formatter', () => {
  describe('.GridFormatter', () => {
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
  describe('.IncrementalGridFormatter', () => {
    it('Grid drops cache cleanly when terminal resizes', () => {
      const formatter = new IncrementalGridFormatter(2)
      // Format at width 100
      formatter.formatGrid(['Hello\nWorld', 'Testing'], 100)
      equal(formatter.getState()[0].lockedLen, 6, 'Locked successfully')
      // Format at width 80 (simulated window shrink)
      formatter.formatGrid(['Hello\nWorld', 'Testing'], 80)
      // Re-evaluating the whole string resets locks for the new dimensions
      equal(formatter.getState()[0].lockedLen, 6, 'Cache rebuilt dynamically')
    })

    it('Hard Word Wrapping gracefully slices impossible strings', () => {
      const formatter = new IncrementalGridFormatter(1)
      const colWidth = 5
      // "abcdef" is 6 chars, but column is 5. It MUST slice.
      const lines = formatter.wrapParagraph('abcdef', colWidth)
      equal(lines[0], 'abcde', 'First part sliced exactly to 5')
      equal(lines[1], 'f', 'Remainder pushed to new line')
    })

    it('Empty Buffers handle gracefully', () => {
      const formatter = new IncrementalGridFormatter(2)
      const result = formatter.formatGrid(['', ''], 80)
      // maxLines should be at least 1 even if completely empty
      equal(result.rows.length, 1)
    })
  })
})
