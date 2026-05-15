// import { styleText } from 'node:util'

// /**
//  * Standard ANSI escape sequences for terminal manipulation.
//  */
// export const ANSI = {
//   CURSOR_HIDE: '\x1b[?25l',
//   CURSOR_SHOW: '\x1b[?25h',
//   CLEAR_SCREEN: '\x1b[2J',
//   CLEAR_LINE_RIGHT: '\x1b[K',
//   CLEAR_DOWN: '\x1b[J',
//   CURSOR_HOME: '\x1b[H',
//   CURSOR_UP: (n) => `\x1b[${n}A`
// }

// /**
//  * Wraps text to a specific column width without breaking words.
//  */
// export function wrapLines (text, width) {
//   if (!text) return ['']
//   const lines = []
//   for (const paragraph of text.split('\n')) {
//     let currentLine = ''
//     const words = paragraph.split(/(\s+)/) // Preserve spaces
//     for (const word of words) {
//       if ((currentLine.length + word.length) > width && currentLine.length > 0) {
//         lines.push(currentLine)
//         currentLine = word.trimStart()
//       } else {
//         currentLine += word
//       }
//     }
//     lines.push(currentLine)
//   }
//   return lines
// }

// /**
//  * Multi-column terminal renderer that works in-place.
//  */
// export class ColumnRenderer {
//   #previousLines = 0

//   constructor () {
//     // Ensure cursor is restored if the process exits unexpectedly
//     process.on('SIGINT', () => {
//       process.stdout.write(ANSI.CURSOR_SHOW + '\n')
//       process.exit(0)
//     })
//   }

//   /**
//    * Renders an array of text strings side-by-side.
//    * It safely redraws over its previous output without clearing the whole screen.
//    */
//   render (buffers, { gap = 3 } = {}) {
//     const batchSize = buffers.length
//     const terminalWidth = process.stdout.columns || 120
//     const colWidth = Math.floor((terminalWidth - (batchSize - 1) * gap) / batchSize)

//     // Move cursor up to overwrite previous render (if this isn't the first frame)
//     if (this.#previousLines > 0) {
//       process.stdout.write(ANSI.CURSOR_UP(this.#previousLines))
//     }

//     let outputString = ''

//     // 1. Build Header Row
//     const headers = Array.from({ length: batchSize }, (_, i) =>
//       styleText(['bold', 'cyan'], ` Stream ${i + 1} `.padEnd(colWidth, '─'))
//     ).join(styleText('dim', ' ┬ '))

//     outputString += ANSI.CLEAR_LINE_RIGHT + headers + '\n'

//     // 2. Build Content Rows
//     const columnLines = buffers.map(text => wrapLines(text, colWidth))
//     const maxLines = Math.max(...columnLines.map(lines => lines.length), 1)

//     for (let i = 0; i < maxLines; i++) {
//       const row = columnLines.map(lines => {
//         const line = lines[i] || ''
//         return line.padEnd(colWidth, ' ')
//       }).join(styleText('dim', ' │ '))

//       outputString += ANSI.CLEAR_LINE_RIGHT + row + '\n'
//     }

//     // 3. Clear any leftover text below if the new render is shorter than the last
//     outputString += ANSI.CLEAR_DOWN

//     // 4. Write to stdout in a single chunk to prevent flickering
//     process.stdout.write(outputString)

//     // 5. Save the height (Headers + Content) for the next frame
//     this.#previousLines = maxLines + 1
//   }

//   /**
//    * Resets the renderer so the next frame starts fresh below the current output.
//    */
//   finalize () {
//     this.#previousLines = 0
//   }
// }

// /*

// import { ColumnRenderer, ANSI } from './terminal-ui.js' // Point to the new file

// // ... inside runTestsUsing() ...

//   const BATCH_SIZE = 3
//   const buffers = Array(BATCH_SIZE).fill('')
//   const ui = new ColumnRenderer()

//   // Hide cursor for smooth rendering (we don't need to clear the screen anymore!)
//   process.stdout.write(ANSI.CURSOR_HIDE)

//   try {
//     while (true) {
//       const { done, value } = await iterator.next()
//       if (done) break

//       const tick = value
//       for (let b = 0; b < BATCH_SIZE; b++) {
//         const token = tick[b]
//         if (token !== -1) {
//           buffers[b] += tokenizer.decode([token], { skip_special_tokens: true })
//         }
//       }

//       // Draw gracefully in-place
//       ui.render(buffers)
//     }
//   } finally {
//     ui.finalize()
//     process.stdout.write(ANSI.CURSOR_SHOW + '\n\n')
//   }

// */
