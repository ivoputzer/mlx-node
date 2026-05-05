import { styleText } from 'node:util'

export function inlineDiff (oldStr, newStr) {
  if (oldStr === newStr) return oldStr

  // Tokenize by words, whitespace (including newlines), and punctuation.
  const tokenize = (str) => str.match(/[\w]+|\s+|[^\w\s]+/g) || []

  const arr1 = tokenize(oldStr)
  const arr2 = tokenize(newStr)

  const len1 = arr1.length
  const len2 = arr2.length

  const dp = Array.from({ length: len1 + 1 }, () => new Int32Array(len2 + 1))

  // 1. Build the DP table for the Longest Common Subsequence
  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      if (arr1[i - 1] === arr2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
  }

  // 2. Backtrack to find the exact insertions, deletions, and matches
  let i = len1
  let j = len2
  const diff = []

  while (i > 0 && j > 0) {
    if (arr1[i - 1] === arr2[j - 1]) {
      diff.unshift({ type: 'equal', value: arr1[i - 1] })
      i--
      j--
    } else if (dp[i - 1][j] > dp[i][j - 1]) {
      diff.unshift({ type: 'removed', value: arr1[i - 1] })
      i--
    } else {
      diff.unshift({ type: 'added', value: arr2[j - 1] })
      j--
    }
  }

  while (i > 0) {
    diff.unshift({ type: 'removed', value: arr1[i - 1] })
    i--
  }
  while (j > 0) {
    diff.unshift({ type: 'added', value: arr2[j - 1] })
    j--
  }

  // 3. Render the output with high-contrast background colors
  return diff
    .map(({ type, value }) => {
      if (type === 'added') {
        // Green background with black text for crisp readability
        return styleText(['bgGreen', 'black'], value)
      }

      if (type === 'removed') {
        // Red background with black text + strikethrough
        return styleText(['bgRed', 'black', 'strikethrough'], value)
      }

      // Kept text remains entirely unmodified (default terminal colors)
      return value
    })
    .join('')
}

// const originalText = `export default function hello() {
//   console.log("Hello World");
//   return true;
// }`

// const updatedText = `export function hello() {
//   console.log("Hello Brave New World");
//   return false;
// }`

// const result = inlineDiff(originalText, updatedText)

// console.log(result)
