// // bridges/swift/test/api.test.js
// import test from 'node:test'
// import assert from 'node:assert'
// import { createRequire } from 'node:module'
// const require = createRequire(import.meta.url)

// // We can test the helper functions we refactored without the GPU
// test('createPaddedBatch correctly left-pads jagged arrays', (t) => {
//   const { createPaddedBatch } = require('../index.js') // Assuming you exported it

//   const prompts = [
//     [1, 2], // Length 2
//     [1, 2, 3, 4] // Length 4
//   ]

//   const { flatTokens, maxLen, batchSize } = createPaddedBatch(prompts, 0)

//   assert.strictEqual(maxLen, 4)
//   assert.strictEqual(batchSize, 2)
//   assert.strictEqual(flatTokens.length, 8)

//   // Row 0 should be [0, 0, 1, 2]
//   assert.strictEqual(flatTokens[0], 0)
//   assert.strictEqual(flatTokens[1], 0)
//   assert.strictEqual(flatTokens[2], 1)
//   assert.strictEqual(flatTokens[3], 2)

//   // Row 1 should be [1, 2, 3, 4]
//   assert.strictEqual(flatTokens[4], 1)
//   assert.strictEqual(flatTokens[7], 4)
// })

// test('configFrom filters only allowed keys', (t) => {
//   const { configFrom } = require('../index.js')
//   const raw = { temperature: 0.7, unknownKey: 'voldemort', maxTokens: 100 }
//   const json = configFrom(raw)
//   const parsed = JSON.parse(json)

//   assert.strictEqual(parsed.temperature, 0.7)
//   assert.strictEqual(parsed.maxTokens, 100)
//   assert.strictEqual(parsed.unknownKey, undefined)
// })
