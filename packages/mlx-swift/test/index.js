import native from '../index.js'

console.log(process.env)

console.log(native.loadModel)
console.log(native.generate)

const loaded = native.loadModel('/Users/ivoputzer/github/models/Jackrong/MLX-Qwopus3.5-9B-v3-8bit')

if (loaded) {
  console.log('Generating...')
  const result = native.generate('Why is Swift better than Rust for MLX on MacOS?')
  console.log('Result:', result)
} else {
  console.log('Failed to load model.')
}
