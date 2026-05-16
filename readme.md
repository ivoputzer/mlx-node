[![build](https://img.shields.io/github/actions/workflow/status/ivoputzer/mlx-node/ci.yml?label=build&style=flat-square)](https://github.com/ivoputzer/mlx-node/actions/workflows/ci.yml)
[![style](https://img.shields.io/badge/style-standard-brightgreen.svg?style=flat-square)](http://standardjs.com/)
[![version](https://img.shields.io/npm/v/mlx-node?style=flat-square&colorB=007EC6)]()
[![node](https://img.shields.io/badge/node-lts-blue.svg?style=flat-square)](https://nodejs.org/en/about/previous-releases)
[![license](https://img.shields.io/badge/license-WTFNMFPL-blue.svg?style=flat-square)](https://spdx.org/licenses/WTFNMFPL)

# MLX Node
The foundational ecosystem for bringing Apple's **MLX** machine learning framework to **Node.js**. Run Large Language Models, Vision Models, and AI agents directly on Apple Silicon (M1/M2/M3/M4) with zero-overhead native bindings.

> [!NOTE]
> **If you just want to generate text in your app, you probably want to use [`mlx-lm`](./packages/lm/readme.md)**

## Architecture
This monorepo is divided into distinct layers to provide both maximum performance and maximum developer experience:
- **Bridges** Low-level, pre-compiled C/C++/Swift Node-API binaries interfacing directly with Apple's Metal GPU APIs.
- **Packages** High-level, ergonomic JavaScript libraries (like `generate()`, `stream()`, etc).
- **Meta** Legacy JavaScript libraries and proxy packages for ecosystem compatibility.
- **Roadmap** Future-facing modules (Vision, Audio, etc.) currently in development.

## Setup
```bash
npm install mlx-lm # This installs mlx-swift, mlx-node, and mlx-lm
```

## Basic Usage
Just a sneak peek. Full documentation and the more examples live in [**`mlx-lm`**](./packages/lm/readme.md).

### Standard Generation
```js
import { generate, loadModel } from 'mlx-lm'

// Automatic resource management (Explicit Resource Management)
using model = await loadModel('path/to/your/model')

// Simple string prompt
const response = await generate(model, 'Count to 10', { maxTokens: 100 })
console.log(response.text, response.stats)

// Model's native chat template
const response = await generate(model, { messages:[{ role: 'user', content: 'Count to 10' }] }, { temperature: 0.75 })
console.log(response.text, response.stats)

// Custom Jinja template context (ie. ChatML)
const response = await generate(model, { text: 'Count to 10' }, { template: '<|im_start|>user\n{{ text | trim }}<|im_end|>\n<|im_start|>assistant' })
console.log(response.text, response.stats)
```

### Streaming Conversations
```js
import { stdout } from 'node:process'
import { stream, loadModel } from 'mlx-lm'

using model = await loadModel('path/to/your/model')
const messages = [
  { role: 'system', content: 'You are a helpful math tutor.' },
  { role: 'user', content: 'What is 5 + 5?' },
  { role: 'assistant', content: '5 + 5 is 10.' },
  { role: 'user', content: 'And what is that times 2?' }
]

for await (const { done, text, stats } of stream(model, { messages })) {
  if (done) {
    console.log(stats)
  } else {
    stdout.write(text)
  }
}
```

### Advanced: Plugins (Reasoning & Tool Calls)
```js
import { stdout } from 'node:process'
import { styleText } from 'node:util'

import { stream, loadModel } from 'mlx-lm'
import { toolCalls, reasoningContent } from 'mlx-lm/plugins'

using model = await loadModel('path/to/your/model')

const controller = new AbortController()
const tools = [/* */]

for await (const event of stream(model, { messages, tools }, { signal: controller.signal }, [reasoningContent, toolCalls])) {
  if (event.done) {
    console.log(event.text, event.reasoningContent, event.toolCalls)
  } else {
    const { text, isReasoning, hasToolCalls } = event
    stdout.write(
      styleText(isReasoning ? 'dim' : 'yellow', text)
    )
  }
}
```

### Advanced: Batching/ABTesting
```js
import { stdin, stdout } from 'node:process'
import { createInterface } from 'node:readline/promises'

import { stream, loadModel } from 'mlx-lm'
import { printBatchHeaders, createBatchRenderer } from 'mlx-cli/helpers'

const batchSize = 2

using model = await loadModel('path/to/your/model')
const readline = createInterface({ input: stdin, output: stdout })

const messages = [
  { role: 'system', content: 'You are a helpful, very brief AI.' },
  { role: 'user', content: await readline.question('> ') }
]

const buffer = Array(batchSize).fill('')
const render = createBatchRenderer(batchSize, { readline })

printBatchHeaders(batchSize, {titles: Array.from({ length: batchSize }, (_, i) => `Branch ${1 + i}`)})

for await (const batches of stream(model, { messages }, { batchSize, temperature: 0.8 }, [reasoningContent, toolCalls])) {
  // In batch mode, the stream yields an array of events
  batches.forEach(({ text }, i) => buffer[i] += text)
  render(buffer)
}

render(buffer, true)
```

## Contributing
Contributing to a native Apple Silicon project shouldn't require you to sacrifice disk space to Xcode unless you're actually touching the core. The workflow is meant to be modular. If you are focusing on packages, you can "Go Lite" by pulling our pre-compiled binaries. If you're here to optimize bridges, you can "Go Full" and rebuild the entire stack from source.

#### Using GitHub CLI (Recommended)
```bash
gh repo clone ivoputzer/mlx-node

gh run download --name mlx.node --dir bridges/swift
gh run download --name default.metallib --dir bridges/swift

npm install
npm test --workspaces --if-present
```

#### Without GitHub CLI
Unlike `git clone`, the Artifacts API requires a valid `GITHUB_TOKEN` exported in your environment.
```bash
git clone https://github.com/ivoputzer/mlx-node.git

fetch_prebuilt () {
  URL=$(curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
    "https://api.github.com/repos/ivoputzer/mlx-node/actions/artifacts?name=$1&per_page=1" \
    | jq -r '.artifacts[0].archive_download_url')

  if [ "$URL" != "null" ]; then
    echo "Downloading $1..."
    curl -L -H "Authorization: Bearer $GITHUB_TOKEN" -o "$1.zip" "$URL"
    unzip -o "$1.zip" -d bridges/swift/ && rm "$1.zip"
  else
    echo "Artifact $1 not found. Check your GITHUB_TOKEN."
  fi
}

fetch_prebuilt  "mlx.node"
fetch_prebuilt  "default.metallib"

npm install
npm test --workspaces --if-present
```

#### Build the project manually
Needed if you plan on working on `bridges/swift/**`. Full Xcode IDE (not just Command Line Tools) required.

```bash
xcode-select --install
sudo xcodebuild -license accept
sudo xcode-select --print-path

# If path is still /Library/Developer/CommandLineTools you have to switch
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

# Verify you've got everything up and running
clang --version
swift --version
xcodebuild -version

npm run build --workspaces --if-present

npm install
npm test --workspaces --if-present
```

## License
[WTFNMFPL](https://spdx.org/licenses/WTFNMFPL)
