[![build](https://img.shields.io/github/actions/workflow/status/ivoputzer/mlx-node/ci.yml?label=build&style=flat-square)](https://github.com/ivoputzer/mlx-node/actions/workflows/ci.yml)
[![style](https://img.shields.io/badge/style-standard-brightgreen.svg?style=flat-square)](http://standardjs.com/)
[![version](https://img.shields.io/npm/v/mlx-node?style=flat-square&colorB=007EC6)]()
[![node](https://img.shields.io/badge/node-lts-blue.svg?style=flat-square)](https://nodejs.org/en/about/previous-releases)
[![license](https://img.shields.io/badge/license-WTFNMFPL-blue.svg?style=flat-square)](https://spdx.org/licenses/WTFNMFPL)



# MLX Node 🚀
The foundational ecosystem for bringing Apple's **MLX** machine learning framework to **Node.js**. Run Large Language Models, Vision Models, and AI agents directly on Apple Silicon (M1/M2/M3/M4) with zero-overhead native bindings.

> [!NOTE]
> If you just want to generate text in your Node app, you want to install [**`mlx-lm`**](./packages/lm/readme.md).

## Setup
```
npm install mlx-lm
```

## Baisc Usage
```js
import { generate, loadModel } from 'mlx-lm'

using model = loadModel('path/to/your/model')

const response = await generate(model, 'Count to 10', { maxTokens: 100 })
console.log(response.text, respose.stats)

const response = await generate(model, { text: 'Count to 10' }, { template: '<|im_start|>user\n{{ text | trim }}<|im_end|>\n<|im_start|>assistant' }) // Jinja
console.log(response.text, respose.stats)
```

```js
import { stream, loadModel } from 'mlx-lm'

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
    process.stdout.write(text)
  }
}
```

```js
import { stream, loadModel } from 'mlx-lm'
import { toolCalls, reasoningContent } from 'mlx-lm/plugins'

const controller = new AbortController()
const tools = [/* */]

for await (const event of stream(model, { messages, tools }, { signal: controller.signal, maxTokens: 512 }), [reasoningContent, toolCalls]) {
  if (event.done) {
    console.log(event.text, event.reasoningContent, event.toolCalls)
  } else {
    const {text, isReasoning, hasToolCalls } = event
    process.stdout.write( 
      styleText(isReasoning ? 'dim' : 'yellow', text) 
    )
  }
}
```

## Architecture
This monorepo is divided into distinct layers to provide both maximum performance and maximum developer experience:
- **Bridges** Low-level, pre-compiled C/C++/Swift Node-API binaries. These interface directly with Apple's Metal GPU APIs.
- **Packages** High-level, ergonomic JavaScript libraries. These provide the APIs you actually want to use (like `generate()`).
- **Meta** High-level, legacy JavaScript libraries. These provide access to the same APIs packages do and technically act as a proxy towards other packages.
- **Roadmap** Packages we havent had time working on yet...

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


<!--
## Project Roadmap

#### Phase 1: Core Reliability (Current)
- [x] Zero-dependency pre-compiled N-API binaries.
- [x] Automatic `.metallib` shader discovery & binding.
- [x] Synchronous Swift-to-V8 threading.
- [x] ~~Dynamic C-string buffers for massive context windows (>32k).~~ Int32Array
- [x] Advanced `GenerationConfig` (stop sequences, logit bias).
- [x] Robust Error Propagation with Swift stack traces.
- [x] Unified build pipeline for GitHub Actions.


#### Phase 2: Usability
- [x] Implement `[{role: "user", content: "..."}]` Chat Templates (OpenAI compatible?).
- [ ] Provide Model Metadata (vocab size, context length).
- [x] Stream Backpressure handling for heavily loaded Node event loops.
- [ ] GGUF format support in `mlx-lm` (NTH).
- [ ] Configurable MLX Logging levels.
- [x] `AbortController` support for killing active inference.

#### Phase 3: Developer Experience & Tooling
- [ ] TypeScript definitions (`index.d.ts`) or jsdoc.
- [ ] CLI tools to download models directly from HuggingFace.
- [ ] Progress callbacks for large model loading.

<!--

### Phase 3.5: Developer Tools
- [ ] `mlx-server` - Cli to run an OpenAI-Like endpoint
- [ ] `mlx-agent` - Agent parser (Agent = Model+Prompt+Tools+Loop -> Wrapper for an agent function definition)
  - Pipeline
  - Tool
  - Queue
  - Agent
- [ ] `mlx-tool` - Tool parser (Tool = Wrapper/Parser for a function with definitions) <- To be honest this doesnt even make much sense and should be in
- [ ] `mlx-mcp` - Tools that allows you to spin up mcp servers that can be passed to an Agent (directory access, git access, docker, etc)


### Phase 4: Ecosystem Expansion
- [ ] `mlx-embed` - Text Embeddings API.
- [ ] `mlx-audio` - Whisper integration for Speech-to-Text.
- [ ] `mlx-image` - Diffusion model integration.

-->
