# MLX Node 🚀
The foundational ecosystem for bringing Apple's **MLX** machine learning framework to **Node.js**. Run Large Language Models, Vision Models, and AI agents directly on Apple Silicon (M1/M2/M3/M4) with zero-overhead native bindings.

## Architecture
This monorepo is divided into two distinct layers to provide both maximum performance and maximum developer experience:
1. **Bridges (`/bridges`)**: Low-level, pre-compiled C/C++/Swift Node-API binaries. These interface directly with Apple's Metal GPU APIs.
2. **Packages (`/packages`)**: High-level, ergonomic JavaScript libraries. These provide the APIs you actually want to use (like `generate()`).
3. **Meta (`/meta`)**: High-level, legacy JavaScript libraries. These provide access to the same APIs packages do and technically act as a proxy towards other packages (like `export * from 'mlx-lm'`).

> [!NOTE]
> If you just want to generate text in your Node app, you want to install [**`mlx-lm`**](./packages/lm/readme.md).

## Project Roadmap

### Phase 1: Core Reliability (Current)
- [x] Zero-dependency pre-compiled N-API binaries.
- [x] Automatic `.metallib` shader discovery & binding.
- [x] Synchronous Swift-to-V8 threading.
- [ ] Dynamic C-string buffers for massive context windows (>32k).
- [ ] Advanced generation configs (stop sequences, logit bias).

### Phase 2: Usability
- [ ] Implement `[{role: "user", content: "..."}]` Chat Templates.
- [ ] Provide Model Metadata (vocab size, context length).
- [ ] Stream Backpressure handling for heavily loaded Node event loops.
- [ ] GGUF format support.

### Phase 3: Developer Experience & Tooling
- [ ] Full TypeScript definitions (`index.d.ts`).
- [ ] CLI tools to download models directly from HuggingFace.
- [ ] Progress callbacks for large model loading.

### Phase 4: Ecosystem Expansion
- [ ] `mlx-embed` - Text Embeddings API.
- [ ] `mlx-audio` - Whisper integration for Speech-to-Text.
- [ ] `mlx-image` - Diffusion model integration.

## License
WTFNMFPL
