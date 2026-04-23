# mlx-node

The primary Native Bridge aggregator for Apple's MLX framework on Node.js.

`mlx-node` acts as the router between JavaScript and the native Apple Silicon GPU drivers. It automatically resolves and exports the best available low-level bridge (`mlx-swift` or `mlx-cpp`) for your environment.

> **Note to Developers:**
> Unless you are building custom AI framework tooling, you probably don't want to use this package directly. For generating text or running models, use [**`mlx-lm`**](https://www.npmjs.com/package/mlx-lm) instead!

## Features
- Universal entry point for MLX on Node.js.
- Apple Silicon (arm64) native acceleration.
- Zero-overhead N-API exports.
