mlx-swift
---

The foundational Node.js bridge to Apple's MLX framework via Swift.

This package contains the pre-compiled C/Swift binaries and Metal GPU shaders required to execute MLX computation graphs on Apple Silicon. It is distributed as a fully standalone `.node` binary bundle.

## How it works
This package statically links Apple's `mlx-swift` libraries into a highly optimized Node-API C bridge. It manages memory safety across the Swift/V8 boundary and handles multi-threaded inference streams safely within the Node.js event loop.

<!--
  FIXME
  ADD A NOTE THAT THE USERS MOST LIKELY WANT TO USE MLX-LM INSTEAD 🤣
-->

*Internal package. Do not consume directly. Use [mlx-lm](https://www.npmjs.com/package/mlx-lm) instead.*
