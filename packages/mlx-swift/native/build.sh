#!/bin/bash
set -e

# Build the static library via CLI (this part always worked fast for you)
swift build -c release

NODE_INCLUDE=$(node -e "const path=require('path'); console.log(path.join(path.dirname(process.execPath), '..', 'include', 'node'))")
LIB_PATH=$(find .build -name "libMLXNative.a" | grep "release" | head -n 1)
LIB_DIR=$(dirname "$LIB_PATH")

echo "🔗 Linking Node-API Bridge..."
clang -O3 -shared \
    -I"$NODE_INCLUDE" \
    -L"$LIB_DIR" \
    -lMLXNative \
    binding.c \
    -o ../mlx_swift.node \
    -undefined dynamic_lookup \
    -Wl,-rpath,"$LIB_DIR" \
    -L/usr/lib/swift \
    -mmacosx-version-min=14.0

echo "✅ Node Bridge Linked!"
