#!/bin/bash
set -e

echo "🧹 Cleaning old caches..."
rm -rf .build

echo "🔨 Building Swift Core (Pure Static)..."
swift build -c release

NODE_INCLUDE=$(node -e "const path=require('path'); console.log(path.join(path.dirname(process.execPath), '..', 'include', 'node'))")

# Find the exact static library file
LIB_PATH=$(find .build -name "libMLXNative.a" | grep "release" | head -n 1)

if [ -z "$LIB_PATH" ]; then
    echo "❌ Error: Static library not found!"
    exit 1
fi

echo "🔗 Linking Node-API Bridge (Forced Static)..."

# By passing "$LIB_PATH" directly (instead of -lMLXNative),
# Clang is FORCED to embed the static library into the .node file.
clang -O3 -shared \
    -I"$NODE_INCLUDE" \
    binding.c \
    "$LIB_PATH" \
    -o ../mlx_swift.node \
    -undefined dynamic_lookup \
    -L/usr/lib/swift \
    -mmacosx-version-min=14.0

# fixme: mlx_swift.node should rather be called bridge.node

echo "✅ Node Bridge Linked Statically!"
