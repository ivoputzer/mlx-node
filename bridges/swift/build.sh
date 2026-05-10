#!/bin/bash
set -e

LIB_NAME="MLXBridge"

echo "🧹 Cleaning old caches..."
# rm -rf .build

#########################################################
# Why did Xcode give you the .metallib but no .a?
# Because when Xcode reads a Package.swift, it compiles the .o (object) files and links them directly into its internal cache.
# It doesn't bother zipping them up into a standalone .a static archive unless you explicitly create a full Xcode Project file 🚀

# echo "🔨 Building XCodeBuild (forces bundle for default.metallib)..."
# xcodebuild build -scheme $LIB_NAME -destination 'generic/platform=macOS' -derivedDataPath ./.build/xcode CONFIGURATION=Release

# echo "📦 Extracting default.metallib from Xcode Bundle..."
# METALLIB_PATH=$(find ./.build/xcode -name "default.metallib" | head -n 1)

# if [ -n "$METALLIB_PATH" ]; then
#   cp "$METALLIB_PATH" ./default.metallib
#   echo "✅ default.metallib extracted successfully!"
# else
#   echo "❌ Error: default.metallib not found!"
#   exit 1
# fi

# Up until here should be cached in the CI
#########################################################

# So swift build gives you the .a but no metal. xcodebuild gives you the metal but no .a 🙈 🔫
echo "🔨 Building Swift Core..."
swift build -c release

# Let's find the static library in the correct output folder
LIB_PATH=$(find ./.build -name "lib$LIB_NAME.a" | grep "release" | head -n 1)

if [ -z "$LIB_PATH" ]; then
  echo "❌ Error: Static library not found!"
  exit 1
fi

NODE_INCLUDE=$(node -pe "path.join(process.execPath, '..', '..', 'include', 'node')")
NPM_PACKAGE_VERSION=$(node -p "require('./package.json').version")
echo "📦 Package version identified as: $NPM_PACKAGE_VERSION"

echo "🔗 Linking Node-API Bridge (Forced Static)..."
clang -O3 -shared -DNPM_PACKAGE_VERSION="\"$NPM_PACKAGE_VERSION\"" -I"$NODE_INCLUDE" binding.c "$LIB_PATH" -o ./mlx.node -undefined dynamic_lookup -L/usr/lib/swift -mmacosx-version-min=14.0
