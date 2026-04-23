#!/usr/bin/env node

import { spawn } from 'node:child_process'
import { rm, copyFile, readdir, access } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { parseArgs, styleText } from 'node:util'
import { execPath } from 'node:process'

const NODE_DIR = resolve(import.meta.dirname, '..')
const SWIFT_DIR = join(NODE_DIR, 'native')
const NODE_INCLUDE_PATH = join(execPath, '..', '..', 'include', 'node')

const { values } = parseArgs({
  options: {
    swift: { type: 'boolean', default: true },
    metal: { type: 'boolean', default: true },
    clean: { type: 'boolean', default: true }
  },
  allowNegative: true
})

try {
  console.log(styleText(['bold', 'cyan'], '\n🚀 Starting MLX Bridge Build Pipeline...\n'))

  const XCODE_DERIVED_DATA_PATH = join(SWIFT_DIR, 'DerivedData')


  if (values.clean) {
    console.log(styleText('yellow', '🧹 Cleaning old caches...'))

    await rm(join(SWIFT_DIR, '.build'), { recursive: true, force: true })
    await rm(XCODE_DERIVED_DATA_PATH, { recursive: true, force: true })
  }

  if (values.metal) {
    // When Xcode reads a Package.swift it compiles the .o (object) files and links them directly into its internal cache thus it gives us the .metallib but no .a
    // It doesn't bother zipping them up into a standalone .a static archive unless you explicitly create a full Xcode Project file which could be a future idea 🚀
    console.log(styleText(['bold', 'magenta'], '\n🍎 [XCode Override] Initiating the Metallib Heist...'))

    await runCommand('xcodebuild', ['build', '-scheme', 'MLXBridge', '-destination', 'generic/platform=macOS', '-derivedDataPath', './DerivedData', 'CONFIGURATION=Release'], SWIFT_DIR)

    console.log(styleText('yellow', '\n📦 Locating default.metallib...'))
    const metalLibPath = await findFile(XCODE_DERIVED_DATA_PATH, 'default.metallib')

    if (!metalLibPath) {
      throw new Error('Failed to extract default.metallib from Xcode! Apple wins this round.')
    }

    await copyFile(metalLibPath, join(NODE_DIR, 'default.metallib'))
    console.log(styleText('green', '✅ Successfully kidnapped default.metallib and moved it to root!'))
  }

  if (values.swift) {
    console.log(styleText(['bold', 'blue'], '\n🔨 Building Swift Core (libMLXBridge.a)...'))
    await runCommand('swift', ['build', '-c', 'release'], SWIFT_DIR)
  }

  console.log(styleText('yellow', '\n🔍 Locating compiled static library...'))
  const staticLibPath = await findFile(join(SWIFT_DIR, '.build'), 'libMLXBridge.a')

  if (!staticLibPath) {
    throw new Error('libMLXBridge.a not found! Swift build failed silently.')
  }
  console.log(styleText('green', `✅ Found: ${staticLibPath}`))

  // 4. Link Node-API Bridge (C -> Swift)
  console.log(styleText(['bold', 'blue'], '\n🔗 Compiling and Linking Node-API Bridge...'))
  const outputPath = join(NODE_DIR, 'mlx.node')
  await runCommand('clang', ['-O3', '-shared', `-I${NODE_INCLUDE_PATH}`, 'binding.c', `${staticLibPath}`, '-o', `${outputPath}`, '-undefined', 'dynamic_lookup', '-L/usr/lib/swift', '-mmacosx-version-min=14.0'], SWIFT_DIR)

  try {
    await access(outputPath)
    console.log(styleText('green', '✅ mlx.node compiled successfully!'))
  } catch {
    throw new Error('mlx.node was not created.')
  }

  try {
    await access(join(NODE_DIR, 'default.metallib'))
    console.log(styleText('green', '✅ default.metallib is present.'))
  } catch {
    console.log(styleText('red', '⚠️  Warning: default.metallib is missing from the root directory!'))
    console.log(styleText('gray', '   Run this script with --add-fucking-metallib if MLX crashes at runtime.'))
  }
  console.log(styleText(['bold', 'green'], '\n🎉 Build Complete! You are ready to run Node.js!\n'))
} catch (err) {
  console.error(styleText(['bold', 'red'], '\n❌ Build Failed:'))
  console.error(styleText('red', err.message || err))
  process.exit(1)
}

// Spawns a child process and streams output directly to the terminal.
function runCommand (command, args, cwd = NODE_DIR) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, stdio: 'inherit' })

    child.on('close', (code) => {
      if (code === 0) resolve()
      else reject(new Error(`Command "${command} ${args.join(' ')}" failed with code ${code}`))
    })
  })
}

// recursively searches a directory for a specific filename.
async function findFile (dir, targetName) {
  try {
    const dirents = await readdir(dir, { withFileTypes: true })
    for (const dirent of dirents) {
      const fullPath = join(dir, dirent.name)
      if (dirent.isDirectory()) {
        const found = await findFile(fullPath, targetName)
        if (found) return found
      } else if (dirent.name === targetName) {
        return fullPath
      }
    }
  } catch (err) {
    // Ignore errors for unreadable directories
  }
  return null
}
