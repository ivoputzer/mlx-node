import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const native = require('mlx-swift')

export const loadModel = native.loadModel // fixme: load
export const unloadModel = native.unloadModel // fixme: unload

export const generate = native.generate // fixme: generate
export const generateStream = native.generateStream // fixme: stream

export default native
