import { createRequire } from 'node:module'
import { join } from 'node:path'

const require = createRequire(import.meta.url)

const metal = join(import.meta.dirname, 'default.metallib')
const binding = join(import.meta.dirname, 'mlx_swift.node')

process.env.MLX_METAL_PATH = metal
export default require(binding)
