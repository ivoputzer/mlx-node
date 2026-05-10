import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const mlx = require('mlx-swift')

// Extract N-API methods into a mutable JS object
export default Object.fromEntries(
  Object.getOwnPropertyNames(mlx).map(key => [key, mlx[key]])
)
