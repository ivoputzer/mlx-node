import { describe, it } from 'node:test'
import { ok, strictEqual } from 'node:assert'

import mlx from 'mlx-node/swift'
import { MLXMetrics } from 'mlx-node'

describe('MLXMetrics', () => {
  it('parses system metrics', ({ mock }) => {
    mock.method(mlx, 'systemMetrics', () => JSON.stringify({
      active: 1024,
      cache: 2048,
      peak: 4096,
      memoryLimit: 10240,
      cacheLimit: 5120
    }))

    const metrics = MLXMetrics.fromSnapshot()

    strictEqual(metrics.active, 1024)
    strictEqual(metrics.total, 3072) // 1024 + 2048
    strictEqual(metrics.usage, 3072 / 10240)
    strictEqual(metrics.usagePercent, (3072 / 10240) * 100)

    const str = metrics.toString()
    ok(str.includes('1 KB')) // 1024 bytes -> 1 KB
    ok(str.includes('30%')) // (3072/10240)*100
  })
})
