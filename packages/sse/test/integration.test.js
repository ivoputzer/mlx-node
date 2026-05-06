import { describe, it } from 'node:test'
import { strictEqual, ok } from 'node:assert/strict'

import http from 'node:http'

const host = 'localhost'
const port = 8080
const url = `http://${host}:${port}`
const user = 'test'
const model = 'MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-8bit' // Change this if your server requires a specific model name

describe('mlx-sse', () => {
  describe('POST /v1/chat/completions', () => {
    it('successfully handles a single-turn text request', async () => {
      const messages = [{ role: 'user', content: 'Say hello!' }]
      const res = await request('/v1/chat/completions', { model, user, messages })

      strictEqual(res.status, 200, 'Expected HTTP 200')
      strictEqual(res.data.object, 'chat.completion', 'Expected standard object type')
      ok(res.data.choices[0].message.content, 'Expected a text response')
      strictEqual(res.data.choices[0].message.role, 'assistant')
    })

    it('successfully handles a multi-turn conversation', async () => {
      const messages = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is 2+2?' },
        { role: 'assistant', content: '4' },
        { role: 'user', content: 'And 4+4?' }
      ]
      const res = await request('/v1/chat/completions', { model, user, messages })

      strictEqual(res.status, 200)
      ok(res.data.choices[0].message.content, 'Expected a text response')
    })

    it('responds with 400 Bad Request if messages array is missing or empty', async () => {
      const messages = [] // Invalid per OpenAI spec
      const res = await request('/v1/chat/completions', { model, user, messages })

      ok(res.status === 400 || res.status === 422, `Expected 400 or 422, got ${res.status}`)
      ok(res.data.error, 'Expected an error object in the response body')
    })

    it('successfully streams chunks via SSE when (stream:true)', async () => {
      const messages = [{ role: 'user', content: 'Count to 3.' }]
      const res = await streamRequest('/v1/chat/completions', { model, user, messages })

      strictEqual(res.status, 200)
      ok(res.events.length > 0, 'Should have received at least one SSE chunk')
      strictEqual(res.events[0].object, 'chat.completion.chunk')
      ok(res.isDone, 'Should have received a final data: [DONE] payload')
      const hasDelta = res.events.some(event => event.choices[0].delta !== undefined)
      ok(hasDelta, 'Expected streaming events to contain a "delta" object')
    })

    it('supports the n parameter (batching) returning multiple choices', async () => {
      const messages = [{ role: 'user', content: 'Give me a random color.' }]
      const n = 2 // Test dual branching
      const res = await request('/v1/chat/completions', { user, messages, n })

      strictEqual(res.status, 200)
      strictEqual(res.data.choices.length, n, `Expected exactly ${n} choices back`)

      ok(res.data.choices[0].message.content, 'Choice 0 has content')
      ok(res.data.choices[1].message.content, 'Choice 1 has content')
      strictEqual(res.data.choices[0].index, 0)
      strictEqual(res.data.choices[1].index, 1)
    })

    it('supports the n parameter (batching) when streaming', async () => {
      const messages = [{ role: 'user', content: 'Say A or B.' }]
      const n = 2
      const res = await streamRequest('/v1/chat/completions', { user, messages, n, stream: true })

      strictEqual(res.status, 200)

      const seenIndices = new Set()
      res.events.forEach(e => {
        if (e.choices && e.choices.length > 0) {
          seenIndices.add(e.choices[0].index)
        }
      })

      ok(seenIndices.has(0), 'Stream should contain chunks for index 0')
      ok(seenIndices.has(1), 'Stream should contain chunks for index 1')
    })
  })

  describe('POST /v1/completions', () => {
    it('should successfully handle a standard text completion', async () => {
      const prompt = 'Once upon a time,'
      const res = await request('/v1/completions', { model, prompt, max_tokens: 10 })

      strictEqual(res.status, 200)
      strictEqual(res.data.object, 'text_completion')
      ok(res.data.choices[0].text !== undefined, 'Expected "text" in choices')
    })

    it('should return 400 Bad Request if prompt is missing', async () => {
      const res = await request('/v1/completions', { model })

      ok(res.status === 400 || res.status === 422, `Expected 400 or 422, got ${res.status}`)
      ok(res.data.error, 'Expected an error object in the response body')
    })

    it('should stream legacy completions via SSE successfully', async () => {
      const prompt = 'The numbers 1 to 3 are:'
      const res = await streamRequest('/v1/completions', { model, prompt, max_tokens: 10 })

      strictEqual(res.status, 200)
      ok(res.events.length > 0, 'Should have received at least one SSE chunk')
      strictEqual(res.events[0].object, 'text_completion')
      ok(res.isDone, 'Should have received a final data: [DONE] payload')
      ok(res.events[0].choices[0].text !== undefined, 'Chunks should contain "text" property')
    })
  })
})

function request (path, payload, options = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(`${url}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer test-key', ...options.headers }
    }, (res) => {
      const body = []
      res.on('data', data => {
        body.push(data)
      })
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, headers: res.headers, data: JSON.parse(Buffer.concat(body).toString('utf8')) })
        } catch {
          resolve({ status: res.statusCode, headers: res.headers, data: Buffer.concat(body).toString('utf8') }) // Body might not be JSON (Legacy, SSE, 404 HTML pages)
        }
      })
    })
    req.on('error', reject)

    if (payload) {
      req.write(JSON.stringify(payload))
    }
    req.end()
  })
}

function streamRequest (path, payload, options = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(`${url}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Bearer test-key',
        ...options.headers
      }
    }, (res) => {
      const events = []
      let buffer = ''
      let isDone = false

      res.on('data', chunk => {
        buffer += chunk.toString()
        // SSE chunks are separated by double newlines or single newlines
        const lines = buffer.split('\n')
        buffer = lines.pop() // Keep the last incomplete line in the buffer

        for (let line of lines) {
          line = line.trim()
          if (line.startsWith('data:')) {
            const dataStr = line.slice(5).trim()
            if (dataStr === '[DONE]') {
              isDone = true
            } else if (dataStr) {
              try {
                events.push(JSON.parse(dataStr))
              } catch (e) {
                // Some servers might send non-JSON comments, ignore them
              }
            }
          }
        }
      })

      res.on('end', () => {
        resolve({ status: res.statusCode, headers: res.headers, events, isDone })
      })
    })

    req.on('error', reject)

    if (payload) {
      req.write(JSON.stringify({ ...payload, stream: true }))
    }
    req.end()
  })
}
