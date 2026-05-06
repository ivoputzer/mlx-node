import http from 'node:http'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { parseArgs, styleText } from 'node:util'
import { MLXModel, MLXCache, MLXMetrics } from 'mlx-swift'
import { loadTokenizer, loadTemplate, stopTokensFrom, padTokenFrom } from 'mlx-lm'

/**
 * 1. CLI ARGUMENT PARSING
 */
const { values } = parseArgs({
  options: {
    model: { short: 'm', type: 'string', multiple: true },
    port: { short: 'p', type: 'string', default: '8080' },
    host: { short: 'h', type: 'string', default: '127.0.0.1' },
    help: { type: 'boolean', default: false }
  }
})

if (values.help || !values.model?.length) {
  console.log(`
Usage: node index.js -m /path/to/model1 -m /path/to/model2
Options:
  -m, --model   Path to MLX model directory (can be used multiple times)
  -p, --port    Port to listen on (default: 8080)
  -h, --host    Host to bind to (default: 0.0.0.0)
  `)
  process.exit(0)
}

/**
 * 2. STATE MANAGEMENT
 */
const MODELS = new Map() // name -> { ref, tokenizer, template, config }
const SESSIONS = new Map() // sessionId -> { cache, lastTokens, modelName }

/**
 * 3. TOOL PARSING LOGIC
 */
const parseToolCall = (text) => {
  // Try XML Format (Qwen/Claude style)
  const xmlMatch = text.match(/<function=(.*?)>([\s\S]*?)<\/function>/)
  if (xmlMatch) {
    const args = {}
    const paramRegex = /<parameter=(.*?)>\n?([\s\S]*?)\n?<\/parameter>/g
    let m
    while ((m = paramRegex.exec(xmlMatch[2])) !== null) args[m[1]] = m[2].trim()
    return { name: xmlMatch[1], arguments: JSON.stringify(args) }
  }
  // Try JSON Format
  try {
    const jsonMatch = text.match(/<tool_call>\s*(\{.*?\})\s*<\/tool_call>/s)
    if (jsonMatch) return JSON.parse(jsonMatch[1])
  } catch (e) { return null }
  return null
}

/**
 * 4. TOKEN DIFFING & CACHE OPTIMIZATION
 */
async function syncCacheAndGetTokens (session, currentTokens, M) {
  const old = session.lastTokens
  const curr = currentTokens

  // Initial case
  if (!old) return { tokens: curr, trimCount: 0 }

  // Find divergence point
  let sharedIdx = 0
  while (sharedIdx < old.length && sharedIdx < curr.length && old[sharedIdx] === curr[sharedIdx]) {
    sharedIdx++
  }

  // Case A: Perfect Continuation
  if (sharedIdx === old.length) {
    const eos = M.tokenizer.config.eos_token_id ?? M.tokenizer.config.eot_token_id
    // Prepend EOS to "close" the previous assistant mouth in the cache
    return { tokens: new Int32Array([eos, ...curr.slice(sharedIdx)]), trimCount: 0 }
  }

  // Case B: Divergence (User edit / Time Travel)
  if (session.cache.isTrimmable) {
    const trimCount = old.length - sharedIdx
    return { tokens: curr.slice(sharedIdx), trimCount }
  } else {
    // Mamba/Non-trimmable cache: Must reset
    session.cache.dispose()
    session.cache = MLXCache.fromModel(M.ref)
    return { tokens: curr, trimCount: 0 }
  }
}

/**
 * 5. SSE STREAMING HANDLER
 */
async function * handleRequest (body) {
  const { model: modelName, messages, temperature = 0.7, max_tokens = 1024, user = 'default' } = body

  const M = MODELS.get(modelName)
  if (!M) throw new Error(`Model ${modelName} not loaded`)

  // Session sticky to model
  if (!SESSIONS.has(user) || SESSIONS.get(user).modelName !== modelName) {
    if (SESSIONS.has(user)) SESSIONS.get(user).cache.dispose()
    SESSIONS.set(user, { cache: MLXCache.fromModel(M.ref), lastTokens: null, modelName })
  }
  const session = SESSIONS.get(user)

  // Render Template & Tokenize
  const rendered = M.template.render({ messages, add_generation_prompt: true })
  const currentTokens = new Int32Array(M.tokenizer.encode(rendered).ids)

  const { tokens, trimCount } = await syncCacheAndGetTokens(session, currentTokens, M)

  if (trimCount > 0) session.cache.trim(trimCount)

  const reqId = `chatcmpl-${randomUUID()}`
  const generator = session.cache.generate(tokens, {
    batchSize: 1,
    temperature,
    maxTokens: max_tokens,
    stopTokenIds: M.stopTokens,
    padTokenId: M.padToken
  })

  const fullResponseTokens = [...currentTokens]
  let toolBuffer = ''
  let inTool = false

  const format = (delta, finish = null) =>
    `data: ${JSON.stringify({ id: reqId, object: 'chat.completion.chunk', model: modelName, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`

  try {
    for await (const [token] of generator) {
      if (token === -1 || M.stopTokens.includes(token)) break

      fullResponseTokens.push(token)
      const text = M.tokenizer.decode([token], { skip_special_tokens: false })

      if (text.includes('<tool_call')) inTool = true

      if (inTool) {
        toolBuffer += text
        if (text.includes('</tool_call>')) {
          const call = parseToolCall(toolBuffer)
          if (call) yield format({ tool_calls: [{ id: `call_${Date.now()}`, type: 'function', function: call }] })
          inTool = false; toolBuffer = ''
        }
        continue
      }

      yield format({ content: text })
    }
    session.lastTokens = new Int32Array(fullResponseTokens)
  } finally {
    yield format({}, 'stop')
    yield 'data: [DONE]\n\n'
  }
}

/**
 * 6. INITIALIZATION & SERVER
 */
(async () => {
  console.log(styleText('cyan', 'Initializing MLX Models...'))

  for (const modelPath of values.model) {
    const name = path.basename(path.resolve(modelPath))
    const ref = await MLXModel.fromPath(modelPath)
    const tokenizer = await loadTokenizer(modelPath)
    const template = await loadTemplate(modelPath)

    MODELS.set(name, {
      ref,
      tokenizer,
      template,
      stopTokens: stopTokensFrom(tokenizer),
      padToken: padTokenFrom(tokenizer)
    })
    console.log(styleText('green', `LOADED: ${name} (${modelPath})`))
  }

  const server = http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Headers', '*')

    if (req.method === 'OPTIONS') return res.writeHead(204).end()

    if (req.url === '/v1/chat/completions' && req.method === 'POST') {
      let body = ''
      for await (const chunk of req) body += chunk

      try {
        const parsed = JSON.parse(body)
        res.writeHead(200, { 'Content-Type': 'text/event-stream' })
        for await (const sse of handleRequest(parsed)) res.write(sse)
        return res.end()
      } catch (error) {
        console.error(error)
        return res.end(JSON.stringify({ error: error.message }))
      }
    }

    if (req.url === '/v1/models') {
      return res.end(JSON.stringify({ data: Array.from(MODELS.keys()).map(id => ({ id, object: 'model' })) }))
    }

    res.writeHead(404).end()
  })

  // Simple GC based on memory pressure
  setInterval(() => {
    const metrics = MLXMetrics.fromSnapshot()
    if (metrics.usage > 0.85 && SESSIONS.size > 0) {
      const oldestId = SESSIONS.keys().next().value
      const s = SESSIONS.get(oldestId)
      s.cache.dispose()
      SESSIONS.delete(oldestId)
      console.log(styleText('yellow', `Memory Pressure: Evicted session ${oldestId}`))
    }
  }, 5000)

  server.listen(parseInt(values.port), values.host, () => {
    console.log(styleText('bold', `\n🚀 MLX Gateway ready at http://${values.host}:${values.port}`))
    console.log(`Available Models: ${Array.from(MODELS.keys()).join(', ')}`)
  })
})()
