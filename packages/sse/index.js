import http from 'node:http'
import fs from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto' // Fix 1: Corrected default import
import { parseArgs } from 'node:util'

import { MLXModel, MLXCache, MLXMetrics } from 'mlx-swift'
import { loadTokenizer, loadTemplate, stopTokensFrom, padTokenFrom } from 'mlx-lm'

// ============================================================================
// CONFIGURATION & CLI ARGUMENTS
// ============================================================================

const { values: args } = parseArgs({
  options: {
    port: { type: 'string', short: 'p', default: '8080' },
    host: { type: 'string', short: 'h', default: '127.0.0.1' },
    model: { type: 'string', short: 'm', multiple: true, default: [] }
  }
})

const CACHE_DIR = path.join(process.cwd(), '.mlx_sessions')
await fs.mkdir(CACHE_DIR, { recursive: true }).catch(() => {})

const MODELS = new Map()
const SESSIONS = new Map()

// ============================================================================
// TOOL PARSERS
// ============================================================================

const ToolParsers = {
  xml: (raw) => {
    try {
      const nameMatch = raw.match(/"name":\s*"([^"]+)"/)
      if (nameMatch) {
        const jsonMatch = raw.match(/{[\s\S]*}/)
        const parsed = JSON.parse(jsonMatch[0])
        return { name: parsed.name, arguments: JSON.stringify(parsed.arguments || {}) }
      }

      const funcMatch = raw.match(/<function=(.*?)>/)
      if (funcMatch) {
        const args = {}
        const paramRegex = /<parameter=(.*?)>\n?([\s\S]*?)\n?<\/parameter>/g
        let m
        while ((m = paramRegex.exec(raw)) !== null) args[m[1]] = m[2].trim()
        return { name: funcMatch[1], arguments: JSON.stringify(args) }
      }
    } catch { return null }
  }
}

// ============================================================================
// SESSION & CACHE MANAGEMENT
// ============================================================================

async function getOrCreateSession (sessionId, modelConfig) {
  const { model, name: modelName } = modelConfig
  const sessionKey = crypto.createHash('sha256').update(`${sessionId}_${modelName}`).digest('hex').slice(0, 16)
  const diskPath = path.join(CACHE_DIR, sessionKey)

  if (SESSIONS.has(sessionKey)) {
    const session = SESSIONS.get(sessionKey)
    session.lastAccess = Date.now()
    return session
  }

  try {
    await fs.access(`${diskPath}.safetensors`)
    const cache = await MLXCache.fromPath(`${diskPath}.safetensors`, model)
    const meta = JSON.parse(await fs.readFile(`${diskPath}.meta.json`, 'utf8'))
    const session = { cache, historyTokens: meta.historyTokens, lastAccess: Date.now(), diskPath }
    SESSIONS.set(sessionKey, session)
    return session
  } catch {
    const cache = MLXCache.fromModel(model)
    const session = { cache, historyTokens: [], lastAccess: Date.now(), diskPath }
    SESSIONS.set(sessionKey, session)
    return session
  }
}

async function saveSessionToDisk (session) {
  if (!session || !session.cache.available) return
  await session.cache.save(`${session.diskPath}.safetensors`)
  await fs.writeFile(`${session.diskPath}.meta.json`, JSON.stringify({ historyTokens: session.historyTokens }))
}

// ============================================================================
// CORE GENERATION
// ============================================================================

async function * generateMLXStream (session, fullPromptString, modelConfig, options) {
  const { tokenizer, model, stopTokens, padTokenId } = modelConfig
  const targetTokens = tokenizer.encode(fullPromptString).ids

  let commonPrefixLen = 0
  while (
    commonPrefixLen < session.historyTokens.length &&
    commonPrefixLen < targetTokens.length &&
    session.historyTokens[commonPrefixLen] === targetTokens[commonPrefixLen]
  ) {
    commonPrefixLen++
  }

  const tokensToTrim = session.historyTokens.length - commonPrefixLen
  if (tokensToTrim > 0) {
    if (session.cache.isTrimmable) session.cache.trim(tokensToTrim)
    else {
      session.cache.dispose()
      session.cache = MLXCache.fromModel(model)
      commonPrefixLen = 0
    }
  }

  session.historyTokens = Array.from(targetTokens)
  const newTokens = new Int32Array(targetTokens.slice(commonPrefixLen))

  const n = options.n || 1
  const stream = session.cache.generate(newTokens, {
    batchSize: n,
    temperature: options.temperature ?? 0.7,
    maxTokens: options.max_tokens ?? 1024,
    stopTokenIds: stopTokens,
    padTokenId
  })

  const startsThinking = fullPromptString.includes('<think>') && !fullPromptString.includes('</think>')

  const branches = Array.from({ length: n }, (_, i) => ({
    index: i,
    active: true,
    inThinkBlock: startsThinking,
    inToolBlock: false,
    toolBuffer: '',
    toolId: `call_${crypto.randomUUID().slice(0, 8)}`,
    tokens: 0
  }))

  let activeCount = n
  for await (const batches of stream) {
    for (let i = 0; i < n; i++) {
      const branch = branches[i]
      if (!branch.active) continue

      const tokenId = batches[i]
      if (tokenId === -1 || stopTokens.includes(tokenId)) {
        branch.active = false
        activeCount--
        continue
      }

      branch.tokens++
      if (i === 0) session.historyTokens.push(tokenId)

      let textChunk = tokenizer.decode([tokenId], { skip_special_tokens: false })

      if (textChunk.includes('<think>')) { branch.inThinkBlock = true; textChunk = textChunk.replace('<think>', '') }
      if (textChunk.includes('</think>')) { branch.inThinkBlock = false; textChunk = textChunk.replace('</think>', '') }

      if (branch.inThinkBlock && textChunk) {
        yield { branchIndex: i, type: 'reasoning', text: textChunk }
        continue
      }

      if (textChunk.includes('<tool_call>')) {
        branch.inToolBlock = true
        textChunk = textChunk.replace('<tool_call>', '')
        yield { branchIndex: i, type: 'tool_start', id: branch.toolId }
      }

      if (branch.inToolBlock) {
        branch.toolBuffer += textChunk
        if (branch.toolBuffer.includes('</tool_call>')) {
          const parsedTool = ToolParsers.xml(branch.toolBuffer)
          if (parsedTool) yield { branchIndex: i, type: 'tool_finish', id: branch.toolId, tool: parsedTool }
          branch.inToolBlock = false
          branch.toolBuffer = ''
        }
        continue
      }

      if (textChunk) yield { branchIndex: i, type: 'text', text: textChunk }
    }
    if (activeCount === 0) break
  }

  await saveSessionToDisk(session)
  yield { type: 'usage', prompt_tokens: targetTokens.length, completion_tokens: Math.max(...branches.map(b => b.tokens)), total_tokens: targetTokens.length + Math.max(...branches.map(b => b.tokens)) }
}

// ============================================================================
// API ROUTES
// ============================================================================

async function handleChatCompletions (req, res, body) {
  if (!body.messages?.length) return sendError(res, 400, 'messages array is required')

  const modelConfig = MODELS.get(body.model) || Array.from(MODELS.values())[0]
  if (!modelConfig) return sendError(res, 404, 'Model not found')

  const fullPromptString = modelConfig.template.render({ messages: body.messages, tools: body.tools, add_generation_prompt: true })
  const session = await getOrCreateSession(body.user || 'default_user', modelConfig)
  const generator = generateMLXStream(session, fullPromptString, modelConfig, body)
  const requestId = `chatcmpl-${crypto.randomUUID()}`

  if (body.stream) {
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' })
    for await (const chunk of generator) {
      if (chunk.type === 'usage' && body.stream_options?.include_usage) {
        res.write(`data: ${JSON.stringify({ id: requestId, object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000), model: modelConfig.name, choices: [], usage: chunk })}\n\n`)
      } else if (chunk.type !== 'usage') {
        const delta = {}
        if (chunk.type === 'text') delta.content = chunk.text
        if (chunk.type === 'reasoning') delta.reasoning_content = chunk.text
        if (chunk.type === 'tool_start') delta.tool_calls = [{ index: 0, id: chunk.id, type: 'function', function: { name: '', arguments: '' } }]
        if (chunk.type === 'tool_finish') delta.tool_calls = [{ index: 0, id: chunk.id, function: chunk.tool }]

        res.write(`data: ${JSON.stringify({
          id: requestId,
object: 'chat.completion.chunk',
created: Math.floor(Date.now() / 1000),
model: modelConfig.name,
          choices: [{ index: chunk.branchIndex, delta, finish_reason: null }]
        })}\n\n`)
      }
    }
    for (let i = 0; i < (body.n || 1); i++) res.write(`data: {"id":"${requestId}","object":"chat.completion.chunk","choices":[{"index":${i},"delta":{},"finish_reason":"stop"}]}\n\n`)
    res.end('data: [DONE]\n\n')
  } else {
    const branches = Array.from({ length: body.n || 1 }, () => ({ content: '', reasoning_content: '', tool_calls: [] }))
    let usage = null
    for await (const chunk of generator) {
      if (chunk.type === 'usage') usage = chunk
      else if (chunk.type === 'text') branches[chunk.branchIndex].content += chunk.text
      else if (chunk.type === 'reasoning') branches[chunk.branchIndex].reasoning_content += chunk.text
      else if (chunk.type === 'tool_finish') branches[chunk.branchIndex].tool_calls.push({ id: chunk.id, type: 'function', function: chunk.tool })
    }
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({
      id: requestId,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: modelConfig.name,
      usage,
      choices: branches.map((b, index) => ({
        index,
        message: { role: 'assistant', content: b.content, reasoning_content: b.reasoning_content || undefined, tool_calls: b.tool_calls.length ? b.tool_calls : undefined },
        finish_reason: b.tool_calls.length ? 'tool_calls' : 'stop'
      }))
    }))
  }
}

async function handleLegacyCompletions (req, res, body) {
  if (!body.prompt) return sendError(res, 400, 'prompt is required')

  const modelConfig = MODELS.get(body.model) || Array.from(MODELS.values())[0]
  if (!modelConfig) return sendError(res, 404, 'Model not found')

  const session = await getOrCreateSession(body.user || 'default_user', modelConfig)
  const generator = generateMLXStream(session, body.prompt, modelConfig, body)
  const requestId = `cmpl-${crypto.randomUUID()}`

  if (body.stream) {
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' })
    for await (const chunk of generator) {
      if (chunk.type === 'text') {
        res.write(`data: ${JSON.stringify({
          id: requestId,
object: 'text_completion',
created: Math.floor(Date.now() / 1000),
model: modelConfig.name,
          choices: [{ index: chunk.branchIndex, text: chunk.text, finish_reason: null }]
        })}\n\n`)
      }
    }
    for (let i = 0; i < (body.n || 1); i++) res.write(`data: {"id":"${requestId}","object":"text_completion","choices":[{"index":${i},"text":"","finish_reason":"stop"}]}\n\n`)
    res.end('data: [DONE]\n\n')
  } else {
    const branches = Array.from({ length: body.n || 1 }, () => ({ text: '' }))
    let usage = null
    for await (const chunk of generator) {
      if (chunk.type === 'usage') usage = chunk
      else if (chunk.type === 'text') branches[chunk.branchIndex].text += chunk.text
    }
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({
      id: requestId,
      object: 'text_completion',
      created: Math.floor(Date.now() / 1000),
      model: modelConfig.name,
      usage,
      choices: branches.map((b, index) => ({ index, text: b.text, finish_reason: 'stop' }))
    }))
  }
}

// ============================================================================
// HTTP SERVER ENTRYPOINT
// ============================================================================

function sendError (res, status, message) {
  res.writeHead(status, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify({ error: { message, type: 'invalid_request_error', code: status } }))
}

const server = http.createServer((req, res) => {
  const start = Date.now()
  res.on('finish', () => console.log(`[${new Date().toISOString()}] ${req.method} ${req.url} ${res.statusCode} ${Date.now() - start}ms`))

  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization')

  if (req.method === 'OPTIONS') return res.writeHead(204).end()

  const url = new URL(req.url, `http://${req.headers.host}`)

  if (url.pathname === '/v1/models' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    return res.end(JSON.stringify({ object: 'list', data: Array.from(MODELS.values()).map(m => ({ id: m.name, object: 'model', created: Date.now() })) }))
  }

  if (req.method === 'POST') {
    let bodyData = ''
    req.on('data', chunk => { bodyData += chunk })
    req.on('end', async () => {
      try {
        const body = JSON.parse(bodyData)
        if (url.pathname === '/v1/chat/completions') await handleChatCompletions(req, res, body)
        else if (url.pathname === '/v1/completions') await handleLegacyCompletions(req, res, body)
        else sendError(res, 404, 'Endpoint not found')
      } catch (err) {
        console.error(err)
        sendError(res, 500, err.message)
      }
    })
    return
  }
  sendError(res, 404, 'Not Found')
})

// ============================================================================
// BOOT & LIFECYCLE
// ============================================================================

setInterval(() => {
  const now = Date.now()
  for (const [key, session] of SESSIONS.entries()) {
    if (session.cache.available && now - session.lastAccess > 30 * 60 * 1000) {
      session.cache.dispose()
      console.log(`[GC] Evicted session ${key.slice(0, 8)} from VRAM.`)
    }
  }
}, 5 * 60 * 1000)

async function boot () {
  for (const modelPath of args.model) {
    const name = path.basename(modelPath)
    MODELS.set(modelPath, {
      name,
      model: await MLXModel.fromPath(modelPath),
      tokenizer: await loadTokenizer(modelPath),
      template: await loadTemplate(modelPath),
      stopTokens: stopTokensFrom(await loadTokenizer(modelPath)),
      padTokenId: padTokenFrom(await loadTokenizer(modelPath))
    })
    MODELS.set(name, MODELS.get(modelPath)) // Allow lookup by basename too
    console.log(`✅ Loaded: ${name}`)
  }
  server.listen(args.port, args.host, () => console.log(`🚀 MLX API Server running on http://${args.host}:${args.port}`))
}

boot().catch(console.error)
