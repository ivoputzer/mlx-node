import http from 'node:http'
import fs from 'node:fs/promises'
import path from 'node:path'
import { createHash, randomUUID } from 'node:crypto'
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

// Registries
// path -> { name, model, tokenizer, template, stopTokens, padTokenId }
const MODELS = new Map()
// sessionId -> { cache, historyTokens: number[], lastAccess, diskPath }
const SESSIONS = new Map()

// ============================================================================
// TOOL PARSERS
// ============================================================================

const ToolParsers = {
  // Parses typical <tool_call>...</tool_call> outputs
  xml: (raw) => {
    try {
      const nameMatch = raw.match(/"name":\s*"([^"]+)"/)
      if (nameMatch) {
        // Handle Qwen-style JSON inside XML tags
        const jsonMatch = raw.match(/{[\s\S]*}/)
        const parsed = JSON.parse(jsonMatch[0])
        return { name: parsed.name, arguments: JSON.stringify(parsed.arguments || {}) }
      }

      // Handle DeepSeek/Generic XML attributes
      const funcMatch = raw.match(/<function=(.*?)>/)
      if (funcMatch) {
        const args = {}
        const paramRegex = /<parameter=(.*?)>\n?([\s\S]*?)\n?<\/parameter>/g
        let m
        while ((m = paramRegex.exec(raw)) !== null) {
          args[m[1]] = m[2].trim()
        }
        return { name: funcMatch[1], arguments: JSON.stringify(args) }
      }
    } catch {
      return null // Fallback if parsing fails
    }
  }
}

// ============================================================================
// SESSION & CACHE MANAGEMENT
// ============================================================================

async function getOrCreateSession (sessionId, modelConfig) {
  const { model, name: modelName } = modelConfig
  const sessionKey = createHash('sha256').update(`${sessionId}_${modelName}`).digest('hex').slice(0, 16)
  const diskPath = path.join(CACHE_DIR, sessionKey)

  // 1. Check RAM (Active Session)
  if (SESSIONS.has(sessionKey)) {
    const session = SESSIONS.get(sessionKey)
    session.lastAccess = Date.now()
    return session
  }

  // 2. Check Disk (Sleeping Session)
  try {
    await fs.access(`${diskPath}.safetensors`)
    const cache = await MLXCache.fromPath(`${diskPath}.safetensors`, model)
    const meta = JSON.parse(await fs.readFile(`${diskPath}.meta.json`, 'utf8'))

    const session = { cache, historyTokens: meta.historyTokens, lastAccess: Date.now(), diskPath }
    SESSIONS.set(sessionKey, session)
    return session
  } catch {
    // 3. Create Fresh Session
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
// CORE GENERATION (Token Diffing & Open Mouth)
// ============================================================================

async function * generateMLXStream (session, fullPromptString, modelConfig, options) {
  const { tokenizer, model, stopTokens, padTokenId } = modelConfig

  // 1. Encode the target prompt
  const targetTokens = tokenizer.encode(fullPromptString).ids

  // 2. Diff against Cache History
  let commonPrefixLen = 0
  while (
    commonPrefixLen < session.historyTokens.length &&
    commonPrefixLen < targetTokens.length &&
    session.historyTokens[commonPrefixLen] === targetTokens[commonPrefixLen]
  ) {
    commonPrefixLen++
  }

  const tokensToTrim = session.historyTokens.length - commonPrefixLen

  // 3. Reconcile State
  if (tokensToTrim > 0) {
    if (session.cache.isTrimmable) {
      session.cache.trim(tokensToTrim)
    } else {
      // Mamba/RNN fallback: Cannot trim, must restart context
      session.cache.dispose()
      session.cache = MLXCache.fromModel(model)
      commonPrefixLen = 0
    }
  }

  // 4. Update memory to strictly mirror the new target context
  session.historyTokens = Array.from(targetTokens)

  // The un-evaluated "new" portion of the prompt (including the previous turn's EOS token!)
  const newTokens = new Int32Array(targetTokens.slice(commonPrefixLen))

  // 5. Generate
  const stream = session.cache.generate(newTokens, {
    batchSize: 1,
    temperature: options.temperature ?? 0.7,
    maxTokens: options.max_tokens ?? 1024,
    stopTokenIds: stopTokens,
    padTokenId
  })

  let toolBuffer = ''
  let inToolBlock = false

  for await (const batches of stream) {
    const tokenId = batches[0]
    if (tokenId === -1) continue // Pad skip
    if (stopTokens.includes(tokenId)) break

    // Track generated tokens to keep diff state perfectly aligned
    session.historyTokens.push(tokenId)

    const textChunk = tokenizer.decode([tokenId], { skip_special_tokens: false })

    // Tool Call Detection Logic
    if (textChunk.includes('<tool_call>')) inToolBlock = true

    if (inToolBlock) {
      toolBuffer += textChunk
      if (toolBuffer.includes('</tool_call>')) {
        const parsedTool = ToolParsers.xml(toolBuffer)
        if (parsedTool) yield { type: 'tool', tool: parsedTool }

        inToolBlock = false
        toolBuffer = ''
      }
      continue // Suppress tool text from standard output
    }

    yield { type: 'text', text: textChunk }
  }

  // Persist after turn
  await saveSessionToDisk(session)
}

// ============================================================================
// API ROUTES
// ============================================================================

async function handleChatCompletions (req, res, body) {
  if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
    return sendError(res, 400, 'messages array is required and must not be empty')
  }

  const requestedModelName = body.model || Array.from(MODELS.values())[0]?.name
  const modelConfig = Array.from(MODELS.values()).find(m => m.name === requestedModelName)

  if (!modelConfig) return sendError(res, 404, `Model '${requestedModelName}' not loaded.`)

  // Render exactly as the tokenizer expects, injecting tools if provided
  const fullPromptString = modelConfig.template.render({
    messages: body.messages,
    tools: body.tools,
    add_generation_prompt: true
  })

  const session = await getOrCreateSession(body.user || 'default_user', modelConfig)
  const generator = generateMLXStream(session, fullPromptString, modelConfig, body)
  const requestId = `chatcmpl-${randomUUID()}`

  if (body.stream) {
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' })

    for await (const chunk of generator) {
      if (chunk.type === 'text') {
        res.write(`data: ${JSON.stringify({
          id: requestId,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model: modelConfig.name,
          choices: [{ index: 0, delta: { content: chunk.text }, finish_reason: null }]
        })}\n\n`)
      } else if (chunk.type === 'tool') {
        res.write(`data: ${JSON.stringify({
          id: requestId,
          object: 'chat.completion.chunk',
          model: modelConfig.name,
          choices: [{
            index: 0,
            delta: {
              tool_calls: [{
                id: `call_${randomUUID().slice(0, 8)}`,
                type: 'function',
                function: chunk.tool
              }]
            },
            finish_reason: 'tool_calls'
          }]
        })}\n\n`)
      }
    }
    res.write(`data: {"id":"${requestId}","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n`)
    res.end('data: [DONE]\n\n')
  } else {
    // Non-Streaming Wrapper
    let fullContent = ''
    const toolCalls = []

    for await (const chunk of generator) {
      if (chunk.type === 'text') fullContent += chunk.text
      if (chunk.type === 'tool') {
        toolCalls.push({
          id: `call_${randomUUID().slice(0, 8)}`,
          type: 'function',
          function: chunk.tool
        })
      }
    }

    const message = { role: 'assistant', content: fullContent }
    if (toolCalls.length > 0) message.tool_calls = toolCalls

    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({
      id: requestId,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: modelConfig.name,
      choices: [{ message, finish_reason: toolCalls.length > 0 ? 'tool_calls' : 'stop', index: 0 }]
    }))
  }
}

async function handleLegacyCompletions (req, res, body) {
  if (!body.prompt) return sendError(res, 400, 'prompt is required')

  const requestedModelName = body.model || Array.from(MODELS.values())[0]?.name
  const modelConfig = Array.from(MODELS.values()).find(m => m.name === requestedModelName)
  if (!modelConfig) return sendError(res, 404, `Model '${requestedModelName}' not loaded.`)

  const session = await getOrCreateSession(body.user || 'default_user', modelConfig)
  // Bypass template rendering, just pass raw text
  const generator = generateMLXStream(session, body.prompt, modelConfig, body)
  const requestId = `cmpl-${randomUUID()}`

  if (body.stream) {
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' })
    for await (const chunk of generator) {
      if (chunk.type === 'text') {
        res.write(`data: ${JSON.stringify({
          id: requestId,
          object: 'text_completion',
          created: Math.floor(Date.now() / 1000),
          model: modelConfig.name,
          choices: [{ text: chunk.text, index: 0, finish_reason: null }]
        })}\n\n`)
      }
    }
    res.write(`data: {"id":"${requestId}","object":"text_completion","choices":[{"text":"","index":0,"finish_reason":"stop"}]}\n\n`)
    res.end('data: [DONE]\n\n')
  } else {
    let fullContent = ''
    for await (const chunk of generator) {
      if (chunk.type === 'text') fullContent += chunk.text
    }
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({
      id: requestId,
      object: 'text_completion',
      created: Math.floor(Date.now() / 1000),
      model: modelConfig.name,
      choices: [{ text: fullContent, index: 0, finish_reason: 'stop' }]
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
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization')

  if (req.method === 'OPTIONS') return res.writeHead(204).end()

  const url = new URL(req.url, `http://${req.headers.host}`)

  if (url.pathname === '/v1/models' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    return res.end(JSON.stringify({
      object: 'list',
      data: Array.from(MODELS.values()).map(m => ({ id: m.name, object: 'model', created: Date.now() }))
    }))
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

// GC: Evict RAM caches idle for > 30 mins to VRAM limit
setInterval(() => {
  const now = Date.now()
  for (const [key, session] of SESSIONS.entries()) {
    if (session.cache.available && now - session.lastAccess > 30 * 60 * 1000) {
      session.cache.dispose() // Unload from RAM (still safe on disk!)
      console.log(`[GC] Evicted session ${key.slice(0, 8)} from VRAM to disk.`)
    }
  }
}, 5 * 60 * 1000)

async function boot () {
  if (args.model.length === 0) {
    console.warn('⚠️ No models specified via --model (-m). Server will start, but requests may fail.')
  }

  for (const modelPath of args.model) {
    console.log(`Loading model: ${modelPath}...`)
    const name = path.basename(modelPath)
    const model = await MLXModel.fromPath(modelPath)
    const tokenizer = await loadTokenizer(modelPath)
    const template = await loadTemplate(modelPath)

    MODELS.set(modelPath, {
      name,
      model,
      tokenizer,
      template,
      stopTokens: stopTokensFrom(tokenizer),
      padTokenId: padTokenFrom(tokenizer)
    })
    console.log(`✅ Loaded: ${name}`)
  }

  server.listen(args.port, args.host, () => {
    console.log(`\n🚀 MLX API Server running on http://${args.host}:${args.port}`)
    console.log(`Metrics: ${MLXMetrics.fromSnapshot().usagePercent.toFixed(1)}% Unified Memory Used`)
  })
}

boot().catch(console.error)
