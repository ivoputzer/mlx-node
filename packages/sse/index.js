import http from 'node:http'
import fs from 'node:fs/promises'
import path from 'node:path'
import crypto from 'node:crypto'
import { parseArgs } from 'node:util'

import { MLXModel, MLXCache } from 'mlx-swift'
import { loadTokenizer, loadTemplate, stopTokensFrom, padTokenFrom } from 'mlx-lm'
import { analyzeToolMarkers } from 'mlx-tool'

// ============================================================================
// CONFIGURATION & CLI ARGUMENTS
// ============================================================================

const { values: args } = parseArgs({
  options: {
    port: { type: 'string', short: 'p', default: '8080' },
    host: { type: 'string', short: 'h', default: '127.0.0.1' },
    model: { type: 'string', short: 'm', multiple: true, default: [] },
    verbose: { type: 'boolean', short: 'v', default: false } // Added verbose flag
  }
})

const CACHE_DIR = path.join(process.cwd(), '.mlx_sessions')
await fs.mkdir(CACHE_DIR, { recursive: true }).catch(() => {})

// Standardized Logger
const log = {
  info: (...msg) => console.log('[INFO]', ...msg),
  verb: (...msg) => { if (args.verbose) console.log('[DEBUG]', ...msg) },
  error: (...msg) => console.error('[ERROR]', ...msg)
}

if (args.verbose) log.info('Verbose logging enabled. Expect detailed output.')

// ============================================================================
// UTILITIES
// ============================================================================

function parseDynamicTool (raw, markers) {
  try {
    const funcMatch = raw.match(/<function=(.*?)>/)
    if (funcMatch) {
      const args = {}
      const paramRegex = /<parameter=(.*?)>\n?([\s\S]*?)\n?<\/parameter>/g
      let m
      while ((m = paramRegex.exec(raw)) !== null) args[m[1]] = m[2].trim()
      return { name: funcMatch[1], arguments: JSON.stringify(args) }
    }

    const start = raw.indexOf('{')
    const end = raw.lastIndexOf('}')
    if (start === -1 || end === -1 || start >= end) return null

    const parsed = JSON.parse(raw.slice(start, end + 1))
    let name = parsed.name
    let args = parsed.arguments || parsed

    if (markers?.topology === 'custom_segmented' && markers?.namePrefix) {
      const nameStart = raw.indexOf(markers.namePrefix)
      if (nameStart !== -1) {
        const afterPrefix = raw.slice(nameStart + markers.namePrefix.length)
        const nameEnd = markers.nameSuffix ? afterPrefix.indexOf(markers.nameSuffix) : afterPrefix.indexOf('{')
        if (nameEnd !== -1) name = afterPrefix.slice(0, nameEnd).trim()
      }
    }

    if (typeof args === 'object' && args !== null && args.name === name && args.arguments) args = args.arguments
    return { name: name || 'unknown', arguments: typeof args === 'string' ? args : JSON.stringify(args) }
  } catch { return null }
}

function getPartialMatchLen (buffer, target) {
  if (!target) return 0
  const maxLen = Math.min(buffer.length, target.length)
  for (let i = maxLen; i > 0; i--) {
    if (target.startsWith(buffer.slice(-i))) return i
  }
  return 0
}

// ============================================================================
// CORE: SESSION MANAGEMENT FACTORY
// ============================================================================

function createSessionManager ({ cacheDir }) {
  const activeSessions = new Map()

  async function getOrCreate (sessionId, modelConfig) {
    const { model, name: modelName } = modelConfig
    const sessionKey = crypto.createHash('sha256').update(`${sessionId}_${modelName}`).digest('hex').slice(0, 16)
    const diskPath = path.join(cacheDir, sessionKey)

    // 1. Return from VRAM if currently active
    if (activeSessions.has(sessionKey)) {
      const session = activeSessions.get(sessionKey)
      clearTimeout(session.gcTimer) // Prevent eviction while in use
      log.verb(`[Cache] Reusing active VRAM session: ${sessionKey}`)
      return session
    }

    // 2. Hydrate from Disk (Very fast on Unified Memory)
    let cache; let historyTokens = []
    try {
      await fs.access(`${diskPath}.safetensors`)
      cache = await MLXCache.fromPath(`${diskPath}.safetensors`, model)
      const meta = JSON.parse(await fs.readFile(`${diskPath}.meta.json`, 'utf8'))
      historyTokens = meta.historyTokens
      log.verb(`[Cache] Restored session from disk to VRAM: ${sessionKey} (${historyTokens.length} tokens)`)
    } catch {
      // 3. Fallback to entirely new Cache
      cache = MLXCache.fromModel(model)
      log.verb(`[Cache] Created fresh empty session: ${sessionKey}`)
    }

    const session = { cache, historyTokens, diskPath, key: sessionKey }
    activeSessions.set(sessionKey, session)
    return session
  }

  async function saveAndRelease (session) {
    if (!session || !session.cache.available) return

    // Immediately persist state to disk just in case
    await session.cache.save(`${session.diskPath}.safetensors`)
    await fs.writeFile(`${session.diskPath}.meta.json`, JSON.stringify({ historyTokens: session.historyTokens }))

    // Aggressive FS Strategy: 15 seconds of idle time = Evict from VRAM
    session.gcTimer = setTimeout(() => {
      if (session.cache.available) {
        session.cache.dispose()
        activeSessions.delete(session.key)
        log.info(`[GC] Evicted session ${session.key} from VRAM to disk to save memory.`)
      }
    }, 15000)
  }

  return { getOrCreate, saveAndRelease }
}

// ============================================================================
// CORE: GENERATION ENGINE
// ============================================================================

async function * generateMLXStream (session, fullPromptString, modelConfig, options) {
  const { tokenizer, model, stopTokens, padTokenId, markers } = modelConfig
  const targetTokens = tokenizer.encode(fullPromptString).ids

  let commonPrefixLen = 0
  while (
    commonPrefixLen < session.historyTokens.length &&
    commonPrefixLen < targetTokens.length &&
    session.historyTokens[commonPrefixLen] === targetTokens[commonPrefixLen]
  ) {
    commonPrefixLen++
  }

  log.verb(`[Context] Match: ${commonPrefixLen} tokens. New: ${targetTokens.length - commonPrefixLen} tokens.`)

  // STRICT APPEND-ONLY BEHAVIOR
  if (commonPrefixLen < session.historyTokens.length) {
    log.verb(`[Context] History diverged (Cached: ${session.historyTokens.length}, Matched: ${commonPrefixLen}). Dropping entire cache due to append-only rule!`)
    session.cache.dispose()
    session.cache = MLXCache.fromModel(model)
    commonPrefixLen = 0
  } else if (commonPrefixLen > 0) {
    log.verb('[Context] Successfully appending to existing cache.')
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
  const triggerStart = markers?.prefix || markers?.namePrefix || '<tool_call>'
  const triggerEnd = markers?.suffix || '</tool_call>'
  const THINK_START = '<think>'
  const THINK_END = '</think>'

  const branches = Array.from({ length: n }, (_, i) => ({
    index: i,
    active: true,
    inThinkBlock: startsThinking,
    inToolBlock: false,
    textBuffer: '',
    toolBuffer: '',
    toolId: `call_${crypto.randomUUID().slice(0, 8)}`,
    tokens: 0
  }))

  const iterator = stream[Symbol.asyncIterator]()
  let mlxStats = null

  while (true) {
    const { done, value } = await iterator.next()
    if (done) {
      mlxStats = value
      break
    }

    const batches = value
    for (let i = 0; i < n; i++) {
      const branch = branches[i]
      if (!branch.active) continue

      const tokenId = batches[i]

      if (tokenId === -1 || stopTokens.includes(tokenId)) {
        branch.active = false
        if (branch.inToolBlock) {
          const parsedTool = parseDynamicTool(branch.toolBuffer, markers)
          if (parsedTool) yield { branchIndex: i, type: 'tool_finish', id: branch.toolId, tool: parsedTool }
        } else if (branch.textBuffer) {
          yield { branchIndex: i, type: branch.inThinkBlock ? 'reasoning' : 'text', text: branch.textBuffer }
        }
        continue
      }

      branch.tokens++
      if (i === 0) session.historyTokens.push(tokenId)

      const textChunk = tokenizer.decode([tokenId], { skip_special_tokens: false })
      branch.textBuffer += textChunk

      // Reasoning
      if (!branch.inThinkBlock && !branch.inToolBlock && branch.textBuffer.includes(THINK_START)) {
        const startIdx = branch.textBuffer.indexOf(THINK_START)
        const before = branch.textBuffer.slice(0, startIdx)
        if (before) yield { branchIndex: i, type: 'text', text: before }
        branch.inThinkBlock = true
        branch.textBuffer = branch.textBuffer.slice(startIdx + THINK_START.length)
      }

      if (branch.inThinkBlock) {
        const endIdx = branch.textBuffer.indexOf(THINK_END)
        if (endIdx !== -1) {
          const thinking = branch.textBuffer.slice(0, endIdx)
          if (thinking) yield { branchIndex: i, type: 'reasoning', text: thinking }
          branch.inThinkBlock = false
          branch.textBuffer = branch.textBuffer.slice(endIdx + THINK_END.length)
        } else {
          const partialLen = getPartialMatchLen(branch.textBuffer, THINK_END)
          if (partialLen === 0) {
            yield { branchIndex: i, type: 'reasoning', text: branch.textBuffer }
            branch.textBuffer = ''
          } else if (partialLen < branch.textBuffer.length) {
            yield { branchIndex: i, type: 'reasoning', text: branch.textBuffer.slice(0, -partialLen) }
            branch.textBuffer = branch.textBuffer.slice(-partialLen)
          }
          continue
        }
      }

      // Tool
      if (!branch.inToolBlock && triggerStart && branch.textBuffer.includes(triggerStart)) {
        const startIdx = branch.textBuffer.indexOf(triggerStart)
        const before = branch.textBuffer.slice(0, startIdx)
        if (before) yield { branchIndex: i, type: 'text', text: before }
        branch.inToolBlock = true
        branch.toolBuffer = branch.textBuffer.slice(startIdx + triggerStart.length)
        branch.textBuffer = ''
        yield { branchIndex: i, type: 'tool_start', id: branch.toolId }
        continue
      }

      if (branch.inToolBlock) {
        branch.toolBuffer += branch.textBuffer
        branch.textBuffer = ''
        if (triggerEnd && branch.toolBuffer.includes(triggerEnd)) {
          const endIdx = branch.toolBuffer.indexOf(triggerEnd)
          branch.toolBuffer = branch.toolBuffer.slice(0, endIdx)
          const parsedTool = parseDynamicTool(branch.toolBuffer, markers)
          if (parsedTool) yield { branchIndex: i, type: 'tool_finish', id: branch.toolId, tool: parsedTool }
          branch.inToolBlock = false
          branch.toolBuffer = ''
        }
        continue
      }

      // Safe Text Output
      const partialLen = Math.max(getPartialMatchLen(branch.textBuffer, THINK_START), triggerStart ? getPartialMatchLen(branch.textBuffer, triggerStart) : 0)
      if (partialLen === 0) {
        yield { branchIndex: i, type: 'text', text: branch.textBuffer }
        branch.textBuffer = ''
      } else if (partialLen < branch.textBuffer.length) {
        yield { branchIndex: i, type: 'text', text: branch.textBuffer.slice(0, -partialLen) }
        branch.textBuffer = branch.textBuffer.slice(-partialLen)
      }
    }
  }

  // Verbose Output for MLX Hardware Stats
  if (args.verbose && mlxStats) {
    log.verb('\n--- Generation Stats ---')
    log.verb(`TTFT (Prompt Eval): ${mlxStats.promptTime?.toFixed(2)}s (${mlxStats.promptTokensPerSecond?.toFixed(2)} t/s)`)
    log.verb(`Generation Time : ${mlxStats.generateTime?.toFixed(2)}s (${mlxStats.tokensPerSecond?.toFixed(2)} t/s)`)
    log.verb('------------------------\n')
  }

  yield {
    type: 'usage',
    prompt_tokens: mlxStats?.promptTokens ?? targetTokens.length,
    completion_tokens: mlxStats?.generatedTokens ?? Math.max(...branches.map(b => b.tokens)),
    total_tokens: (mlxStats?.promptTokens ?? targetTokens.length) + (mlxStats?.generatedTokens ?? Math.max(...branches.map(b => b.tokens))),
    time_to_first_token: mlxStats?.promptTime,
    generation_time: mlxStats?.generateTime,
    tokens_per_second: mlxStats?.tokensPerSecond
  }
}

// ============================================================================
// API ADAPTERS (OpenAI)
// ============================================================================

function createOpenAIAdapter () {
  return {
    headers: {
      stream: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' },
      json: { 'Content-Type': 'application/json' }
    },

    formatChatChunk: (id, model, chunk, includeUsage) => {
      if (chunk.type === 'usage') {
        if (!includeUsage) return null
        return `data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000), model, choices: [], usage: chunk })}\n\n`
      }
      const delta = {}
      if (chunk.type === 'text') delta.content = chunk.text
      if (chunk.type === 'reasoning') delta.reasoning_content = chunk.text
      if (chunk.type === 'tool_start') delta.tool_calls = [{ index: 0, id: chunk.id, type: 'function', function: { name: '', arguments: '' } }]
      if (chunk.type === 'tool_finish') delta.tool_calls = [{ index: 0, id: chunk.id, function: chunk.tool }]

      return `data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000), model, choices: [{ index: chunk.branchIndex, delta, finish_reason: null }] })}\n\n`
    },

    formatChatStops: (id, model, n = 1) => {
      return Array.from({ length: n }, (_, i) => `data: {"id":"${id}","object":"chat.completion.chunk","choices":[{"index":${i},"delta":{},"finish_reason":"stop"}]}\n\n`)
    },

    formatChatResponse: (id, model, branches, usage) => {
      return JSON.stringify({
        id,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        usage,
        choices: branches.map((b, index) => ({
          index,
          message: { role: 'assistant', content: b.content, reasoning_content: b.reasoning_content || undefined, tool_calls: b.tool_calls.length ? b.tool_calls : undefined },
          finish_reason: b.tool_calls.length ? 'tool_calls' : 'stop'
        }))
      })
    },

    formatCompletionChunk: (id, model, chunk) => {
      if (chunk.type === 'text') return `data: ${JSON.stringify({ id, object: 'text_completion', created: Math.floor(Date.now() / 1000), model, choices: [{ index: chunk.branchIndex, text: chunk.text, finish_reason: null }] })}\n\n`
      return null
    },

    formatCompletionStops: (id, model, n = 1) => {
      return Array.from({ length: n }, (_, i) => `data: {"id":"${id}","object":"text_completion","choices":[{"index":${i},"text":"","finish_reason":"stop"}]}\n\n`)
    },

    formatCompletionResponse: (id, model, branches, usage) => {
      return JSON.stringify({
        id,
        object: 'text_completion',
        created: Math.floor(Date.now() / 1000),
        model,
        usage,
        choices: branches.map((b, index) => ({ index, text: b.text, finish_reason: 'stop' }))
      })
    }
  }
}

// ============================================================================
// HANDLER FACTORIES
// ============================================================================

function createChatCompletionHandler ({ models, sessionManager, generateStream, adapter }) {
  return async function handle (req, res, body) {
    if (!body.messages?.length) throw new Error('messages array is required')

    const modelConfig = models.get(body.model) || Array.from(models.values())[0]
    if (!modelConfig) throw new Error('Model not found')

    const fullPromptString = modelConfig.template.render({ messages: body.messages, tools: body.tools, add_generation_prompt: true })
    const session = await sessionManager.getOrCreate(body.user || 'default_user', modelConfig)
    const generator = generateStream(session, fullPromptString, modelConfig, body)
    const requestId = `chatcmpl-${crypto.randomUUID()}`

    if (body.stream) {
      res.writeHead(200, adapter.headers.stream)
      for await (const chunk of generator) {
        const formatted = adapter.formatChatChunk(requestId, modelConfig.name, chunk, body.stream_options?.include_usage)
        if (formatted) res.write(formatted)
      }
      adapter.formatChatStops(requestId, modelConfig.name, body.n).forEach(stop => res.write(stop))
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
      res.writeHead(200, adapter.headers.json)
      if (args.verbose) log.verb('[Response] Sending non-streaming JSON block back to client.')
      res.end(adapter.formatChatResponse(requestId, modelConfig.name, branches, usage))
    }

    await sessionManager.saveAndRelease(session)
  }
}

function createLegacyCompletionHandler ({ models, sessionManager, generateStream, adapter }) {
  return async function handle (req, res, body) {
    if (!body.prompt) throw new Error('prompt is required')

    const modelConfig = models.get(body.model) || Array.from(models.values())[0]
    if (!modelConfig) throw new Error('Model not found')

    const session = await sessionManager.getOrCreate(body.user || 'default_user', modelConfig)
    const generator = generateStream(session, body.prompt, modelConfig, body)
    const requestId = `cmpl-${crypto.randomUUID()}`

    if (body.stream) {
      res.writeHead(200, adapter.headers.stream)
      for await (const chunk of generator) {
        const formatted = adapter.formatCompletionChunk(requestId, modelConfig.name, chunk)
        if (formatted) res.write(formatted)
      }
      adapter.formatCompletionStops(requestId, modelConfig.name, body.n).forEach(stop => res.write(stop))
      res.end('data: [DONE]\n\n')
    } else {
      const branches = Array.from({ length: body.n || 1 }, () => ({ text: '' }))
      let usage = null
      for await (const chunk of generator) {
        if (chunk.type === 'usage') usage = chunk
        else if (chunk.type === 'text') branches[chunk.branchIndex].text += chunk.text
      }
      res.writeHead(200, adapter.headers.json)
      res.end(adapter.formatCompletionResponse(requestId, modelConfig.name, branches, usage))
    }

    await sessionManager.saveAndRelease(session)
  }
}

// ============================================================================
// ROUTER & SERVER FACTORY
// ============================================================================

function createRequestHandler ({ models, chatHandler, legacyHandler }) {
  function sendError (res, status, message) {
    res.writeHead(status, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: { message, type: 'invalid_request_error', code: status } }))
  }

  return async (req, res) => {
    const start = Date.now()
    res.on('finish', () => log.info(`${req.method} ${req.url} ${res.statusCode} ${Date.now() - start}ms`))

    res.setHeader('Access-Control-Allow-Origin', '*')
    res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    if (req.method === 'OPTIONS') return res.writeHead(204).end()

    const url = new URL(req.url, `http://${req.headers.host}`)

    if (url.pathname === '/v1/models' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      return res.end(JSON.stringify({ object: 'list', data: Array.from(models.values()).map(m => ({ id: m.name, object: 'model', created: Date.now() })) }))
    }

    if (req.method === 'POST') {
      let bodyData = ''
      req.on('data', chunk => { bodyData += chunk })
      req.on('end', async () => {
        try {
          const body = JSON.parse(bodyData)

          if (args.verbose) {
            log.verb(`\n--- Incoming Payload (${url.pathname}) ---`)
            log.verb(JSON.stringify(body, null, 2))
            log.verb('-------------------------------------------\n')
          }

          if (url.pathname === '/v1/chat/completions') await chatHandler(req, res, body)
          else if (url.pathname === '/v1/completions') await legacyHandler(req, res, body)
          else sendError(res, 404, 'Endpoint not found')
        } catch (err) {
          log.error(err)
          sendError(res, err.message.includes('required') || err.message.includes('not found') ? 400 : 500, err.message)
        }
      })
      return
    }
    sendError(res, 404, 'Not Found')
  }
}

// ============================================================================
// BOOT & WIRING
// ============================================================================

async function boot () {
  const modelsMap = new Map()
  const sessionManager = createSessionManager({ cacheDir: CACHE_DIR })
  const openAIAdapter = createOpenAIAdapter()

  const chatHandler = createChatCompletionHandler({
    models: modelsMap, sessionManager, generateStream: generateMLXStream, adapter: openAIAdapter
  })

  const legacyHandler = createLegacyCompletionHandler({
    models: modelsMap, sessionManager, generateStream: generateMLXStream, adapter: openAIAdapter
  })

  const requestHandler = createRequestHandler({
    models: modelsMap, chatHandler, legacyHandler
  })

  for (const modelPath of args.model) {
    const name = path.basename(modelPath)
    const tokenizer = await loadTokenizer(modelPath)
    const template = await loadTemplate(modelPath, tokenizer)
    const markers = analyzeToolMarkers(template, tokenizer)

    modelsMap.set(modelPath, {
      name,
      model: await MLXModel.fromPath(modelPath),
      tokenizer,
      template,
      markers,
      stopTokens: stopTokensFrom(tokenizer),
      padTokenId: padTokenFrom(tokenizer)
    })
    modelsMap.set(name, modelsMap.get(modelPath))
    log.info(`✅ Loaded: ${name} (Tool Topology: ${markers.topology || markers.type})`)
  }

  const server = http.createServer(requestHandler)
  server.listen(args.port, args.host, () => log.info(`🚀 MLX API Server running on http://${args.host}:${args.port}`))
}

boot().catch(log.error)
