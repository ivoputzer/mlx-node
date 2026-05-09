/**
 * Core tool marker extraction utility for mlx-tool.
 * Uses a Bidirectional "Mock Execution & Divergence" strategy (Skeleton Key).
 */
export function analyzeToolMarkers (template, tokenizer = null) {
  const MOCK_TOOL_NAME = 'mock_tool_name_777'
  const MOCK_ARG_KEY = 'mock_arg_key_888'
  const MOCK_ARG_VAL = 'mock_arg_val_999'
  const MOCK_TOOL_ID = '123456789'

  const mockArguments = { [MOCK_ARG_KEY]: MOCK_ARG_VAL }
  Object.defineProperty(mockArguments, 'toString', { value: function () { return JSON.stringify(this) }, enumerable: false })
  Object.defineProperty(mockArguments, 'toJSON', { value: function () { return { [MOCK_ARG_KEY]: MOCK_ARG_VAL } }, enumerable: false })

  const tools = [{
    type: 'function',
    function: { name: MOCK_TOOL_NAME, description: 'Mock', parameters: { type: 'object', properties: { [MOCK_ARG_KEY]: { type: 'string' } }, required: [MOCK_ARG_KEY] } }
  }]

  // 💡 FIX: Base has content, Tool has empty content.
  // This physically bypasses CoT/Reasoning branches in templates like GPT-OSS.
  const baseMessages = [
    { role: 'system', content: 'MOCK_SYSTEM' },
    { role: 'user', content: 'MOCK_USER' },
    { role: 'assistant', content: 'MOCK_BASE' }
  ]

  const toolMessages = [
    { role: 'system', content: 'MOCK_SYSTEM' },
    { role: 'user', content: 'MOCK_USER' },
    { role: 'assistant', content: '', tool_calls: [{ id: MOCK_TOOL_ID, type: 'function', function: { name: MOCK_TOOL_NAME, arguments: mockArguments } }] }
  ]

  const commonContext = {
    tools,
    add_generation_prompt: false,
    bos_token: tokenizer?.bos_token || '<bos>',
    eos_token: tokenizer?.eos_token || '<eos>'
  }

  function renderContext (ctx) {
    if (template && typeof template.render === 'function') return template.render(ctx)
    if (tokenizer && typeof tokenizer.apply_chat_template === 'function') {
      return tokenizer.apply_chat_template(ctx.messages, { ...ctx, chat_template: typeof template === 'string' ? template : undefined, tokenize: false })
    }
    if (typeof template === 'function') return template(ctx)
    throw new Error('No valid renderer found.')
  }

  let baseString, toolString
  try {
    baseString = renderContext({ ...commonContext, messages: baseMessages })
    toolString = renderContext({ ...commonContext, messages: toolMessages })
  } catch (e) { return { type: 'error', reason: e.message } }

  if (baseString === toolString) return { type: 'unsupported', topology: 'unsupported' }

  // Bidirectional Diffing
  let divergePrefixIdx = 0
  const minLen = Math.min(baseString.length, toolString.length)
  while (divergePrefixIdx < minLen && baseString[divergePrefixIdx] === toolString[divergePrefixIdx]) divergePrefixIdx++

  let divergeSuffixIdx = 0
  while (divergeSuffixIdx < (minLen - divergePrefixIdx) && baseString[baseString.length - 1 - divergeSuffixIdx] === toolString[toolString.length - 1 - divergeSuffixIdx]) divergeSuffixIdx++

  const isolateStr = toolString.slice(divergePrefixIdx, toolString.length - divergeSuffixIdx).trim()

  // Robust JSON boundary search (Handles arrays & objects natively)
  let bestJson = null
  for (const [startChar, endChar] of [['{', '}'], ['[', ']']]) {
    const startIdx = isolateStr.indexOf(startChar)
    if (startIdx === -1) continue
    let endIdx = isolateStr.lastIndexOf(endChar)
    while (endIdx > startIdx) {
      try {
        bestJson = { parsed: JSON.parse(isolateStr.slice(startIdx, endIdx + 1)), start: startIdx, end: endIdx }
        break
      } catch (e) { endIdx = isolateStr.lastIndexOf(endChar, endIdx - 1) }
    }
    if (bestJson) break
  }

  const result = { type: 'success' }

  if (bestJson) {
    const jsonStr = JSON.stringify(bestJson.parsed)
    if (jsonStr.includes(MOCK_TOOL_NAME) && jsonStr.includes(MOCK_ARG_KEY)) {
      result.prefix = isolateStr.slice(0, bestJson.start).trim()
      result.suffix = isolateStr.slice(bestJson.end + 1).trim()
      result.topology = result.prefix.length === 0 ? 'implicit_json' : 'json_encapsulated'
      return result
    }
  }

  // custom_segmented (Qwen, Gemma, GPT-OSS)
  result.topology = 'custom_segmented'
  const nameIdx = isolateStr.indexOf(MOCK_TOOL_NAME)
  result.namePrefix = nameIdx !== -1 ? isolateStr.slice(0, nameIdx).trim() : isolateStr.split('{')[0].trim()

  if (nameIdx !== -1) {
    const afterName = isolateStr.slice(nameIdx + MOCK_TOOL_NAME.length)
    result.nameSuffix = afterName.slice(0, afterName.indexOf('{') + 1).trim() // Captures the exact bridge to the JSON args
  }

  return result
}
