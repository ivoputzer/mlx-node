/**
 * Core tool marker extraction utility for mlx-tool.
 * Uses a Bidirectional "Mock Execution & Divergence" strategy (Skeleton Key)
 * to bypass hardcoded regex and extract exact template boundaries.
 *
 * @param {Object|String|Function} template - The Jinja template instance, string, or callable
 * @param {Object} [tokenizer=null] - Optional tokenizer instance for context alignment
 * @returns {Object} Extracted topological boundaries
 */
export function analyzeToolMarkers (template, tokenizer = null) {
  const MOCK_TOOL_NAME = 'mock_tool_name_777'
  const MOCK_ARG_KEY = 'mock_arg_key_888'
  const MOCK_ARG_VAL = 'mock_arg_val_999'
  const MOCK_TOOL_ID = '123456789' // Exactly 9 chars to bypass strict Mistral/Ministral validations

  // Trick strict Jinja templates into iterating keys OR formatting as JSON cleanly
  const mockArguments = { [MOCK_ARG_KEY]: MOCK_ARG_VAL }
  Object.defineProperty(mockArguments, 'toString', {
    value: function () { return JSON.stringify(this) },
    enumerable: false
  })
  Object.defineProperty(mockArguments, 'toJSON', {
    value: function () { return { [MOCK_ARG_KEY]: MOCK_ARG_VAL } },
    enumerable: false
  })

  const tools = [{
    type: 'function',
    function: {
      name: MOCK_TOOL_NAME,
      description: 'Mock',
      parameters: { type: 'object', properties: { [MOCK_ARG_KEY]: { type: 'string' } }, required: [MOCK_ARG_KEY] }
    }
  }]

  // By giving Base message content but Tool message no content, we physically
  // bypass CoT/Reasoning branches and isolate purely the tool-call formatting.
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
    custom_tools: tools,       // Added for templates expecting 'custom_tools'
    available_tools: tools,    // Added for templates expecting 'available_tools'
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
  } catch (e) {
    return { type: 'error', reason: e.message }
  }

  if (baseString === toolString) return { type: 'unsupported', topology: 'unsupported' }

  // Bidirectional Diffing: Find exactly where the tool wrapper was injected
  let divergePrefixIdx = 0
  const minLen = Math.min(baseString.length, toolString.length)
  while (divergePrefixIdx < minLen && baseString[divergePrefixIdx] === toolString[divergePrefixIdx]) divergePrefixIdx++

  let divergeSuffixIdx = 0
  while (divergeSuffixIdx < (minLen - divergePrefixIdx) && baseString[baseString.length - 1 - divergeSuffixIdx] === toolString[toolString.length - 1 - divergeSuffixIdx]) divergeSuffixIdx++

  const isolateStr = toolString.slice(divergePrefixIdx, toolString.length - divergeSuffixIdx).trim()

  // Robust JSON boundary search: O(N^2) scanner that continuously advances `startIdx`.
  // This bypasses the old bug where stray `{` characters inside <thought> blocks bricked XML parsing.
  let bestJson = null
  for (const [startChar, endChar] of [['{', '}'], ['[', ']']]) {
    let startIdx = isolateStr.indexOf(startChar)
    while (startIdx !== -1) {
      let endIdx = isolateStr.lastIndexOf(endChar)
      while (endIdx > startIdx) {
        try {
          const parsed = JSON.parse(isolateStr.slice(startIdx, endIdx + 1))
          const jsonStr = JSON.stringify(parsed)
          // Must contain our arg key to confirm it's the payload
          if (jsonStr.includes(MOCK_ARG_KEY)) {
            bestJson = {
              parsed,
              start: startIdx,
              end: endIdx,
              hasName: jsonStr.includes(MOCK_TOOL_NAME) // Determines encapsulated vs segmented
            }
            break
          }
        } catch (e) {
          // Ignore and shrink the trailing edge
        }
        endIdx = isolateStr.lastIndexOf(endChar, endIdx - 1)
      }
      if (bestJson) break
      startIdx = isolateStr.indexOf(startChar, startIdx + 1) // Crucial: Advance search cursor
    }
    if (bestJson) break
  }

  const result = { type: 'success' }

  // Topology 1: Standard Encapsulated JSON (Llama 3.1, Granite, Ministral)
  if (bestJson && bestJson.hasName) {
    result.prefix = isolateStr.slice(0, bestJson.start).trim()
    result.suffix = isolateStr.slice(bestJson.end + 1).trim()
    result.topology = result.prefix.length === 0 ? 'implicit_json' : 'json_encapsulated'
    return result
  }

  // Topology 2: Custom Segmented / Strict XML (Gemma-4, GPT-OSS, Qwen3.6)
  result.topology = 'custom_segmented'
  const nameIdx = isolateStr.indexOf(MOCK_TOOL_NAME)
  result.namePrefix = nameIdx !== -1 ? isolateStr.slice(0, nameIdx).trim() : isolateStr.split('{')[0].trim()

  if (nameIdx !== -1) {
    // const afterName = isolateStr.slice(nameIdx + MOCK_TOOL_NAME.length)

    if (bestJson) {
      // It's segmented, but safely backed by JSON arguments. Extract all 3 segments so server knows what to strip.
      result.nameSuffix = isolateStr.slice(nameIdx + MOCK_TOOL_NAME.length, bestJson.start + 1).trim()
      result.argsSuffix = isolateStr.slice(bestJson.end + 1).trim()
    } else {
      // Pure XML / Syntax lacking valid JSON strings (e.g., Qwen3.6 <parameter=> structure)
      const argKeyIdx = isolateStr.indexOf(MOCK_ARG_KEY)

      if (argKeyIdx !== -1) {
        const braceIdx = isolateStr.lastIndexOf('{', argKeyIdx)
        if (braceIdx !== -1 && braceIdx > nameIdx) {
          result.nameSuffix = isolateStr.slice(nameIdx + MOCK_TOOL_NAME.length, braceIdx + 1).trim()
          const valIdx = isolateStr.indexOf(MOCK_ARG_VAL, argKeyIdx)
          const closeBraceIdx = valIdx !== -1 ? isolateStr.indexOf('}', valIdx) : -1
          result.argsSuffix = closeBraceIdx !== -1 ? isolateStr.slice(closeBraceIdx + 1).trim() : ''
        } else {
          // Strictly XML structure
          result.nameSuffix = isolateStr.slice(nameIdx + MOCK_TOOL_NAME.length, argKeyIdx).trim()
          const valIdx = isolateStr.indexOf(MOCK_ARG_VAL, argKeyIdx)
          if (valIdx !== -1) {
            const afterVal = isolateStr.slice(valIdx + MOCK_ARG_VAL.length)
            // Drop any closing quotes injected strictly around the parameter value itself
            const quoteMatch = afterVal.match(/^["']?\s*(.*)$/s)
            result.argsSuffix = quoteMatch ? quoteMatch[1].trim() : afterVal.trim()
          } else {
            result.argsSuffix = ''
          }
        }
      } else {
        result.nameSuffix = ''
        result.argsSuffix = ''
      }
    }
  } else {
    result.nameSuffix = ''
    result.argsSuffix = ''
  }

  return result
}
