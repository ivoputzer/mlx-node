/**
 * Extracts Qwen/Mistral style XML-wrapped JSON tool calls.
 * Returns an array of parsed objects: [{ name: 'get_weather', arguments: { city: 'Paris' } }]
 */
export function parseToolCalls (text, options = {}) {
  const tag = options.tag || 'tool_call'

  // Non-greedy match for anything between the tags
  const toolRegex = new RegExp(`<${tag}>\\s*([\\s\\S]*?)\\s*<\\/${tag}>`, 'g')

  const calls = []
  let match

  while ((match = toolRegex.exec(text)) !== null) {
    try {
      const jsonString = match[1].trim()
      calls.push(JSON.parse(jsonString))
    } catch (err) {
      console.warn('Failed to parse tool call JSON:', err.message)
      // Soft fail: Don't let one hallucinated comma crash the agent
    }
  }

  return calls
}

/**
 * Safely extracts reasoning/thought processes using exact token IDs.
 * Handles missing opening tags and mid-generation thoughts.
 */
export function parseThinking (target, tokens, options = {}) {
  const startTag = options.startTag || '<think>'
  const endTag = options.endTag || '</think>'

  const startId = target.encode(startTag, { add_special_tokens: false }).ids[0]
  const endId = target.encode(endTag, { add_special_tokens: false }).ids[0]

  const endIndex = tokens.indexOf(endId)

  if (endIndex !== -1) {
    // 1. Found closing tag! Look backwards for the opening tag.
    const startIndex = tokens.lastIndexOf(startId, endIndex)

    let thoughtTokens, responseTokens

    if (startIndex !== -1) {
      // Thought was mid-generation: <response>...<think>...</think>...<response>
      thoughtTokens = tokens.slice(startIndex + 1, endIndex)
      responseTokens = [
        ...tokens.slice(0, Math.max(0, startIndex)),
        ...tokens.slice(endIndex + 1)
      ]
    } else {
      // Thought started at token 0 (Prompt injected the <think> tag)
      thoughtTokens = tokens.slice(0, endIndex)
      responseTokens = tokens.slice(endIndex + 1)
    }

    return {
      thought: target.decode(thoughtTokens, { skip_special_tokens: true }).trim(),
      response: target.decode(responseTokens, { skip_special_tokens: true }).trim()
    }
  }

  // 2. No closing tag. Did it start thinking but got interrupted?
  const startIndex = tokens.indexOf(startId)
  if (startIndex !== -1) {
    return {
      thought: target.decode(tokens.slice(startIndex + 1), { skip_special_tokens: true }).trim(),
      response: target.decode(tokens.slice(0, startIndex), { skip_special_tokens: true }).trim()
    }
  }

  // 3. Pure response, no reasoning.
  return {
    thought: null,
    response: target.decode(tokens, { skip_special_tokens: true }).trim()
  }
}

/**
 * Extracts all Markdown code blocks from a response.
 * Returns an array of objects: { language: 'js', content: 'console.log("hi")' }
 */
export function parseMarkdownBlocks (text) {
  // Matches ```language\n content \n```
  // (\w+|\w+[+-]\w+)? handles languages like "c++" or "objective-c"
  const blockRegex = /```([\w+-]*)\n([\s\S]*?)```/g

  const blocks = []
  let match

  while ((match = blockRegex.exec(text)) !== null) {
    blocks.push({
      language: match[1].trim() || 'text', // Fallback if no language is specified
      content: match[2].trim()
    })
  }

  return blocks
}
