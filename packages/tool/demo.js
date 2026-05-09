import { analyzeToolMarkers } from './index.js'

// ------------------------------------------------------------------
// 1. Regex Generator Utility (for stream: false)
// ------------------------------------------------------------------

/**
 * Escapes characters with special meaning in Regular Expressions.
 */
function escapeRegExp (string) {
  if (!string) return ''
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * Compiles a highly optimized RegExp based on the discovered topology.
 * @param {Object} markers - The output of analyzeToolMarkers()
 * @returns {Function} A parsing function that takes a full completion string
 */
function createNonStreamingExtractor (markers) {
  if (markers.topology === 'unsupported' || markers.topology === 'error') {
    return (text) => ({ content: text, toolCalls: [] })
  }

  // TOPOLOGY 1: <tool_call> {"name": "foo"} </tool_call>
  if (markers.topology === 'json_encapsulated') {
    const prefix = escapeRegExp(markers.prefix)
    const suffix = escapeRegExp(markers.suffix) || '$' // Fallback to end-of-string

    // Match anything between the prefix and suffix globally
    const regex = new RegExp(`${prefix}([\\s\\S]*?)${suffix}`, 'g')

    return (text) => {
      const toolCalls = []
      let cleanContent = text

      let match
      while ((match = regex.exec(text)) !== null) {
        try {
          const payload = JSON.parse(match[1].trim())
          toolCalls.push({ name: payload.name, arguments: payload.arguments || payload.parameters })
          // Strip the tool call from the text meant for the user
          cleanContent = cleanContent.replace(match[0], '')
        } catch (e) {
          // Ignore invalid JSON (model hallucinated)
        }
      }
      return { content: cleanContent.trim(), toolCalls }
    }
  }

  // TOPOLOGY 2: <|tool_call|>name{"args": 1}<|end_call|>
  if (markers.topology === 'custom_segmented') {
    const namePrefix = escapeRegExp(markers.namePrefix)
    const nameSuffix = escapeRegExp(markers.nameSuffix)
    const argsSuffix = escapeRegExp(markers.argsSuffix) || '$'

    // Group 1: Name, Group 2: JSON Arguments
    const regex = new RegExp(`${namePrefix}(.*?)${nameSuffix}([\\s\\S]*?)${argsSuffix}`, 'g')

    return (text) => {
      const toolCalls = []
      let cleanContent = text

      let match
      while ((match = regex.exec(text)) !== null) {
        try {
          const name = match[1].trim()
          const args = JSON.parse(match[2].trim())
          toolCalls.push({ name, arguments: args })
          cleanContent = cleanContent.replace(match[0], '')
        } catch (e) {
          // Ignore invalid JSON
        }
      }
      return { content: cleanContent.trim(), toolCalls }
    }
  }

  // TOPOLOGY 3: implicit_json (Llama 3.1)
  if (markers.topology === 'implicit_json') {
    return (text) => {
      try {
        // Find the first { and last }
        const start = text.indexOf('{')
        const end = text.lastIndexOf('}')
        if (start !== -1 && end > start) {
          const payload = JSON.parse(text.slice(start, end + 1))
          return { content: text.slice(0, start).trim(), toolCalls: [{ name: payload.name, arguments: payload.arguments }] }
        }
      } catch (e) { /* ignore */ }
      return { content: text, toolCalls: [] }
    }
  }
}

// ------------------------------------------------------------------
// 2. Demo Execution
// ------------------------------------------------------------------

// Simulated Jinja templates (Just standard JS functions for the demo)
const mockGraniteTemplate = (ctx) => {
  // Simulates JSON Encapsulated (XML tags around JSON)
  const lastMsg = ctx.messages[ctx.messages.length - 1]
  if (lastMsg.tool_calls) {
    const tc = lastMsg.tool_calls[0].function
    return `Let me fetch that for you.\n<tool_call>\n{"name": "${tc.name}", "arguments": ${JSON.stringify(tc.arguments)}}\n</tool_call>`
  }
  return lastMsg.content
}

const mockGemmaTemplate = (ctx) => {
  // Simulates Custom Segmented (Name outside the JSON, specific syntax)
  const lastMsg = ctx.messages[ctx.messages.length - 1]
  if (lastMsg.tool_calls) {
    const tc = lastMsg.tool_calls[0].function
    return `I will use a tool.<|tool_call>call:${tc.name}${JSON.stringify(tc.arguments)}<tool_call|>`
  }
  return lastMsg.content
}

// --- START DEMO ---

console.log('=== Boot Sequence: Compiling Extractors ===')

// 1. At startup, analyze the template boundaries
const graniteMarkers = analyzeToolMarkers(mockGraniteTemplate)
console.log('\n[Granite Analysis]:', graniteMarkers)

const gemmaMarkers = analyzeToolMarkers(mockGemmaTemplate)
console.log('\n[Gemma Analysis]:', gemmaMarkers)

// 2. Compile our optimized non-streaming RegExp extractors
const extractGranite = createNonStreamingExtractor(graniteMarkers)
const extractGemma = createNonStreamingExtractor(gemmaMarkers)

console.log('\n=== Runtime Sequence: Processing stream:false Completions ===')

// 3. Simulate a raw string response from the LLM via MLX/C++ backend
const rawGraniteCompletion = `Certainly! I'll check the weather for Paris.
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris, FR"}}
</tool_call>
I will let you know once I have the data.`

const rawGemmaCompletion = '<|channel>thought\nI need to look this up.\n<channel|>I will use a tool.<|tool_call>call:get_weather{"location":"Paris, FR"}<tool_call|>'

// 4. Extract instantly using the compiled regex
console.time('Granite Extraction Time')
const graniteResult = extractGranite(rawGraniteCompletion)
console.timeEnd('Granite Extraction Time')
console.log('\n[Granite Final Output]:')
console.log('  Clean Content:', JSON.stringify(graniteResult.content))
console.log('  Extracted Tools:', graniteResult.toolCalls)

console.time('Gemma Extraction Time')
const gemmaResult = extractGemma(rawGemmaCompletion)
console.timeEnd('Gemma Extraction Time')
console.log('\n[Gemma Final Output]:')
console.log('  Clean Content:', JSON.stringify(gemmaResult.content))
console.log('  Extracted Tools:', gemmaResult.toolCalls)
