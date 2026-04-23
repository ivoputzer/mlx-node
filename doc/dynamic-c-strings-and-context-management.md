## Technical Summary: Dynamic C-String Buffers & Context Window Management

### 1. TODO Status (`readme.md`, line 19)

"Dynamic C-string buffers for massive context windows (>32k)" is listed as an **uncompleted Phase 1 task**. The checkbox `[ ]` indicates this has not yet been implemented. This limitation directly stems from the fixed-size stack buffers in `binding.c`.

### 2. Current Swift-Side Streaming Buffering (`MLXBridge.swift`)

The streaming path (lines 151–220) uses a **Swift `String` accumulator** (`chunkBuffer`) that collects decoded token chunks and flushes them to the C callback when `tokenCount >= chunkSize` (default: 4 tokens). Each flush converts the Swift string via `.withCString { cChunk in callback(...) }`, which creates a temporary null-terminated C-string view over the Swift buffer's storage. The remaining text is flushed after the stream ends. This approach works fine for small chunks but has no mechanism to handle payloads exceeding ~32k characters — the downstream NAPI layer cannot accept them (see below).

### 3. NAPI/C-Side Memory Management (`binding.c`) — **The Core Bottleneck**

This is where the 32k ceiling originates:

- **Incoming prompts**: `char prompt[32768]` on the stack (line 198 for `Generate`, line 226 for `GenerateStream`). The call to `napi_get_value_string_utf8(env, args[1], prompt, sizeof(prompt), &result)` silently truncates any JavaScript string longer than ~32k characters. This is a **hard limit** — prompts exceeding this size are truncated without error.

- **Incoming config JSON**: `char config_json[4096]` on the stack (line 202/230). Same truncation risk for large configuration objects.

- **Outgoing payloads**: When Swift returns a generated response, it passes a C-string pointer through the callback chain. The C bridge allocates heap memory via `malloc(sizeof(PromiseResult))` and `strdup(payload)` (lines 84–87, 95–97) to own the string across thread boundaries. The JS-side TSFN callback (`CallJsPromise`, line 51) then calls `napi_create_string_utf8(env, result->payload, NAPI_AUTO_LENGTH, &js_result)` and finally `free(result->payload)` (line 76). This pattern is correct for memory safety but **assumes the payload fits within a single C-string** — there's no streaming or chunked transfer mechanism from Swift back to JS for large outputs.

- **Stream chunks**: Similarly, each stream chunk is `strdup`'d into a `StreamResult` (line 143) and freed after delivery (lines 130–134). No cumulative buffer management exists on the C side.

### 4. Context Window Delegation to MLX Native Libraries

The Node.js bridge **does not manage context windows at all** — it delegates entirely to Apple's MLX Swift libraries:

- **Tokenization**: `BridgeTokenizer` (Swift, line 35) wraps `Tokenizers.Tokenizer`, encoding the prompt string into token IDs via `tokenizer.encode(text: promptStr)` and decoding output tokens back to strings.
- **KV Cache**: Created per-generation via `model.newCache(parameters: parameters)` (line 129/178), which allocates the attention key-value cache on Metal GPU memory. The size of this cache is determined by the model's architecture, not the bridge.
- **Generation Loop**: `TokenIterator` and `MLXLMCommon.generateTask()` handle the autoregressive decoding loop internally within MLX. The bridge only observes the output stream — it never inspects or manipulates token sequences, attention states, or context length.

### Summary of the Gap

The 32k limit is a **NAPI boundary problem**, not an MLX limitation. MLX itself supports arbitrarily large context windows (determined by model architecture). The bridge's fixed stack buffers (`char prompt[32768]`) and single-shot C-string return paths cannot accommodate prompts or outputs exceeding ~32k characters. Solving this would require replacing the static `char[]` buffers with dynamically allocated, size-aware buffers — potentially using `napi_get_value_string_utf8`'s length query to allocate precisely-sized heap memory, and/or implementing a chunked/streaming return path for large generation results.
