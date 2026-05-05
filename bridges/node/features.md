# Features

### 1. Core Architecture & Interop
- Swift/C/JS FFI Bridge: Zero-overhead pointers passing between Metal, C, and V8.
- Headless Tokenization: NullTokenizer in Swift delegates all BPE/SentencePiece vocabulary logic to the JS layer (@huggingface/tokenizers), allowing decoupled updates.
- Explicit Memory Management: RAII-style MLXResource with dispose() and Symbol.dispose to prevent VRAM memory leaks.
- Task Cancellation: Native AbortController support in JS that propagates instantly to Swift Task cancellation, killing Metal compute immediately.
- System Metrics: Real-time RAM/VRAM introspection (MLXMetrics) tracking Active Memory, Cache Pool, Peak Watermarks, and Limits.

### 2. Generation & Inference Mechanics
- Homogeneous Batch Generation (batchSize > 1): Evaluating the same prompt multiple times simultaneously via Metal matrix broadcasting.
- Heterogeneous Batching: Packing different length prompts into a single tensor via JS left-padding (createPaddedBatch).
- Chunked Prefilling: prefillStepSize support to prevent VRAM OOM on massive prompts.
- Async Token Streaming: Native JS AsyncGenerator yielding tokens tick-by-tick.
- Advanced Samplers: Temperature, Top-P, Top-K, Min-P, Repetition/Presence/Frequency Penalties, and custom Context Sizes for penalties.

### 3. State-of-the-Art KV Cache Management
- The "Open Mouth" Paradigm: Stop tokens are intentionally omitted from the cache, enabling seamless Assistant Continuation, Output Forcing, and Mid-Flight injection.
- Gibberish Prevention: Swift traps completed sequences in a batch and feeds them <pad> or <unk> tokens, preserving cache math and protecting RoPE.
- Cache Slicing (Branching): Extracting specific sequences from a batched cache (e.g., extracting the best branch of a Monte Carlo Tree Search).
- Auto-Trimming: Swift automatically tracks and trims padded tokens when slicing to restore the cache to a pristine state.
- Time Travel (Manual Trimming): cache.trim(N) allows deleting the last N tokens to undo mistakes and regenerate without re-evaluating the prompt.
- Cache I/O: Saving to and loading from disk (.safetensors).
- Cache Cloning: In-memory duplication of KV state.
- KV Quantization: On-the-fly KV cache quantization (Bit size, Group Size, Quantized Start Index) to save VRAM on massive contexts.

### 4. Prompting & LLM UX
- Jinja Chat Templates: Hugging Face spec compliant chat_template rendering.
- Dynamic Stop Tokens: Extracting EOS, EOT, and custom stop strings directly from the tokenizer config.


# Backlog
- Token Logit Inspector: Add an option to yield the Top 5 token probabilities alongside the chosen token. Essential for confidence scoring and debugging.
- Logit Bias: Allow users to pass a dictionary {"token_id": 100, "token_id_2": -100} to mathematically force or ban specific words.
- JSON Schema / Grammar Forcing: The #1 feature developers want. Constrain logits so the model can only output valid JSON matching a specific schema.

- Speculative Decoding: Run a tiny, fast model (e.g., 1.5B) to draft 5 tokens, and use the big model (e.g., 32B) to verify them in a single pass. 2x-3x speedup.
- Continuous In-Flight Batching: Instead of waiting for the longest sequence to finish, dynamically eject finished sequences and inject new incoming requests into the active GPU batch.
