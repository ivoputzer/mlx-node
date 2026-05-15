// Extracts and deduplicates all stop tokens (eos, eot, pad) from the tokenizer config.
// Supports string tokens, numeric IDs, and arrays of IDs (e.g., Llama 3).
export function stopTokensFrom (tokenizer) {
  const stopTokens = new Map() // Use Map to deduplicate by ID

  const addToken = (id, text) => {
    if (id !== undefined && id !== null && !stopTokens.has(id)) {
      // If text wasn't provided, try to resolve it. Fallback to empty string if missing.
      const resolvedText = text ?? tokenizer.id_to_token(id) ?? '<unknown>'
      stopTokens.set(id, { id, text: resolvedText })
    }
  }

  for (const [key, value] of Object.entries(tokenizer.config)) {
    if (key.includes('eos_token') || key.includes('eot_token')) {
      // Normalize to array so we can handle single values and Llama-3 style arrays identically
      const values = Array.isArray(value) ? value : [value]

      for (const val of values) {
        if (typeof val === 'string' && tokenizer.model.tokens_to_ids.has(val)) {
          addToken(tokenizer.token_to_id(val), val)
        } else if (typeof val === 'number') {
          addToken(val, undefined) // We'll look up the text inside `addToken`
        }
      }
    }
  }

  return Array.from(stopTokens.values())
}

// Determines the most appropriate padding token and its ID from a given tokenizer.
// Safely resolves the padding token in the following order: pad_token_id, pad_token, unk_token_id, unk_token (safest "white noise" fallback for kvcache)
// Note: This intentionally does NOT fallback to `eos_token` or `0` to prevent poisoning the KV Cache during assistant continuation or heterogeneous batching!
export function padTokenFrom (tokenizer) {
  if (tokenizer.config.pad_token_id !== undefined && tokenizer.config.pad_token_id !== null) {
    return {
      id: tokenizer.config.pad_token_id,
      text: tokenizer.config.pad_token ?? tokenizer.id_to_token(tokenizer.config.pad_token_id)
    }
  }

  if (tokenizer.config.pad_token && tokenizer.model.tokens_to_ids.has(tokenizer.config.pad_token)) {
    return {
      id: tokenizer.config.pad_token_id ?? tokenizer.token_to_id(tokenizer.config.pad_token),
      text: tokenizer.config.pad_token
    }
  }

  // The safest "white noise" for Mamba KVCache
  if (tokenizer.config.unk_token_id !== undefined && tokenizer.config.unk_token_id !== null) {
    return {
      id: tokenizer.config.unk_token_id,
      text: tokenizer.config.unk_token ?? tokenizer.id_to_token(tokenizer.config.unk_token_id)
    }
  }

  if (tokenizer.config.unk_token && tokenizer.model.tokens_to_ids.has(tokenizer.config.unk_token)) {
    return {
      id: tokenizer.config.unk_token_id ?? tokenizer.token_to_id(tokenizer.config.unk_token),
      text: tokenizer.config.unk_token
    }
  }

  return null
}
