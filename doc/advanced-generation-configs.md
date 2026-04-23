# Advanced Generation Configs

## Overview

"Advanced generation configs (stop sequences, logit bias)" is listed as an **uncompleted Phase 1 task** in the project roadmap (`readme.md`, line 20). This document summarizes the current state of generation configuration support across the codebase and outlines how stop sequences and logit bias would integrate into the existing architecture.

## Current State

### Implemented Config Options

The `BridgeGenerateConfig` struct in `bridges/swift/native/Sources/MLXBridge/MLXBridge.swift` (lines 8–23) currently supports three generation parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `temperature` | Float? | 0.6 | Sampling temperature for token selection |
| `topP` | Float? | 1.0 | Nucleus sampling threshold |
| `repetitionPenalty` | Float? | 1.0 | Penalty applied to repeated tokens |
| `repetitionContextSize` | Int? | 20 | Window size for repetition penalty calculation |
| `streamChunkSize` | Int? | 4 | Tokens accumulated per stream flush (JS-side only) |

These map directly to Apple's MLX `GenerateParameters` struct via the `toGenerateParameters()` method:

```swift
func toGenerateParameters() -> GenerateParameters {
    return GenerateParameters(
        temperature: temperature ?? 0.6,
        topP: topP ?? 1.0,
        repetitionPenalty: repetitionPenalty ?? 1.0,
        repetitionContextSize: repetitionContextSize ?? 20
    )
}
```

### Config Data Flow

Configuration flows through three layers:

1. **JavaScript API** (`packages/lm/index.js`): Accepts a plain JS object as the third argument to `generate()` and `stream()`, serialized via `JSON.stringify(config)`.

2. **N-API Bridge** (`bridges/swift/native/binding.c`): Reads the JSON string into a fixed-size stack buffer (`char config_json[4096]`) at lines 202–203 (Generate) and 230–231 (GenerateStream), then passes it as a C-string to Swift.

3. **Swift Layer** (`bridges/swift/native/Sources/MLXBridge/MLXBridge.swift`): Deserializes the JSON string via `JSONDecoder().decode(BridgeGenerateConfig.self, from: data)` in the `parseConfig()` function (lines 25–31), then converts to MLX's native `GenerateParameters`.

## Stop Sequences

### What They Are

Stop sequences are token strings that, when detected in the generated output, cause generation to terminate early. For example, setting a stop sequence of `"\\nUser:"` would halt generation as soon as the model produces that exact string.

### Current Status: Not Implemented

The `BridgeGenerateConfig` struct has no field for stop sequences, and `toGenerateParameters()` does not pass any stop-related values to MLX's `GenerateParameters`. The underlying MLX library supports this natively — it would require adding a `stopSequences: [String]?` property to the config struct.

### Proposed Implementation

```swift
struct BridgeGenerateConfig: Decodable {
    var temperature: Float?
    var topP: Float?
    var repetitionPenalty: Float?
    var repetitionContextSize: Int?
    var streamChunkSize: Int?
    var stopSequences: [String]?  // NEW

    func toGenerateParameters() -> GenerateParameters {
        return GenerateParameters(
            temperature: temperature ?? 0.6,
            topP: topP ?? 1.0,
            repetitionPenalty: repetitionPenalty ?? 1.0,
            repetitionContextSize: repetitionContextSize ?? 20,
            stopSequences: stopSequences ?? []  // NEW
        )
    }
}
```

### Usage Example (JavaScript)

```javascript
import { load, generate } from 'mlx-lm';

const model = await load('/path/to/model');

const response = await generate(model, "Write a story about a robot.", {
  temperature: 0.7,
  stopSequences: ["\\n\\n", "\\nStory:", "<|end|>"]
});

console.log(response);
unload(model);
```

### Implementation Notes

- Stop sequences are checked against the **decoded text output**, not raw token IDs. The bridge would need to compare each newly decoded chunk (in both `mlx_swift_generate` and `mlx_swift_generate_stream`) against the configured stop strings.
- In the streaming path, partial matches require buffering — if a generated chunk partially overlaps with a stop sequence, the remaining suffix must be retained for the next iteration.

## Logit Bias

### What It Is

Logit bias allows injecting per-token ID adjustments into the generation process. Positive values increase the probability of selecting a specific token; negative values decrease it. This is useful for steering output toward or away from certain words, enforcing vocabulary constraints, or implementing custom sampling strategies.

### Current Status: Not Implemented

No logit bias field exists in `BridgeGenerateConfig`, and no bias array is passed to MLX's generation pipeline. The underlying MLX framework supports this via a token ID-to-bias mapping in `GenerateParameters`.

### Proposed Implementation

```swift
struct BridgeGenerateConfig: Decodable {
    var temperature: Float?
    var topP: Float?
    var repetitionPenalty: Float?
    var repetitionContextSize: Int?
    var streamChunkSize: Int?
    var stopSequences: [String]?
    var logitBias: [Int: Float]?  // NEW — maps token ID to bias value

    func toGenerateParameters() -> GenerateParameters {
        return GenerateParameters(
            temperature: temperature ?? 0.6,
            topP: topP ?? 1.0,
            repetitionPenalty: repetitionPenalty ?? 1.0,
            repetitionContextSize: repetitionContextSize ?? 20,
            stopSequences: stopSequences ?? [],
            logitBias: Dictionary(uniqueKeysWithValues: logitBias?.map { ($0, $1) } ?? [])  // NEW
        )
    }
}
```

### Usage Example (JavaScript)

```javascript
import { load, generate } from 'mlx-lm';

const model = await load('/path/to/model');

// Force the model to use specific tokens by ID
const response = await generate(model, "The capital of France is", {
  temperature: 0.1,
  logitBias: {
    12345: 5.0,   // Strongly bias toward token 12345 ("Paris")
    67890: -10.0  // Suppress token 67890 (some unrelated word)
  }
});

console.log(response);
unload(model);
```

### Implementation Notes

- Token IDs must be resolved from the model's vocabulary before bias application. The JS API would accept numeric token IDs directly, requiring the Swift layer to validate them against the loaded tokenizer.
- Bias values are typically in the range of -10 to +10 (MLX convention). Values outside this range may cause numerical instability.

## Architecture Summary

```
JavaScript (mlx-lm)
  └── generate(modelId, prompt, config)
        │ JSON.stringify(config)
        ▼
N-API Bridge (binding.c)
  └── char config_json[4096]   ← fixed buffer, truncation risk for large configs
        │ C-string to Swift
        ▼
Swift Layer (MLXBridge.swift)
  └── BridgeGenerateConfig → GenerateParameters
        │ MLX generation API
        ▼
Apple MLX Framework
  └── TokenIterator + generateTask()
```

## Related Limitations

### Fixed Buffer Sizes (`binding.c`)

The N-API bridge uses fixed-size stack buffers that impose hard limits:

- **Prompt buffer**: `char prompt[32768]` — silently truncates prompts exceeding ~32k characters (documented in `doc/dynamic-c-strings-and-context-management.md`).
- **Config JSON buffer**: `char config_json[4096]` — large configuration objects with many stop sequences or logit bias entries could exceed this limit.

These buffers are listed as a Phase 1 TODO item and would need to be replaced with dynamically allocated, size-aware buffers before advanced configs can support large arrays of stop sequences or extensive logit bias maps.

### No Streaming Return Path for Large Outputs

Generation results are returned via single-shot C-string callbacks (`strdup` + `napi_create_string_utf8`). There is no chunked transfer mechanism from Swift back to JS, which limits output size regardless of prompt buffer capacity.
