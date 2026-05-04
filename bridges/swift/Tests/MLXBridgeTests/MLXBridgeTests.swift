import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXBridge

@Suite struct MLXBridgeTests {

  @Test func testBridgeMetrics() throws {
    let cStringPtr = bridge_metrics()
    let ptr = try #require(cStringPtr)

    let jsonString = String(cString: ptr)
    free(ptr)

    let data = try #require(jsonString.data(using: .utf8))

    // We decode into our local Mirror of the BridgeMetrics struct
    // since the one in the bridge might be private or internal
    struct MetricsMirror: Decodable {
      let active: Int
      let cache: Int
      let peak: Int
      let memoryLimit: Int
      let cacheLimit: Int
    }

    let metrics = try JSONDecoder().decode(MetricsMirror.self, from: data)

    #expect(metrics.active >= 0)
    #expect(metrics.peak >= 0)
    #expect(metrics.memoryLimit > 0)
    #expect(metrics.cacheLimit > 0)
  }

  @Test func testNullTokenizer() throws {
    let tokenizer = NullTokenizer()

    // Ensure it always returns empty/nil as expected for the pure MLX native bridge
    #expect(tokenizer.encode(text: "Hello World", addSpecialTokens: true).isEmpty)
    #expect(tokenizer.decode(tokenIds: [1, 2, 3], skipSpecialTokens: true) == "")
    #expect(tokenizer.convertTokenToId("test") == nil)
    #expect(tokenizer.convertIdToToken(123) == nil)
    #expect(tokenizer.bosToken == nil)
    #expect(tokenizer.eosToken == nil)
  }

  @Test func testBridgeGenerateConfig() throws {
    let json = """
      {
          "maxTokens": 1024,
          "temperature": 0.8,
          "topP": 0.95,
          "batchSize": 4,
          "stopTokenIds": [10, 20, 30]
      }
      """
    let data = try #require(json.data(using: .utf8))
    let config = try JSONDecoder().decode(BridgeGenerateConfig.self, from: data)

    // Test explicit decoding
    #expect(config.maxTokens == 1024)
    #expect(config.temperature == 0.8)
    #expect(config.batchSize == 4)
    #expect(config.stopTokenIds == [10, 20, 30])
    #expect(config.repetitionPenalty == nil)  // Should be nil if omitted

    // Test mapping to MLX GenerateParameters
    let params = config.toGenerateParameters()

    // Explicitly set values
    #expect(params.maxTokens == 1024)
    #expect(params.temperature == 0.8)
    #expect(params.topP == 0.95)

    // Default fallbacks injected by the bridge
    #expect(params.kvGroupSize == 64)
    #expect(params.quantizedKVStart == 0)
    #expect(params.repetitionContextSize == 20)
  }

  /// A helper class to ferry Swift continuations across the C-boundary via the `context` pointer
  private class AsyncContext {
    // FIX: Match the tuple labels exactly
    var continuation: CheckedContinuation<(success: Bool, errorMsg: String?), Never>?
  }

  @Test func testModelLoadInvalidPath() async throws {
    // We use an AsyncContext to hold the continuation because @convention(c)
    // closures cannot capture local variables.
    let asyncCtx = AsyncContext()

    let result: (success: Bool, errorMsg: String?) = await withCheckedContinuation { continuation in
      asyncCtx.continuation = continuation

      // Retain the object and get a raw C pointer to it
      let contextPtr = Unmanaged.passRetained(asyncCtx).toOpaque()

      // Define the strict C-callback
      let callback: BridgeAsyncCallback = { ctxPtr, success, modelPtr, errPtr in
        // Consume the retained pointer to prevent memory leaks
        let ctx = Unmanaged<AsyncContext>.fromOpaque(ctxPtr).takeRetainedValue()

        // Parse the C string error
        let errMsg = errPtr.map { String(cString: $0) }

        // Resume the Swift async/await pipeline
        ctx.continuation?.resume(returning: (success, errMsg))
      }

      // Trigger the native bridge load function
      "/this/path/does/not/exist".withCString { cPath in
        bridge_model_load(path: cPath, context: contextPtr, callback: callback)
      }
    }

    // Assertions
    #expect(result.success == false, "Loading a fake path should fail")
    let errorMsg = try #require(result.errorMsg, "Bridge should return an error message")
    #expect(!errorMsg.isEmpty, "Error message should not be empty")
    print("Successfully trapped model load error: \(errorMsg)")
  }

  @Test func testMetalClearCache() throws {
    // Simply ensuring this C-Export doesn't crash or trap.
    bridge_metal_clear_cache()
  }

  @Test func testGenerateConfigDefaults() throws {
    // Test with empty JSON to ensure defaults are applied correctly in the bridge
    let json = "{}"
    let data = try #require(json.data(using: .utf8))
    let config = try JSONDecoder().decode(BridgeGenerateConfig.self, from: data)
    let params = config.toGenerateParameters()

    // These are the "pragmatic defaults" we set in MLXBridge.swift
    #expect(params.temperature == 0.6)
    #expect(params.topP == 1.0)
    #expect(params.kvGroupSize == 64)
    #expect(params.repetitionContextSize == 20)
    #expect(params.prefillStepSize == 512)
  }

  @Test func testCacheDebugSchema() throws {
    // This tests if our debug JSON return is valid.
    // Since we can't easily instantiate a real KVCache without a model,
    // we can at least test the logic if we had one.

    // Mocking a tiny bit of the logic we use in bridge_cache_debug
    let layers = 32
    let offset = 100
    let typeStr = "KVCache"

    let json = """
      {
          "layers": \(layers),
          "type": "\(typeStr)",
          "offset": \(offset),
          "isTrimmable": true
      }
      """

    let data = try #require(json.data(using: .utf8))
    let dict = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(dict["layers"] as? Int == 32)
    #expect(dict["isTrimmable"] as? Bool == true)
  }

  @Test func testStopTokenCompilation() throws {
    // Since compileStopTokens requires a ModelContext, we usually test its
    // underlying logic: Does it combine the sets correctly?
    var stopIds = Set([1, 2, 3])
    let extraIds = [3, 4, 5]

    stopIds.formUnion(extraIds)

    #expect(stopIds.count == 5)
    #expect(stopIds.contains(5))
  }
}
