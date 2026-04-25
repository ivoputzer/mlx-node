import Testing
import Foundation
import MLX
@testable import MLXBridge

@Suite struct MLXBridgeTests {

    @Test func testBridgeMetrics() throws {
        // 1. Call the C-bridged function
        let cStringPtr = bridge_metrics()

        // 2. #require is the proper way to unwrap in Swift Testing
        // If this is nil, the test stops and fails here.
        let ptr = try #require(cStringPtr)

        // 3. Convert to String and ensure we free the C memory
        let jsonString = String(cString: ptr)
        free(ptr)

        print("Received JSON: \(jsonString)")

        // 4. Validate the JSON by decoding it back into an MLX Memory Snapshot
        // This is the ultimate test: does our JSON match MLX's own data structure?
        let data = try #require(jsonString.data(using: .utf8))

        // We decode directly into Memory.Snapshot because it is Codable!
        let decodedSnapshot = try JSONDecoder().decode(Memory.Snapshot.self, from: data)

        // 5. Assertions
        #expect(decodedSnapshot.activeMemory >= 0)
        #expect(decodedSnapshot.peakMemory >= 0)
        #expect(decodedSnapshot.cacheMemory >= 0)
    }
}
