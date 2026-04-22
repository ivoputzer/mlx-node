import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

struct BridgeTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
    let tokenizer: Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        // swift-transformers handles special tokens internally based on config
        return tokenizer.encode(text: text)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        return tokenizer.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    // Fix: Using encode/decode for single token conversions
    func convertTokenToId(_ token: String) -> Int? {
        return tokenizer.encode(text: token).first
    }

    func convertIdToToken(_ id: Int) -> String? {
        return tokenizer.decode(tokens: [id])
    }

    // These properties exist in the Tokenizers.Tokenizer protocol
    var bosToken: String? { tokenizer.bosToken }
    var eosToken: String? { tokenizer.eosToken }
    var unknownToken: String? { tokenizer.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        // Convert the generic [String: any Sendable] to [String: String]
        // which the HuggingFace library expects for messages.
        let hfMessages = messages.map { dict in
            dict.reduce(into: [String: String]()) { (result, item) in
                if let value = item.value as? String {
                    result[item.key] = value
                }
            }
        }

        // Fix: This returns [Int] directly, no need to encode() again
        return try tokenizer.applyChatTemplate(messages: hfMessages)
    }
}

struct BridgeTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return BridgeTokenizer(tokenizer: tokenizer)
    }
}

// --- Bridge Logic ---

private var modelContext: ModelContext?

@_cdecl("mlx_swift_load_model")
public func loadModel(path: UnsafePointer<CChar>) -> Int32 {
    let pathString = String(cString: path)
    let url = URL(fileURLWithPath: pathString)
    let semaphore = DispatchSemaphore(value: 0)
    var success: Int32 = 0

    Task {
        do {
            modelContext = try await LLMModelFactory.shared.load(
                from: url,
                using: BridgeTokenizerLoader()
            )
            success = 1
        } catch {
            print("❌ Swift Error: \(error)")
            success = 0
        }
        semaphore.signal()
    }
    semaphore.wait()
    return success
}

@_cdecl("mlx_swift_generate")
public func generate(prompt: UnsafePointer<CChar>) -> UnsafeMutablePointer<CChar>? {
    let promptString = String(cString: prompt)
    let semaphore = DispatchSemaphore(value: 0)
    var resultString = ""

    Task {
        guard let context = modelContext else {
            resultString = "Error: Model not loaded"
            semaphore.signal()
            return
        }
        do {
            let session = ChatSession(context)
            resultString = try await session.respond(to: promptString)
        } catch {
            resultString = "Error: \(error.localizedDescription)"
        }
        semaphore.signal()
    }
    semaphore.wait()
    return strdup(resultString)
}
