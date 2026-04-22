// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXNative",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MLXNative", type: .static, targets: ["MLXNative"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MLXNative",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources"
        )
    ]
)
