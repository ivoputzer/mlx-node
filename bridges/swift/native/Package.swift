// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MLXBridge",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MLXBridge", type: .static, targets: ["MLXBridge"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MLXBridge",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm")
            ],
            path: "Sources"
        )
    ]
)
