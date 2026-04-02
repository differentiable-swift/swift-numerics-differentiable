// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "swift-numerics-differentiable",
    platforms: [
        .macOS("26.0"),
        .iOS("26.0"),
    ],
    products: [
        .library(
            name: "NumericsDifferentiable",
            targets: ["NumericsDifferentiable"]
        ),
        .library(
            name: "RealModuleDifferentiable",
            targets: ["RealModuleDifferentiable"]
        ),
        .library(
            name: "ComplexModuleDifferentiable",
            targets: ["ComplexModuleDifferentiable"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", from: "1.1.0"),
        .package(url: "https://github.com/differentiable-swift/swift-differentiation.git", from: "2.0.0"),
    ],
    targets: [
        .executableTarget(name: "CodeGeneratorExecutable"),
        .plugin(
            name: "CodeGeneratorPlugin",
            capability: .buildTool,
            dependencies: ["CodeGeneratorExecutable"]
        ),
        .target(
            name: "NumericsDifferentiable",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics"),
                "RealModuleDifferentiable",
                "ComplexModuleDifferentiable",
            ]
        ),
        .target(
            name: "RealModuleDifferentiable",
            dependencies: [
                .product(name: "RealModule", package: "swift-numerics"),
                .product(name: "Differentiation", package: "swift-differentiation"),
            ],
            plugins: [
                "CodeGeneratorPlugin",
            ]
        ),
        .target(
            name: "ComplexModuleDifferentiable",
            dependencies: [
                .product(name: "ComplexModule", package: "swift-numerics"),
                .product(name: "Differentiation", package: "swift-differentiation"),
            ]
        ),
        .testTarget(
            name: "RealModuleDifferentiableTests",
            dependencies: [
                "RealModuleDifferentiable",
            ]
        ),
        .testTarget(
            name: "ComplexModuleDifferentiableTests",
            dependencies: [
                "ComplexModuleDifferentiable",
            ]
        ),
    ]
)
