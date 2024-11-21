// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "swift-numerics-differentiable",
    products: [
        .library(
            name: "NumericsDifferentiable",
            targets: ["NumericsDifferentiable"]
        ),
        .library(
            name: "RealModuleDifferentiable",
            targets: ["RealModuleDifferentiable"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.2")
    ],
    targets: [
        .target(
            name: "NumericsDifferentiable",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics"),
                "RealModuleDifferentiable",
            ]
        ),
        .target(
            name: "RealModuleDifferentiable",
            dependencies: [
                .product(name: "RealModule", package: "swift-numerics")
            ]
        ),
        .testTarget(
            name: "RealModuleDifferentiableTests",
            dependencies: [
                "RealModuleDifferentiable"
            ]
        ),
    ]
)
