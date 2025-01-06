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
        .library(
            name: "ComplexModuleDifferentiable",
            targets: ["ComplexModuleDifferentiable"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.2"),
    ],
    targets: [
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
            ]
        ),
        .target(
            name: "ComplexModuleDifferentiable",
            dependencies: [
                .product(name: "ComplexModule", package: "swift-numerics"),
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
