import Foundation
import PackagePlugin

@main
struct CodeGeneratorPlugin: BuildToolPlugin {
    func createBuildCommands(context: PackagePlugin.PluginContext, target _: PackagePlugin.Target) async throws -> [PackagePlugin.Command] {
        let output = context.pluginWorkDirectoryURL

        let floatingPointTypes: [String] = ["Float", "Double"]
        let simdWidths = [2, 4, 8, 16, 32, 64]

        let outputFiles = floatingPointTypes.flatMap { floatingPointType in
            simdWidths.flatMap { simdWidth in
                [
                    output.appending(component: "SIMD\(simdWidth)+\(floatingPointType)+RealFunctions.swift"),
                    output.appending(component: "SIMD\(simdWidth)+\(floatingPointType)+RealFunctions+Derivatives.swift"),
                ]
            } + [
                output.appending(component: "\(floatingPointType)+RealFunctions+Derivatives.swift"),
            ]
        } + [
            output.appending(component: "SIMD+RealFunctions.swift"),
        ]

        return [
            .buildCommand(
                displayName: "Generate Code",
                executable: try context.tool(named: "CodeGeneratorExecutable").url,
                arguments: [output.relativePath],
                environment: [:],
                inputFiles: [],
                outputFiles: outputFiles
            ),
        ]
    }
}
