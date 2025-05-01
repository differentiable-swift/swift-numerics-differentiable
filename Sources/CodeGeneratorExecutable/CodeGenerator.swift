import Foundation

@main
struct CodeGenerator {
    static func main() throws {
        // Use swift-argument-parser or just CommandLine, here we just imply that 2 paths are passed in: input and output
        guard CommandLine.arguments.count == 2 else {
            throw CodeGeneratorError.invalidArguments
        }
        // arguments[0] is the path to this command line tool
        let output = URL(filePath: CommandLine.arguments[1])

        // generate default implementations of RealFunctions for SIMD protocol
        let realFunctionSIMDFileURL = output.appending(component: "SIMD+RealFunctions.swift")
        let realFunctionsSIMDExtension = RealFunctionsGenerator.realFunctionsExtension(objectType: "SIMD", type: "Self", whereClause: true, simdAccelerated: false)
        try realFunctionsSIMDExtension.write(to: realFunctionSIMDFileURL, atomically: true, encoding: .utf8)
        
        let floatingPointTypes: [String] = ["Float", "Double"]
        let simdSizes: [Int] = [2, 4, 8, 16, 32, 64]

        for floatingPointType in floatingPointTypes {
            // Generator Derivatives for RealFunctions for floating point types
            let realFunctionDerivativesFileURL = output.appending(
                component: "\(floatingPointType)+RealFunctions+Derivatives.swift",
                directoryHint: .notDirectory
            )
            let type = floatingPointType
            let realFunctionsDerivativesExtensionCode = RealFunctionsDerivativesGenerator.realFunctionsDerivativesExtension(type: type, floatingPointType: floatingPointType)
            try realFunctionsDerivativesExtensionCode.write(to: realFunctionDerivativesFileURL, atomically: true, encoding: .utf8)
            
            for simdSize in simdSizes {
                let realFunctionFileURL = output.appending(
                    component: "SIMD\(simdSize)+\(floatingPointType)+RealFunctions.swift",
                    directoryHint: .notDirectory
                )
                let simdType = "SIMD\(simdSize)<\(floatingPointType)>"
                
                // no simd methods exist for simd size >= 16 and scalar > Float so we don't add acceleration to those.
                var simdAccelerated: Bool
                if simdSize > 16 || (simdSize == 16 && floatingPointType == "Double") {
                    simdAccelerated = false
                } else {
                    simdAccelerated = true
                }
                
                // Generate RealFunctions implementations on concrete SIMD types to attach derivatives to
                let realFunctionsExtensionCode = RealFunctionsGenerator.realFunctionsExtension(objectType: simdType, type: simdType, whereClause: false, simdAccelerated: simdAccelerated)
                try realFunctionsExtensionCode.write(to: realFunctionFileURL, atomically: true, encoding: .utf8)
                
                // Generate RealFunctions derivatives for concrete SIMD types
                let realFunctionDerivativesFileURL = output.appending(component: "SIMD\(simdSize)+\(floatingPointType)+RealFunctions+Derivatives.swift")
                let type = "SIMD\(simdSize)<\(floatingPointType)>"
                let realFunctionsDerivativesExtensionCode = RealFunctionsDerivativesGenerator.realFunctionsDerivativesExtension(type: type, floatingPointType: floatingPointType)
                try realFunctionsDerivativesExtensionCode.write(to: realFunctionDerivativesFileURL, atomically: true, encoding: .utf8)
            }
        }
    }
}

struct RealFunction {
    var name: String
    var simdName: String?
    var arguments: [Argument]
    
    struct Argument {
        var name: String
        var label: String?
        var type: String? = nil
    }
    
    init(name: String, simdName: String? = nil, arguments: [Argument] = [.init(name: "x", label: "_")]) {
        self.name = name
        self.simdName = simdName
        self.arguments = arguments
    }
}

enum CodeGeneratorError: Error {
    case invalidArguments
    case invalidData
}
