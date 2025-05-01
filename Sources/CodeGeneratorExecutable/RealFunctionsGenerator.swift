struct RealFunctionsGenerator {
    static func realFunctionsExtension(objectType: String, type: String, whereClause: Bool, simdAccelerated: Bool) -> String {
        let elementaryFunctions = [
            RealFunction(name: "exp", simdName: "exp"),
            RealFunction(name: "expMinusOne", simdName: "expm1"),
            RealFunction(name: "cosh", simdName: "cosh"),
            RealFunction(name: "sinh", simdName: "sinh"),
            RealFunction(name: "tanh", simdName: "tanh"),
            RealFunction(name: "cos", simdName: "cos"),
            RealFunction(name: "sin", simdName: "sin"),
            RealFunction(name: "tan", simdName: "tan"),
            RealFunction(name: "log", simdName: "log"),
            RealFunction(name: "log", simdName: "log1p", arguments: [.init(name: "x", label: "onePlus")]),
            RealFunction(name: "acosh", simdName: "acosh"),
            RealFunction(name: "asinh", simdName: "asinh"),
            RealFunction(name: "atanh", simdName: "atanh"),
            RealFunction(name: "acos", simdName: "acos"),
            RealFunction(name: "asin", simdName: "asin"),
            RealFunction(name: "atan", simdName: "atan"),
            RealFunction(name: "pow", simdName: "pow", arguments: [.init(name: "x", label: "_"), .init(name: "n", label: "_", type: "Int")]),
            RealFunction(name: "pow", simdName: "pow", arguments: [.init(name: "x", label: "_"), .init(name: "y", label: "_")]),
            RealFunction(name: "sqrt"),
            RealFunction(name: "root", arguments: [.init(name: "x", label: "_"), .init(name: "n", label: "_", type: "Int")]),
        ]
        
        let realFunctions = [
            RealFunction(name: "atan2", simdName: "atan2", arguments: [.init(name: "y"), .init(name: "x")]),
            RealFunction(name: "erf", simdName: "erf"),
            RealFunction(name: "erfc", simdName: "erfc"),
            RealFunction(name: "exp2", simdName: "exp2"),
            RealFunction(name: "exp10", simdName: "exp10"),
            RealFunction(name: "hypot", simdName: "hypot", arguments: [.init(name: "x", label: "_"), .init(name: "y", label: "_")]),
            RealFunction(name: "gamma", simdName: "tgamma"),
            RealFunction(name: "log2", simdName: "log2"),
            RealFunction(name: "log10", simdName: "log10"),
            RealFunction(name: "logGamma", simdName: "lgamma"),
        ]
        
        let floatingPointFunctions = [
            RealFunction(name: "abs", simdName: "simd_abs"),
        ]

        let elementaryFunctionsCode = elementaryFunctions.map {
            realFunctionTemplate(for: $0, type: type, simdAccelerated: simdAccelerated)
        }.joined(separator: "\n\n")
        
        let realFunctionsCode = realFunctions.map {
            realFunctionTemplate(for: $0, type: type, simdAccelerated: simdAccelerated)
        }.joined(separator: "\n\n")
        
        let floatingPointFunctionsCode = floatingPointFunctions.map {
            realFunctionTemplate(for: $0, type: type, simdAccelerated: simdAccelerated)
        }.joined(separator: "\n\n")
        
        let acceleratedHeader = """
        #if canImport(simd)
        import simd
        #endif
        """
        
        return """
        \(simdAccelerated ? acceleratedHeader : "")
        import RealModule
        
        // MARK: ElementaryFunctions
        extension \(objectType)\(whereClause ? " where Scalar: ElementaryFunctions" : "") {
        \(elementaryFunctionsCode)
        }
        
        // MARK: RealFunctions
        extension \(objectType)\(whereClause ? " where Scalar: RealFunctions" : "") {
        \(realFunctionsCode)
        
            // signGamma is missing here since we cannot return a SIMDX<FloatingPointSign> Otherwise we could also conform SIMD types to the RealFunctions protocol.
            //    @_transparent
            //    public static func signGamma(_ x: Self) -> SIMDX<FloatingPointSign> {
            //        fatalError()
            //    }
        }
        
        // MARK: FloatingPointFunctions
        extension \(objectType)\(whereClause ? " where Scalar: Real" : "") {
        \(floatingPointFunctionsCode)
        }
        """
    }
    
    static func realFunctionTemplate(for function: RealFunction, type: String, simdAccelerated: Bool) -> String {
        let interfaceArguments: String = function.arguments.map {
            if let label = $0.label {
                return "\(label) \($0.name): \($0.type ?? type)"
            } else {
                return "\($0.name): \($0.type ?? type)"
            }
        }.joined(separator: ", ")
        
        let implementationArguments = function.arguments.map {
            if let label = $0.label {
                "\(label == "_" ? "" : "\(label): ")\($0.name)\($0.type == nil ? "[i]" : "")"
            } else {
                "\($0.name): \($0.name)\($0.type == nil ? "[i]" : "")"
            }
        }.joined(separator: ", ")
        
        let regularImplementation = """
            @_transparent
            public static func \(function.name)(\(interfaceArguments)) -> \(type) {
                var v = Self()
                for i in v.indices {
                    v[i] = .\(function.name)(\(implementationArguments))
                }
                return v
            }
        """
        
        guard simdAccelerated else { return regularImplementation }

        // we return the regular implementation if no simd equivalent is present (currently only true for sqrt and root)
        guard let simdName = function.simdName else { return regularImplementation }
        
        let acceleratedArguments = function.arguments.map { arg in "\(arg.type.map { _ in ".init(repeating: .init(\(arg.name)))" } ?? arg.name)" }.joined(separator: ", ")
        
        let acceleratedImplementation = """
            #if canImport(simd)
            @_transparent
            public static func \(function.name)(\(interfaceArguments)) -> \(type) {
                simd.\(simdName)(\(acceleratedArguments))
            }
            #else
        \(regularImplementation)
            #endif
        """
        
        return acceleratedImplementation
    }
}
