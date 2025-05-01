// These extensions are here due to sqrt(_:) being defined on the `Real` protocol and we currently can't define
// default derivatives for protocol requirements. So we have to create a concrete implementation for each type
// to attach the derivatives to.
extension Float {
    @_transparent
    public static func sqrt(_ x: Float) -> Float {
        x.squareRoot()
    }
}

extension Double {
    @_transparent
    public static func sqrt(_ x: Double) -> Double {
        x.squareRoot()
    }
}

#if !(os(macOS) || os(iOS) || os(tvOS) || os(watchOS))
// This is a concrete version of the default implementation on the `Real` protocol
// this exists here so we can associate a derivative with this function on platforms that do not have a math library that provides exp10
extension Float {
    @_transparent
    public static func exp10(_ x: Float) -> Float {
        pow(10, x)
    }
}

extension Double {
    @_transparent
    public static func exp10(_ x: Double) -> Double {
        pow(10, x)
    }
}
#endif

// Extensions so that SIMD can have a fallback `abs` implementation with similar api as the RealFunctions protocol
extension Float {
    @_transparent
    public static func abs(_ x: Float) -> Float {
        Swift.abs(x)
    }
}

extension Double {
    @_transparent
    public static func abs(_ x: Double) -> Double {
        Swift.abs(x)
    }
}

extension Real {
    @_transparent
    public static func abs(_ x: Self) -> Self {
        Swift.abs(x)
    }
}
