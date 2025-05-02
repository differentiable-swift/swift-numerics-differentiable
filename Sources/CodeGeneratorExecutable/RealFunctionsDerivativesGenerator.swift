enum RealFunctionsDerivativesGenerator {
    static func realFunctionsDerivativesExtension(type: String, floatingPointType: String) -> String {
        """
        #if canImport(_Differentiation)
        import _Differentiation
        import RealModule

        // MARK: ElementaryFunctions derivatives
        extension \(type) {
            @derivative(of: exp)
            public static func _vjpExp(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                let value = exp(x)
                return (value: value, pullback: { v in v * value })
            }

            @derivative(of: expMinusOne)
            public static func _vjpExpMinusOne(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                return (value: expMinusOne(x), pullback: { v in v * exp(x) })
            }

            @derivative(of: cosh)
            public static func _vjpCosh(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: cosh(x), pullback: { v in sinh(x) })
            }

            @derivative(of: sinh)
            public static func _vjpSinh(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: sinh(x), pullback: { v in cosh(x) })
            }

            @derivative(of: tanh)
            public static func _vjpTanh(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (
                    value: tanh(x),
                    pullback: { v in
                        let coshx = cosh(x)
                        return v / (coshx * coshx)
                    }
                )
            }

            @derivative(of: cos)
            public static func _vjpCos(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: cos(x), pullback: { v in -v * sin(x) })
            }

            @derivative(of: sin)
            public static func _vjpSin(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: sin(x), pullback: { v in v * cos(x) })
            }

            @derivative(of: tan)
            public static func _vjpTan(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (
                    value: tan(x), 
                    pullback: { v in 
                        let cosx = cos(x)
                        return v / (cosx * cosx)
                    }
                )
            }

            @derivative(of: log(_:))
            public static func _vjpLog(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: log(x), pullback: { v in v / x })
            }

            @derivative(of: acosh)
            public static func _vjpAcosh(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                // only valid for x > 1
                return (value: acosh(x), pullback: { v in v / sqrt(x * x - 1) })
            }

            @derivative(of: asinh)
            public static func _vjpAsinh(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: asinh(x), pullback: { v in v / sqrt(x * x + 1) })
            }

            @derivative(of: atanh)
            public static func _vjpAtanh(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: atanh(x), pullback: { v in v / (1 - x * x) })
            }

            @derivative(of: acos)
            public static func _vjpAcos(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: acos(x), pullback: { v in -v / (1 - x * x) })
            }

            @derivative(of: asin)
            public static func _vjpAsin(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: asin(x), pullback: { v in v / (1 - x * x) })
            }

            @derivative(of: atan)
            public static func _vjpAtan(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: atan(x), pullback: { v in v / (x * x + 1) })
            }

            @derivative(of: log(onePlus:))
            public static func _vjpLog(onePlus x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: log(onePlus: x), pullback: { v in v / (1 + x) })
            }

            @derivative(of: pow)
            public static func _vjpPow(_ x: \(type), _ y: \(type)) -> (value: \(type), pullback: (\(type)) -> (\(type), \(type))) {
                let value = pow(x, y)
                // pullback wrt y is not defined for (x < 0) and (x = 0, y = 0)
                return (value: value, pullback: { v in (v * y * pow(x, y - 1), v * value * log(x)) })
            }

            @derivative(of: pow)
            public static func _vjpPow(_ x: \(type), _ n: Int) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: pow(x, n), pullback: { v in v * \(floatingPointType)(n) * pow(x, n - 1) })
            }

            @derivative(of: sqrt)
            public static func _vjpSqrt(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                let value = sqrt(x)
                return (value: value, pullback: { v in v / (2 * value) })
            }

            @derivative(of: root)
            public static func _vjpRoot(_ x: \(type), _ n: Int) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                let value = root(x, n)
                return (value: value, pullback: { v in v * value / (x * \(floatingPointType)(n)) })
            }
        }

        // MARK: RealFunctions derivatives
        extension \(type) {
            @derivative(of: erf)
            public static func _vjpErf(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: erf(x), pullback: { v in 2 * exp(-x * x) / .sqrt(\(floatingPointType).pi) })
            }

            @derivative(of: erfc)
            public static func _vjpErfc(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: erfc(x), pullback: { v in -2 * exp(-x * x) / .sqrt(\(floatingPointType).pi) })
            }

            @derivative(of: exp2)
            public static func _vjpExp2(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                let value = exp2(x)
                return (value, { v in v * value * .log(2) })
            }

            @derivative(of: exp10)
            public static func _vjpExp10(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                let value = exp10(x)
                return (value, { v in v * value * .log(10) })
            }

            @derivative(of: gamma)
            public static func _vjpGamma(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                fatalError("unimplemented")
            }

            @derivative(of: log2)
            public static func _vjpLog2(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: log2(x), pullback: { v in v / (.log(2) * x) })
            }

            @derivative(of: log10)
            public static func _vjpLog10(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                (value: log10(x), pullback: { v in v / (.log(10) * x) })
            }

            @derivative(of: logGamma)
            public static func _vjpLogGamma(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                fatalError("unimplemented")
            }

            @derivative(of: atan2)
            public static func _vjpAtan2(y: \(type), x: \(type)) -> (value: \(type), pullback: (\(type)) -> (\(type), \(type))) {
                (
                    value: atan2(y: y, x: x), 
                    pullback: { v in 
                        let c = x * x + y * y
                        return (v * x / c, -v * y / c)
                    }
                )
            }

            @derivative(of: hypot)
            public static func _vjpHypot(_ x: \(type), _ y: \(type)) -> (value: \(type), pullback: (\(type)) -> (\(type), \(type))) {
                (
                    value: hypot(x, y), 
                    pullback: { v in 
                        let c = sqrt(x * x + y * y)
                        return (v * x / c, v * y / c)
                    }
                )
            }
        }

        // MARK: FloatingPoint functions derivatives
        extension \(type) {
            @derivative(of: abs)
            public static func _vjpAbs(_ x: \(type)) -> (value: \(type), pullback: (\(type)) -> \(type)) {
                \({
                    if type == floatingPointType {
                        "x < 0 ? (value: -x, pullback: { v in .zero - v }) : (value: x, pullback: { v in v })"
                    }
                    else {
                        "(value: abs(x), pullback: { v in v.replacing(with: -v, where: x .< .zero) })"
                    }
                }())
            }
        }
        #endif
        """
    }
}
