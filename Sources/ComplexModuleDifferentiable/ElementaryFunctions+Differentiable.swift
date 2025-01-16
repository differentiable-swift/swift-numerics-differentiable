#if canImport(_Differentiation)
import _Differentiation
@_exported import ComplexModule
import RealModule

extension Complex
    where RealType: Differentiable,
    RealType.TangentVector == RealType
{
    // --------------------------------------

    // MARK: - VJP of exp(_:)

    // --------------------------------------
    @derivative(of: exp)
    @inlinable
    public static func _vjpExp(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.exp(z)
        return (val, { v in v * val })
    }

    // --------------------------------------

    // MARK: - VJP of expMinusOne(_:)

    // --------------------------------------
    @derivative(of: expMinusOne)
    @inlinable
    public static func _vjpExpMinusOne(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        // expMinusOne(z) = exp(z) - 1
        let val = Complex.expMinusOne(z)
        let expz = Complex.exp(z)
        return (val, { v in v * expz })
    }

    // --------------------------------------

    // MARK: - VJP of cosh(_:)

    // --------------------------------------
    @derivative(of: cosh)
    @inlinable
    public static func _vjpCosh(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.cosh(z)
        let sinhz = Complex.sinh(z)
        return (val, { v in v * sinhz })
    }

    // --------------------------------------

    // MARK: - VJP of sinh(_:)

    // --------------------------------------
    @derivative(of: sinh)
    @inlinable
    public static func _vjpSinh(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.sinh(z)
        let coshz = Complex.cosh(z)
        return (val, { v in v * coshz })
    }

    // --------------------------------------

    // MARK: - VJP of tanh(_:)

    // --------------------------------------
    @derivative(of: tanh)
    @inlinable
    public static func _vjpTanh(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.tanh(z)
        // derivative(tanh(z)) = 1 - tanh(z)^2
        return (val, { v in v * (1 - val * val) })
    }

    // --------------------------------------

    // MARK: - VJP of cos(_:)

    // --------------------------------------
    @derivative(of: cos)
    @inlinable
    public static func _vjpCos(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.cos(z)
        // derivative(cos(z)) = -sin(z)
        let minusSinz = -Complex.sin(z)
        return (val, { v in v * minusSinz })
    }

    // --------------------------------------

    // MARK: - VJP of sin(_:)

    // --------------------------------------
    @derivative(of: sin)
    @inlinable
    public static func _vjpSin(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.sin(z)
        let cosz = Complex.cos(z)
        return (val, { v in v * cosz })
    }

    // --------------------------------------

    // MARK: - VJP of tan(_:)

    // --------------------------------------
    @derivative(of: tan)
    @inlinable
    public static func _vjpTan(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.tan(z)
        // derivative(tan(z)) = 1 + tan^2(z)
        let derivative = 1 + val * val
        return (val, { v in v * derivative })
    }

    // --------------------------------------

    // MARK: - VJP of log(_:)

    // --------------------------------------
    @derivative(of: log(_:))
    @inlinable
    public static func _vjpLog(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.log(z)
        // d/dz log(z) = 1 / z
        return (val, { v in v / z })
    }

    // --------------------------------------

    // MARK: - VJP of log(onePlus:)

    // --------------------------------------
    @derivative(of: log(onePlus:))
    @inlinable
    public static func _vjpLogOnePlus(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.log(onePlus: z)
        // d/dz log(1 + z) = 1 / (1 + z)
        return (val, { v in v / (1 + z) })
    }

    // --------------------------------------

    // MARK: - VJP for acos(_:)

    // --------------------------------------
    @derivative(of: acos)
    @inlinable
    public static func _vjpAcos(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.acos(z)
        let denom = -Complex.sqrt(1 - z * z)
        return (val, { v in v / denom })
    }

    // --------------------------------------

    // MARK: - VJP for asin(_:)

    // --------------------------------------
    @derivative(of: asin)
    @inlinable
    public static func _vjpAsin(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.asin(z)
        let denom = Complex.sqrt(1 - z * z)
        return (val, { v in v / denom })
    }

    // --------------------------------------

    // MARK: - VJP for atan(_:)

    // --------------------------------------
    @derivative(of: atan)
    @inlinable
    public static func _vjpAtan(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.atan(z)
        let denom = 1 + z * z
        return (val, { v in v / denom })
    }

    // --------------------------------------

    // MARK: - VJP for acosh(_:)

    // --------------------------------------
    @derivative(of: acosh)
    @inlinable
    public static func _vjpAcosh(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.acosh(z)
        let denom = Complex.sqrt(z * z - 1)
        return (val, { v in v / denom })
    }

    // --------------------------------------

    // MARK: - VJP for asinh(_:)

    // --------------------------------------
    @derivative(of: asinh)
    @inlinable
    public static func _vjpAsinh(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.asinh(z)
        let denom = Complex.sqrt(1 + z * z)
        return (val, { v in v / denom })
    }

    // --------------------------------------

    // MARK: - VJP for atanh(_:)

    // --------------------------------------
    @derivative(of: atanh)
    @inlinable
    public static func _vjpAtanh(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.atanh(z)
        let denom = 1 - z * z
        return (val, { v in v / denom })
    }

    // --------------------------------------

    // MARK: - VJP for pow(z, w)

    // --------------------------------------
    @derivative(of: pow(_:_:))
    @inlinable
    public static func _vjpPowZW(
        _ z: Complex,
        _ w: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> (
            Complex.TangentVector,
            Complex.TangentVector
        )
    ) {
        let val = Complex.pow(z, w) // = exp(w * log(z))
        return (val, { v in
            // derivative wrt z => val * (w / z)
            let dZ = v * (val * (w / z))
            // derivative wrt w => val * log(z)
            let dW = v * (val * Complex.log(z))
            return (dZ, dW)
        })
    }

    // --------------------------------------

    // MARK: - VJP of pow(_:_:)

    // --------------------------------------
    @derivative(of: pow(_:_:), wrt: z)
    @inlinable
    public static func _vjpPowN(
        _ z: Complex,
        _ n: Int
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        // Forward value
        let val = pow(z, n)
        return (
            value: val,
            pullback: { v in
                // We treat `n` as a constant, so the partial derivative
                // w.r.t. `z` is n * z^(n-1).
                // Mathematically: d/dz [z^n] = n * z^(n-1).
                //
                // That means the pullback multiplies the upstream 'v'
                // by n * z^(n-1).

                // Handle edge cases if desired (e.g. z=0, n<=0, etc.);
                // but in typical usage, this is enough:
                if n == 0 {
                    // d/dz [z^0 = 1] = 0
                    return .zero
                }
                else if z.isZero, n > 1 {
                    // 0^n => derivative might be 0 or undefined for n<=0
                    return .zero
                }
                // For general z != 0 or n>0:
                let partial = Complex(RealType(n)) * pow(z, n - 1)
                return v * partial
            }
        )
    }

    // --------------------------------------

    // MARK: - VJP of sqrt(_:)

    // --------------------------------------
    @derivative(of: sqrt)
    @inlinable
    public static func _vjpSqrt(
        _ z: Complex
    ) -> (
        value: Complex,
        pullback: (Complex.TangentVector) -> Complex.TangentVector
    ) {
        let val = Complex.sqrt(z)
        // derivative(sqrt(z)) = 1 / (2 * sqrt(z))
        return (val, { v in
            let denom = val * 2
            return v / denom
        })
    }

    // --------------------------------------

    // MARK: - VJP of root(_:_:)

    // --------------------------------------
    @derivative(of: root(_:_:), wrt: z)
    @inlinable
    public static func _vjpRootN(
        _ z: Complex<RealType>,
        _ n: Int
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
    ) {
        // Forward value
        let val = root(z, n) // = exp(log(z)/n)

        return (
            value: val,
            pullback: { v in
                // derivative of root(z, n) w.r.t. z => val * (1 / (n*z))
                // = z^(1/n) * [1 / (n*z)] = z^(1/n - 1) / n
                //
                // If n=0 or z=0, we might have undefined behavior mathematically.
                // We'll do a simple guard for typical usage: n != 0, z != 0.
                if n == 0 {
                    // Not well-defined. Return .zero:
                    return .zero
                }
                if z.isZero {
                    // Return .zero for z=0:
                    return .zero
                }
                // General case:
                let partial = val / (Complex(RealType(n)) * z)
                return v * partial
            }
        )
    }
}

#endif
