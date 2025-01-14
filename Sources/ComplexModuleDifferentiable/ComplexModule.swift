#if swift(>=5.3) && canImport(_Differentiation)
import _Differentiation
@_exported import ComplexModule
import RealModule

extension Complex
    where
    RealType: Differentiable,
    RealType.TangentVector == RealType
{
    // --------------------------------------

    // MARK: - Derivative of exp(_:)

    // --------------------------------------
    @derivative(of: exp)
    @inlinable
    public static func _derivativeExp(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = exp(z)
        return (val, { v in v * val })
    }

    // --------------------------------------

    // MARK: - Derivative of expMinusOne(_:)

    // --------------------------------------
    @derivative(of: expMinusOne)
    @inlinable
    public static func _derivativeExpMinusOne(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        // expMinusOne(z) = exp(z) - 1
        let val = expMinusOne(z)
        let expz = exp(z)
        return (val, { v in v * expz })
    }

    // --------------------------------------

    // MARK: - Derivative of log(_:)

    // --------------------------------------
    @derivative(of: log(_:))
    @inlinable
    public static func _derivativeLog(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = log(z)
        // Derivative: d/dz log(z) = 1 / z
        return (val, { v in v / z })
    }

    // --------------------------------------

    // MARK: - Derivative of log(onePlus:)

    // --------------------------------------
    @derivative(of: log(onePlus:))
    @inlinable
    public static func _derivativeLogOnePlus(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = log(onePlus: z)
        // Derivative: d/dz log(1 + z) = 1 / (1 + z)
        return (val, { v in v / (1 + z) })
    }

    // --------------------------------------

    // MARK: - Derivative for pow(z, w)

    // --------------------------------------
    @derivative(of: pow(_:_:))
    @inlinable
    public static func _derivativePowZW(
        _ z: Self,
        _ w: Self
    ) -> (value: Self, pullback: (TangentVector) -> (TangentVector, TangentVector)) {
        let val = pow(z, w) // = exp(w * log(z))
        return (val, { v in
            let dZ = v * (val * (w / z)) // partial wrt z
            let dW = v * (val * log(z)) // partial wrt w
            return (dZ, dW)
        })
    }

    // We do NOT define a derivative for pow(_: _: Int), because Int is not Differentiable.

    // --------------------------------------

    // MARK: - Derivative of cosh(_:)

    // --------------------------------------
    @derivative(of: cosh)
    @inlinable
    public static func _derivativeCosh(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = cosh(z)
        let sinhz = sinh(z)
        return (val, { v in v * sinhz })
    }

    // --------------------------------------

    // MARK: - Derivative of sinh(_:)

    // --------------------------------------
    @derivative(of: sinh)
    @inlinable
    public static func _derivativeSinh(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = sinh(z)
        let coshz = cosh(z)
        return (val, { v in v * coshz })
    }

    // --------------------------------------

    // MARK: - Derivative of tanh(_:)

    // --------------------------------------
    @derivative(of: tanh)
    @inlinable
    public static func _derivativeTanh(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        // derivative(tanh(z)) = 1 - tanh(z)^2
        let val = tanh(z)
        return (val, { v in v * (1 - val * val) })
    }

    // --------------------------------------

    // MARK: - Derivative of cos(_:)

    // --------------------------------------
    @derivative(of: cos)
    @inlinable
    public static func _derivativeCos(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = cos(z)
        // derivative(cos(z)) = -sin(z)
        let minusSinz = -sin(z)
        return (val, { v in v * minusSinz })
    }

    // --------------------------------------

    // MARK: - Derivative of sin(_:)

    // --------------------------------------
    @derivative(of: sin)
    @inlinable
    public static func _derivativeSin(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = sin(z)
        let cosz = cos(z)
        return (val, { v in v * cosz })
    }

    // --------------------------------------

    // MARK: - Derivative of tan(_:)

    // --------------------------------------
    @derivative(of: tan)
    @inlinable
    public static func _derivativeTan(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = tan(z)
        // derivative(tan(z)) = sec^2(z) = 1 + tan^2(z)
        let derivative = 1 + val * val
        return (val, { v in v * derivative })
    }

    // --------------------------------------

    // MARK: - Derivative of sqrt(_:)

    // --------------------------------------
    @derivative(of: sqrt)
    @inlinable
    public static func _derivativeSqrt(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = sqrt(z)
        // derivative(sqrt(z)) = 1 / (2 * sqrt(z))
        return (val, { v in
            let denom = val * 2
            return v / denom
        })
    }

    // --------------------------------------

    // MARK: - Derivatives for inverse trig/hyperbolic

    // --------------------------------------
    @derivative(of: asin)
    @inlinable
    public static func _derivativeAsin(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = asin(z)
        let denom = sqrt(1 - z * z)
        return (val, { v in v / denom })
    }

    @derivative(of: acos)
    @inlinable
    public static func _derivativeAcos(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = acos(z)
        let denom = -sqrt(1 - z * z)
        return (val, { v in v / denom })
    }

    @derivative(of: atan)
    @inlinable
    public static func _derivativeAtan(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = atan(z)
        let denom = 1 + z * z
        return (val, { v in v / denom })
    }

    @derivative(of: asinh)
    @inlinable
    public static func _derivativeAsinh(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = asinh(z)
        let denom = sqrt(1 + z * z)
        return (val, { v in v / denom })
    }

    @derivative(of: acosh)
    @inlinable
    public static func _derivativeAcosh(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = acosh(z)
        let denom = sqrt(z * z - 1)
        return (val, { v in v / denom })
    }

    @derivative(of: atanh)
    @inlinable
    public static func _derivativeAtanh(_ z: Self)
        -> (value: Self, pullback: (TangentVector) -> TangentVector)
    {
        let val = atanh(z)
        let denom = 1 - z * z
        return (val, { v in v / denom })
    }
}

/*
 * No Derivative for pow(z, n: Int)
 * The Swift compiler requires all parameters of the original function to be
 * Differentiable if we want a pullback of type
 * (TangentVector) -> (TangentVector, TangentVector).
 * Since Int is not differentiable, hard to define a conforming derivative
 * for that overload.
 */

#endif
