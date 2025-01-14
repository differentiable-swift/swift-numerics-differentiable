#if swift(>=5.3) && canImport(_Differentiation)
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
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
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
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
    ) {
        // expMinusOne(z) = exp(z) - 1
        let val = Complex.expMinusOne(z)
        let expz = Complex.exp(z)
        return (val, { v in v * expz })
    }

    // --------------------------------------

    // MARK: - VJP of log(_:)

    // --------------------------------------
    @derivative(of: log(_:))
    @inlinable
    public static func _vjpLog(
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
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
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
    ) {
        let val = Complex.log(onePlus: z)
        // d/dz log(1 + z) = 1 / (1 + z)
        return (val, { v in v / (1 + z) })
    }

    // --------------------------------------

    // MARK: - VJP for pow(z, w)

    // --------------------------------------
    @derivative(of: pow(_:_:))
    @inlinable
    public static func _vjpPowZW(
        _ z: Complex<RealType>,
        _ w: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> (
            Complex<RealType>.TangentVector,
            Complex<RealType>.TangentVector
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

    // MARK: - VJP of cosh(_:)

    // --------------------------------------
    @derivative(of: cosh)
    @inlinable
    public static func _vjpCosh(
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
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
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
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
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
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
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
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
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
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
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
    ) {
        let val = Complex.tan(z)
        // derivative(tan(z)) = 1 + tan^2(z)
        let derivative = 1 + val * val
        return (val, { v in v * derivative })
    }

    // --------------------------------------

    // MARK: - VJP of sqrt(_:)

    // --------------------------------------
    @derivative(of: sqrt)
    @inlinable
    public static func _vjpSqrt(
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
    ) {
        let val = Complex.sqrt(z)
        // derivative(sqrt(z)) = 1 / (2 * sqrt(z))
        return (val, { v in
            let denom = val * 2
            return v / denom
        })
    }

    // --------------------------------------

    // MARK: - VJPs for inverse trig/hyperbolic

    // --------------------------------------
    @derivative(of: asin)
    @inlinable
    public static func _vjpAsin(
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
    ) {
        let val = Complex.asin(z)
        let denom = Complex.sqrt(1 - z * z)
        return (val, { v in v / denom })
    }

    @derivative(of: acos)
    @inlinable
    public static func _vjpAcos(
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
    ) {
        let val = Complex.acos(z)
        let denom = -Complex.sqrt(1 - z * z)
        return (val, { v in v / denom })
    }

    @derivative(of: atan)
    @inlinable
    public static func _vjpAtan(
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
    ) {
        let val = Complex.atan(z)
        let denom = 1 + z * z
        return (val, { v in v / denom })
    }

    @derivative(of: asinh)
    @inlinable
    public static func _vjpAsinh(
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
    ) {
        let val = Complex.asinh(z)
        let denom = Complex.sqrt(1 + z * z)
        return (val, { v in v / denom })
    }

    @derivative(of: acosh)
    @inlinable
    public static func _vjpAcosh(
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
    ) {
        let val = Complex.acosh(z)
        let denom = Complex.sqrt(z * z - 1)
        return (val, { v in v / denom })
    }

    @derivative(of: atanh)
    @inlinable
    public static func _vjpAtanh(
        _ z: Complex<RealType>
    ) -> (
        value: Complex<RealType>,
        pullback: (Complex<RealType>.TangentVector) -> Complex<RealType>.TangentVector
    ) {
        let val = Complex.atanh(z)
        let denom = 1 - z * z
        return (val, { v in v / denom })
    }
}

/*
 * No Derivative for pow(z, n: Int)
 * The Swift compiler requires all parameters of the original function to be
 * Differentiable if we want a pullback of type
 * (TangentVector) -> (TangentVector, TangentVector).
 * Since Int is not Differentiable, we cannot define a conforming derivative
 * for that overload.
 */

#endif

/*
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

 */
