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
  @usableFromInline
  static func _derivativeExp(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    let val = exp(z)
    return (val, { v in v * val })
  }

  // --------------------------------------
  // MARK: - Derivative of expMinusOne(_:)
  // --------------------------------------
  @derivative(of: expMinusOne)
  @usableFromInline
  static func _derivativeExpMinusOne(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    // expMinusOne(z) = exp(z) - 1
    let val = expMinusOne(z)
    let expz = exp(z)
    return (val, { v in v * expz })
  }

  // --------------------------------------
  // MARK: - Derivative of cosh(_:)
  // --------------------------------------
  @derivative(of: cosh)
  @usableFromInline
  static func _derivativeCosh(_ z: Self)
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
  @usableFromInline
  static func _derivativeSinh(_ z: Self)
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
  @usableFromInline
  static func _derivativeTanh(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    // derivative(tanh(z)) = 1 - tanh(z)^2
    let val = tanh(z)
    return (val, { v in v * (1 - val*val) })
  }

  // --------------------------------------
  // MARK: - Derivative of cos(_:)
  // --------------------------------------
  @derivative(of: cos)
  @usableFromInline
  static func _derivativeCos(_ z: Self)
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
  @usableFromInline
  static func _derivativeSin(_ z: Self)
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
  @usableFromInline
  static func _derivativeTan(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    let val = tan(z)
    // derivative(tan(z)) = sec^2(z) = 1 + tan^2(z)
    let derivative = 1 + val*val
    return (val, { v in v * derivative })
  }

  // ----------------------------------------------------------------
  // MARK: - Workaround for ambiguous log and log(onePlus:) overloads
  // ----------------------------------------------------------------
  
  /// A local wrapper for `Complex.log(z)`.
  /// We'll define a derivative for `_logOfZ(_:)` instead of `log(_:)` directly.
  @usableFromInline
  static func _logOfZ(_ z: Self) -> Self {
    Self.log(z)
  }
  
  @derivative(of: _logOfZ)
  @usableFromInline
  static func _derivativeLogOfZ(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    // derivative(log z) = 1 / z
    let val = _logOfZ(z)
    return (val, { v in v / z })
  }

  /// A local wrapper for `Complex.log(onePlus: z)`.
  @usableFromInline
  static func _logOnePlusZ(_ z: Self) -> Self {
    Self.log(onePlus: z)
  }
  
  @derivative(of: _logOnePlusZ)
  @usableFromInline
  static func _derivativeLogOnePlusZ(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    // derivative(log(1+z)) = 1 / (1 + z)
    let val = _logOnePlusZ(z)
    return (val, { v in v / (1 + z) })
  }

  // --------------------------------------
  // MARK: - Derivative of sqrt(_:)
  // --------------------------------------
  @derivative(of: sqrt)
  @usableFromInline
  static func _derivativeSqrt(_ z: Self)
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
  @usableFromInline
  static func _derivativeAsin(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    let val = asin(z)
    let denom = sqrt(1 - z*z)
    return (val, { v in v / denom })
  }

  @derivative(of: acos)
  @usableFromInline
  static func _derivativeAcos(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    let val = acos(z)
    let denom = -sqrt(1 - z*z)
    return (val, { v in v / denom })
  }

  @derivative(of: atan)
  @usableFromInline
  static func _derivativeAtan(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    let val = atan(z)
    let denom = 1 + z*z
    return (val, { v in v / denom })
  }

  @derivative(of: asinh)
  @usableFromInline
  static func _derivativeAsinh(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    let val = asinh(z)
    let denom = sqrt(1 + z*z)
    return (val, { v in v / denom })
  }

  @derivative(of: acosh)
  @usableFromInline
  static func _derivativeAcosh(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    let val = acosh(z)
    let denom = sqrt(z*z - 1)
    return (val, { v in v / denom })
  }

  @derivative(of: atanh)
  @usableFromInline
  static func _derivativeAtanh(_ z: Self)
    -> (value: Self, pullback: (TangentVector) -> TangentVector)
  {
    let val = atanh(z)
    let denom = 1 - z*z
    return (val, { v in v / denom })
  }

  // ----------------------------------------------------------------
  // MARK: - Workaround for ambiguous pow(z, w) overload
  // ----------------------------------------------------------------
  /// A local wrapper for `Complex.pow(z, w)`.
  @usableFromInline
  static func _powZW(_ z: Self, _ w: Self) -> Self {
    Self.pow(z, w)
  }

  /// Derivative for `pow(z, w)` w.r.t. both z & w.
  @derivative(of: _powZW)
  @usableFromInline
  static func _derivativePowZW(_ z: Self, _ w: Self)
    -> (value: Self, pullback: (TangentVector) -> (TangentVector, TangentVector))
  {
    let val = _powZW(z, w)  // = exp(w * log(z))
    return (val, { v in
      // d/dz => val * (w / z)
      // d/dw => val * log(z)
      let dZ = v * (val * (w / z))
      let dW = v * (val * _logOfZ(z))
      return (dZ, dW)
    })
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
