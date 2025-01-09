@_exported import ComplexModule
import _Differentiation
import RealModule

/// We do NOT redeclare `extension Complex: Differentiable` here,
/// we only attach custom derivatives to existing members.
extension Complex
where RealType: Differentiable & FloatingPoint,
      RealType == RealType.TangentVector
{
  // MARK: - Derivative for the existing initializer init(_:_)
  @derivative(of: init(_:_:))
  @usableFromInline
  static func _derivativeInit(
    _ real: RealType,
    _ imaginary: RealType
  ) -> (value: Self, pullback: (Self) -> (RealType, RealType)) {
    let value = Self(real, imaginary)
    // The pullback is the identity (v.real, v.imaginary).
    return (value, { v in (v.real, v.imaginary) })
  }

  // MARK: - Derivative for '+'
  @derivative(of: +)
  @usableFromInline
  static func _derivativeAdd(
    lhs: Self, rhs: Self
  ) -> (value: Self, pullback: (Self) -> (Self, Self))
  {
    let value = lhs + rhs
    // Pullback: gradient just distributes to both lhs & rhs identically.
    return (value, { v in (v, v) })
  }

  // MARK: - Derivative for '-'
  @derivative(of: -)
  @usableFromInline
  static func _derivativeSubtract(
    lhs: Self, rhs: Self
  ) -> (value: Self, pullback: (Self) -> (Self, Self))
  {
    let value = lhs - rhs
    // Pullback: wrt lhs is +v, wrt rhs is -v.
    return (value, { v in (v, .init(-v.real, -v.imaginary)) })
  }

  // MARK: - Derivative for '*'
  @derivative(of: *)
  @usableFromInline
  static func _derivativeMultiply(
    lhs: Self, rhs: Self
  ) -> (value: Self, pullback: (Self) -> (Self, Self))
  {
    let value = lhs * rhs
    // If we treat complex multiplication as (lhs * rhs), the derivative is:
    // d/dlhs => v * conj(rhs), d/drhs => v * conj(lhs).
    // We'll implement v * conj(rhs) manually using public .real/.imaginary:

    return (value, { v in
      // conj(rhs) = (rhs.real, -rhs.imaginary)
      // so v * conj(rhs) = (v.real * rhs.real - v.imaginary * (-rhs.imaginary),
      //                     v.real * (-rhs.imaginary) + v.imaginary * rhs.real)
      // => real part = v.real*rhs.real + v.imaginary*rhs.imaginary
      // => imag part = v.imaginary*rhs.real - v.real*rhs.imaginary
      let dLHS = Self(
        v.real * rhs.real + v.imaginary * rhs.imaginary,
        v.imaginary * rhs.real - v.real * rhs.imaginary
      )

      // conj(lhs) = (lhs.real, -lhs.imaginary)
      // v * conj(lhs)
      let dRHS = Self(
        v.real * lhs.real + v.imaginary * lhs.imaginary,
        v.imaginary * lhs.real - v.real * lhs.imaginary
      )
      return (dLHS, dRHS)
    })
  }

  // MARK: - Derivative for '/'
  @derivative(of: /)
  @usableFromInline
  static func _derivativeDivide(
    lhs: Self, rhs: Self
  ) -> (value: Self, pullback: (Self) -> (Self, Self))
  {
    let value = lhs / rhs
    // We'll do a manual approach:
    // d/dlhs => v / rhs
    // d/drhs => - (lhs * v / rhs^2)
    return (value, { v in
      let denom = rhs.real * rhs.real + rhs.imaginary * rhs.imaginary
      
      // v / rhs = (v * conj(rhs)) / |rhs|^2
      // conj(rhs) = (rhs.real, -rhs.imaginary)
      let vTimesConjR = Self(
        v.real*rhs.real + v.imaginary*rhs.imaginary,
        v.imaginary*rhs.real - v.real*rhs.imaginary
      )
      let dLHS = Self(
        vTimesConjR.real / denom,
        vTimesConjR.imaginary / denom
      )

      // -(lhs * v)/|rhs|^2 => multiply lhs*v first, then divide, then negate
      // lhs*v = (lhs.real*v.real - lhs.imaginary*v.imaginary,
      //          lhs.real*v.imaginary + lhs.imaginary*v.real)
      let lhsTimesV = Self(
        lhs.real*v.real - lhs.imaginary*v.imaginary,
        lhs.real*v.imaginary + lhs.imaginary*v.real
      )
      let dRHSpart = Self(
        lhsTimesV.real / denom,
        lhsTimesV.imaginary / denom
      )
      let dRHS = Self(-dRHSpart.real, -dRHSpart.imaginary)
      
      return (dLHS, dRHS)
    })
  }

  // MARK: - Derivative for '.conjugate'
  @derivative(of: conjugate)
  @usableFromInline
  func _derivativeConjugate() -> (value: Self, pullback: (Self) -> Self)
  {
    let value = conjugate
    // pullback: just flip the sign of the imaginary part
    return (value, { v in .init(v.real, -v.imaginary) })
  }
}

// NOTE: We are not providing a derivative for prefix '-'
// because the compiler (6.2-dev) suggests is declared by SignedNumeric, 
// and Swift's AD cannot attach a derivative to a protocol 
// requirement in another module. 




