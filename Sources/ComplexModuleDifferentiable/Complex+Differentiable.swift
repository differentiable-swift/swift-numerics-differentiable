#if canImport(_Differentiation)

import ComplexModule

extension Complex: @retroactive Differentiable where RealType: Differentiable, RealType.TangentVector == RealType {
    public typealias TangentVector = Self

    @inlinable
    public mutating func move(by offset: Complex<RealType>) {
        self += offset
    }
}

extension Complex where RealType: Differentiable, RealType.TangentVector == RealType {
    @derivative(of: init(_:_:))
    @_transparent
    public static func _vjpInit(_ real: RealType, _ imaginary: RealType) -> (value: Complex, pullback: (Complex) -> (RealType, RealType)) {
        (
            value: .init(real, imaginary),
            pullback: { v in (v.real, v.imaginary) }
        )
    }

    @derivative(of: init(_:))
    @_transparent
    public static func _vjpInit(_ real: RealType) -> (value: Complex, pullback: (Complex) -> RealType) {
        (
            value: .init(real),
            pullback: { v in v.real }
        )
    }

    @derivative(of: init(imaginary:))
    @_transparent
    public static func _vjpInit(imaginary: RealType) -> (value: Complex, pullback: (Complex) -> RealType) {
        (
            value: .init(imaginary: imaginary),
            pullback: { v in v.imaginary }
        )
    }

    @derivative(of: real)
    @_transparent
    public func _vjpReal() -> (value: RealType, pullback: (RealType) -> Complex) {
        (value: real, pullback: { v in Complex(v, .zero) })
    }

    @derivative(of: real.set)
    @_transparent
    public mutating func _vjpRealSet(_ newValue: RealType) -> (value: Void, pullback: (inout Complex) -> RealType) {
        self.real = newValue
        return (
            value: (),
            pullback: { v in
                let real = v.real
                v.real = .zero
                return real
            }
        )
    }

    @derivative(of: imaginary)
    @_transparent
    public func _vjpImaginary() -> (value: RealType, pullback: (RealType) -> Complex) {
        (value: imaginary, pullback: { v in Complex(.zero, v) })
    }

    @derivative(of: imaginary.set)
    @_transparent
    public mutating func _vjpImaginarySet(_ newValue: RealType) -> (value: Void, pullback: (inout Complex) -> RealType) {
        self.imaginary = newValue
        return (
            value: (),
            pullback: { v in
                let imaginary = v.imaginary
                v.imaginary = .zero
                return imaginary
            }
        )
    }

    @derivative(of: +)
    @_transparent
    public static func _vjpAdd(z: Complex, w: Complex) -> (value: Complex, pullback: (Complex) -> (Complex, Complex)) {
        (value: z + w, pullback: { v in (v, v) })
    }

    @derivative(of: +=)
    @_transparent
    public static func _vjpAddAssign(z: inout Complex, w: Complex) -> (value: Void, pullback: (inout Complex) -> (Complex)) {
        z += w
        return (value: (), pullback: { v in v })
    }

    @derivative(of: -)
    @_transparent
    public static func _vjpSubtract(z: Complex, w: Complex) -> (value: Complex, pullback: (Complex) -> (Complex, Complex)) {
        (value: z - w, pullback: { v in (v, -v) })
    }

    @derivative(of: -=)
    @_transparent
    public static func _vjpSubtractAssign(z: inout Complex, w: Complex) -> (value: Void, pullback: (inout Complex) -> (Complex)) {
        z -= w
        return (value: (), pullback: { v in -v })
    }

    @derivative(of: *)
    @_transparent
    public static func _vjpMultiply(z: Complex, w: Complex) -> (value: Complex, pullback: (Complex) -> (Complex, Complex)) {
        (value: z * w, pullback: { v in (w * v, z * v) })
    }

    @derivative(of: *=)
    @_transparent
    public static func _vjpMultiplyAssign(z: inout Complex, w: Complex) -> (value: Void, pullback: (inout Complex) -> (Complex)) {
        defer { z *= w }
        return (
            value: (),
            pullback: { [z = z] v in
                let drhs = z * v
                v *= w
                return drhs
            }
        )
    }

    @derivative(of: /)
    @_transparent
    public static func _vjpDivide(z: Complex, w: Complex) -> (value: Complex, pullback: (Complex) -> (Complex, Complex)) {
        (value: z / w, pullback: { v in (v / w, -z / (w * w) * v) })
    }

    @derivative(of: /=)
    @_transparent
    public static func _vjpDivideAssign(z: inout Complex, w: Complex) -> (value: Void, pullback: (inout Complex) -> (Complex)) {
        defer { z /= w }
        return (
            value: (),
            pullback: { [z = z] v in
                let drhs = -z / (w * w) * v
                v /= w
                return drhs
            }
        )
    }

    @derivative(of: conjugate)
    @_transparent
    public func _vjpConjugate() -> (value: Complex, pullback: (Complex) -> Complex) {
        (value: conjugate, pullback: { v in v.conjugate })
    }
}

#endif
