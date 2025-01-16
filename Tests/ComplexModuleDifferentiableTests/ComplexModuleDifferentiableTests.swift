
@testable import ComplexModuleDifferentiable
import Testing

// A helper function that compares two Complex<Double> values
// within a specified absolute tolerance using `#expect`.
func expectClose(
    _ lhs: Complex<Double>,
    _ rhs: Complex<Double>,
    accuracy: Double = 1E-15
) {
    // Real-part check
    #expect(
        abs(lhs.real - rhs.real) < accuracy,
        "Real parts differ: \(lhs.real) vs \(rhs.real) (tolerance: \(accuracy))"
    )
    // Imag-part check
    #expect(
        abs(lhs.imaginary - rhs.imaginary) < accuracy,
        "Imag parts differ: \(lhs.imaginary) vs \(rhs.imaginary) (tolerance: \(accuracy))"
    )
}

// MARK: - Test Suite for Complex Derivatives

struct ComplexDerivativesTests {
    // --------------------------------------------

    // MARK: 1) Test derivative of init(_:_)

    // --------------------------------------------
    @Test(arguments: zip(
        [0.0, 1.5, -2.0],
        [3.0, -1.0, 0.0]
    ))
    func testInit(real: Double, imag: Double) {
        let (value, pullback) = valueWithPullback(
            at: real, imag,
            of: { r, i -> Complex<Double> in
                Complex(r, i)
            }
        )

        // 1) Forward check
        #expect(value.real == real, "Init: real part mismatch")
        #expect(value.imaginary == imag, "Init: imaginary part mismatch")

        // 2) Pullback check
        let upstream = Complex<Double>(2, -3)
        let (dReal, dImag) = pullback(upstream)
        #expect(dReal == upstream.real, "d/dreal mismatch")
        #expect(dImag == upstream.imaginary, "d/dimag mismatch")
    }

    // --------------------------------------------

    // MARK: 2) Test derivative of '+'

    // --------------------------------------------
    @Test(arguments: zip(
        [(0.0, 0.0), (1.0, 2.0), (-3.5, 4.0)],
        [(0.0, 2.0), (1.0, -1.0), (2.5, -3.0)]
    ))
    func testAdd(lhs: (Double, Double), rhs: (Double, Double)) {
        let z1 = Complex(lhs.0, lhs.1)
        let z2 = Complex(rhs.0, rhs.1)

        let (value, pullback) = valueWithPullback(at: z1, z2) { a, b in a + b }

        // 1) Forward check
        #expect(value.real == z1.real + z2.real, "Add: real mismatch")
        #expect(value.imaginary == z1.imaginary + z2.imaginary, "Add: imag mismatch")

        // 2) Pullback check
        let upstream = Complex<Double>(1, -1)
        let (dLHS, dRHS) = pullback(upstream)
        #expect(dLHS == upstream, "Add: dLHS mismatch")
        #expect(dRHS == upstream, "Add: dRHS mismatch")
    }

    // --------------------------------------------

    // MARK: 3) Test derivative of '-'

    // --------------------------------------------
    @Test(arguments: zip(
        [(0.0, 0.0), (1.0, 2.0), (-3.5, 4.0)],
        [(0.0, 2.0), (1.0, -1.0), (2.5, -3.0)]
    ))
    func testSubtract(lhs: (Double, Double), rhs: (Double, Double)) {
        let z1 = Complex(lhs.0, lhs.1)
        let z2 = Complex(rhs.0, rhs.1)

        let (value, pullback) = valueWithPullback(at: z1, z2) { a, b in a - b }

        // 1) Forward check
        #expect(value.real == z1.real - z2.real, "Sub: real mismatch")
        #expect(value.imaginary == z1.imaginary - z2.imaginary, "Sub: imag mismatch")

        // 2) Pullback check
        let upstream = Complex<Double>(1, 2)
        let (dLHS, dRHS) = pullback(upstream)
        #expect(dLHS == upstream, "Sub: dLHS mismatch")
        #expect(dRHS == Complex(-upstream.real, -upstream.imaginary), "Sub: dRHS mismatch")
    }

    // --------------------------------------------

    // MARK: 4) Test derivative of '*'

    // --------------------------------------------
    @Test(arguments: zip(
        [(1.0, 0.0), (2.0, 3.0), (-1.5, 2.5)],
        [(0.0, 1.0), (4.0, -1.0), (1.5, 1.5)]
    ))
    func testMultiply(lhs: (Double, Double), rhs: (Double, Double)) {
        let z1 = Complex(lhs.0, lhs.1)
        let z2 = Complex(rhs.0, rhs.1)

        let (value, pullback) = valueWithPullback(at: z1, z2) { a, b in a * b }

        // 1) Forward check
        let realProd = z1.real * z2.real - z1.imaginary * z2.imaginary
        let imagProd = z1.real * z2.imaginary + z1.imaginary * z2.real
        #expect(value == Complex(realProd, imagProd), "Mul: forward mismatch")

        // 2) Pullback check
        let upstream = Complex<Double>(1, 1)
        let (dLHS, dRHS) = pullback(upstream)
        let expectedLHS = upstream * z2
        let expectedRHS = upstream * z1
        #expect(dLHS == expectedLHS, "Mul: dLHS mismatch")
        #expect(dRHS == expectedRHS, "Mul: dRHS mismatch")
    }

    // --------------------------------------------

    // MARK: 5) Test derivative of '/'

    // --------------------------------------------
    @Test(arguments: zip(
        [(1.0, 0.0), (2.0, 3.0), (2.5, -1.5)],
        [(0.5, 0.0), (1.0, 2.0), (2.0, -3.0)]
    ))
    func testDivide(lhs: (Double, Double), rhs: (Double, Double)) {
        let z1 = Complex(lhs.0, lhs.1)
        let z2 = Complex(rhs.0, rhs.1)

        // Avoid dividing by zero in tests
        #expect(!(z2.real == 0 && z2.imaginary == 0), "Zero rhs not allowed")

        let (value, pullback) = valueWithPullback(at: z1, z2) { a, b in a / b }

        // 1) Forward check (with tolerance)
        let denom = z2.real * z2.real + z2.imaginary * z2.imaginary
        let realQuot = (z1.real * z2.real + z1.imaginary * z2.imaginary) / denom
        let imagQuot = (z1.imaginary * z2.real - z1.real * z2.imaginary) / denom
        expectClose(value, Complex(realQuot, imagQuot))

        // 2) Pullback checks
        let upstream = Complex<Double>(-1, 0.5)
        let (dLHS, dRHS) = pullback(upstream)

        // a) dLHS = v / z2
        let computedLHS = upstream / z2
        expectClose(dLHS, computedLHS)

        // b) dRHS = -lhs / (z2*z2) * v
        let lhsOverZ2squared = z1 / (z2 * z2)
        let computedRHS = -lhsOverZ2squared * upstream
        expectClose(dRHS, computedRHS)
    }

    // --------------------------------------------

    // MARK: 6) Test derivative of '.conjugate'

    // --------------------------------------------
    @Test(arguments: [(1.0, 2.0), (3.0, -4.0), (0.0, -1.0)])
    func testConjugate(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (value, pullback) = valueWithPullback(at: c) { $0.conjugate }

        // 1) Forward check
        #expect(value == Complex(c.real, -c.imaginary), "Conjugate mismatch")

        // 2) Pullback check
        let upstream = Complex<Double>(2, 1)
        let d = pullback(upstream)
        // derivative => v.conjugate
        #expect(d == Complex(upstream.real, -upstream.imaginary), "dConjugate mismatch")
    }

    // ---------------------------------------------------

    // MARK: 7) Test derivative of 'exp'

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (1.0, -0.5),
        (-0.5, 1.5)
    ])
    func testExp(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pullback) = valueWithPullback(at: c) { Complex.exp($0) }

        // 1) Forward check
        let expectedVal = Complex.exp(c)
        expectClose(val, expectedVal)

        // 2) Derivative: d/dz exp(z) = exp(z)
        let upstream = Complex<Double>(1, -1)
        let dZ = pullback(upstream)
        expectClose(dZ, upstream * expectedVal)
    }

    // ---------------------------------------------------

    // MARK: 8) expMinusOne

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, 0.5)
    ])
    func testExpMinusOne(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pullback) = valueWithPullback(at: c) { Complex.expMinusOne($0) }

        // 1) Forward
        let expectedVal = Complex.exp(c) - Complex(1, 0)
        expectClose(val, expectedVal)

        // 2) Derivative
        let upstream = Complex<Double>(-0.5, 2)
        let dZ = pullback(upstream)
        // derivative is exp(z)
        expectClose(dZ, upstream * Complex.exp(c))
    }

    // ---------------------------------------------------

    // MARK: 9) cosh

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (1.0, 0.5),
        (-0.5, 2.0)
    ])
    func testCosh(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.cosh($0) }

        let expectedVal = Complex.cosh(c)
        expectClose(val, expectedVal)

        // derivative => sinh(z)
        let upstream = Complex<Double>(1, 1)
        let dZ = pb(upstream)
        expectClose(dZ, upstream * Complex.sinh(c))
    }

    // ---------------------------------------------------

    // MARK: 10) sinh

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (2.0, -1.0),
        (-0.5, 0.5)
    ])
    func testSinh(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.sinh($0) }

        let expectedVal = Complex.sinh(c)
        expectClose(val, expectedVal)

        // derivative => cosh(z)
        let upstream = Complex<Double>(-1, 2)
        let dZ = pb(upstream)
        expectClose(dZ, upstream * Complex.cosh(c))
    }

    // ---------------------------------------------------

    // MARK: 11) tanh

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (1.0, 1.0),
        (-0.5, -2.0)
    ])
    func testTanh(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.tanh($0) }

        let expectedVal = Complex.tanh(c)
        expectClose(val, expectedVal)

        // derivative => 1 - tanh^2(z)
        let upstream = Complex<Double>(-1, 1)
        let factor = Complex(1, 0) - expectedVal * expectedVal
        let dZ = pb(upstream)
        expectClose(dZ, upstream * factor)
    }

    // ---------------------------------------------------

    // MARK: 12) cos

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (1.0, 2.0),
        (-0.5, -1.0)
    ])
    func testCos(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.cos($0) }

        let expectedVal = Complex.cos(c)
        expectClose(val, expectedVal)

        // derivative => -sin(z)
        let upstream = Complex<Double>(1, -1)
        let dZ = pb(upstream)
        expectClose(dZ, upstream * (-Complex.sin(c)))
    }

    // ---------------------------------------------------

    // MARK: 13) sin

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (2.0, -1.5),
        (-1.0, 1.0)
    ])
    func testSin(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.sin($0) }

        let expectedVal = Complex.sin(c)
        expectClose(val, expectedVal)

        // derivative => cos(z)
        let upstream = Complex<Double>(-1, 2)
        let dZ = pb(upstream)
        expectClose(dZ, upstream * Complex.cos(c))
    }

    // ---------------------------------------------------

    // MARK: 14) tan

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (1.0, 1.0),
        (-0.5, 0.5)
    ])
    func testTan(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.tan($0) }

        let expectedVal = Complex.tan(c)
        expectClose(val, expectedVal)

        // derivative => sec^2(z) = 1 + tan^2(z)
        let upstream = Complex<Double>(1, -1)
        let factor = Complex(1, 0) + expectedVal * expectedVal
        let dZ = pb(upstream)
        expectClose(dZ, upstream * factor)
    }

    // ---------------------------------------------------

    // MARK: 15) sqrt

    // ---------------------------------------------------
    @Test(arguments: [
        (4.0, 0.0),
        (1.0, 1.0),
        (0.5, -1.0)
    ])
    func testSqrt(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.sqrt($0) }

        let expectedVal = Complex.sqrt(c)
        expectClose(val, expectedVal)

        // derivative => 1 / (2 * sqrt(z))
        let upstream = Complex<Double>(1, -1)
        let dZ = pb(upstream)
        let denom = expectedVal * 2
        expectClose(dZ, upstream / denom)
    }

    // ---------------------------------------------------

    // MARK: 16) log

    // ---------------------------------------------------
    @Test(arguments: [
        (1.0, 0.0), // log(1) = 0
        (0.5, 1.0),
        (2.0, -1.0)
    ])
    func testLog(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        #expect(!c.isZero && c.isFinite, "Avoid zero or infinite log")

        let (val, pb) = valueWithPullback(at: c) { Complex.log($0) }

        let expectedVal = Complex.log(c)
        expectClose(val, expectedVal)

        // derivative => 1 / z
        let upstream = Complex<Double>(-1, 2)
        let dZ = pb(upstream)
        expectClose(dZ, upstream / c)
    }

    // ---------------------------------------------------

    // MARK: 17) log(onePlus:)

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (0.5, 1.0),
        (-0.25, 0.75)
    ])
    func testLogOnePlus(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.log(onePlus: $0) }

        // forward
        let expectedVal = Complex.log(onePlus: c)
        expectClose(val, expectedVal)

        // derivative => 1 / (1 + z)
        let upstream = Complex<Double>(1, 1)
        let dZ = pb(upstream)
        expectClose(dZ, upstream / (Complex(1, 0) + c))
    }

    // ---------------------------------------------------

    // MARK: 18) asin

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (0.5, 0.0),
        (-0.25, 1.0)
    ])
    func testAsin(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.asin($0) }

        let expectedVal = Complex.asin(c)
        expectClose(val, expectedVal)

        // derivative => 1 / sqrt(1 - z^2)
        let upstream = Complex<Double>(1, 1)
        let denom = Complex.sqrt(Complex(1, 0) - c * c)
        let dZ = pb(upstream)
        expectClose(dZ, upstream / denom)
    }

    // ---------------------------------------------------

    // MARK: 19) acos

    // ---------------------------------------------------
    @Test(arguments: [
        (1.0, 0.0),
        (0.5, 1.0),
        (-0.5, 0.5)
    ])

    func testAcos(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.acos($0) }

        // 1) Forward check
        let expectedVal = Complex.acos(c)
        expectClose(val, expectedVal)

        // 2) Derivative check:
        //    derivative(acos z) = -1 / sqrt(1 - z^2)
        let upstream = Complex<Double>(1, 2)
        let denom = -Complex.sqrt(Complex(1, 0) - c * c)
        let dZ = pb(upstream)

        // If z == 1.0 (real axis), derivative is infinite => NaN in floating-point
        if c == Complex(1, 0) {
            #expect(dZ.real.isNaN, "acos'(1).real should be NaN (∞ in math).")
            #expect(dZ.imaginary.isNaN, "acos'(1).imag should be NaN.")
        }
        else {
            // Normal case
            expectClose(dZ, upstream / denom)
        }
    }

    // ---------------------------------------------------

    // MARK: 20) atan

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (1.0, 1.0),
        (-0.5, 0.25)
    ])
    func testAtan(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.atan($0) }

        let expectedVal = Complex.atan(c)
        expectClose(val, expectedVal)

        // derivative => 1 / (1 + z^2)
        let upstream = Complex<Double>(2, -1)
        let dZ = pb(upstream)
        expectClose(dZ, upstream / (Complex(1, 0) + c * c))
    }

    // ---------------------------------------------------

    // MARK: 21) asinh

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (1.0, -1.0),
        (-1.5, 0.5)
    ])
    func testAsinh(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.asinh($0) }

        let expectedVal = Complex.asinh(c)
        expectClose(val, expectedVal)

        // derivative => 1 / sqrt(1 + z^2)
        let upstream = Complex<Double>(0.5, 1)
        let denom = Complex.sqrt(Complex(1, 0) + c * c)
        let dZ = pb(upstream)
        expectClose(dZ, upstream / denom)
    }

    // ---------------------------------------------------

    // MARK: 22) acosh

    // ---------------------------------------------------
    @Test(arguments: [
        (1.5, 0.0),
        (2.0, 1.0),
        (3.0, -2.0) // note: real domain for acosh is Re(z) >= 1, but complex is fine
    ])
    func testAcosh(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.acosh($0) }

        let expectedVal = Complex.acosh(c)
        expectClose(val, expectedVal)

        // derivative => 1 / sqrt(z^2 - 1)
        let upstream = Complex<Double>(-0.5, 2)
        let denom = Complex.sqrt(c * c - Complex(1, 0))
        let dZ = pb(upstream)
        expectClose(dZ, upstream / denom)
    }

    // ---------------------------------------------------

    // MARK: 23) atanh

    // ---------------------------------------------------
    @Test(arguments: [
        (0.0, 0.0),
        (0.5, 1.0),
        (-1.0, 0.5)
    ])
    func testAtanh(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (val, pb) = valueWithPullback(at: c) { Complex.atanh($0) }

        let expectedVal = Complex.atanh(c)
        expectClose(val, expectedVal)

        // derivative => 1/(1 - z^2)
        let upstream = Complex<Double>(1, -1)
        let denom = Complex(1, 0) - c * c
        let dZ = pb(upstream)
        expectClose(dZ, upstream / denom)
    }

    // ---------------------------------------------------

    // MARK: 24) pow

    // ---------------------------------------------------
    @Test(arguments: zip(
        // A few test pairs for (z1, z2)
        [(1.0, 0.0), (2.0, 3.0), (-1.5, 2.5)],
        [(0.0, 1.0), (1.0, -1.0), (2.5, 1.5)]
    ))
    func testPowZW(lhs: (Double, Double), rhs: (Double, Double)) {
        let z1 = Complex(lhs.0, lhs.1)
        let z2 = Complex(rhs.0, rhs.1)

        // 1) Forward check: valueWithPullback on `Complex.pow(a, b)`
        let (value, pullback) = valueWithPullback(at: z1, z2) { a, b in
            Complex.pow(a, b)
        }
        // Compare with direct call
        let expectedVal = Complex.pow(z1, z2)
        #expect(value == expectedVal, "Pow: forward mismatch")

        // 2) Pullback check
        //    derivative wrt z => val * (z2 / z1)
        //    derivative wrt w => val * log(z1)
        let upstream = Complex<Double>(1, -1)
        let (dLHS, dRHS) = pullback(upstream)

        let expectedLHS = upstream * (expectedVal * (z2 / z1))
        let expectedRHS = upstream * (expectedVal * Complex.log(z1))

        #expect(dLHS == expectedLHS, "Pow: dLHS mismatch")
        #expect(dRHS == expectedRHS, "Pow: dRHS mismatch")
    }

    // ---------------------------------------------------

    // MARK: 25) pow(z, n:Int) (wr.t z only)

    // ---------------------------------------------------
    @Test(arguments: zip(
        // Some sample complex values for z
        [(1.0, 0.0), (2.0, 1.0), (0.5, -1.5)],
        // Corresponding integer exponents
        [0, 3, -2]
    ))
    func testPowZIntWrtZ(zInput: (Double, Double), n: Int) {
        let z = Complex(zInput.0, zInput.1)

        // We do *not* call `valueWithPullback(at: z, n)` because 'n' is not Differentiable.
        // Instead, we treat 'n' as a captured constant and only differentiate wrt 'z'.
        let (value, pullback) = valueWithPullback(at: z) { zVal in
            // Here, Swift sees a single-parameter function (zVal) -> Complex.
            // `n` is captured as a constant integer, so the derivative is purely wrt zVal.
            Complex.pow(zVal, n)
        }

        // 1) Forward check
        let expectedVal = Complex.pow(z, n)
        #expect(
            value == expectedVal,
            "pow(z, n): forward mismatch for z=\(z), n=\(n). Got \(value), expected \(expectedVal)."
        )

        // 2) Pullback check => derivative wrt z => n * z^(n-1).
        // Multiply that by our 'upstream' test gradient.
        let upstream = Complex<Double>(1, -1)
        let dZ = pullback(upstream)

        let partial: Complex<Double>
        if n == 0 {
            // derivative of z^0 = 0
            partial = .zero
        }
        else {
            // n * z^(n - 1), ignoring edge cases if negative or zero exponent
            partial = Complex(Double(n)) * Complex.pow(z, n - 1)
        }
        let expectedDZ = upstream * partial

        #expect(
            dZ == expectedDZ,
            "pow(z, n): derivative mismatch for z=\(z), n=\(n). Got \(dZ), expected \(expectedDZ)."
        )
    }

    // ---------------------------------------------------

    // MARK: 26) root(z, n:Int) (w.r.t. z only)

    // ---------------------------------------------------
    @Test(arguments: zip(
        // Some sample complex values for z
        [(1.0, 0.0), (4.0, 0.0), (0.5, 2.0)],
        // Corresponding integer exponents
        [2, 3, -1]
    ))
    func testRootZIntWrtZ(zInput: (Double, Double), n: Int) {
        let z = Complex(zInput.0, zInput.1)

        // We do NOT call `valueWithPullback(at: z, n)` because `n` is not Differentiable.
        // Instead, treat n as a captured constant and only differentiate w.r.t. z.
        let (value, pullback) = valueWithPullback(at: z) { zVal in
            // single-parameter function zVal -> Complex
            // n is a captured constant
            Complex.root(zVal, n) // -> zVal^(1/n)
        }

        // 1) Forward check
        let expectedVal = Complex.root(z, n)
        #expect(
            value == expectedVal,
            "root(z, n): forward mismatch for z=\(z), n=\(n). Got \(value), expected \(expectedVal)."
        )

        // 2) Pullback check => derivative wrt z => z^(1/n) / (n * z)
        // Multiply that by the upstream.
        let upstream = Complex<Double>(1, -1)
        let dZ = pullback(upstream)

        // If n=0 or z=0 => derivative is undefined in pure math
        // (currently returns .zero in those cases).
        if n == 0 || z.isZero {
            // We can check if it returned .zero.
            #expect(
                dZ == .zero,
                "Expected derivative .zero for n=0 or z=0, got \(dZ)"
            )
        }
        else {
            // derivative => val / (n*z)
            let partial = expectedVal / (Complex(Double(n)) * z)
            let expectedDZ = upstream * partial
            #expect(
                dZ == expectedDZ,
                "root(z, n): derivative mismatch for z=\(z), n=\(n). Got \(dZ), expected \(expectedDZ)."
            )
        }
    }
}
