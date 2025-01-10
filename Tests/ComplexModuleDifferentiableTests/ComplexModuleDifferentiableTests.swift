import _Differentiation
@testable import ComplexModule
import Testing

// MARK: - Test Suite for Complex Derivatives

struct ComplexDerivativesTests {
    // --------------------------------------------

    // MARK: 1) Test derivative of init(_:_)

    // --------------------------------------------
    //
    // The derivative of Complex(real, imaginary) w.r.t. (real, imaginary)
    // is the identity map. We check that using valueWithPullback.
    //
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
        #expect(value.real == real)
        #expect(value.imaginary == imag)

        // 2) Pullback check:
        //    derivative wrt real -> (1, 0)
        //    derivative wrt imag -> (0, 1)
        //    i.e. if upstream "v", then pullback(v) = (v.real, v.imaginary)
        let upstream = Complex<Double>(2, -3) // arbitrary test gradient
        let (dReal, dImag) = pullback(upstream)

        #expect(dReal == upstream.real)
        #expect(dImag == upstream.imaginary)
    }

    // --------------------------------------------

    // MARK: 2) Test derivative of '+'

    // --------------------------------------------
    //
    // The derivative of (lhs + rhs) w.r.t. lhs is identity,
    // w.r.t. rhs is identity as well.
    //
    @Test(arguments: zip(
        [(0.0, 0.0), (1.0, 2.0), (-3.5, 4.0)],
        [(0.0, 2.0), (1.0, -1.0), (2.5, -3.0)]
    ))
    func testAdd(lhs: (Double, Double), rhs: (Double, Double)) {
        let z1 = Complex(lhs.0, lhs.1)
        let z2 = Complex(rhs.0, rhs.1)

        let (value, pullback) = valueWithPullback(at: z1, z2) { a, b in a + b }

        // 1) Forward check
        #expect(value.real == z1.real + z2.real)
        #expect(value.imaginary == z1.imaginary + z2.imaginary)

        // 2) Pullback check
        //    derivative wrt lhs -> v, wrt rhs -> v
        let upstream = Complex<Double>(1, -1)
        let (dLHS, dRHS) = pullback(upstream)
        #expect(dLHS == upstream)
        #expect(dRHS == upstream)
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
        #expect(value.real == z1.real - z2.real)
        #expect(value.imaginary == z1.imaginary - z2.imaginary)

        // 2) Pullback check
        //    derivative wrt lhs -> +v, wrt rhs -> -v
        let upstream = Complex<Double>(1, 2)
        let (dLHS, dRHS) = pullback(upstream)
        #expect(dLHS == upstream)
        #expect(dRHS == Complex(-upstream.real, -upstream.imaginary))
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
        // (x1 + i y1)*(x2 + i y2) = (x1*x2 - y1*y2) + i(x1*y2 + y1*x2)
        let realProd = z1.real * z2.real - z1.imaginary * z2.imaginary
        let imagProd = z1.real * z2.imaginary + z1.imaginary * z2.real
        #expect(value == Complex(realProd, imagProd))

        // 2) Pullback check:
        //    d/dlhs -> v * conj(rhs)
        //    d/drhs -> v * conj(lhs)
        let upstream = Complex<Double>(1, 1)
        let (dLHS, dRHS) = pullback(upstream)

        // We'll just check that this matches the formula for v*conjugate(...)
        // Not rewriting the entire math; do a spot check for correctness:
        // dLHS = (1 + i1) * (rhs.real - i rhs.imag)
        let conjR = Complex(z2.real, -z2.imaginary)
        let expectedLHS = upstream * conjR
        #expect(dLHS == expectedLHS)

        let conjL = Complex(z1.real, -z1.imaginary)
        let expectedRHS = upstream * conjL
        #expect(dRHS == expectedRHS)
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
        #expect(!(z2.real == 0 && z2.imaginary == 0))

        let (value, pullback) = valueWithPullback(at: z1, z2) { a, b in a / b }

        // 1) Forward check
        let denom = (z2.real * z2.real) + (z2.imaginary * z2.imaginary)
        let realQuot = (z1.real * z2.real + z1.imaginary * z2.imaginary) / denom
        let imagQuot = (z1.imaginary * z2.real - z1.real * z2.imaginary) / denom
        #expect(value == Complex(realQuot, imagQuot))

        // 2) Pullback checks
        // d/dlhs -> v / rhs
        // d/drhs -> -(lhs * v)/|rhs|^2
        let upstream = Complex<Double>(-1, 0.5)
        let (dLHS, dRHS) = pullback(upstream)

        // quick check: dLHS should be upstream * conj(rhs) / denom
        let conjR = Complex(z2.real, -z2.imaginary)
        let uv = upstream * conjR
        let expectedLHS = Complex(uv.real / denom, uv.imaginary / denom)
        #expect(dLHS == expectedLHS)

        // dRHS => -( lhs * upstream ) / denom
        let lv = Complex(
            z1.real * upstream.real - z1.imaginary * upstream.imaginary,
            z1.real * upstream.imaginary + z1.imaginary * upstream.real
        )
        let partial = Complex(lv.real / denom, lv.imaginary / denom)
        let expectedRHS = Complex(-partial.real, -partial.imaginary)
        #expect(dRHS == expectedRHS)
    }

    // --------------------------------------------

    // MARK: 6) Test derivative of '.conjugate'

    // --------------------------------------------
    @Test(arguments: [(1.0, 2.0), (3.0, -4.0), (0.0, -1.0)])
    func testConjugate(z: (Double, Double)) {
        let c = Complex(z.0, z.1)
        let (value, pullback) = valueWithPullback(at: c) { $0.conjugate }

        // 1) Forward check
        #expect(value == Complex(c.real, -c.imaginary))

        // 2) Pullback check
        let upstream = Complex<Double>(2, 1)
        // derivative => flip sign of imaginary
        let d = pullback(upstream)
        #expect(d == Complex(upstream.real, -upstream.imaginary))
    }
}
