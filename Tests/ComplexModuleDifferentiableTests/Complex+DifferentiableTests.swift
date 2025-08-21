import ComplexModuleDifferentiable
import Testing

#if canImport(_Differentiation)
@Suite
struct ComplexDifferentiableTests {
    @Test
    func componentGetter() {
        #expect(gradient(at: Complex<Float>(5, 5)) { $0.real * 2 } == Complex(2, 0))
        #expect(gradient(at: Complex<Float>(5, 5)) { $0.imaginary * 2 } == Complex(0, 2))
        #expect(gradient(at: Complex<Float>(5, 5)) {
            $0.real * 5 + $0.imaginary * 2
        } == Complex(5, 2))
    }

    @Test
    func initializer() {
        let pb1 = pullback(at: 4, -3) { r, i in Complex<Float>(r, i) }
        let tan1 = pb1(Complex(-1, 2))
        #expect(tan1.0 == -1)
        #expect(tan1.1 == 2)

        let pb2 = pullback(at: 4, -3) { r, i in Complex<Float>(r * r, i + i)
        }
        let tan2 = pb2(Complex(-1, 1))
        #expect(tan2.0 == -8)
        #expect(tan2.1 == 2)
    }

    @Test
    func conjugate() {
        let pullback = pullback(at: Complex<Float>(20, -4)) { x in x.conjugate }
        #expect(pullback(Complex(1, 0)) == Complex(1, 0))
        #expect(pullback(Complex(0, 1)) == Complex(0, -1))
        #expect(pullback(Complex(-1, 1)) == Complex(-1, -1))
    }

    @Test
    func arithmetics() {
        let additionPullback = pullback(at: Complex<Float>(2, 3)) { x in
            x + Complex(5, 6)
        }
        #expect(additionPullback(Complex(1, 1)) == Complex(1, 1))

        let subtractPullback = pullback(at: Complex<Float>(2, 3)) { x in
            Complex(5, 6) - x
        }
        #expect(subtractPullback(Complex(1, 1)) == Complex(-1, -1))

        let multiplyPullback = pullback(at: Complex<Float>(2, 3)) { x in x * x }
        #expect(multiplyPullback(Complex(1, 0)) == Complex(4, 6))
        #expect(multiplyPullback(Complex(0, 1)) == Complex(-6, 4))
        #expect(multiplyPullback(Complex(1, 1)) == Complex(-2, 10))

        let dividePullback = pullback(at: Complex<Float>(20, -4)) { x in
            x / Complex(2, 2)
        }
        #expect(dividePullback(Complex(1, 0)) == Complex(0.25, -0.25))
        #expect(dividePullback(Complex(0, 1)) == Complex(0.25, 0.25))
    }
}

#endif
