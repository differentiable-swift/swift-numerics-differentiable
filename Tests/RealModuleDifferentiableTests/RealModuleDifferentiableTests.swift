import _Differentiation
@testable import RealModuleDifferentiable
import Testing

@Suite
struct TestRegisteredDerivatives {
    // These tests are more about the derivatives being correctly registered than the correct values. Should probably use a form of
    // `.isApproximatelyEqual(to:)` for the results in the future but that doesn't combine too well with a SIMD vector comparison.
    @Test
    func
        testExp()
    {
        let vwpb = valueWithPullback(at: 2.0, of: Float.exp)
        #expect(vwpb.value == 7.38905609893065)
        #expect(vwpb.pullback(1) == 7.38905609893065)
    }

    @Test
    func testExpMinusOne() {
        let vwpb = valueWithPullback(at: 2.0, of: Double.expMinusOne(_:))
        #expect(vwpb.value == 6.38905609893065)
        #expect(vwpb.pullback(1) == 7.38905609893065)
    }

    @Test
    func testCosh() {
        let vwpb = valueWithPullback(at: SIMD2<Float>(repeating: 2.0), of: SIMD2<Float>.cosh)
        #expect(vwpb.value == .init(repeating: 3.7621956))
        #expect(vwpb.pullback(.one) == .init(repeating: 3.6268604))
    }

    @Test
    func testSinh() {
        let vwpb = valueWithPullback(at: SIMD4<Float>(repeating: 2.0), of: SIMD4<Float>.sinh)
        #expect(vwpb.value == .init(repeating: 3.6268604))
        #expect(vwpb.pullback(.one) == .init(repeating: 3.7621956))
    }

    @Test
    func testTanh() {
        let vwpb = valueWithPullback(at: SIMD8<Float>(repeating: 2.0), of: SIMD8<Float>.tanh)
        #expect(vwpb.value == .init(repeating: 0.9640276))
        #expect(vwpb.pullback(.one) == .init(repeating: 0.07065083))
    }

    @Test
    func testCos() {
        let vwpb = valueWithPullback(at: SIMD16<Float>(repeating: .pi / 2), of: SIMD16<Float>.cos)
        #expect(vwpb.value == .init(repeating: 7.54979E-08))
        #expect(vwpb.pullback(.one) == .init(repeating: -1))
    }

    @Test
    func testSin() {
        let vwpb = valueWithPullback(at: SIMD32<Float>(repeating: .pi / 2), of: SIMD32<Float>.sin)
        #expect(vwpb.value == .init(repeating: 1))
        #expect(vwpb.pullback(.one) == .init(repeating: 7.54979E-08))
    }

    @Test
    func testTan() {
        let vwpb = valueWithPullback(at: SIMD64<Float>(repeating: .pi / 4), of: SIMD64<Float>.tan)
        #expect(vwpb.value == .init(repeating: 0.99999994))
        #expect(vwpb.pullback(.one) == .init(repeating: 1.9999998))
    }

    @Test
    func testLog() {
        let vwpb = valueWithPullback(at: SIMD2<Double>(repeating: 2), of: SIMD2<Double>.log(_:))
        #expect(vwpb.value == SIMD2<Double>(repeating: 0.6931471805599453))
        #expect(vwpb.pullback(SIMD2<Double>.one) == .init(repeating: 0.5))
    }

    @Test
    func testLogOnePlus() {
        let vwpb = valueWithPullback(at: SIMD4<Double>(repeating: 3), of: SIMD4<Double>.log(onePlus:))
        #expect(vwpb.value == .init(repeating: 1.3862943611198906))
        #expect(vwpb.pullback(.one) == .init(repeating: 0.25))
    }

    @Test
    func testAcosh() {
        let vwpb = valueWithPullback(at: SIMD8<Double>(repeating: 2), of: SIMD8<Double>.acosh)
        #expect(vwpb.value == .init(repeating: 1.3169578969248166))
        #expect(vwpb.pullback(.one) == .init(repeating: 1 / .sqrt(3)))
    }

    @Test
    func testAsinh() {
        let vwpb = valueWithPullback(at: SIMD16<Double>(repeating: 2), of: SIMD16<Double>.asinh)
        #expect(vwpb.value == .init(repeating: 1.4436354751788103))
        #expect(vwpb.pullback(.one) == .init(repeating: 1 / .sqrt(5)))
    }

    @Test
    func testAtanh() {
        let vwpb = valueWithPullback(at: SIMD32<Double>(repeating: 0.5), of: SIMD32<Double>.atanh)
        #expect(vwpb.value == .init(repeating: 0.5493061443340549))
        #expect(vwpb.pullback(.one) == .init(repeating: 4 / 3))
    }

    @Test
    func testaCos() {
        let vwpb = valueWithPullback(at: SIMD64<Double>(repeating: 0.5), of: SIMD64<Double>.acos)
        #expect(vwpb.value == .init(repeating: 1.0471975511965976))
        #expect(vwpb.pullback(.one) == .init(repeating: -1.1547005383792517))
    }

    @Test
    func testaSin() {
        let vwpb = valueWithPullback(at: 0.5, of: Float.asin)
        #expect(vwpb.value == 0.5235988)
        #expect(vwpb.pullback(1) == 1.1547005383792517)
    }

    @Test
    func testaTan() {
        let vwpb = valueWithPullback(at: 0.5, of: Double.atan)
        #expect(vwpb.value == 0.46364760900080615)
        #expect(vwpb.pullback(1) == 0.8)
    }

    @Test
    func testPow() {
        let vwpb = valueWithPullback(at: SIMD2<Float>(repeating: 0.5), SIMD2<Float>(repeating: 2), of: SIMD2<Float>.pow(_:_:))
        #expect(vwpb.value == .init(repeating: 0.25))
        #expect(vwpb.pullback(.one) == (.init(repeating: 1.0), .init(repeating: -0.1732868)))
    }

    @Test
    func testPowInt() {
        let vwpb = valueWithPullback(at: SIMD4<Float>(repeating: 0.5), of: { x in SIMD4<Float>.pow(x, 2) })
        #expect(vwpb.value == .init(repeating: 0.25))
        #expect(vwpb.pullback(.one) == .init(repeating: 1.0))
    }

    @Test
    func testSqrt() {
        let vwpb = valueWithPullback(at: SIMD8<Float>(repeating: 4), of: SIMD8<Float>.sqrt)
        #expect(vwpb.value == .init(repeating: 2))
        #expect(vwpb.pullback(.one) == .init(repeating: 0.25))
    }

    @Test
    func testRoot() {
        let vwpb = valueWithPullback(at: SIMD16<Float>(repeating: 16), of: { x in SIMD16<Float>.root(x, 4) })
        #expect(vwpb.value == .init(repeating: 2))
        #expect(vwpb.pullback(.one) == .init(repeating: 1 / 32))
    }

    @Test
    func testAtan2() {
        let vwpb = valueWithPullback(at: SIMD32<Float>(repeating: 1), SIMD32<Float>(repeating: 0), of: SIMD32<Float>.atan2)
        #expect(vwpb.value == .init(repeating: 1.5707964)) // .pi / 2
        #expect(vwpb.pullback(.one) == (.init(repeating: 0), .init(repeating: -1)))
    }

    @Test
    func testErf() {
        let vwpb = valueWithPullback(at: SIMD64<Float>(repeating: 0.5), of: SIMD64<Float>.erf)
        #expect(vwpb.value == .init(repeating: 0.5204999))
        #expect(vwpb.pullback(.one) == .init(repeating: 0.87878263))
    }

    @Test
    func testErfc() {
        let vwpb = valueWithPullback(at: SIMD2<Double>(repeating: 0.5), of: SIMD2<Double>.erfc)
        #expect(vwpb.value == .init(repeating: 0.4795001221869535))
        #expect(vwpb.pullback(.one) == .init(repeating: -0.8787825789354449))
    }

    @Test
    func testExp2() {
        let vwpb = valueWithPullback(at: SIMD4<Double>(repeating: 2), of: SIMD4<Double>.exp2)
        #expect(vwpb.value == .init(repeating: 4))
        #expect(vwpb.pullback(.one) == .init(repeating: 4 * .log(2)))
    }

    @Test
    func testExp10() {
        let vwpb = valueWithPullback(at: SIMD8<Double>(repeating: 2), of: SIMD8<Double>.exp10)
        #expect(vwpb.value == .init(repeating: 100))
        #expect(vwpb.pullback(.one) == .init(repeating: 100 * .log(10)))
    }

    @Test
    func testHypot() {
        let vwpb = valueWithPullback(at: SIMD16<Double>(repeating: 3), SIMD16<Double>(repeating: 4), of: SIMD16<Double>.hypot)
        #expect(vwpb.value == .init(repeating: 5))
        #expect(vwpb.pullback(.one) == (.init(repeating: 3 / 5), .init(repeating: 4 / 5)))
    }

    @Test(.disabled("derivative not implemented"))
    func testGamma() {
        let vwpb = valueWithPullback(at: SIMD32<Double>(repeating: 2), of: SIMD32<Double>.gamma)
        #expect(vwpb.value == .init(repeating: 1))
        #expect(vwpb.pullback(.one) == .init(repeating: 0))
    }

    @Test
    func testLog2() {
        let vwpb = valueWithPullback(at: SIMD64<Double>(repeating: 2), of: SIMD64<Double>.log2)
        #expect(vwpb.value == .init(repeating: 1))
        #expect(vwpb.pullback(.one) == .init(repeating: 1 / .log(4)))
    }

    @Test
    func testLog10() {
        let vwpb = valueWithPullback(at: 2.0, of: Float.log10)
        #expect(vwpb.value == 0.30103) // .log(2) / .log(10)
        #expect(vwpb.pullback(1) == 1 / .log(100))
    }

    @Test(.disabled("derivative not implemented"))
    func testLogGamma() {
        let vwpb = valueWithPullback(at: 2, of: Double.logGamma)
        #expect(vwpb.value == 0)
        #expect(vwpb.pullback(1) == 0)
    }

    @Test
    func testAbs() {
        let vwpb = valueWithPullback(at: SIMD2<Float>(repeating: -2), of: SIMD2<Float>.abs)
        #expect(vwpb.value == .init(repeating: 2))
        #expect(vwpb.pullback(.one) == .init(repeating: -1))
    }
}
