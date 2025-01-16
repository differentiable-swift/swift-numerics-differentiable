// Export the ComplexModule API from Swift Numerics which this module extends with `Differentiable` support
@_exported import ComplexModule

// Export the differentiation module since we're trying to use its api
#if canImport(_Differentiation)
@_exported import _Differentiation
#endif
