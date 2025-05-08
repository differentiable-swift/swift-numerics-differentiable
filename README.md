# swift-numerics-differentiable

This package attempts to add more Differentiable capabilities to the existing [swift-numerics](https://github.com/apple/swift-numerics) package. Every target in swift-numerics has a Differentiable counterpart that `@_exported import`s the original module such that when you import `NumericsDifferentiable` you will also get all the contents of the `Numerics` module from swift-numerics. 

## RealModule Differentiable
- Registers derivatives to the `Float` and `Double` conformances to `ElementaryFunctions` and `RealFunctions` from swift-numerics.
- Conforms all `SIMD{n}` types to `ElementaryFunctions` and adds most of the protocol requirements from `RealFunctions` as well (`signGamma` is not implementable)
- Registers derivatives for all the provided `ElementaryFunctions` and `RealFunctions` implementations on SIMD{n}
- Tries to leverage Apple's `simd` framework to accelerate these operations where possible on Apple platforms.

## Contributing
### Code Formatting
This package makes use of [SwiftFormat](https://github.com/nicklockwood/SwiftFormat?tab=readme-ov-file#command-line-tool), which you can install
from [homebrew](https://brew.sh/). 

To apply formatting rules to all files, which you should do before submitting a PR, run from the root of the repository:

```sh
swiftformat .
```
Formatting is validated with the `--strict` flag on every PR

