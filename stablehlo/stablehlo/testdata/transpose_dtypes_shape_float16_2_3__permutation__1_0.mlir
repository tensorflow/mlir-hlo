// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xf16>
    %1 = call @expected() : () -> tensor<3x2xf16>
    %2 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xf16>) -> tensor<3x2xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xf16>, tensor<3x2xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<[[6.273440e+00, -7.539060e-01, -2.914430e-02], [5.761710e-01, 1.934570e+00, -2.343750e+00]]> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
  func.func private @expected() -> tensor<3x2xf16> {
    %0 = stablehlo.constant dense<[[6.273440e+00, 5.761710e-01], [-7.539060e-01, 1.934570e+00], [-2.914430e-02, -2.343750e+00]]> : tensor<3x2xf16>
    return %0 : tensor<3x2xf16>
  }
}
