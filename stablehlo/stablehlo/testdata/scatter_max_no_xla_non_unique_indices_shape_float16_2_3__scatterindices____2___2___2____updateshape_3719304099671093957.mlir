// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<2> : tensor<1x3x1xi32>
    %1:2 = call @inputs() : () -> (tensor<2x3xf16>, tensor<2x1x3xf16>)
    %2 = call @expected() : () -> tensor<2x3xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>} : (tensor<2x3xf16>, tensor<1x3x1xi32>, tensor<2x1x3xf16>) -> tensor<2x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf16>, tensor<2x1x3xf16>) {
    %0 = stablehlo.constant dense<[[-2.086180e-01, 2.542970e+00, 2.121090e+00], [-5.576170e-01, -1.132810e+00, -2.250000e+00]]> : tensor<2x3xf16>
    %1 = stablehlo.constant dense<[[[-1.725590e+00, -1.166990e+00, 2.515630e+00]], [[-3.292970e+00, -1.618650e-01, 1.017580e+00]]]> : tensor<2x1x3xf16>
    return %0, %1 : tensor<2x3xf16>, tensor<2x1x3xf16>
  }
  func.func private @expected() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<[[-2.086180e-01, 2.542970e+00, 2.515630e+00], [-5.576170e-01, -1.132810e+00, 1.017580e+00]]> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
}
