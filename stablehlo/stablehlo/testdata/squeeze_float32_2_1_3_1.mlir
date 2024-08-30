// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x1x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x1x3x1xf32>
    %1 = call @expected() : () -> tensor<2x1x3xf32>
    %2 = stablehlo.reshape %0 : (tensor<2x1x3x1xf32>) -> tensor<2x1x3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x1x3xf32>, tensor<2x1x3xf32>) -> ()
    return %2 : tensor<2x1x3xf32>
  }
  func.func private @inputs() -> (tensor<2x1x3x1xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[[-1.59504378], [-2.05178547], [3.58855987]]], [[[-0.277097464], [3.28455472], [-1.11533749]]]]> : tensor<2x1x3x1xf32>
    return %cst : tensor<2x1x3x1xf32>
  }
  func.func private @expected() -> (tensor<2x1x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-1.59504378, -2.05178547, 3.58855987]], [[-0.277097464, 3.28455472, -1.11533749]]]> : tensor<2x1x3xf32>
    return %cst : tensor<2x1x3xf32>
  }
}