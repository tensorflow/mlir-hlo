// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[2.67503524, -1.21969211]> : tensor<2xf32>
    %cst_0 = stablehlo.constant dense<[-1.51913178, -1.08529949]> : tensor<2xf32>
    %cst_1 = stablehlo.constant dense<[181.823883, 91.2684555]> : tensor<2xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.0039059886745378084:-128>>
    %2 = stablehlo.divide %1, %0 : (tensor<2x!quant.uniform<i8:f32, 0.0039059886745378084:-128>>, tensor<2x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>) -> tensor<2x!quant.uniform<i8:f32, 0.7130348355162377:-128>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<2x!quant.uniform<i8:f32, 0.7130348355162377:-128>>) -> tensor<2xf32>
    %4 = stablehlo.custom_call @check.eq(%cst_1, %3) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
}
