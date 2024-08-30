// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[[-2.70618963], [-1.87663591], [-2.06809068], [3.13180709], [-5.647770e-01], [-6.40476512], [-3.12457585], [4.29680347], [-4.77720356], [-2.14421678], [-4.62567759], [-4.25790119], [-0.796836376], [2.05816221], [1.44689882], [3.84717488], [-3.75239062], [1.88363218], [4.10074186], [-1.20101893], [5.12631035], [-1.34320879], [7.59781837], [6.43065262], [-0.807578682], [-2.5147438], [-1.34953046], [-0.217908949]]]> : tensor<1x28x1xf32>
    %cst_0 = stablehlo.constant dense<[[[5.97360468, 0.565488756, 1.90250754, 1.16698813, -1.0267024, 0.539800227, 1.02096438, 7.50488281, 3.29065299, 2.80925131, -1.00724733, 2.50891948, 1.9983573, -3.58216739, -1.08878911, -0.526236176]], [[1.58082891, -2.48857307, -0.902140557, 2.64325571, 4.98374081, 1.6286974, -0.793671488, 2.73143911, -0.955305993, 4.765710e-02, 2.02940774, 1.60887933, 2.17747688, 3.31030536, -0.812408626, -3.16653347]], [[-2.33335423, -0.416958928, 1.38901448, 0.764225184, -2.18225026, -2.79969263, -1.53878593, -6.03682947, 1.99186349, 4.33513737, -5.87947845, 0.59931469, 1.81237674, -4.62853718, 1.41328144, 0.307831258]]]> : tensor<3x1x16xf32>
    %cst_1 = stablehlo.constant dense<"0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000C72A803F0000000000000000C72A803FC72A803FC72A803F00000000C72A803F000000009D40513DC72A803FC72A803FC72A803FC72A803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000C72A803F6CDC0F3FC72A00401041E23F0000000035A10A3FC72A803FC72A803FC72A0040C72A0040000000006605CC3FC72A004000000000C72A803F76F09C3E00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000C72A803F0000000000000000C72A803FC72A803FC72A803F00000000C72A803F000000009D40513DC72A803FC72A803FC72A803FC72A803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000C72A803F6CDC0F3FC72A803FC72A803F0000000035A10A3FC72A803FC72A803FC72A803FC72A803F00000000C72A803FC72A803F000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000C72A803F932C443F00000000000000000000000000000000C72A803FC72A803F00000000DA521A3FC72A803F00000000C72A803F76F09C3E0000000000000000C72A803F932C443F00000000000000000000000000000000C72A803FC72A803F00000000DA521A3FC72A803F00000000C72A803F76F09C3EC72A803F00000000C72A803F1041E23FC72A803FC72A803F00000000C72A803FC72A803FFE65853FC72A803F6605CC3FC72A0040C72A803FC72A803F76F09C3EC72A803F0000000000000000C72A803FC72A803FC72A803F00000000C72A803F000000009D40513DC72A803FC72A803FC72A803FC72A803F0000000000000000C72A00406CDC0F3FC72A00407DBF2640C72A803F617BC53FC72A803FC72A0040C72A004062C80240C72A803F161826407DBF2640C72A803FC72A803F76F09C3EC72A803F6CDC0F3FC72A00401041E23F0000000035A10A3FC72A803FC72A803FC72A0040C72A0040000000006605CC3FC72A004000000000C72A803F76F09C3EC72A00406CDC0F3FC72A803FC72A0040C72A803F617BC53FC72A803FC72A0040C72A803FFE65853FC72A803FC72A0040C72A0040C72A803F0000000000000000C72A803F00000000C72A803F1041E23FC72A803FC72A803F00000000C72A803FC72A803FFE65853FC72A803F6605CC3FC72A0040C72A803FC72A803F76F09C3EC72A803F6CDC0F3FC72A803FC72A803F0000000035A10A3FC72A803FC72A803FC72A803FC72A803F00000000C72A803FC72A803F000000000000000000000000C72A00406CDC0F3FC72A00407DBF2640C72A803F617BC53FC72A803FC72A0040C72A004062C80240C72A803F161826407DBF2640C72A803FC72A803F76F09C3E0000000000000000C72A803F932C443F00000000000000000000000000000000C72A803FC72A803F00000000DA521A3FC72A803F00000000C72A803F76F09C3EC72A00406CDC0F3FC72A803FC72A0040C72A803F617BC53FC72A803FC72A0040C72A803FFE65853FC72A803FC72A0040C72A0040C72A803F0000000000000000C72A803F0000000000000000C72A803FC72A803FC72A803F00000000C72A803F000000009D40513DC72A803FC72A803FC72A803FC72A803F0000000000000000C72A803F6CDC0F3FC72A803FC72A803F0000000035A10A3FC72A803FC72A803FC72A803FC72A803F00000000C72A803FC72A803F000000000000000000000000C72A803F6CDC0F3FC72A803FC72A803F0000000035A10A3FC72A803FC72A803FC72A803FC72A803F00000000C72A803FC72A803F000000000000000000000000"> : tensor<1x24x16xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<3x1x16xf32>) -> tensor<3x1x16x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<1x28x1xf32>) -> tensor<1x28x1x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>
    %2 = stablehlo.convolution(%1, %0) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window = {rhs_dilate = [2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x1x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>, tensor<3x1x16x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>) -> tensor<1x24x16x!quant.uniform<i32:f32, 1.5367804432676217E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<1x24x16x!quant.uniform<i32:f32, 1.5367804432676217E-5>>) -> tensor<1x24x16x!quant.uniform<i8:f32, 0.0102174020281025:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<1x24x16x!quant.uniform<i8:f32, 0.0102174020281025:-128>>) -> tensor<1x24x16xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<1x24x16xf32>, tensor<1x24x16xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}