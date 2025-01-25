// RUN: stablehlo-opt %s | FileCheck %s
// RUN: stablehlo-opt %s | stablehlo-opt | FileCheck %s

// Test encodings first since aliases are printed at top of file.
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

// CHECK: #[[$CSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
// CHECK: #[[$DCSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>
// CHECK-LABEL: func @encodings
func.func @encodings(%arg0: tensor<10x20xf32, #CSR>,
                     %arg1: tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32> {
  // CHECK:      %0 = stablehlo.add %arg0, %arg1 : (tensor<10x20xf32, #[[$CSR]]>, tensor<10x20xf32, #[[$DCSR]]>) -> tensor<10x20xf32>
  // CHECK-NEXT: %1 = stablehlo.add %arg1, %arg1 : tensor<10x20xf32, #[[$DCSR]]>
  // CHECK-NEXT: %2 = stablehlo.abs %arg0 : (tensor<10x20xf32, #[[$CSR]]>) -> tensor<10x20xf32>
  // CHECK-NEXT: %3 = stablehlo.abs %arg0 : tensor<10x20xf32, #[[$CSR]]>
  // CHECK-NEXT: %4 = stablehlo.complex %arg0, %arg0 : (tensor<10x20xf32, #[[$CSR]]>, tensor<10x20xf32, #[[$CSR]]>) -> tensor<10x20xcomplex<f32>>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<10x20xf32, #CSR>,
                                   tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32>
  %1 = "stablehlo.add"(%arg1, %arg1) : (tensor<10x20xf32, #DCSR>,
                                   tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32, #DCSR>
  %2 = "stablehlo.abs"(%arg0) : (tensor<10x20xf32, #CSR>) -> tensor<10x20xf32>
  %3 = "stablehlo.abs"(%arg0) : (tensor<10x20xf32, #CSR>) -> tensor<10x20xf32, #CSR>
  %4 = "stablehlo.complex"(%arg0, %arg0) : (tensor<10x20xf32, #CSR>, tensor<10x20xf32, #CSR>) -> tensor<10x20xcomplex<f32>>
  func.return %0 : tensor<10x20xf32>
}

// CHECK-LABEL: func @zero_input
func.func @zero_input() -> !stablehlo.token {
  // CHECK:      %0 = stablehlo.replica_id : tensor<ui32>
  // CHECK-NEXT: %1 = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %2 = stablehlo.create_token : !stablehlo.token
  %0 = "stablehlo.replica_id"() : () -> tensor<ui32>
  %1 = "stablehlo.partition_id"() : () -> tensor<ui32>
  %2 = "stablehlo.create_token"() : () -> !stablehlo.token
  return %2 : !stablehlo.token
}

// CHECK-LABEL: func @zero_output_ret2
func.func @zero_output_ret2(%arg0 : tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
  // CHECK: stablehlo.return %arg0, %arg0 : tensor<3xi64>, tensor<3xi64>
  "stablehlo.return"(%arg0, %arg0) : (tensor<3xi64>, tensor<3xi64>) -> ()
}

// CHECK-LABEL: func @zero_output_ret1
func.func @zero_output_ret1(%arg0 : tensor<3xi64>) -> (tensor<3xi64>) {
  // CHECK:     stablehlo.return %arg0 : tensor<3xi64>
  "stablehlo.return"(%arg0) : (tensor<3xi64>) -> ()
}

// CHECK-LABEL: func @zero_output_ret0
func.func @zero_output_ret0(%arg0 : tensor<3xi64>) -> () {
  // CHECK:     stablehlo.return
  "stablehlo.return"() : () -> ()
}

func.func @constants() -> () {
  // CHECK:      %c = stablehlo.constant dense<-1> : tensor<1xi64>
  // CHECK-NEXT: %c_0 = stablehlo.constant {attr = 1 : i32} dense<[-2, 4]> : tensor<2xi64>
  // CHECK-NEXT: %cst = stablehlo.constant() <{value = dense<[1, 2]> : tensor<2xi8>}> : () -> tensor<2x!quant.uniform<i8:f32, 2.000000e+00:15>>
  // CHECK-NEXT: %cst_1 = stablehlo.constant() <{value = dense<3> : tensor<1xi8>}> : () -> tensor<1x!quant.uniform<i8:f32, 2.000000e+00:15>>
  // CHECK-NEXT: %cst_2 = stablehlo.constant() <{value = dense<4> : tensor<1xi8>}> {attr = 1 : i32} : () -> tensor<1x!quant.uniform<i8:f32, 2.000000e+00:15>>
  %cst = "stablehlo.constant"() <{value = dense<[-1]> : tensor<1xi64>}> : () -> tensor<1xi64>
  %cst_attrs = "stablehlo.constant"() <{value = dense<[-2, 4]> : tensor<2xi64>}> {attr = 1 : i32} : () -> tensor<2xi64>
  %cst_q = "stablehlo.constant"() {value = dense<[1, 2]> : tensor<2xi8>} : () -> tensor<2x!quant.uniform<i8:f32, 2.000000e+00:15>>
  %cst_q_attr = stablehlo.constant() {value = dense<[3]> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 2.000000e+00:15>>
  %cst_q_attrs = stablehlo.constant() {value = dense<[4]> : tensor<1xi8>, attr = 1 : i32} : () -> tensor<1x!quant.uniform<i8:f32, 2.000000e+00:15>>
  return
}

// CHECK-LABEL: func @unary_ops
func.func @unary_ops(%arg0 : tensor<2xi32>, %arg1 : tensor<2xf32>) -> () {
  // CHECK:      %0 = stablehlo.abs %arg0 : tensor<2xi32>
  // CHECK-NEXT: %1 = stablehlo.ceil %arg1 : tensor<2xf32>
  // CHECK-NEXT: %2 = stablehlo.count_leading_zeros %arg0 : tensor<2xi32>
  // CHECK-NEXT: %3 = stablehlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
  // CHECK-NEXT: %4 = stablehlo.cosine %arg1 : tensor<2xf32>
  // CHECK-NEXT: %5 = stablehlo.exponential %arg1 : tensor<2xf32>
  // CHECK-NEXT: %6 = stablehlo.exponential_minus_one %arg1 : tensor<2xf32>
  // CHECK-NEXT: %7 = stablehlo.floor %arg1 : tensor<2xf32>
  // CHECK-NEXT: %8 = stablehlo.imag %arg1 : tensor<2xf32>
  // CHECK-NEXT: %9 = stablehlo.is_finite %arg1 : (tensor<2xf32>) -> tensor<2xi1>
  // CHECK-NEXT: %10 = stablehlo.log %arg1 : tensor<2xf32>
  // CHECK-NEXT: %11 = stablehlo.log_plus_one %arg1 : tensor<2xf32>
  // CHECK-NEXT: %12 = stablehlo.logistic %arg1 : tensor<2xf32>
  // CHECK-NEXT: %13 = stablehlo.not %arg0 : tensor<2xi32>
  // CHECK-NEXT: %14 = stablehlo.negate %arg1 : tensor<2xf32>
  // CHECK-NEXT: %15 = stablehlo.popcnt %arg0 : tensor<2xi32>
  // CHECK-NEXT: %16 = stablehlo.real %arg1 : tensor<2xf32>
  // CHECK-NEXT: %17 = stablehlo.round_nearest_afz %arg1 : tensor<2xf32>
  // CHECK-NEXT: %18 = stablehlo.round_nearest_even %arg1 : tensor<2xf32>
  // CHECK-NEXT: %19 = stablehlo.sign %arg1 : tensor<2xf32>
  // CHECK-NEXT: %20 = stablehlo.sine %arg1 : tensor<2xf32>
  // CHECK-NEXT: %21 = stablehlo.sqrt %arg1 : tensor<2xf32>
  // CHECK-NEXT: %22 = stablehlo.tan %arg1 : tensor<2xf32>
  // CHECK-NEXT: %23 = stablehlo.tanh %arg1 : tensor<2xf32>
  %0 = "stablehlo.abs"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %1 = "stablehlo.ceil"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %2 = "stablehlo.count_leading_zeros"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %3 = "stablehlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  %4 = "stablehlo.cosine"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %5 = "stablehlo.exponential"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %6 = "stablehlo.exponential_minus_one"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %7 = "stablehlo.floor"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %8 = "stablehlo.imag"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %9 = "stablehlo.is_finite"(%arg1) : (tensor<2xf32>) -> tensor<2xi1>
  %10 = "stablehlo.log"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %11 = "stablehlo.log_plus_one"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %12 = "stablehlo.logistic"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %13 = "stablehlo.not"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %14 = "stablehlo.negate"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %15 = "stablehlo.popcnt"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %16 = "stablehlo.real"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %17 = "stablehlo.round_nearest_afz"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %18 = "stablehlo.round_nearest_even"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %19 = "stablehlo.sign"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %20 = "stablehlo.sine"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %21 = "stablehlo.sqrt"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %22 = "stablehlo.tan"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %23 = "stablehlo.tanh"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  "stablehlo.return"(%0) : (tensor<2xi32>) -> ()
}

// CHECK-LABEL: func @binary_ops
func.func @binary_ops(%arg0: tensor<2xi1>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xi32>) -> tensor<2xi1> {
  // CHECK:      %0 = stablehlo.add %arg0, %arg0 : tensor<2xi1>
  // CHECK-NEXT: %1 = stablehlo.and %arg0, %arg0 : tensor<2xi1>
  // CHECK-NEXT: %2 = stablehlo.atan2 %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %3 = stablehlo.divide %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %4 = stablehlo.maximum %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %5 = stablehlo.minimum %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %6 = stablehlo.multiply %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %7 = stablehlo.or %arg0, %arg0 : tensor<2xi1>
  // CHECK-NEXT: %8 = stablehlo.power %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %9 = stablehlo.remainder %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %10 = stablehlo.shift_left %arg2, %arg2 : tensor<2xi32>
  // CHECK-NEXT: %11 = stablehlo.shift_right_arithmetic %arg2, %arg2 : tensor<2xi32>
  // CHECK-NEXT: %12 = stablehlo.shift_right_logical %arg2, %arg2 : tensor<2xi32>
  // CHECK-NEXT: %13 = stablehlo.subtract %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %14 = stablehlo.xor %arg0, %arg0 : tensor<2xi1>
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %1 = "stablehlo.and"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %2 = "stablehlo.atan2"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %3 = "stablehlo.divide"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %4 = "stablehlo.maximum"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %5 = "stablehlo.minimum"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %6 = "stablehlo.multiply"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %7 = "stablehlo.or"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %8 = "stablehlo.power"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %9 = "stablehlo.remainder"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %10 = "stablehlo.shift_left"(%arg2, %arg2) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %11 = "stablehlo.shift_right_arithmetic"(%arg2, %arg2) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %12 = "stablehlo.shift_right_logical"(%arg2, %arg2) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %13 = "stablehlo.subtract"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %14 = "stablehlo.xor"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK-LABEL: func @type_convert_ops
func.func @type_convert_ops(%arg0 : tensor<2xf32>) -> () {
  // CHECK:      %0 = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xf64>
  // CHECK-NEXT: %1 = stablehlo.reshape %arg0 : (tensor<2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: %2 = stablehlo.bitcast_convert %arg0 : (tensor<2xf32>) -> tensor<2xi32>
  %0 = "stablehlo.convert"(%arg0) : (tensor<2xf32>) -> tensor<2xf64>
  %1 = "stablehlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x2xf32>
  %2 = "stablehlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @multiple_attr_ops
func.func @multiple_attr_ops(%arg0 : tensor<3x4xf32>) -> () {
  // CHECK:      %0 = stablehlo.reduce_precision %arg0, format = e8m10 : tensor<3x4xf32>
  // CHECK-NEXT: %1 = stablehlo.custom_call @foo(%arg0, %arg0) {backend_config = "bar", has_side_effect = true} : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<1x2x3xf32>
  // CHECK-NEXT: %2 = stablehlo.custom_call @"foo-not-id"(%arg0, %arg0) {backend_config = "bar", has_side_effect = true} : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<1x2x3xf32>
  %0 = "stablehlo.reduce_precision"(%arg0) {exponent_bits = 8 : i32, mantissa_bits = 10 : i32} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %1 = "stablehlo.custom_call"(%arg0, %arg0) {backend_config = "bar", call_target_name = "foo", has_side_effect = true} : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<1x2x3xf32>
  %2 = "stablehlo.custom_call"(%arg0, %arg0) {backend_config = "bar", call_target_name = "foo-not-id", has_side_effect = true} : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<1x2x3xf32>
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @select_op
func.func @select_op(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>,
                  %arg2: tensor<2x?xi32>, %arg3: tensor<?x2xi32>) -> () {
  // CHECK      %0 = stablehlo.select %arg0, %arg1, %arg1 : tensor<2x3xi1>, tensor<2x3xi32>
  // CHECK-NEXT %1 = stablehlo.select %arg0, %arg2, %arg3 : (tensor<2x3xi1>, tensor<2x?xi32>, tensor<?x2xi32>) -> tensor<2x?xi32>
  %0 = "stablehlo.select"(%arg0, %arg1, %arg1) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  %1 = "stablehlo.select"(%arg0, %arg2, %arg3) : (tensor<2x3xi1>, tensor<2x?xi32>, tensor<?x2xi32>) -> tensor<2x?xi32>
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @single_attr_enums
func.func @single_attr_enums(%arg0: tensor<1x2xf32>,
                             %arg1: tensor<f32>,
                             %arg2: tensor<3xi64>) -> () {
  // CHECK:      %0 = stablehlo.dot %arg2, %arg2, precision = [DEFAULT, DEFAULT] : (tensor<3xi64>, tensor<3xi64>) -> tensor<i64>
  // CHECK-NEXT: %1 = stablehlo.dot %arg2, %arg2, precision = [] : (tensor<3xi64>, tensor<3xi64>) -> tensor<i64>
  // CHECK-NEXT: %2 = stablehlo.dot %arg2, %arg2 : (tensor<3xi64>, tensor<3xi64>) -> tensor<i64>
  // CHECK-NEXT: %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = PHILOX : (tensor<1x2xf32>) -> (tensor<1x2xf32>, tensor<2x2xui32>)
  // CHECK-NEXT: %3 = stablehlo.rng %arg1, %arg1, %arg2, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  %0 = "stablehlo.dot"(%arg2, %arg2) {precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<3xi64>, tensor<3xi64>) -> tensor<i64>
  %1 = "stablehlo.dot"(%arg2, %arg2) {precision_config = []} : (tensor<3xi64>, tensor<3xi64>) -> tensor<i64>
  %2 = "stablehlo.dot"(%arg2, %arg2) : (tensor<3xi64>, tensor<3xi64>) -> tensor<i64>
  %3, %4 = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm PHILOX>} : (tensor<1x2xf32>) -> (tensor<1x2xf32>, tensor<2x2xui32>)
  %5 = "stablehlo.rng"(%arg1, %arg1, %arg2) {rng_distribution = #stablehlo<rng_distribution NORMAL>} : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @single_attr_scalar_ops
func.func @single_attr_scalar_ops(%arg0 : tensor<2x2xf32>,
                                  %arg1 : tensor<4x1xf32>,
                                  %arg2 : tensor<4x2xf32>,
                                  %arg3 : tensor<1xindex>,
                                  %arg4 : tensor<i32>) -> () {
  // CHECK:      %0 = stablehlo.cholesky %arg0 : tensor<2x2xf32>
  // CHECK-NEXT: %1 = stablehlo.cholesky %arg0, lower = true : tensor<2x2xf32>
  // CHECK-NEXT: %2 = stablehlo.concatenate %arg1, %arg2, dim = 1 : (tensor<4x1xf32>, tensor<4x2xf32>) -> tensor<4x3xf32>
  // CHECK-NEXT: %3 = stablehlo.dynamic_iota %arg3, dim = 0 : (tensor<1xindex>) -> tensor<4xi32>
  // CHECK-NEXT: %4 = stablehlo.iota dim = 1 : tensor<1x10xf32>
  // CHECK-NEXT: %5 = stablehlo.get_dimension_size %arg2, dim = 1 : (tensor<4x2xf32>) -> tensor<i32>
  // CHECK-NEXT: %6 = stablehlo.set_dimension_size %arg2, %arg4, dim = 1 : (tensor<4x2xf32>, tensor<i32>) -> tensor<4x2xf32>
  %0 = "stablehlo.cholesky"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = "stablehlo.cholesky"(%arg0) { lower = true } : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = "stablehlo.concatenate"(%arg1, %arg2) {dimension = 1 : i64} : (tensor<4x1xf32>, tensor<4x2xf32>) -> tensor<4x3xf32>
  %3 = "stablehlo.dynamic_iota"(%arg3) {iota_dimension = 0 : i64} : (tensor<1xindex>) -> tensor<4xi32>
  %4 = "stablehlo.iota"() {iota_dimension = 1 : i64}  : () -> tensor<1x10xf32>
  %5 = "stablehlo.get_dimension_size"(%arg2) {dimension = 1 : i64} : (tensor<4x2xf32>) -> tensor<i32>
  %6 = "stablehlo.set_dimension_size"(%arg2, %arg4) {dimension = 1 : i64} : (tensor<4x2xf32>, tensor<i32>) -> tensor<4x2xf32>
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @tuple_ops
func.func @tuple_ops(%arg0 : tensor<i32>) -> () {
  // CHECK:      %0 = stablehlo.tuple %arg0, %arg0 : tuple<tensor<i32>, tensor<i32>>
  // CHECK-NEXT: %1 = stablehlo.tuple %arg0 : tuple<tensor<i32>>
  // CHECK-NEXT: %2 = stablehlo.tuple : tuple<>
  // CHECK-NEXT: %3 = stablehlo.get_tuple_element %1[0] : (tuple<tensor<i32>>) -> tensor<i32>
  %0 = "stablehlo.tuple"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>>
  %1 = "stablehlo.tuple"(%arg0) : (tensor<i32>) -> tuple<tensor<i32>>
  %2 = "stablehlo.tuple"() : () -> tuple<>
  %3 = "stablehlo.get_tuple_element"(%1) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @pairwise_ops
func.func @pairwise_ops(%arg0 : tensor<4xf32>) -> () {
  // CHECK:      stablehlo.optimization_barrier()
  // CHECK-NEXT: %0 = stablehlo.optimization_barrier %arg0 : tensor<4xf32>
  // CHECK-NEXT: %1:2 = stablehlo.optimization_barrier %arg0, %arg0 : tensor<4xf32>, tensor<4xf32>
  "stablehlo.optimization_barrier"() : () -> ()
  %0 = "stablehlo.optimization_barrier"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1:2 = "stablehlo.optimization_barrier"(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @compare_op
func.func @compare_op(%arg0 : tensor<3xi32>) -> () {
  // CHECK:      %0 = stablehlo.compare LT, %arg0, %arg0 : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  // CHECK-NEXT: %1 = stablehlo.compare LT, %arg0, %arg0, TOTALORDER : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
   %0 = "stablehlo.compare"(%arg0, %arg0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
   %1 = "stablehlo.compare"(%arg0, %arg0) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @dimension_attr
func.func @dimension_attr(%arg0 : tensor<1x2xf32>, %arg1 : tensor<3xi32>, %arg2 : tensor<3x4xi32>, %arg3 : tensor<i64>, %arg4 : tensor<8xf32>, %arg5 : tensor<f32>) -> () {
  // CHECK:      %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x3xf32>
  // CHECK-NEXT: %1 = stablehlo.broadcast %arg1, sizes = [1, 2] : (tensor<3xi32>) -> tensor<1x2x3xi32>
  // CHECK-NEXT: %2 = stablehlo.reverse %arg0, dims = [0, 1] : tensor<1x2xf32>
  // CHECK-NEXT: %3 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<1x2xf32>) -> tensor<2x1xf32>
  // CHECK-NEXT: %4 = stablehlo.dynamic_slice %arg2, %arg3, %arg3, sizes = [1, 4] : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  // CHECK-NEXT: %5 = stablehlo.pad %arg4, %arg5, low = [4], high = [4], interior = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<16xf32>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 0, 1>} : (tensor<1x2xf32>) -> tensor<1x2x3xf32>
  %1 = "stablehlo.broadcast"(%arg1) {broadcast_sizes = array<i64: 1, 2>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  %2 = "stablehlo.reverse"(%arg0) {dimensions = array<i64: 0, 1>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %3 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0>} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %4 = "stablehlo.dynamic_slice"(%arg2, %arg3, %arg3) {slice_sizes = array<i64: 1, 4>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  %5 = "stablehlo.pad"(%arg4, %arg5) { edge_padding_high = array<i64: 4>, edge_padding_low = array<i64: 4>, interior_padding = array<i64: 0>} : (tensor<8xf32>, tensor<f32>) -> tensor<16xf32>
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @op_einsum
func.func @op_einsum(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8xf32> {
  // CHECK:      %0 = stablehlo.einsum %arg0, %arg1, config = "ab,bc->ac" : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.unary_einsum %arg0, config = "ab->a" : (tensor<8x16xf32>) -> tensor<8xf32>
  %0 = "stablehlo.einsum"(%arg0, %arg1) { einsum_config = "ab,bc->ac" } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  %1 = "stablehlo.unary_einsum"(%arg0) { einsum_config = "ab->a" } : (tensor<8x16xf32>) -> tensor<8xf32>
  func.return %1 : tensor<8xf32>
}

// CHECK-LABEL: func @fft_op
func.func @fft_op(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  // CHECK: %0 = stablehlo.fft %arg0, type = FFT, length = [16] : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  %0 = "stablehlo.fft"(%arg0) {fft_type = #stablehlo<fft_type FFT>, fft_length = array<i64: 16>} : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}

// CHECK-LABEL: func @extensions
func.func @extensions(%arg0 : tensor<?x?xf32, #stablehlo.bounds<3, ?>>,
                      %arg1 : tensor<i32>,
                      %arg2 : tensor<f32, #stablehlo.bounds<>>) -> () {
  // CHECK:      %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 1 : (tensor<?x?xf32, #stablehlo.bounds<3, ?>>, tensor<i32>) -> tensor<?x?xf32>
  // CHECK-NEXT: %1 = stablehlo.set_dimension_size %arg0, %arg1, dim = 1 : (tensor<?x?xf32, #stablehlo.bounds<3, ?>>, tensor<i32>) -> tensor<?x?xf32>
  %0 = "stablehlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x?xf32, #stablehlo.bounds<3, ?>>, tensor<i32>) -> tensor<?x?xf32>
  %1 = "stablehlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, ?]>>, tensor<i32>) -> tensor<?x?xf32>
  "stablehlo.return"() : () -> ()
}

func.func @dot_general(%arg0: tensor<2x2x2xi8>, %arg1: tensor<2x2x3xi8>, %arg2: tensor<2x2xi8>, %arg3: tensor<2x3xi8>) -> tensor<2x2x3xi32> {
  //      CHECK: {{%.*}} = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x2x2xi8>, tensor<2x2x3xi8>) -> tensor<2x2x3xi32>
  // CHECK-NEXT: {{%.*}} = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x2x2xi8>, tensor<2x2x3xi8>) -> tensor<2x2x3xi32>
  // CHECK-NEXT: {{%.*}} = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] : (tensor<2x2xi8>, tensor<2x3xi8>) -> tensor<2x3xi32>
  // CHECK-NEXT: {{%.*}} = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xi8>, tensor<2x3xi8>) -> tensor<2x3xi32>
  // CHECK-NEXT: {{%.*}} = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<2x2xi8>, tensor<2x3xi8>) -> tensor<2x3xi32>
  // CHECK-NEXT: {{%.*}} = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<2x2xi8>, tensor<2x3xi8>) -> tensor<2x3xi32>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x2x2xi8>, tensor<2x2x3xi8>) -> tensor<2x2x3xi32>
  %1 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x2x2xi8>, tensor<2x2x3xi8>) -> tensor<2x2x3xi32>
  %2 = "stablehlo.dot_general"(%arg2, %arg3) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<2x2xi8>, tensor<2x3xi8>) -> tensor<2x3xi32>
  %3 = "stablehlo.dot_general"(%arg2, %arg3) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x2xi8>, tensor<2x3xi8>) -> tensor<2x3xi32>
  %4 = "stablehlo.dot_general"(%arg2, %arg3) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = tf32,
      rhs_precision_type = tf32,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  } : (tensor<2x2xi8>, tensor<2x3xi8>) -> tensor<2x3xi32>
  %5 = "stablehlo.dot_general"(%arg2, %arg3) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [0]
    >,
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = tf32,
      rhs_precision_type = tf32,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  } : (tensor<2x2xi8>, tensor<2x3xi8>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x2x3xi32>
}


func.func @slice(%arg0: tensor<3x8xf32>, %arg1: tensor<8xf32>)
  -> (tensor<1xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>, tensor<2x2xf32>, tensor<2x2xf32>) {
  // CHECK: stablehlo.slice %arg1 [0:1:2] : (tensor<8xf32>) -> tensor<1xf32>
  %slice1 = stablehlo.slice %arg1 [0:1:2] : (tensor<8xf32>) -> tensor<1xf32>
  // CHECK: stablehlo.slice %arg1 [0:4:2] : (tensor<8xf32>) -> tensor<2xf32>
  %slice2 = stablehlo.slice %arg1 [0:4:2] : (tensor<8xf32>) -> tensor<2xf32>
  // CHECK: stablehlo.slice %arg1 [0:1] : (tensor<8xf32>) -> tensor<1xf32>
  %slice3 = stablehlo.slice %arg1 [0:1:1] : (tensor<8xf32>) -> tensor<1xf32>
  // CHECK: stablehlo.slice %arg1 [0:1] : (tensor<8xf32>) -> tensor<1xf32>
  %slice4 = stablehlo.slice %arg1 [0:1] : (tensor<8xf32>) -> tensor<1xf32>
  // CHECK: stablehlo.slice %arg0 [1:3, 4:8:2] : (tensor<3x8xf32>) -> tensor<2x2xf32>
  %slice5 = stablehlo.slice %arg0 [1:3, 4:8:2] : (tensor<3x8xf32>) -> tensor<2x2xf32>
  // CHECK: stablehlo.slice %arg0 [1:3, 4:8:2] : (tensor<3x8xf32>) -> tensor<2x2xf32>
  %slice6 = stablehlo.slice %arg0 [1:3:1, 4:8:2] : (tensor<3x8xf32>) -> tensor<2x2xf32>
  return %slice1, %slice2, %slice3, %slice4, %slice5, %slice6 : tensor<1xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>, tensor<2x2xf32>, tensor<2x2xf32>
}

func.func @result_accuracy_default() -> () attributes {
  // CHECK: mode.default = #stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<DEFAULT>>
  // CHECK: mode.highest = #stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<HIGHEST>>
  // CHECK: mode.tolerance_full = #stablehlo.result_accuracy<atol = 1.000000e-05, rtol = 1.000000e-04, ulps = 2, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: mode.tolerance_partial = #stablehlo.result_accuracy<atol = 1.000000e-05, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  mode.default = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>,
  mode.highest = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<HIGHEST>>,
  mode.tolerance_full = #stablehlo.result_accuracy<atol = 1.0e-5, rtol = 1.0e-4, ulps = 2, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>,
  mode.tolerance_partial = #stablehlo.result_accuracy<atol = 1.0e-5, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
} {
  func.return
}
