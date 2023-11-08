// RUN: experimental-stablehlo-opt --experimental-stablehlo-canonicalize-dynamism --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @dynamic_reduce_window_success_static_result_type
func.func @dynamic_reduce_window_success_static_result_type(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>) -> tensor<2x2xf32> {
  //           CHECK-NOT: stablehlo.dynamic_reduce_window
  //               CHECK: "stablehlo.reduce_window"(%arg0, %arg1) ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG2:arg.*]]: tensor<f32>, %[[ARG3:arg.*]]: tensor<f32>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = stablehlo.add %arg2, %arg3 : tensor<f32>
  //          CHECK-NEXT:     stablehlo.return %[[VAL1]] : tensor<f32>
  //          CHECK-NEXT: }) {
  //          CHECK-SAME:   base_dilations = dense<[2, 1]> : tensor<2xi64>,
  // CHECK-SAME{LITERAL}:   padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>,
  //          CHECK-SAME:   window_dilations = dense<[3, 1]> : tensor<2xi64>,
  //          CHECK-SAME:   window_dimensions = dense<[2, 1]> : tensor<2xi64>,
  //          CHECK-SAME:   window_strides = dense<[4, 1]> : tensor<2xi64>
  //          CHECK-SAME: } : (tensor<3x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  %0 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[4, 1]> : tensor<2xi64>
  %2 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %3 = stablehlo.constant dense<[3, 1]> : tensor<2xi64>
  %4 = stablehlo.constant dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
  %5 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg0, %arg1, %0, %1, %2, %3, %4) {
    called_computations = [@dynamic_reduce_window0]
  } : (tensor<3x2xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<2x2xf32>
  func.return %5 : tensor<2x2xf32>
}

func.func private @dynamic_reduce_window0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @dynamic_reduce_window_success_dynamic_result_type
func.func @dynamic_reduce_window_success_dynamic_result_type(%arg0: tensor<?x2xf32>, %arg1: tensor<f32>) -> tensor<?x2xf32> {
  //           CHECK-NOT: stablehlo.dynamic_reduce_window
  //               CHECK: "stablehlo.reduce_window"(%arg0, %arg1) ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG2:arg.*]]: tensor<f32>, %[[ARG3:arg.*]]: tensor<f32>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = stablehlo.add %arg2, %arg3 : tensor<f32>
  //          CHECK-NEXT:     stablehlo.return %[[VAL1]] : tensor<f32>
  //          CHECK-NEXT: }) {
  //          CHECK-SAME:   base_dilations = dense<[2, 1]> : tensor<2xi64>,
  // CHECK-SAME{LITERAL}:   padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>,
  //          CHECK-SAME:   window_dilations = dense<[3, 1]> : tensor<2xi64>,
  //          CHECK-SAME:   window_dimensions = dense<[2, 1]> : tensor<2xi64>,
  //          CHECK-SAME:   window_strides = dense<[4, 1]> : tensor<2xi64>
  //          CHECK-SAME: } : (tensor<?x2xf32>, tensor<f32>) -> tensor<?x2xf32>
  %0 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[4, 1]> : tensor<2xi64>
  %2 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %3 = stablehlo.constant dense<[3, 1]> : tensor<2xi64>
  %4 = stablehlo.constant dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
  %5 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg0, %arg1, %0, %1, %2, %3, %4) {
    called_computations = [@dynamic_reduce_window0]
  } : (tensor<?x2xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<?x2xf32>
  func.return %5 : tensor<?x2xf32>
}

func.func private @dynamic_reduce_window0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
  func.return %0 : tensor<f32>
}

// TODO(burmako): Implement tests for verification failures for dynamic_reduce_window.

// -----

// CHECK-LABEL: func @dynamic_reduce_window_inapplicable_dynamic_window_dimensions
func.func @dynamic_reduce_window_inapplicable_dynamic_window_dimensions(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>, %arg2: tensor<2xi64>) -> tensor<2x2xf32> {
  // CHECK: stablehlo.dynamic_reduce_window
  %0 = stablehlo.constant dense<[4, 1]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %2 = stablehlo.constant dense<[3, 1]> : tensor<2xi64>
  %3 = stablehlo.constant dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
  %4 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg0, %arg1, %arg2, %0, %1, %2, %3) {
    called_computations = [@dynamic_reduce_window0]
  } : (tensor<3x2xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<2x2xf32>
  func.return %4 : tensor<2x2xf32>
}

func.func private @dynamic_reduce_window0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @dynamic_reduce_window_inapplicable_dynamic_window_strides
func.func @dynamic_reduce_window_inapplicable_dynamic_window_strides(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>, %arg2: tensor<2xi64>) -> tensor<2x2xf32> {
  // CHECK: stablehlo.dynamic_reduce_window
  %0 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %2 = stablehlo.constant dense<[3, 1]> : tensor<2xi64>
  %3 = stablehlo.constant dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
  %4 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg0, %arg1, %0, %arg2, %1, %2, %3) {
    called_computations = [@dynamic_reduce_window0]
  } : (tensor<3x2xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<2x2xf32>
  func.return %4 : tensor<2x2xf32>
}

func.func private @dynamic_reduce_window0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @dynamic_reduce_window_inapplicable_dynamic_base_dilations
func.func @dynamic_reduce_window_inapplicable_dynamic_base_dilations(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>, %arg2: tensor<2xi64>) -> tensor<2x2xf32> {
  // CHECK: stablehlo.dynamic_reduce_window
  %0 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[4, 1]> : tensor<2xi64>
  %2 = stablehlo.constant dense<[3, 1]> : tensor<2xi64>
  %3 = stablehlo.constant dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
  %4 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg0, %arg1, %0, %1, %arg2, %2, %3) {
    called_computations = [@dynamic_reduce_window0]
  } : (tensor<3x2xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<2x2xf32>
  func.return %4 : tensor<2x2xf32>
}

func.func private @dynamic_reduce_window0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @dynamic_reduce_window_inapplicable_dynamic_window_dilations
func.func @dynamic_reduce_window_inapplicable_dynamic_window_dilations(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>, %arg2: tensor<2xi64>) -> tensor<2x2xf32> {
  // CHECK: stablehlo.dynamic_reduce_window
  %0 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[4, 1]> : tensor<2xi64>
  %2 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %3 = stablehlo.constant dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
  %4 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg0, %arg1, %0, %1, %2, %arg2, %3) {
    called_computations = [@dynamic_reduce_window0]
  } : (tensor<3x2xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<2x2xf32>
  func.return %4 : tensor<2x2xf32>
}

func.func private @dynamic_reduce_window0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @dynamic_reduce_window_inapplicable_dynamic_padding
func.func @dynamic_reduce_window_inapplicable_dynamic_padding(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>, %arg2: tensor<2x2xi64>) -> tensor<2x2xf32> {
  // CHECK: stablehlo.dynamic_reduce_window
  %0 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[4, 1]> : tensor<2xi64>
  %2 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %3 = stablehlo.constant dense<[3, 1]> : tensor<2xi64>
  %4 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg0, %arg1, %0, %1, %2, %3, %arg2) {
    called_computations = [@dynamic_reduce_window0]
  } : (tensor<3x2xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<2x2xf32>
  func.return %4 : tensor<2x2xf32>
}

func.func private @dynamic_reduce_window0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @dynamic_rng_bit_generator_success
func.func @dynamic_rng_bit_generator_success(%arg0: tensor<2xui64>) -> tensor<1x4xf32> {
  // CHECK-NOT: stablehlo.dynamic_rng_bit_generator
  // CHECK: stablehlo.rng_bit_generator %arg0, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<1x4xf32>)
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_rng_bit_generator(%arg0, %0) {
    rng_algorithm = #stablehlo<rng_algorithm DEFAULT>
  } : (tensor<2xui64>, tensor<2xi64>) -> (tensor<2xui64>, tensor<1x4xf32>)
  return %1#1 : tensor<1x4xf32>
}

// TODO(burmako): Implement tests for verification failures for dynamic_rng_bit_generator.

// -----

// CHECK-LABEL: func @dynamic_rng_bit_generator_inapplicable_dynamic_output_shape
func.func @dynamic_rng_bit_generator_inapplicable_dynamic_output_shape(%arg0: tensor<2xui64>, %arg1: tensor<2xi64>) -> tensor<1x4xf32> {
  // CHECK: stablehlo.dynamic_rng_bit_generator
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_rng_bit_generator(%arg0, %arg1) {
    rng_algorithm = #stablehlo<rng_algorithm DEFAULT>
  } : (tensor<2xui64>, tensor<2xi64>) -> (tensor<2xui64>, tensor<1x4xf32>)
  return %1#1 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_rng_bit_generator_inapplicable_dynamic_output_type
func.func @dynamic_rng_bit_generator_inapplicable_dynamic_output_type(%arg0: tensor<2xui64>) -> tensor<?x?xf32> {
  // CHECK: stablehlo.dynamic_rng_bit_generator
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_rng_bit_generator(%arg0, %0) {
    rng_algorithm = #stablehlo<rng_algorithm DEFAULT>
  } : (tensor<2xui64>, tensor<2xi64>) -> (tensor<2xui64>, tensor<?x?xf32>)
  return %1#1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_top_k_success
func.func @dynamic_top_k_success(%arg0: tensor<16xf32>) -> (tensor<3xf32>, tensor<3xi32>) {
  // CHECK: chlo.top_k
  %k = stablehlo.constant dense<3> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<ui64>) -> (tensor<3xf32>, tensor<3xi32>)
  return %1#0, %1#1 : tensor<3xf32>, tensor<3xi32>
}

// -----

// CHECK-LABEL: func @dynamic_top_k_failure_k_mismatch
func.func @dynamic_top_k_failure_k_mismatch(%arg0: tensor<16xf32>) -> (tensor<3xf32>, tensor<3xi32>) {
  // CHECK: @stablehlo.dynamic_top_k
  %k = stablehlo.constant dense<4> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<ui64>) -> (tensor<3xf32>, tensor<3xi32>)
  return %1#0, %1#1 : tensor<3xf32>, tensor<3xi32>
}

// -----

// dynamic_top_k I1
// CHECK-LABEL: func @dynamic_top_k_error_operand_not_float
func.func @dynamic_top_k_error_operand_not_float(%arg0: tensor<16xcomplex<f64>>) -> (tensor<3xcomplex<f64>>, tensor<3xi32>) {
  // expected-error@+2{{expects operand #0 to be a tensor of integer or floating-point type}}
  %k = stablehlo.constant dense<3> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xcomplex<f64>>, tensor<ui64>) -> (tensor<3xcomplex<f64>>, tensor<3xi32>)
  return %1#0, %1#1 : tensor<3xcomplex<f64>>, tensor<3xi32>
}

// -----

// dynamic_top_k I1
// CHECK-LABEL: func @dynamic_top_k_error_operand_unranked
func.func @dynamic_top_k_error_operand_unranked(%arg0: tensor<*xf32>) -> (tensor<3xf32>, tensor<3xi32>) {
  // expected-error@+2{{expects operand #0 to be a tensor of integer or floating-point type of rank at least 1}}
  %k = stablehlo.constant dense<3> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<*xf32>, tensor<ui64>) -> (tensor<3xf32>, tensor<3xi32>)
  return %1#0, %1#1 : tensor<3xf32>, tensor<3xi32>
}

// -----

// dynamic_top_k I1
// CHECK-LABEL: func @dynamic_top_k_error_scalar_operand
func.func @dynamic_top_k_error_scalar_operand(%arg0: tensor<f32>) -> (tensor<3xf32>, tensor<3xi32>) {
  // expected-error@+2{{expects operand #0 to be a tensor of integer or floating-point type of rank at least 1}}
  %k = stablehlo.constant dense<3> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<f32>, tensor<ui64>) -> (tensor<3xf32>, tensor<3xi32>)
  return %1#0, %1#1 : tensor<3xf32>, tensor<3xi32>
}

// -----

// dynamic_top_k I2
// CHECK-LABEL: func @dynamic_top_k_error_k_not_integer
func.func @dynamic_top_k_error_k_not_integer(%arg0: tensor<16xf32>) -> (tensor<3xf32>, tensor<3xi32>) {
  // expected-error@+2{{expects k (operand #1) to be a 0-dimensional tensor of integer or index type}}
  %k = stablehlo.constant dense<3.> : tensor<f32>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<f32>) -> (tensor<3xf32>, tensor<3xi32>)
  return %1#0, %1#1 : tensor<3xf32>, tensor<3xi32>
}

// -----

// dynamic_top_k I2
// CHECK-LABEL: func @dynamic_top_k_error_k_not_scalar
func.func @dynamic_top_k_error_k_not_scalar(%arg0: tensor<16xf32>) -> (tensor<3xf32>, tensor<3xi32>) {
  // expected-error@+2{{expects k (operand #1) to be a 0-dimensional tensor of integer or index type}}
  %k = stablehlo.constant dense<3> : tensor<1xui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<1xui64>) -> (tensor<3xf32>, tensor<3xi32>)
  return %1#0, %1#1 : tensor<3xf32>, tensor<3xi32>
}

// -----

// dynamic_top_k O1
// CHECK-LABEL: func @dynamic_top_k_error_values_not_float
func.func @dynamic_top_k_error_values_not_float(%arg0: tensor<16xf32>) -> (tensor<3xcomplex<f64>>, tensor<3xi32>) {
  // expected-error@+2{{expects values (result #0) to be a tensor of integer or floating-point type}}
  %k = stablehlo.constant dense<3> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<ui64>) -> (tensor<3xcomplex<f64>>, tensor<3xi32>)
  return %1#0, %1#1 : tensor<3xcomplex<f64>>, tensor<3xi32>
}

// -----

// dynamic_top_k O2
// CHECK-LABEL: func @dynamic_top_k_error_indices_not_i32
func.func @dynamic_top_k_error_indices_not_i32(%arg0: tensor<16xf32>) -> (tensor<3xf32>, tensor<3xi64>) {
  // expected-error@+2{{expects indices (result #1) to be a tensor of si32}}
  %k = stablehlo.constant dense<3> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<ui64>) -> (tensor<3xf32>, tensor<3xi64>)
  return %1#0, %1#1 : tensor<3xf32>, tensor<3xi64>
}

// -----

// dynamic_top_k C1
// CHECK-LABEL: func @dynamic_top_k_error_values_bad_rank
func.func @dynamic_top_k_error_values_bad_rank(%arg0: tensor<16xf32>) -> (tensor<3x4xf32>, tensor<3xi32>) {
  // expected-error@+2{{expects the values shape to match the operand shape in all but the last dimension}}
  %k = stablehlo.constant dense<3> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<ui64>) -> (tensor<3x4xf32>, tensor<3xi32>)
  return %1#0, %1#1 : tensor<3x4xf32>, tensor<3xi32>
}

// -----

// dynamic_top_k C2
// CHECK-LABEL: func @dynamic_top_k_error_values_bad_element_type
func.func @dynamic_top_k_error_values_bad_element_type(%arg0: tensor<16xf32>) -> (tensor<3xf64>, tensor<3xi32>) {
  // expected-error@+2{{expects the values element type to be the same as the operand element type}}
  %k = stablehlo.constant dense<3> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<ui64>) -> (tensor<3xf64>, tensor<3xi32>)
  return %1#0, %1#1 : tensor<3xf64>, tensor<3xi32>
}

// -----

// dynamic_top_k C3
// CHECK-LABEL: func @dynamic_top_k_error_values_last_dim_too_large
func.func @dynamic_top_k_error_values_last_dim_too_large(%arg0: tensor<16xf32>) -> (tensor<17xf32>, tensor<3xi32>) {
  // expected-error@+2{{expects the values last dimension to have size at least as large as operand last dimension}}
  %k = stablehlo.constant dense<17> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<ui64>) -> (tensor<17xf32>, tensor<3xi32>)
  return %1#0, %1#1 : tensor<17xf32>, tensor<3xi32>
}

// -----

// dynamic_top_k C4
// CHECK-LABEL: func @dynamic_top_k_error_indices_shape_mismatch
func.func @dynamic_top_k_error_indices_shape_mismatch(%arg0: tensor<16xf32>) -> (tensor<3xf32>, tensor<4xi32>) {
  // expected-error@+2{{expects the indices shape to match the values shape}}
  %k = stablehlo.constant dense<3> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<ui64>) -> (tensor<3xf32>, tensor<4xi32>)
  return %1#0, %1#1 : tensor<3xf32>, tensor<4xi32>
}
