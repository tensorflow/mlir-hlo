// RUN: experimental-stablehlo-opt --experimental-stablehlo-refine-shapes --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @main
func.func @main(%arg0: tensor<3x2xf32>, %arg1: tensor<f32>) -> tensor<*xf32> {
  // CHECK: stablehlo.dynamic_reduce_window{{.*}} -> tensor<2x2xf32>
  %0 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[4, 1]> : tensor<2xi64>
  %2 = stablehlo.constant dense<[2, 1]> : tensor<2xi64>
  %3 = stablehlo.constant dense<[3, 1]> : tensor<2xi64>
  %4 = stablehlo.constant dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
  %5 = stablehlo.custom_call @stablehlo.dynamic_reduce_window(%arg0, %arg1, %0, %1, %2, %3, %4) {
    called_computations = [@dynamic_reduce_window0]
  } : (tensor<3x2xf32>, tensor<f32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<*xf32>
  func.return %5 : tensor<*xf32>
}

func.func private @dynamic_reduce_window0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @refine_dynamic_rng_bit_generator
func.func @refine_dynamic_rng_bit_generator(%arg0: tensor<2xui64>) -> (tensor<?xui64>, tensor<*xf32>) {
  // CHECK: stablehlo.dynamic_rng_bit_generator{{.*}} -> (tensor<2xui64>, tensor<1x4xf32>)
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_rng_bit_generator(%arg0, %0) {
    rng_algorithm = #stablehlo<rng_algorithm DEFAULT>
  } : (tensor<2xui64>, tensor<2xi64>) -> (tensor<?xui64>, tensor<*xf32>)
  func.return %1#0, %1#1 : tensor<?xui64>, tensor<*xf32>
}

// -----

// CHECK-LABEL: func @refine_dynamic_top_k
func.func @refine_dynamic_top_k(%arg0: tensor<16xf32>) -> (tensor<?xf32>, tensor<?xi32>) {
  // CHECK: stablehlo.dynamic_top_k{{.*}} -> (tensor<4xf32>, tensor<4xi32>)
  %k = stablehlo.constant dense<4> : tensor<ui64>
  %1:2 = stablehlo.custom_call @stablehlo.dynamic_top_k(%arg0, %k) : (tensor<16xf32>, tensor<ui64>) -> (tensor<?xf32>, tensor<?xi32>)
  return %1#0, %1#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

// CHECK-LABEL: func @refine_mhlo_topk
func.func @refine_mhlo_topk(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // CHECK: mhlo.topk{{.*}} -> (tensor<5x4xf32>, tensor<5x4xi32>)
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = true}
  } : (tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @refine_mhlo_error_too_many_operands
func.func @refine_mhlo_error_too_many_operands(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // expected-error@+1{{expects size(operands) = 1}}
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0, %arg0) {
    mhlo.attributes = { k = 4 : i64, largest = true}
  } : (tensor<5x16xf32>, tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @refine_mhlo_error_too_few_results
func.func @refine_mhlo_error_too_few_results(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>) {
  // expected-error@+1{{expects size(results) = 2}}
  %0 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = true}
  } : (tensor<5x16xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @refine_mhlo_error_wrong_output_1_type
func.func @refine_mhlo_error_wrong_output_1_type(%arg0: tensor<5x16xf32>) -> (tensor<f32>, tensor<?x?xi32>) {
  // expected-error@+1{{expects values (result #0) to be a tensor of integer or floating-point type of rank at least 1}}
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = true}
  } : (tensor<5x16xf32>) -> (tensor<f32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<f32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @refine_mhlo_error_wrong_output_2_type
func.func @refine_mhlo_error_wrong_output_2_type(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // expected-error@+1{{expects indices (result #1) to be a tensor of si32 of rank at least 1}}
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = true}
  } : (tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @refine_mhlo_error_c1_wrong_output_shape
func.func @refine_mhlo_error_c1_wrong_output_shape(%arg0: tensor<5x16xf32>) -> (tensor<?x?x?xf32>, tensor<?x?xi32>) {
  // expected-error@+1{{expects the values shape to match the operand}}
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = true}
  } : (tensor<5x16xf32>) -> (tensor<?x?x?xf32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x?x?xf32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @refine_mhlo_error_c2_last_dim_not_k
func.func @refine_mhlo_error_c2_last_dim_not_k(%arg0: tensor<5x16xf32>) -> (tensor<?x5xf32>, tensor<?x?xi32>) {
  // expected-error@+1{{expects the values shape to match the operand}}
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = true}
  } : (tensor<5x16xf32>) -> (tensor<?x5xf32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x5xf32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @refine_mhlo_error_c3_wrong_output_type
func.func @refine_mhlo_error_c3_wrong_output_type(%arg0: tensor<5x16xf32>) -> (tensor<?x?xi32>, tensor<?x?xi32>) {
  // expected-error@+1{{expects the values element type to be the same as the operand element type}}
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = true}
  } : (tensor<5x16xf32>) -> (tensor<?x?xi32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x?xi32>, tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @refine_mhlo_error_c4_outputs_shape_mismatch
func.func @refine_mhlo_error_c4_outputs_shape_mismatch(%arg0: tensor<5x16xf32>) -> (tensor<?x4xf32>, tensor<?x5xi32>) {
  // expected-error@+1{{expects the indices shape to match the values shape}}
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = 4 : i64, largest = true}
  } : (tensor<5x16xf32>) -> (tensor<?x4xf32>, tensor<?x5xi32>)
  return %0#0, %0#1 : tensor<?x4xf32>, tensor<?x5xi32>
}

// -----

// CHECK-LABEL: func @refine_mhlo_error_c5_negative_k
func.func @refine_mhlo_error_c5_negative_k(%arg0: tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // expected-error@+1{{expects k >= 0}}
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = { k = -4 : i64, largest = true}
  } : (tensor<5x16xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}
