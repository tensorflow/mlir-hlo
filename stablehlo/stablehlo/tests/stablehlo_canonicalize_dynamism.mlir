// RUN: stablehlo-opt --stablehlo-canonicalize-dynamism --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @custom_call_success_one_output
func.func @custom_call_success_one_output(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // CHECK: stablehlo.custom_call @foo(%arg0) : (tensor<4xf32>) -> tensor<1x2xf32>
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @custom_call_success_multiple_outputs
func.func @custom_call_success_multiple_outputs(%arg0: tensor<4xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>) {
  // CHECK: stablehlo.custom_call @foo(%arg0) : (tensor<4xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %2:2 = stablehlo.custom_call @foo(%arg0, %0, %1) {
    indices_of_shape_operands = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<4xf32>, tensor<2xi64>, tensor<2xi64>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  return %2#0, %2#1 : tensor<1x2xf32>, tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @custom_call_success_mixed_positions
func.func @custom_call_success_mixed_positions(%arg0: tensor<4xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>) {
  // CHECK: stablehlo.custom_call @foo(%arg0) : (tensor<4xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %2:2 = stablehlo.custom_call @foo(%0, %arg0, %1) {
    indices_of_shape_operands = dense<[0, 2]> : tensor<2xi64>
  } : (tensor<2xi64>, tensor<4xf32>, tensor<2xi64>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  return %2#0, %2#1 : tensor<1x2xf32>, tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @custom_call_success_mixed_positions_layouts
func.func @custom_call_success_mixed_positions_layouts(%arg0: tensor<4x3xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>) {
  //      CHECK: stablehlo.custom_call @foo(%arg0) {
  // CHECK-SAME:   operand_layouts = [dense<[1, 0]> : tensor<2xindex>],
  // CHECK-SAME:   result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]
  // CHECK-SAME: } : (tensor<4x3xf32>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %2:2 = stablehlo.custom_call @foo(%0, %arg0, %1) {
    indices_of_shape_operands = dense<[0, 2]> : tensor<2xi64>,
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>],
    result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]
  } : (tensor<2xi64>, tensor<4x3xf32>, tensor<2xi64>) -> (tensor<1x2xf32>, tensor<3x4xf32>)
  return %2#0, %2#1 : tensor<1x2xf32>, tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @custom_call_success_repeating_operands
func.func @custom_call_success_repeating_operands(%arg0: tensor<4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>) {
  // CHECK: stablehlo.custom_call @foo(%arg0) : (tensor<4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1:2 = stablehlo.custom_call @foo(%arg0, %0, %0) {
    indices_of_shape_operands = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<4xf32>, tensor<2xi64>, tensor<2xi64>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
  return %1#0, %1#1 : tensor<1x2xf32>, tensor<1x2xf32>
}

// -----

func.func @custom_call_failure_attr_number_of_elements(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // expected-error@+3{{indices_of_shape_operands: number of elements (2) must be equal to the number of operation results (1)}}
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %2 = stablehlo.custom_call @foo(%arg0, %0, %1) {
    indices_of_shape_operands = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
  return %2 : tensor<1x2xf32>
}

// -----

func.func @custom_call_failure_attr_rank(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // expected-error@+2{{indices_of_shape_operands: must have rank = 1}}
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[[1]]> : tensor<1x1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

func.func @custom_call_failure_attr_element_type(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // expected-error@+2{{indices_of_shape_operands: must have i64 element type}}
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi32>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

func.func @custom_call_failure_out_of_bounds_operand_index(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // expected-error@+2{{indices_of_shape_operands: index #0 (2) must be within bounds for operation operands (from 0 to 2)}}
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[2]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

func.func @custom_call_failure_incompatible_result_type(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // expected-error@+2{{refinement #0 ([1, 1]) must be compatible with operation result #0 ('tensor<1x2xf32>')}}
  %0 = stablehlo.constant dense<[1, 1]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @custom_call_inapplicable_dynamic_shape_operand
func.func @custom_call_inapplicable_dynamic_shape_operand(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<1x?xf32> {
  // CHECK: stablehlo.custom_call @foo(%arg0, %arg1)
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x?xf32>
  return %0 : tensor<1x?xf32>
}

// -----

// CHECK-LABEL: func @custom_call_inapplicable_missing_indices_of_shape_operands
func.func @custom_call_inapplicable_missing_indices_of_shape_operands(%arg0: tensor<4xf32>) -> tensor<1x2xf32> {
  // CHECK: stablehlo.custom_call @foo(%arg0, %0)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @custom_call_inapplicable_dynamic_result_type
func.func @custom_call_inapplicable_dynamic_result_type(%arg0: tensor<4xf32>) -> tensor<1x?xf32> {
  // CHECK: stablehlo.custom_call @foo(%arg0, %0)
  %0 = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
  %1 = stablehlo.custom_call @foo(%arg0, %0) {
    indices_of_shape_operands = dense<[1]> : tensor<1xi64>
  } : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x?xf32>
  return %1 : tensor<1x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_success
func.func @dynamic_broadcast_in_dim_success(%arg0: tensor<4xf32>) -> tensor<3x4xf32> {
  // CHECK-NOT: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<4xf32>) -> tensor<3x4xf32>
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_inapplicable_dynamic_operand_type
func.func @dynamic_broadcast_in_dim_inapplicable_dynamic_operand_type(%arg0: tensor<?xf32>) -> tensor<3x4xf32> {
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_inapplicable_dynamic_output_dimensions
func.func @dynamic_broadcast_in_dim_inapplicable_dynamic_output_dimensions(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<3x4xf32> {
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_inapplicable_dynamic_result_type
func.func @dynamic_broadcast_in_dim_inapplicable_dynamic_result_type(%arg0: tensor<4xf32>) -> tensor<3x?xf32> {
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<3x?xf32>
  return %1 : tensor<3x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_conv_success_static_result_type
func.func @dynamic_conv_success_static_result_type(%arg0: tensor<100x26x26x32xf32>, %arg1: tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  //  CHECK-NOT: stablehlo.dynamic_conv
  //      CHECK: stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME:  dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f],
  // CHECK-SAME:  window = {
  // CHECK-SAME:    stride = [1, 1],
  // CHECK-SAME:    pad = {{\[}}[2, 2], [2, 2]],
  // CHECK-SAME:    lhs_dilate = [1, 1],
  // CHECK-SAME:    rhs_dilate = [1, 1]
  // CHECK-SAME: } {
  // CHECK-SAME:   batch_group_count = 1 : i64,
  // CHECK-SAME:   feature_group_count = 1 : i64
  // CHECK-SAME: } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32>
  %0 = stablehlo.constant dense<2> : tensor<2x2xi32>
  %1 = "stablehlo.dynamic_conv"(%arg0, %arg1, %0) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]>,
    window_strides = dense<1> : tensor<2xi64>,
    lhs_dilation = dense<1> : tensor<2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi32>) -> tensor<100x28x28x1xf32>
  return %1 : tensor<100x28x28x1xf32>
}

// -----

// CHECK-LABEL: func @dynamic_conv_success_dynamic_result_type
func.func @dynamic_conv_success_dynamic_result_type(%arg0: tensor<100x26x26x32xf32>, %arg1: tensor<3x3x1x32xf32>) -> tensor<?x28x28x1xf32> {
  //  CHECK-NOT: stablehlo.dynamic_conv
  //      CHECK: stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME:  dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f],
  // CHECK-SAME:  window = {
  // CHECK-SAME:    stride = [1, 1],
  // CHECK-SAME:    pad = {{\[}}[2, 2], [2, 2]],
  // CHECK-SAME:    lhs_dilate = [1, 1],
  // CHECK-SAME:    rhs_dilate = [1, 1]
  // CHECK-SAME: } {
  // CHECK-SAME:   batch_group_count = 1 : i64,
  // CHECK-SAME:   feature_group_count = 1 : i64
  // CHECK-SAME: } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) -> tensor<?x28x28x1xf32>
  %0 = stablehlo.constant dense<2> : tensor<2x2xi32>
  %1 = "stablehlo.dynamic_conv"(%arg0, %arg1, %0) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]>,
    window_strides = dense<1> : tensor<2xi64>,
    lhs_dilation = dense<1> : tensor<2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi32>) -> tensor<?x28x28x1xf32>
  return %1 : tensor<?x28x28x1xf32>
}

// -----

// CHECK-LABEL: func @dynamic_conv_inapplicable_dynamic_padding
func.func @dynamic_conv_inapplicable_dynamic_padding(%arg0: tensor<100x26x26x32xf32>, %arg1: tensor<3x3x1x32xf32>, %arg2: tensor<2x2xi32>) -> tensor<100x28x28x1xf32> {
  // CHECK: stablehlo.dynamic_conv
  %0 = "stablehlo.dynamic_conv"(%arg0, %arg1, %arg2) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f]>,
    window_strides = dense<1> : tensor<2xi64>,
    lhs_dilation = dense<1> : tensor<2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>, tensor<2x2xi32>) -> tensor<100x28x28x1xf32>
  return %0 : tensor<100x28x28x1xf32>
}

// -----

// CHECK-LABEL: @dynamic_gather_success_static_result_type
func.func @dynamic_gather_success_static_result_type(%arg0 : tensor<2x4x9xi32>, %arg1 : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  //  CHECK-NOT: stablehlo.dynamic_gather
  //      CHECK: "stablehlo.gather"(%arg0, %arg1) {
  // CHECK-SAME:   dimension_numbers = #stablehlo.gather<
  // CHECK-SAME:     offset_dims = [2],
  // CHECK-SAME:     collapsed_slice_dims = [0, 1],
  // CHECK-SAME:     start_index_map = [0, 1],
  // CHECK-SAME:     index_vector_dim = 2
  // CHECK-SAME:   >,
  // CHECK-SAME:   slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  // CHECK-SAME: } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  %0 = stablehlo.constant dense<[1, 1, 8]> : tensor<3xi32>
  %1 = "stablehlo.dynamic_gather"(%arg0, %arg1, %0) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xi32>
  return %1 : tensor<1x5x8xi32>
}

// -----

// CHECK-LABEL: @dynamic_gather_success_dynamic_result_type
func.func @dynamic_gather_success_dynamic_result_type(%arg0 : tensor<2x4x9xi32>, %arg1 : tensor<1x5x2xi32>) -> tensor<1x5x?xi32> {
  //  CHECK-NOT: stablehlo.dynamic_gather
  //      CHECK: "stablehlo.gather"(%arg0, %arg1) {
  // CHECK-SAME:   dimension_numbers = #stablehlo.gather<
  // CHECK-SAME:     offset_dims = [2],
  // CHECK-SAME:     collapsed_slice_dims = [0, 1],
  // CHECK-SAME:     start_index_map = [0, 1],
  // CHECK-SAME:     index_vector_dim = 2
  // CHECK-SAME:   >,
  // CHECK-SAME:   slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  // CHECK-SAME: } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x?xi32>
  %0 = stablehlo.constant dense<[1, 1, 8]> : tensor<3xi32>
  %1 = "stablehlo.dynamic_gather"(%arg0, %arg1, %0) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x?xi32>
  return %1 : tensor<1x5x?xi32>
}

// -----

// CHECK-LABEL: @dynamic_gather_inapplicable_dynamic_slice_sizes
func.func @dynamic_gather_inapplicable_dynamic_slice_sizes(%arg0 : tensor<2x4x9xi32>, %arg1 : tensor<1x5x2xi32>, %arg2 : tensor<3xi32>) -> tensor<1x5x8xi32> {
  // CHECK: stablehlo.dynamic_gather
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %arg2) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xi32>
  return %0 : tensor<1x5x8xi32>
}

// -----

// CHECK-LABEL: func @dynamic_iota_success
func.func @dynamic_iota_success() -> tensor<4xf32> {
  // CHECK-NOT: stablehlo.dynamic_iota
  // CHECK: stablehlo.iota dim = 0 : tensor<4xf32>
  %0 = stablehlo.constant dense<4> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_iota_inapplicable_dynamic_output_shape
func.func @dynamic_iota_inapplicable_dynamic_output_shape(%arg0: tensor<1xi64>) -> tensor<4xf32> {
  // CHECK: stablehlo.dynamic_iota
  %0 = stablehlo.dynamic_iota %arg0, dim = 0 : (tensor<1xi64>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_iota_inapplicable_dynamic_result_type
func.func @dynamic_iota_inapplicable_dynamic_result_type() -> tensor<?xf32> {
  // CHECK: stablehlo.dynamic_iota
  %0 = stablehlo.constant dense<4> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_pad_success_static_result_type
func.func @dynamic_pad_success_static_result_type(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> tensor<11xf32> {
  // CHECK-NOT: stablehlo.dynamic_pad
  // CHECK: stablehlo.pad %arg0, %arg1, low = [0], high = [1], interior = [2] : (tensor<4xf32>, tensor<f32>) -> tensor<11xf32>
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.constant dense<2> : tensor<1xi64>
  %3 = stablehlo.dynamic_pad %arg0, %arg1, %0, %1, %2 : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<11xf32>
  return %3 : tensor<11xf32>
}

// -----

// CHECK-LABEL: func @dynamic_pad_success_dynamic_result_type
func.func @dynamic_pad_success_dynamic_result_type(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> tensor<?xf32> {
  // CHECK-NOT: stablehlo.dynamic_pad
  // CHECK: stablehlo.pad %arg0, %arg1, low = [0], high = [1], interior = [2] : (tensor<4xf32>, tensor<f32>) -> tensor<?xf32>
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.constant dense<2> : tensor<1xi64>
  %3 = stablehlo.dynamic_pad %arg0, %arg1, %0, %1, %2 : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  return %3 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_pad_inapplicable_dynamic_low
func.func @dynamic_pad_inapplicable_dynamic_low(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<1xi64>) -> tensor<11xf32> {
  // CHECK: stablehlo.dynamic_pad
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.constant dense<2> : tensor<1xi64>
  %2 = stablehlo.dynamic_pad %arg0, %arg1, %arg2, %0, %1 : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<11xf32>
  return %2 : tensor<11xf32>
}

// -----

// CHECK-LABEL: func @dynamic_pad_inapplicable_dynamic_high
func.func @dynamic_pad_inapplicable_dynamic_high(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<1xi64>) -> tensor<11xf32> {
  // CHECK: stablehlo.dynamic_pad
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<2> : tensor<1xi64>
  %2 = stablehlo.dynamic_pad %arg0, %arg1, %0, %arg2, %1 : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<11xf32>
  return %2 : tensor<11xf32>
}

// -----

// CHECK-LABEL: func @dynamic_pad_inapplicable_dynamic_interior
func.func @dynamic_pad_inapplicable_dynamic_interior(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<1xi64>) -> tensor<11xf32> {
  // CHECK: stablehlo.dynamic_pad
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.dynamic_pad %arg0, %arg1, %0, %1, %arg2 : (tensor<4xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<11xf32>
  return %2 : tensor<11xf32>
}

// -----

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

// CHECK-LABEL: func @dynamic_reshape_success
func.func @dynamic_reshape_success(%arg0: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK-NOT: stablehlo.dynamic_reshape
  // CHECK: stablehlo.reshape %arg0 : (tensor<4xf32>) -> tensor<1x4xf32>
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_reshape_inapplicable_dynamic_output_shape
func.func @dynamic_reshape_inapplicable_dynamic_output_shape(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<1x4xf32> {
  // CHECK: stablehlo.dynamic_reshape
  %0 = stablehlo.dynamic_reshape %arg0, %arg1 : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: func @dynamic_reshape_inapplicable_dynamic_result_type
func.func @dynamic_reshape_inapplicable_dynamic_result_type(%arg0: tensor<4xf32>) -> tensor<1x?xf32> {
  // CHECK: stablehlo.dynamic_reshape
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x?xf32>
  return %1 : tensor<1x?xf32>
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

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_dynamic_slice_success_static_result_type
func.func @real_dynamic_slice_to_dynamic_slice_success_static_result_type(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<1xf32> {
  //  CHECK-NOT: stablehlo.real_dynamic_slice
  //      CHECK: [[SIZE0_1D:%.*]] = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
  // CHECK-NEXT: [[SIZE0_0D:%.*]] = stablehlo.reshape [[SIZE0_1D]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK-NEXT: stablehlo.dynamic_slice %arg0, [[SIZE0_0D]], sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.add %arg1, %0 : tensor<1xi64>
  %2 = stablehlo.constant dense<1> : tensor<1xi64>
  %3 = stablehlo.real_dynamic_slice %arg0, %arg1, %1, %2 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %3 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_dynamic_slice_success_dynamic_result_type
func.func @real_dynamic_slice_to_dynamic_slice_success_dynamic_result_type(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<?xf32> {
  //  CHECK-NOT: stablehlo.real_dynamic_slice
  //      CHECK: [[SIZE0_1D:%.*]] = stablehlo.slice %arg1 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
  // CHECK-NEXT: [[SIZE0_0D:%.*]] = stablehlo.reshape [[SIZE0_1D]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK-NEXT: stablehlo.dynamic_slice %arg0, [[SIZE0_0D]], sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<?xf32>
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.add %arg1, %0 : tensor<1xi64>
  %2 = stablehlo.constant dense<1> : tensor<1xi64>
  %3 = stablehlo.real_dynamic_slice %arg0, %arg1, %1, %2 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  return %3 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_dynamic_slice_inapplicable_non_unit_strides
func.func @real_dynamic_slice_to_dynamic_slice_inapplicable_non_unit_strides(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>, %arg2: tensor<1xi64>) -> tensor<1xf32> {
  // CHECK: stablehlo.real_dynamic_slice
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.add %arg1, %0 : tensor<1xi64>
  %2 = stablehlo.real_dynamic_slice %arg0, %arg1, %1, %arg2 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_dynamic_slice_inapplicable_unsupported_limit
func.func @real_dynamic_slice_to_dynamic_slice_inapplicable_unsupported_limit(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>, %arg2: tensor<1xi64>, %arg3: tensor<1xi64>) -> tensor<1xf32> {
  // CHECK: stablehlo.real_dynamic_slice
  %0 = stablehlo.real_dynamic_slice %arg0, %arg1, %arg2, %arg3 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_slice_success_static_result_type
func.func @real_dynamic_slice_to_slice_success_static_result_type(%arg0: tensor<4xf32>) -> tensor<1xf32> {
  //  CHECK-NOT: stablehlo.real_dynamic_slice
  //      CHECK: stablehlo.slice %arg0 [0:1:2] : (tensor<4xf32>) -> tensor<1xf32>
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.constant dense<2> : tensor<1xi64>
  %3 = stablehlo.real_dynamic_slice %arg0, %0, %1, %2 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %3 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_slice_success_dynamic_result_type
func.func @real_dynamic_slice_to_slice_success_dynamic_result_type(%arg0: tensor<4xf32>) -> tensor<?xf32> {
  //  CHECK-NOT: stablehlo.real_dynamic_slice
  //      CHECK: stablehlo.slice %arg0 [0:1:2] : (tensor<4xf32>) -> tensor<?xf32>
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.constant dense<2> : tensor<1xi64>
  %3 = stablehlo.real_dynamic_slice %arg0, %0, %1, %2 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
  return %3 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_slice_inapplicable_dynamic_start
func.func @real_dynamic_slice_to_slice_inapplicable_dynamic_start(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<1xf32> {
  // CHECK: stablehlo.real_dynamic_slice
  %0 = stablehlo.constant dense<1> : tensor<1xi64>
  %1 = stablehlo.constant dense<2> : tensor<1xi64>
  %2 = stablehlo.real_dynamic_slice %arg0, %arg1, %0, %1 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_slice_inapplicable_dynamic_limit
func.func @real_dynamic_slice_to_slice_inapplicable_dynamic_limit(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<1xf32> {
  // CHECK: stablehlo.real_dynamic_slice
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<2> : tensor<1xi64>
  %2 = stablehlo.real_dynamic_slice %arg0, %0, %arg1, %1 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @real_dynamic_slice_to_slice_inapplicable_dynamic_strides
func.func @real_dynamic_slice_to_slice_inapplicable_dynamic_strides(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>) -> tensor<1xf32> {
  // CHECK: stablehlo.real_dynamic_slice
  %0 = stablehlo.constant dense<0> : tensor<1xi64>
  %1 = stablehlo.constant dense<1> : tensor<1xi64>
  %2 = stablehlo.real_dynamic_slice %arg0, %0, %1, %arg1 : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xf32>
  return %2 : tensor<1xf32>
}
