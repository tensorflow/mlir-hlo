// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s
// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect -emit-bytecode -debug-only=stablehlo-bytecode 2>&1 | FileCheck --check-prefix=CHECK-WARN %s

// CHECK-WARN-NOT: Not Implemented

// Tests for types, ops with custom constraints, verifiers, printer or parser
// methods.

// CHECK-LABEL: func private @token_type() -> !stablehlo.token
func.func private @token_type() -> !stablehlo.token

// -----

// expected-error@+1 {{unknown stablehlo type: foobar}}
func.func private @invalid_type() -> !stablehlo.foobar

// -----

// TODO(#498): AllReduceOp replica groups does not need to be rank 2.
func.func @all_reduce(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{replica groups should be a rank 2 tensor}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<0> : tensor<1xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

// CHECK-LABEL: func @all_reduce_with_promotable_types
func.func @all_reduce_with_promotable_types(%operand: tensor<f32>) -> tensor<f64> {

  %result = "stablehlo.all_reduce"(%operand) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%0) : (tensor<f64>) -> ()
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<f32>) -> tensor<f64>

  func.return %result : tensor<f64>
}

// -----

// CHECK-LABEL: func @all_reduce_variadic
func.func @all_reduce_variadic(%operand0: tensor<f32>, %operand1: tensor<f32>) -> (tensor<f64>, tensor<f64>) {
  %results:2 = "stablehlo.all_reduce"(%operand0, %operand1) ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%0) : (tensor<f64>) -> ()
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<f32>, tensor<f32>) -> (tensor<f64>, tensor<f64>)
  func.return %results#0, %results#1 : tensor<f64>, tensor<f64>
}

// -----

// CHECK-LABEL: func @all_reduce_with_promotable_quantized_types
func.func @all_reduce_with_promotable_quantized_types(%operand: tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>)
    -> tensor<!quant.uniform<i16:f32, 2.000000e+00:15>> {

  %result = "stablehlo.all_reduce"(%operand) ({
    ^bb0(%arg0: tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>, %arg1: tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>
      "stablehlo.return"(%0) : (tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>) -> ()
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>) -> tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>

  func.return %result : tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>
}

// -----

func.func @all_reduce_c1(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{replica id #1 seen more than once}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 1, 1, 3]]> : tensor<1x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c3(%operand: tensor<10xf32>) -> tensor<10xf32> {
  //  expected-error@+1 {{replica groups cannot be empty}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<0> : tensor<0x2xi64>,
    use_global_device_ids
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c3(%operand: tensor<10xf32>) -> tensor<10xf32> {
  //  expected-error@+1 {{replica id #2 not seen in replica groups}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 1, 3]]> : tensor<1x3xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c4(%operand: tensor<10xf32>) -> tensor<10xf32> {
  //  expected-error@+1 {{channel_id must be positive when useGlobalDeviceIds is set but got: -1}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = #stablehlo.channel_handle<handle = -1, type = 0>,
    use_global_device_ids
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c5(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{Reduction-region must take 2 parameters, but takes 3 parameter(s)}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, -1], [1, 3, -1, -1]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c5(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{The reduction-region expected to return some value(s)}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    "stablehlo.return"() : () -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c5(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{Reduction-region here must produce 1 tensors, but produces 2 instead}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    "stablehlo.return"(%max, %max) : (tensor<f32>, tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c5(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'tuple<tensor<f32>, tensor<f32>>' instead}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    %tup = "stablehlo.tuple"(%max, %max) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>>
    "stablehlo.return"(%tup) : (tuple<tensor<f32>, tensor<f32>>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c5(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{The type of reduction-region's parameter at index 1 is different than the corresponding result type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<i32>):
    %max = stablehlo.maximum %arg0, %arg0 : tensor<f32>
    "stablehlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c5(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{The type of reduction-region's parameter at index 0 is different than the corresponding result type: 'tensor<f32>' vs 'tensor<i32>'}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    %maxint = "stablehlo.convert"(%max) : (tensor<f32>) -> tensor<i32>
    "stablehlo.return"(%maxint) : (tensor<i32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c5(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<i32>
    "stablehlo.return"(%max) : (tensor<i32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c5(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{The shape of reduction-region's result type at index 0 differs from the op's corresponding init-value type: 'tensor<4xf32>' vs 'tensor<f32>'}}
  %0 = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<4xf32>
    "stablehlo.return"(%max) : (tensor<4xf32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_c5(%operand: tensor<i32>) -> tensor<i8> {

  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<i8>' vs 'tensor<i32>'}}
  %result = "stablehlo.all_reduce"(%operand) ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%0) : (tensor<i8>) -> ()
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<i32>) -> tensor<i8>

  func.return %result : tensor<i8>
}
// -----

func.func @all_reduce_c5(%operand: tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>)
    -> tensor<!quant.uniform<i32:f64, 2.000000e+00:15>> {

  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>' vs 'tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>'}}
  %result = "stablehlo.all_reduce"(%operand) ({
    ^bb0(%arg0: tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>, %arg1: tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>):
      %0 = stablehlo.add %arg0, %arg1 : tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>
      "stablehlo.return"(%0) : (tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>) -> ()
  }) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>) -> tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>

  func.return %result : tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>
}

// -----

// CHECK-LABEL: func @reduce_scatter
func.func @reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @reduce_scatter_dynamic
func.func @reduce_scatter_dynamic(%data: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @reduce_scatter_with_promotable_types
func.func @reduce_scatter_with_promotable_types(%data: tensor<4x16xf32>) -> tensor<4x4xf64> {
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f64>
    "stablehlo.return"(%1) : (tensor<f64>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16xf32>) -> tensor<4x4xf64>
  func.return %0 : tensor<4x4xf64>
}

// -----

// CHECK-LABEL: func @reduce_scatter_with_promotable_quantized_types
func.func @reduce_scatter_with_promotable_quantized_types(
    %data: tensor<4x16x!quant.uniform<i8:f32, 2.000000e+00:15>>) ->
    tensor<4x4x!quant.uniform<i16:f32, 2.000000e+00:15>> {
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>, %arg3: tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>
    "stablehlo.return"(%1) : (tensor<!quant.uniform<i16:f32, 2.000000e+00:15>>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16x!quant.uniform<i8:f32, 2.000000e+00:15>>) -> tensor<4x4x!quant.uniform<i16:f32, 2.000000e+00:15>>
  func.return %0 : tensor<4x4x!quant.uniform<i16:f32, 2.000000e+00:15>>
}

// -----

func.func @reduce_scatter_c2(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{op attribute 'scatter_dimension' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = -1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c2(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{scatter dim should be less than operand/result rank}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 4 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c3(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  //  expected-error@+1 {{replica id #1 seen more than once}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 1, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c5(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  //  expected-error@+1 {{Invalid replica id -1}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, -1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c5(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  //  expected-error@+1 {{replica id #2 not seen in replica groups}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 3]]> : tensor<1x3xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c6(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  //  expected-error@+1 {{channel_id must be positive when useGlobalDeviceIds is set but got: 0}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      use_global_device_ids} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c7(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{Reduction-region must take 2 parameters, but takes 3 parameter(s)}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"() : () -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c7(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{The reduction-region expected to return some value(s)}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"() : () -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c7(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{Reduction-region here must produce 1 tensors, but produces 2 instead}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1, %1) : (tensor<f32>, tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c7(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'tuple<tensor<f32>, tensor<f32>>' instead}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = "stablehlo.tuple"(%arg2, %arg2) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>>
    "stablehlo.return"(%1) : (tuple<tensor<f32>, tensor<f32>>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c7(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{The type of reduction-region's parameter at index 1 is different than the corresponding result type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<i32>):
    %1 = stablehlo.add %arg2, %arg2 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c7(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{The type of reduction-region's parameter at index 0 is different than the corresponding result type: 'tensor<f32>' vs 'tensor<i32>'}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    %2 = "stablehlo.convert"(%1) : (tensor<f32>) -> tensor<i32>
    "stablehlo.return"(%2) : (tensor<i32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c7(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_c7(%data: tensor<4x16xi32>) -> tensor<4x4xi8> {

  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<i8>' vs 'tensor<i32>'}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<i8>, %arg3: tensor<i8>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<i8>
    "stablehlo.return"(%1) : (tensor<i8>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16xi32>) -> tensor<4x4xi8>
  func.return %0 : tensor<4x4xi8>
}

// -----

func.func @reduce_scatter_c7(%data: tensor<4x16x!quant.uniform<i8:f32, 2.000000e+00:15>>)
  -> tensor<4x4x!quant.uniform<i32:f64, 2.000000e+00:15>> {

  // expected-error@+1 {{The element-type of reduction-region's result type at index 0 is expected to be promotable from the op's corresponding init-value element-type: 'tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>' vs 'tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>'}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>, %arg3: tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>
    "stablehlo.return"(%1) : (tensor<!quant.uniform<i32:f64, 2.000000e+00:15>>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16x!quant.uniform<i8:f32, 2.000000e+00:15>>) -> tensor<4x4x!quant.uniform<i32:f64, 2.000000e+00:15>>
  func.return %0 : tensor<4x4x!quant.uniform<i32:f64, 2.000000e+00:15>>
}

// -----

func.func @reduce_scatter_c8(%data: tensor<4x16xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{operand and result should have same rank}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

func.func @reduce_scatter_c8(%data: tensor<4x16xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{operand scatter dimension has size 16, expected to be a multiple of result scatter dimension size 5}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @reduce_scatter_c8(%data: tensor<4x16xf32>) -> tensor<3x4xf32> {
  // expected-error@+1 {{non scatter dimensions should be same for operand (4) and result (3)}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// -----

func.func @reduce_scatter_c9(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{result element-type is expected to be 'f64', but got 'f32'}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f64>
    "stablehlo.return"(%1) : (tensor<f64>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter_i3(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{replica groups should be a rank 2 tensor}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<0> : tensor<1xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// TODO(#1746): Sync verification of ReduceScatter with HLO.
func.func @reduce_scatter_invalid(%data: tensor<4x16xf32>) -> tensor<4x0xf32> {
  // expected-error@+1 {{result dimension size at scatter_dimension cannot be zero}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x0xf32>
  func.return %0 : tensor<4x0xf32>
}

// -----

// TODO(#1746): Sync verification of ReduceScatter with HLO.
func.func @reduce_scatter_invalid(%data: tensor<4x0xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{operand dimension size at scatter_dimension cannot be zero}}
  %0 = "stablehlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x0xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @all_to_all
func.func @all_to_all(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

// CHECK-LABEL: func @all_to_all_variadic
func.func @all_to_all_variadic(%data0: tensor<4x16xf32>, %data1: tensor<5x16xf64>) -> (tensor<16x4xf32>, tensor<20x4xf64>) {
  %0:2 = "stablehlo.all_to_all"(%data0, %data1) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<4x16xf32>, tensor<5x16xf64>) -> (tensor<16x4xf32>, tensor<20x4xf64>)
  func.return %0#0, %0#1 : tensor<16x4xf32>, tensor<20x4xf64>
}

// -----

// CHECK-LABEL: func @all_to_all_same_split_concat_dim
func.func @all_to_all_same_split_concat_dim(%data: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 0 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<4x16xf32>) -> tensor<4x16xf32>
  func.return %0 : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: func @all_to_all_dynamic_split_dim
func.func @all_to_all_dynamic_split_dim(%data: tensor<4x?xf32>) -> tensor<20x?xf32> {
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 5 : i64,
    replica_groups = dense<[[0, 1, 2, 3, 4]]> : tensor<1x5xi64>
  } : (tensor<4x?xf32>) -> tensor<20x?xf32>
  func.return %0 : tensor<20x?xf32>
}

// -----

// CHECK-LABEL: func @all_to_all_dynamic_concat_dim
func.func @all_to_all_dynamic_concat_dim(%data: tensor<?x16xf32>) -> tensor<?x4xf32> {
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<?x16xf32>) -> tensor<?x4xf32>
  func.return %0 : tensor<?x4xf32>
}

// -----

func.func @all_to_all_c1(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{op attribute 'split_dimension' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = -1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @all_to_all_c1(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{AllToAll split_dimension 2 is out-of-bounds for input rank 2}}
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 2 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @all_to_all_c2(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{split dimension has size 16, expected to be a multiple of split_count 5}}
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 5 : i64,
    replica_groups = dense<[[0, 1, 2, 3, 4]]> : tensor<1x5xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @all_to_all_c3(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{op attribute 'concat_dimension' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = -1 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @all_to_all_c3(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{AllToAll concat_dimension 2 is out-of-bounds for input rank 2}}
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 2 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @all_to_all_c4(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{op attribute 'split_count' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 0 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @all_to_all_c5(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{replica id #2 seen more than once}}
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 2]]> : tensor<2x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @all_to_all_c7(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{replica id #1 not seen in replica groups}}
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[-5, -4, -3, 0]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @all_to_all_c8(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{group size of replica_groups must be 4}}
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 2, 4], [1, 3, 5]]> : tensor<2x3xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @all_to_all_c9(%data: tensor<4x16xf32>) -> tensor<16x4xf64> {
  // expected-error@+1 {{op requires the same element type for operand and result at index 0}}
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<4x16xf32>) -> tensor<16x4xf64>
  func.return %0 : tensor<16x4xf64>
}

// -----

func.func @all_to_all_c9_mismatch_count(%data0: tensor<4x16xf32>, %data1: tensor<4x16xf32>) -> tensor<16x4xf64> {
  // expected-error@+1 {{op requires the same number of operands and results}}
  %0 = "stablehlo.all_to_all"(%data0, %data1) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<16x4xf64>
  func.return %0 : tensor<16x4xf64>
}

// -----

func.func @all_to_all_i5(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{replica groups should be a rank 2 tensor}}
  %0 = "stablehlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[[0], [1], [2], [3]]]> : tensor<1x4x1xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @all_gather_variadic(%arg0: tensor<8x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<8x8xf32>, tensor<2x4xf32>) {
  %0:2 = "stablehlo.all_gather"(%arg0, %arg1) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>, tensor<2x2xf32>) -> (tensor<8x8xf32>, tensor<2x4xf32>)
  func.return %0#0, %0#1 : tensor<8x8xf32>, tensor<2x4xf32>
}

// -----

func.func @allgather_gather_along_zero_dimension(%arg0: tensor<128x0xf32>) -> tensor<128x100xf32> {
  // expected-error@+1 {{dimension size of operand at 'all_gather_dim' cannot be zero}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<128x0xf32>) -> tensor<128x100xf32>
  func.return %0 : tensor<128x100xf32>
}

// -----

func.func @all_gather_c1(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{op attribute 'all_gather_dim' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = -1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_c1(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{all_gather_dim must be a valid index of operand}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 2 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_c2(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{replica id #2 seen more than once}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 2]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_c4(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{Invalid replica id -1}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, -1]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_c4(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{replica id #4 not seen in replica groups}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 6, 8], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_c5(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{channel_id cannot be negative when useGlobalDeviceIds is set}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_handle = #stablehlo.channel_handle<handle = -1, type = 0>,
    use_global_device_ids
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_c6(%arg0: tensor<8x2x32xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{operand and result must have the same rank}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2x32xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_c6(%arg0: tensor<8x2xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{operand and result should have the same shape except for the dimension size at 'all_gather_dim'}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// -----

func.func @all_gather_c6(%arg0: tensor<128x32xf32>) -> tensor<128x100xf32> {
  // expected-error@+1 {{result gather dimension has size 100, expected to be a multiple of operand gather dimension size 32}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<128x32xf32>) -> tensor<128x100xf32>
  func.return %0 : tensor<128x100xf32>
}

// -----

func.func @all_gather_i3(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{replica groups should be a rank 2 tensor}}
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[[0], [1], [2], [3]]]> : tensor<1x4x1xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func @broadcast
func.func @broadcast(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  %0 = "stablehlo.broadcast"(%arg0) {broadcast_sizes = array<i64: 1, 2>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_bad_result_rank(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.broadcast' op inferred type(s) 'tensor<2x3xi32>' are incompatible with return type(s) of operation 'tensor<1x2x3xi32>'}}
  %0 = "stablehlo.broadcast"(%arg0) {broadcast_sizes = array<i64: 2>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_bad_first_part_result_shape(%arg0: tensor<3xi32>) -> tensor<1x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.broadcast' op inferred type(s) 'tensor<2x3xi32>' are incompatible with return type(s) of operation 'tensor<1x3xi32>'}}
  %0 = "stablehlo.broadcast"(%arg0) {broadcast_sizes = array<i64: 2>} : (tensor<3xi32>) -> tensor<1x3xi32>
  func.return %0 : tensor<1x3xi32>
}

// -----

func.func @broadcast_bad_second_part_result_shape(%arg0: tensor<3xi32>) -> tensor<2x1xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.broadcast' op inferred type(s) 'tensor<2x3xi32>' are incompatible with return type(s) of operation 'tensor<2x1xi32>'}}
  %0 = "stablehlo.broadcast"(%arg0) {broadcast_sizes = array<i64: 2>} : (tensor<3xi32>) -> tensor<2x1xi32>
  func.return %0 : tensor<2x1xi32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim
func.func @dynamic_broadcast_in_dim(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_unknown_dim
func.func @dynamic_broadcast_in_dim_unknown_dim(%arg0: tensor<32xf32>, %shape: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 2>} : (tensor<32xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_ok_dim
func.func @dynamic_broadcast_in_dim_ok_dim(%arg0: tensor<1xf32>, %shape: tensor<3xi64>) -> tensor<7x8x9xf32> {
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 2>} : (tensor<1xf32>, tensor<3xi64>) -> tensor<7x8x9xf32>
  func.return %0 : tensor<7x8x9xf32>
}

// -----

func.func @dynamic_broadcast_in_dim_output_dimensions_match_result(%arg0: tensor<4xf32>) -> tensor<3x4xf32> {
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

func.func @dynamic_broadcast_in_dim_output_dimensions_compatible_with_result(%arg0: tensor<4xf32>) -> tensor<?x?xf32> {
  %0 = stablehlo.constant dense<[3, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

func.func @dynamic_broadcast_in_dim_c1(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi62> {
  // expected-error@+1 {{expects operand and result to have compatible element type}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi64>
  func.return %0 : tensor<?x?x?xi64>
}

// -----

func.func @dynamic_broadcast_in_dim_c2(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  // expected-error@+1 {{broadcast_dimensions size (1) does not match operand rank (2)}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1>} : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_c3_negative_size(%arg0: tensor<1xf32>, %shape: tensor<3xi64>) -> tensor<7x8x9xf32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value -1 for result with rank 3}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: -1>} : (tensor<1xf32>, tensor<3xi64>) -> tensor<7x8x9xf32>
  func.return %0 : tensor<7x8x9xf32>
}

// -----

func.func @dynamic_broadcast_in_dim_c3_too_large(%arg0: tensor<1xf32>, %shape: tensor<3xi64>) -> tensor<7x8x9xf32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value 3 for result with rank 3}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 3>} : (tensor<1xf32>, tensor<3xi64>) -> tensor<7x8x9xf32>
  func.return %0 : tensor<7x8x9xf32>
}

// -----

func.func @dynamic_broadcast_in_dim_c4(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  // expected-error@+1 {{broadcast_dimensions should not have duplicates}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1, 1>} : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_c5_shape_mismatch(%arg0: tensor<32xf32>, %shape: tensor<3xi64>) -> tensor<7x8x9xf32> {
  // expected-error@+1 {{size of operand dimension 0 (32) is not compatible with size of result dimension 2 (9)}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 2>} : (tensor<32xf32>, tensor<3xi64>) -> tensor<7x8x9xf32>
  func.return %0 : tensor<7x8x9xf32>
}

// -----

func.func @dynamic_broadcast_in_dim_c5_too_large(%arg0: tensor<1xf32>, %shape: tensor<3xi64>) -> tensor<7x8x9xf32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value 3 for result with rank 3}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 3>} : (tensor<1xf32>, tensor<3xi64>) -> tensor<7x8x9xf32>
  func.return %0 : tensor<7x8x9xf32>
}

// -----

func.func @dynamic_broadcast_in_dim_c5_input_mismatch_with_shape(%arg0: tensor<1x3xi32>) {
  %shape = stablehlo.constant dense<[2, 1, 1]> : tensor<3xi32>
  // expected-error@+1 {{size of operand dimension 1 (3) is not equal to 1 or value of shape at index 2 (1)}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<1x3xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

func.func @dynamic_broadcast_in_dim_c7_output_dimensions_negative_size(%arg0: tensor<4xf32>) -> tensor<3x4xf32> {
  // expected-error@+2 {{output shape [-1, 4] is incompatible with return type of operation 'tensor<3x4xf32>'}}
  %0 = stablehlo.constant dense<[-1, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

func.func @dynamic_broadcast_in_dim_c7_output_dimensions_mismatching_size(%arg0: tensor<4xf32>) -> tensor<3x4xf32> {
  // expected-error@+2 {{output shape [1, 4] is incompatible with return type of operation 'tensor<3x4xf32>'}}
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_broadcast_in_dim %arg0, %0, dims = [1] : (tensor<4xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

func.func @dynamic_broadcast_in_dim_c8(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  // expected-error@+1 {{duplicate expansion hint for at least one operand dimension}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {
    broadcast_dimensions = array<i64: 1, 2>,
    known_expanding_dimensions = array<i64: 0, 0>
  } : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_c9_c10(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  // expected-error@+1 {{hint for expanding dimension 3 does not refer to a valid operand dimension}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {
    broadcast_dimensions = array<i64: 1, 2>,
    known_expanding_dimensions = array<i64: 3>
  } : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_broadcast_in_dim_dynamic_output_shape(%arg0: tensor<?x?xi32>, %shape: tensor<?xi64>) -> tensor<7x8x9xi32> {
  // expected-error@+1 {{op operand #1 must be statically shaped}}
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<?x?xi32>, tensor<?xi64>) -> tensor<7x8x9xi32>
  func.return %0 : tensor<7x8x9xi32>
}

// -----

// CHECK-LABEL: func @broadcast_in_dim
func.func @broadcast_in_dim(%arg0: tensor<1x2xi32>) -> tensor<1x2x2xi32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<1x2xi32>) -> tensor<1x2x2xi32>
  func.return %0 : tensor<1x2x2xi32>
}

// -----

func.func @broadcast_in_dim_c2(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions size (1) does not match operand rank (2)}}
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 1>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_in_dim_c3(%arg0: tensor<1x2xi32>) -> tensor<1x2x2xi32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value -1 for result with rank 3}}
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: -1, 2>} : (tensor<1x2xi32>) -> tensor<1x2x2xi32>
  func.return %0 : tensor<1x2x2xi32>
}

// -----

func.func @broadcast_in_dim_c3(%arg0: tensor<1x2x3xi32>) -> tensor<3xi32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value 1 for result with rank 1}}
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 0,1,2>} : (tensor<1x2x3xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @broadcast_in_dim_c4(%arg0: tensor<1x1x3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions should not have duplicates}}
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 0,0,2>} : (tensor<1x1x3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_in_dim_c5(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{size of operand dimension 0 (3) is not equal to 1 or size of result dimension 1 (2)}}
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 1>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: func @broadcast_in_dim_dynamic_i1
func.func @broadcast_in_dim_dynamic_i1(%arg0: tensor<?xi32>) -> tensor<1x3xi32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<?xi32>) -> tensor<1x3xi32>
  return %0 : tensor<1x3xi32>
}

// -----

func.func @broadcast_in_dim_dynamic_result(%arg0: tensor<3xi32>) -> tensor<?x3xi32> {
  // expected-error@+1 {{must be statically shaped or single bounded dimension tensor}}
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = array<i64: 1>} : (tensor<3xi32>) -> tensor<?x3xi32>
  func.return %0 : tensor<?x3xi32>
}

// -----

// Regression test for b/180052624, where this was improperly marked as an
// invalid stablehlo.broadcast_in_dim op.
// CHECK-LABEL: func @broadcast_in_dim_dynamic_shaped_operand
func.func @broadcast_in_dim_dynamic_shaped_operand(%arg0 : tensor<?xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0>
  } : (tensor<?xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @if
func.func @if(%pred : tensor<i1>, %branch_operand : tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
    }, {
      "stablehlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
    }) : (tensor<i1>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

func.func @if_c1(%pred : tensor<i1>, %branch_operand : tensor<f32>) -> tensor<f32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{branch 0 must have 0 arguments, but found 1}}
  %0 = "stablehlo.if"(%pred) ({
      ^bb0(%arg0: tensor<f32>):
        "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @if_c1(%pred : tensor<i1>, %branch_operand : tensor<f32>) -> tensor<f32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{branch 1 must have 0 arguments, but found 1}}
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }, {
      ^bb0(%arg0: tensor<f32>):
        "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @if_c2(%pred : tensor<i1>, %branch_operand : tensor<f32>) -> tensor<f32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{branch 0 and branch 1 have mismatched return types: 'tensor<f32>', 'tensor<f32>' vs 'tensor<f32>'}}
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%branch_operand, %branch_operand) : (tensor<f32>, tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @if_c3(%pred : tensor<i1>, %branch_operand : tensor<f32>) -> tensor<i32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<f32>' are incompatible with return type(s) of operation 'tensor<i32>'}}
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: if_dynamic_branch_result
func.func @if_dynamic_branch_result(%pred : tensor<i1>, %true_branch_operand: tensor<2xf32>, %false_branch_operand : tensor<?xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%true_branch_operand) : (tensor<2xf32>) -> ()
    }, {
      "stablehlo.return"(%false_branch_operand) : (tensor<?xf32>) -> ()
    }) : (tensor<i1>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: if_dynamic_op_result
func.func @if_dynamic_op_result(%pred : tensor<i1>, %branch_operand: tensor<2xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
    }, {
      "stablehlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
    }) : (tensor<i1>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @if_i1(%pred : tensor<1xi1>, %branch_operand : tensor<f32>) -> tensor<f32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operand should be rank 0 tensor but got rank 1}}
  %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }) : (tensor<1xi1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @case
func.func @case(%index : tensor<i32>, %branch_operand : tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0, %1 = "stablehlo.case"(%index) ({
    "stablehlo.return"(%branch_operand, %branch_operand) : (tensor<f32>, tensor<f32>) -> ()
  }, {
    "stablehlo.return"(%branch_operand, %branch_operand) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i32>) -> (tensor<f32>, tensor<f32>)
  func.return %0, %1 : tensor<f32>, tensor<f32>
}

// -----

func.func @case_c1(%index : tensor<i32>, %branch_operand : tensor<2xf32>) -> tensor<2xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expect at least one branch}}
  %0 = "stablehlo.case"(%index) : (tensor<i32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

func.func @case_c2(%index : tensor<i32>, %branch_operand : tensor<f32>) -> tensor<f32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{branch 1 must have 0 arguments, but found 1}}
  %0 = "stablehlo.case"(%index) ({
      "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
  }, {
      ^bb0(%arg0: tensor<f32>):
        "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @case_c3(%index: tensor<i32>, %operand_1: tensor<f32>, %operand_2: tensor<f32>, %operand_3: tensor<f32>) -> tensor<f32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{branch 0 and branch 1 have mismatched return types: 'tensor<f32>' vs 'tensor<i32>'}}
  %0 = "stablehlo.case"(%index) ({
      %1 = "stablehlo.negate"(%operand_1) : (tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    },  {
      %1 = stablehlo.constant dense<2> : tensor<i32>
      "stablehlo.return"(%1) : (tensor<i32>) -> ()
    }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @case_c4(%index : tensor<i32>, %branch_operand : tensor<f32>) -> tensor<i32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<f32>' are incompatible with return type(s) of operation 'tensor<i32>'}}
  %0 = "stablehlo.case"(%index) ({
      "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
  }, {
      "stablehlo.return"(%branch_operand) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: @case_dynamic_branch_result
func.func @case_dynamic_branch_result(%index : tensor<i32>, %branch_operand : tensor<?xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.case"(%index) ({
      "stablehlo.return"(%branch_operand) : (tensor<?xf32>) -> ()
  }, {
      "stablehlo.return"(%branch_operand) : (tensor<?xf32>) -> ()
  }) : (tensor<i32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @case_dynamic_op_result
func.func @case_dynamic_op_result(%index : tensor<i32>, %branch_operand : tensor<2xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.case"(%index) ({
      "stablehlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }, {
      "stablehlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }) : (tensor<i32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @case_i1(%index : tensor<1xi32>, %branch_operand : tensor<2xf32>) -> tensor<2xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operand should be rank 0 tensor but got rank 1}}
  %0 = "stablehlo.case"(%index) ({
      "stablehlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }, {
      "stablehlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }) : (tensor<1xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @compare
func.func @compare(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    compare_type = #stablehlo<comparison_type SIGNED>
  } : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  func.return %0 : tensor<3xi1>
}

// -----

// CHECK-LABEL: func @compare_compatible_types
func.func @compare_compatible_types(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<?xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// -----

// CHECK-LABEL: func @compare_compatible_operand_types
func.func @compare_compatible_operand_types(%arg0: tensor<3xi32>, %arg1: tensor<?xi32>) -> tensor<?xi1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<3xi32>, tensor<?xi32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// -----

// CHECK-LABEL: func @collective_permute
func.func @collective_permute(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

// -----

func.func @collective_permute_c1(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{expect source_target_pairs attribute of shape (N, 2), but got (2, 3)}}
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}


// -----

func.func @collective_permute_c2(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{duplicate sources not allowed}}
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [0, 2], [2, 3]]> : tensor<3x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

// -----

func.func @collective_permute_c3(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{duplicate targets not allowed}}
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 1]]> : tensor<3x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

// -----

func.func @collective_permute_c4(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{replica ids in source_target_pairs must be >= 0}}
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [-1, 0]]> : tensor<2x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

// -----

func.func @collective_permute_i2(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{expect source_target_pairs attribute to be of rank 2, but got rank 1}}
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

// -----

// CHECK-LABEL: @concatenate_1D
func.func @concatenate_1D(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>)  -> tensor<3xi32> {
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: @concatenate_1D
// Verifies that an error is not thrown if the inferred type is compatible with
// the result type.
func.func @concatenate_1D(%arg0: tensor<1xi32>, %arg1: tensor<?xi32>)  -> tensor<3xi32> {
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<?xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concatenate_c1_c5(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>)  -> tensor<4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{op inferred type(s) 'tensor<3xi32>' are incompatible with return type(s) of operation 'tensor<4xi32>'}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

func.func @concatenate_c2(%arg0: tensor<1xi32>, %arg1: tensor<2x2xi32>)  -> tensor<3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operands (0) and (1) do not match rank}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2x2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concatenate_c3()  -> tensor<2xi32> {
  // expected-error@+1 {{expected 1 or more operands, but found 0}}
  %0 = "stablehlo.concatenate"() { dimension = 0 : i64 } : () -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @concatenate_c4(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>)  -> tensor<3xi32> {
  // expected-error@+1 {{op attribute 'dimension' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = -1 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concatenate_c4(%arg0: tensor<i32>, %arg1: tensor<i32>)  -> tensor<2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{rank-0 values cannot be concatenated}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @concatenate_c4(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>)  -> tensor<3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{dimension 10 is out-of-bounds for input rank 1}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 10 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concatenate_c6(%arg0: tensor<1x3xi32>, %arg1: tensor<2x2xi32>)  -> tensor<3x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{shapes of operand (0) and (1) are not compatible at non-concat index 1: (1, 3) != (2, 2)}}
  %0 = "stablehlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x3xi32>, tensor<2x2xi32>) -> tensor<3x3xi32>
  func.return %0 : tensor<3x3xi32>
}

// -----

// CHECK-LABEL: func @clamp
func.func @clamp(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "stablehlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @clamp_compatible_dynamic
func.func @clamp_compatible_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<i32>, %arg2: tensor<3xi32>) -> tensor<?xi32> {
  %0 = "stablehlo.clamp"(%arg1, %arg0, %arg2) : (tensor<i32>, tensor<?xi32>, tensor<3xi32>) -> tensor<?xi32>
  func.return %0: tensor<?xi32>
}

// CHECK-LABEL: func @clamp_compatible_dynamic_match_static
func.func @clamp_compatible_dynamic_match_static(%arg0: tensor<?xi32>, %arg1: tensor<i32>, %arg2: tensor<3xi32>) -> tensor<3xi32> {
  %0 = "stablehlo.clamp"(%arg1, %arg0, %arg2) : (tensor<i32>, tensor<?xi32>, tensor<3xi32>) -> tensor<3xi32>
  func.return %0: tensor<3xi32>
}

// -----

func.func @clamp_c1(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<1xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{min shape [2] is not scalar and is not compatible to operand shape [1]}}
  %0 = "stablehlo.clamp"(%arg1, %arg0, %arg0) : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

func.func @clamp_c2(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<1xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{max shape [2] is not scalar and is not compatible to operand shape [1]}}
  %0 = "stablehlo.clamp"(%arg0, %arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

func.func @clamp_c4(%arg0: tensor<1xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<1xi32>' are incompatible with return type(s) of operation 'tensor<1x2xi32>'}}
  %0 = "stablehlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2xi32>
  func.return %0: tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @clamp_scalar
func.func @clamp_scalar(%arg0: tensor<1xi32>, %arg1: tensor<i32>) -> tensor<1xi32> {
  %0 = "stablehlo.clamp"(%arg1, %arg0, %arg1) : (tensor<i32>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @cholesky
func.func @cholesky(%arg0: tensor<1x2x2xf32>) -> tensor<1x2x2xf32> {
  %0 = "stablehlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  func.return %0: tensor<1x2x2xf32>
}

// -----

func.func @cholesky_c2(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{argument 'a' must have rank >= 2, got shape 1}}
  %0 = "stablehlo.cholesky"(%arg0) { lower = true } : (tensor<1xf32>) -> tensor<1xf32>
  func.return %0: tensor<1xf32>
}

// -----

func.func @cholesky_c3(%arg0: tensor<1x2x1xf32>) -> tensor<1x2x1xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{minor dimensions of 'a' must have equal size, got shape 1, 2, 1}}
  %0 = "stablehlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
  func.return %0: tensor<1x2x1xf32>
}

// -----

func.func @create_token() -> !stablehlo.token {
  %0 = "stablehlo.create_token"() : () -> !stablehlo.token
  func.return %0: !stablehlo.token
}

// -----

// CHECK-LABEL: func @dot_vector
func.func @dot_vector(%arg0: tensor<1x2xi32>, %arg1: tensor<2x1xi32>) -> tensor<1x1xi32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<1x2xi32>, tensor<2x1xi32>) -> tensor<1x1xi32>
  func.return %0: tensor<1x1xi32>
}

// -----

// CHECK-LABEL: func @dot_matrix
func.func @dot_matrix(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0: tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @dot_precision_config
func.func @dot_precision_config(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGHEST>]} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0: tensor<2x2xi32>
}

// -----

func.func @dot_bad_precision_config(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // expected-error@+1 {{'precision_config' failed to satisfy constraint}}
  %0 = "stablehlo.dot"(%arg0, %arg1) {precision_config = ["FOO", #stablehlo<precision HIGHEST>]} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0: tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @test_unary_result_accuracy_tol
func.func @test_unary_result_accuracy_tol(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
  %cbrt = "stablehlo.cbrt"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  %cosine = "stablehlo.cosine"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  %exponential = "stablehlo.exponential"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  %exponential_minus_one = "stablehlo.exponential_minus_one"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  %log = "stablehlo.log"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  %log_plus_one = "stablehlo.log_plus_one"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  %logistic = "stablehlo.logistic"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  %rsqrt = "stablehlo.rsqrt"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  %sine = "stablehlo.sine"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  %sqrt = "stablehlo.sqrt"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  %tan = "stablehlo.tan"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  %tanh = "stablehlo.tanh"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e+00, rtol = 1.000000e+00, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : (tensor<2xf32>) -> tensor<2xf32>
  func.return %cbrt, %cosine, %exponential, %exponential_minus_one, %log, %log_plus_one, %logistic, %rsqrt, %sine, %sqrt, %tan, %tanh : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
}

// -----

// CHECK-LABEL: func @test_unary_result_accuracy_default
func.func @test_unary_result_accuracy_default(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
  %cbrt = "stablehlo.cbrt"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  %cosine = "stablehlo.cosine"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  %exponential = "stablehlo.exponential"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  %exponential_minus_one = "stablehlo.exponential_minus_one"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  %log = "stablehlo.log"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  %log_plus_one = "stablehlo.log_plus_one"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  %logistic = "stablehlo.logistic"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  %rsqrt = "stablehlo.rsqrt"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  %sine = "stablehlo.sine"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  %sqrt = "stablehlo.sqrt"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  %tan = "stablehlo.tan"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  %tanh = "stablehlo.tanh"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 0.0, rtol = 0.0, ulps = 0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>} : (tensor<2xf32>) -> tensor<2xf32>
  func.return %cbrt, %cosine, %exponential, %exponential_minus_one, %log, %log_plus_one, %logistic, %rsqrt, %sine, %sqrt, %tan, %tanh : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
}

// -----

func.func @test_cbrt_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %cbrt = "stablehlo.cbrt"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %cbrt: tensor<f32>
}

// -----

func.func @test_cosine_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %cosine = "stablehlo.cosine"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %cosine: tensor<f32>
}

// -----

func.func @test_exponential_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %exponential = "stablehlo.exponential"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %exponential: tensor<f32>
}

// -----

func.func @test_exponential_minus_one_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %exponential_minus_one = "stablehlo.exponential_minus_one"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %exponential_minus_one: tensor<f32>
}

// -----

func.func @test_log_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %log = "stablehlo.log"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %log: tensor<f32>
}

// -----

func.func @test_log_plus_one_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %log_plus_one = "stablehlo.log_plus_one"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %log_plus_one: tensor<f32>
}

// -----

func.func @test_logistic_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %logistic = "stablehlo.logistic"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %logistic: tensor<f32>
}

// -----

func.func @test_rsqrt_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %rsqrt = "stablehlo.rsqrt"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %rsqrt: tensor<f32>
}

// -----

func.func @test_sine_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %sine = "stablehlo.sine"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %sine: tensor<f32>
}

// -----

func.func @test_sqrt_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %sqrt = "stablehlo.sqrt"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %sqrt: tensor<f32>
}

// -----

func.func @test_tan_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %tan = "stablehlo.tan"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %tan: tensor<f32>
}

// -----

func.func @test_tanh_highest_error(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Invalid tolerances for ResultAccuracyAttr with mode HIGHEST, must be all zero.}}
  %tanh = "stablehlo.tanh"(%arg0) {result_accuracy = #stablehlo.result_accuracy<atol = 1.0, rtol = 0.0, ulps = 4, mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : (tensor<f32>) -> tensor<f32>
  func.return %tanh: tensor<f32>
}

// -----

func.func @dot_more_dynamic_output_type(%arg0: tensor<3xf32>, %arg1: tensor<?x3xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<3xf32>, tensor<?x3xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @dot_cannot_infer_type(%arg0: tensor<?x?x3xf32>, %arg1: tensor<?x3x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error@+1 {{expected both lhs/rhs ranks to be either 1 or 2}}
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<?x?x3xf32>, tensor<?x3x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_result_type_mismatch_with_inferred_type(%arg0: tensor<?x3xf32>, %arg1: tensor<3xf32>) -> tensor<3x?xf32> {
  // expected-error@+1 {{inferred shape '[?]' is incompatible with return type of operation 'tensor<3x?xf32>'}}
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<?x3xf32>, tensor<3xf32>) -> tensor<3x?xf32>
  func.return %0 : tensor<3x?xf32>
}

// -----

func.func @dot_result_type_match_with_inferred_type(%arg0: tensor<?x3xf32>, %arg1: tensor<3xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<?x3xf32>, tensor<3xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @imag_c2(%arg0: tensor<2xf32>) -> tensor<2xf16> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<2xf32>' are incompatible with return type(s) of operation 'tensor<2xf16>'}}
  %0 = "stablehlo.imag"(%arg0) : (tensor<2xf32>) -> tensor<2xf16>
  func.return %0 : tensor<2xf16>
}

// -----

// CHECK-LABEL: func @imag_complex_input
func.func @imag_complex_input(%arg0: tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32> {
  %0 = "stablehlo.imag"(%arg0) : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

// -----

func.func @infeed(%arg0: !stablehlo.token) -> tensor<3x3xi32> {
  // expected-error@+1 {{layout-attribute size must be 2 (which is the number of op-results - 1 (for token result)), but got 1}}
  %0:3 = "stablehlo.infeed"(%arg0) {infeed_config = "foobar", layout=[[0, 1]]} : (!stablehlo.token) -> (tensor<3x3xi32>, tensor<i1>, !stablehlo.token)
  func.return %0#0 : tensor<3x3xi32>
}

// -----

func.func @infeed(%arg0: !stablehlo.token) -> !stablehlo.token {
  // expected-error@+1 {{layout-attribute size must be 0 (which is the number of op-results - 1 (for token result)), but got 1}}
  %0:1 = "stablehlo.infeed"(%arg0) {infeed_config = "foobar", layout=[[]]} : (!stablehlo.token) -> (!stablehlo.token)
  func.return %0#0 : !stablehlo.token
}

// -----

func.func @infeed(%arg0: !stablehlo.token) -> tensor<3x3xi32> {
  // expected-error@+1 {{layout-attribute expected to have elements of type array, but got 0 : i64}}
  %0:2 = "stablehlo.infeed"(%arg0) {infeed_config = "foobar", layout=[0]} : (!stablehlo.token) -> (tensor<3x3xi32>, !stablehlo.token)
  func.return %0#0 : tensor<3x3xi32>
}

// -----

func.func @infeed(%arg0: !stablehlo.token) -> tensor<3x3xi32> {
  // expected-error@+1 {{layout-attribute's leaf elements are expected to be of type integer, but got []}}
  %0:2 = "stablehlo.infeed"(%arg0) {infeed_config = "foobar", layout=[[0,[]]]} : (!stablehlo.token) -> (tensor<3x3xi32>, !stablehlo.token)
  func.return %0#0 : tensor<3x3xi32>
}

// -----

func.func @infeed_c1(%token: !stablehlo.token) -> tensor<3x3xi32> {
  // expected-error@+1 {{result is expected to be at least of size 1, but got 0}}
  "stablehlo.infeed"(%token) {infeed_config = "foobar", layout=[[[0]], [0]]} : (!stablehlo.token) -> ()
  func.return
}

// -----

func.func @infeed_c2(%token: !stablehlo.token) -> tuple<!stablehlo.token, !stablehlo.token> {
  // expected-error@+1 {{all elements of result types, except the last element, are expected to be of tensor type, but got '!stablehlo.token'}}
  %0:2 = "stablehlo.infeed"(%token) {infeed_config = "foobar", layout = [[[0]], [0]]} : (!stablehlo.token) -> (!stablehlo.token, !stablehlo.token)
  %1 = "stablehlo.tuple"(%0#0, %0#1) : (!stablehlo.token, !stablehlo.token) -> tuple<!stablehlo.token, !stablehlo.token>
  func.return %1 : tuple<!stablehlo.token, !stablehlo.token>
}

// -----

func.func @infeed_c3(%token: !stablehlo.token) -> tuple<tensor<i32>, tensor<i32>> {
  // expected-error@+1 {{last element of result types is expected to be of token type, but got 'tensor<i32>'}}
  %0:2 = "stablehlo.infeed"(%token) {infeed_config = "foobar", layout = [[[0]], [0]]} : (!stablehlo.token) -> (tensor<i32>, tensor<i32>)
  %1 = "stablehlo.tuple"(%0#0, %0#1) : (tensor<i32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>>
  func.return %1 : tuple<tensor<i32>, tensor<i32>>
}

// -----

func.func @iota_scalar() -> tensor<i32> {
  // expected-error@+1 {{does not support scalars}}
  %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @iota_invalid_iota_dimension() -> tensor<4xi32> {
  // expected-error@+1 {{iota dimension cannot go beyond the output rank}}
  %0 = "stablehlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @map
func.func @map(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @map_c3(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires monotonically increasing dimension numbers, but got: 1, 0}}
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1, 0>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @map_c3(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{applied to a subset of dimensions currently not supported: operand dimensions = 2, requested map dimensions size = 3}}
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0, 1, 2>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @map_c4(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects number of operands to match the arity of map computation, but got: 2 and 1}}
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg: tensor<f32>):
    %1 = stablehlo.add %arg, %arg : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

func.func @map_c4(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{computation arguments must be 0-rank tensor, but got: arg #1 of type 'tensor<5xf32>'}}
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<5xf32>):
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @map_c4(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{element type of operands and computation arguments must match, but got: 'f32' and 'i32'}}
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @map_c4(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{computation must return single output, but got: 0}}
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    "stablehlo.return"() : () -> ()
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @map_c4(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{computation must return 0-rank tensor, but got: 'tensor<5xf32>'}}
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.constant dense<2.0> : tensor<5xf32>
    "stablehlo.return"(%1) : (tensor<5xf32>) -> ()
  }) {dimensions = array<i64: 0, 1>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

// CHECK-LABEL: func @map_heterogeneous_inputs
func.func @map_heterogeneous_inputs(%arg0: tensor<2xf32>, %arg1: tensor<2xi32>) -> tensor<2xf32> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<i32>):
    "stablehlo.return"(%arg2) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<2xf32>, tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @map_scalar_operands
func.func @map_scalar_operands(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = array<i64>} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @optimization_barrier
func.func @optimization_barrier(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0, %1 = "stablehlo.optimization_barrier"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0, %1 : tensor<f32>, tensor<f32>
}

// -----

func.func @real_c2(%arg0: tensor<2x3xcomplex<f32>>) -> tensor<2x3xf16> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<2x3xf32>' are incompatible with return type(s) of operation 'tensor<2x3xf16>'}}
  %0 = "stablehlo.real"(%arg0) : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xf16>
  func.return %0 : tensor<2x3xf16>
}

// -----

// CHECK-LABEL: func @real_complex_input
func.func @real_complex_input(%arg0: tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32> {
  %0 = "stablehlo.real"(%arg0) : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

// -----

func.func @recv_c1(%token: !stablehlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>> {
  // expected-error@+1 {{channel_type should be DEVICE_TO_DEVICE when is_host_transfer is false}}
  %0:2 = "stablehlo.recv"(%token) {
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 3  // Host to device channel
    >
  } : (!stablehlo.token) -> (tensor<3x4xi32>, tensor<i32>)
  %1 =  "stablehlo.tuple"(%0#0, %0#1) : (tensor<3x4xi32>, tensor<i32>) -> tuple<tensor<3x4xi32>, tensor<i32>>
  func.return %1 : tuple<tensor<3x4xi32>, tensor<i32>>
}

// -----

func.func @recv_c1(%token: !stablehlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>> {
  // expected-error@+1 {{channel_type should be HOST_TO_DEVICE when is_host_transfer is true}}
  %0:2 = "stablehlo.recv"(%token) {
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 1  // Device to device channel
    >,
    is_host_transfer = true
  } : (!stablehlo.token) -> (tensor<3x4xi32>, tensor<i32>)
  %1 =  "stablehlo.tuple"(%0#0, %0#1) : (tensor<3x4xi32>, tensor<i32>) -> tuple<tensor<3x4xi32>, tensor<i32>>
  func.return %1 : tuple<tensor<3x4xi32>, tensor<i32>>
}

// -----

func.func @recv_c2(%token: !stablehlo.token) {
  // expected-error@+1 {{result is expected to be at least of size 1, but got 0}}
  "stablehlo.recv"(%token) {
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 1  // Device to device channel
    >
  } : (!stablehlo.token) -> ()
  func.return
}

// -----

func.func @recv_c3(%token: !stablehlo.token) -> (!stablehlo.token, !stablehlo.token) {
  // expected-error@+1 {{everything but the last element of result types is expected to be of tensor type, but got '!stablehlo.token'}}
  %0:2 = "stablehlo.recv"(%token) {
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 1  // Device to device channel
    >
  } : (!stablehlo.token) -> (!stablehlo.token, !stablehlo.token)
  func.return %0#0 : !stablehlo.token
}

// -----

func.func @recv_c4(%token: !stablehlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>> {
  // expected-error@+1 {{last element of result types is expected to be of token type, but got 'tensor<i32>'}}
  %0:2 = "stablehlo.recv"(%token) {
    channel_handle = #stablehlo.channel_handle<
      handle = 5,
      type = 3  // Host to device channel
    >,
    is_host_transfer = true
  } : (!stablehlo.token) -> (tensor<3x4xi32>, tensor<i32>)
  %1 =  "stablehlo.tuple"(%0#0, %0#1) : (tensor<3x4xi32>, tensor<i32>) -> tuple<tensor<3x4xi32>, tensor<i32>>
  func.return %1 : tuple<tensor<3x4xi32>, tensor<i32>>
}

// -----

// CHECK-LABEL: func @replica_id
func.func @replica_id() -> tensor<ui32> {
  %0 = "stablehlo.replica_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}

// -----

// CHECK-LABEL: func @partition_id
func.func @partition_id() -> tensor<ui32> {
  %0 = "stablehlo.partition_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}

// -----

// CHECK-LABEL: func @rng_bit_generator
func.func @rng_bit_generator(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>) {
  %0, %1 = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm DEFAULT>} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>)
  func.return %0, %1 : tensor<2xui64>, tensor<10x12xui32>
}

// -----

func.func @rng_bit_generator(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>) {
  // expected-error@+1 {{output state shape must be compatible with initial state shape. Got: 'tensor<2xui64>' and 'tensor<3xui64>'}}
  %0, %1 = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm DEFAULT>} : (tensor<2xui64>) -> (tensor<3xui64>, tensor<10x12xui32>)
  func.return %0, %1 : tensor<3xui64>, tensor<10x12xui32>
}

// -----

// CHECK-LABEL: func @rng_bit_generator_dynamic
func.func @rng_bit_generator_dynamic(%arg0: tensor<?xui64>) -> (tensor<?xui64>, tensor<10x12xui32>) {
  %0, %1 = "stablehlo.rng_bit_generator"(%arg0) {rng_algorithm = #stablehlo<rng_algorithm DEFAULT>} : (tensor<?xui64>) -> (tensor<?xui64>, tensor<10x12xui32>)
  func.return %0, %1 : tensor<?xui64>, tensor<10x12xui32>
}

// -----

// CHECK-LABEL: func @rng_normal
func.func @rng_normal(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<2x3x5xf32> {
  %cst = stablehlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %0 = "stablehlo.rng"(%arg0, %arg1, %cst) {rng_distribution = #stablehlo<rng_distribution NORMAL>}: (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

// CHECK-LABEL: func @rng_normal_no_constant
func.func @rng_normal_no_constant(%a: tensor<f32>, %b: tensor<f32>, %shape: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.rng"(%a, %b, %shape) {rng_distribution = #stablehlo<rng_distribution NORMAL>}: (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @rng_normal_invalid_shape(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  %cst = stablehlo.constant dense<7> : tensor<1xi64>
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error @+1 {{inferred type(s) 'tensor<7xf32>' are incompatible with return type(s) of operation 'tensor<12xf32>'}}
  %0 = "stablehlo.rng"(%arg0, %arg1, %cst) {rng_distribution = #stablehlo<rng_distribution NORMAL>}: (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<12xf32>
  func.return
}

// -----

func.func @rng_normal_invalid_mu_rank(%mu: tensor<1xf32>, %sigma: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = stablehlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error-re@+1 {{#0 must be 0D tensor of {{.*}}, but got 'tensor<1xf32>'}}
  %0 = "stablehlo.rng"(%mu, %sigma, %shape) {rng_distribution = #stablehlo<rng_distribution NORMAL>}: (tensor<1xf32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

func.func @rng_normal_invalid_sigma_rank(%mu: tensor<f32>, %sigma: tensor<1xf32>) -> tensor<2x3x5xf32> {
  %shape = stablehlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error-re@+1 {{#1 must be 0D tensor of {{.*}}, but got 'tensor<1xf32>'}}
  %0 = "stablehlo.rng"(%mu, %sigma, %shape) {rng_distribution = #stablehlo<rng_distribution NORMAL>}: (tensor<f32>, tensor<1xf32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

func.func @rng_normal_invalid_shape_rank(%mu: tensor<f32>, %sigma: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = stablehlo.constant dense<[[2, 3, 5]]> : tensor<1x3xi64>
  // expected-error-re@+1 {{operand #2 must be statically shaped 1-dimensional tensor of {{.*}}, but got 'tensor<1x3xi64>'}}
  %0 = "stablehlo.rng"(%mu, %sigma, %shape) {rng_distribution = #stablehlo<rng_distribution NORMAL>}: (tensor<f32>, tensor<f32>, tensor<1x3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

func.func @rng_normal_invalid_type(%arg0: tensor<complex<f32>>, %arg1: tensor<f32>) {
  %cst = stablehlo.constant dense<7> : tensor<1xi64>
  // expected-error-re@+1 {{#0 must be 0D tensor of {{.*}}, but got 'tensor<complex<f32>>'}}
  %0 = "stablehlo.rng"(%arg0, %arg1, %cst) {rng_distribution = #stablehlo<rng_distribution NORMAL>}: (tensor<complex<f32>>, tensor<f32>, tensor<1xi64>) -> tensor<7xf32>
  func.return
}

// -----

// CHECK-LABEL: func @rng_uniform
func.func @rng_uniform(%a: tensor<f32>, %b: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = stablehlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %0 = "stablehlo.rng"(%a, %b, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>}: (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

// CHECK-LABEL: func @rng_uniform_no_constant
func.func @rng_uniform_no_constant(%a: tensor<f32>, %b: tensor<f32>, %shape: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.rng"(%a, %b, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>}: (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @rng_uniform_invalid_shape(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<7xi64>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error @+1 {{inferred type(s) 'tensor<?x?x?x?x?x?x?xf32>' are incompatible with return type(s) of operation 'tensor<?xf32>'}}
  %0 = "stablehlo.rng"(%arg0, %arg1, %arg2) {rng_distribution = #stablehlo<rng_distribution UNIFORM>}: (tensor<f32>, tensor<f32>, tensor<7xi64>) -> tensor<?xf32>
  func.return
}

// -----

func.func @rng_uniform_invalid_a_rank(%a: tensor<1xf32>, %b: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = stablehlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error-re@+1 {{operand #0 must be 0D tensor of {{.*}}, but got 'tensor<1xf32>'}}
  %0 = "stablehlo.rng"(%a, %b, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>}: (tensor<1xf32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}


// -----

func.func @rng_uniform_invalid_b_rank(%a: tensor<f32>, %b: tensor<1xf32>) -> tensor<2x3x5xf32> {
  %shape = stablehlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error-re@+1 {{operand #1 must be 0D tensor of {{.*}}, but got 'tensor<1xf32>'}}
  %0 = "stablehlo.rng"(%a, %b, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>}: (tensor<f32>, tensor<1xf32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

func.func @rng_uniform_invalid_shape_rank(%a: tensor<f32>, %b: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = stablehlo.constant dense<[[2, 3, 5]]> : tensor<1x3xi64>
  // expected-error-re@+1 {{operand #2 must be statically shaped 1-dimensional tensor of {{.*}}, but got 'tensor<1x3xi64>'}}
  %0 = "stablehlo.rng"(%a, %b, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>}: (tensor<f32>, tensor<f32>, tensor<1x3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

func.func @rng_uniform_invalid_type(%a: tensor<complex<f32>>, %b: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = stablehlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error-re@+1 {{operand #0 must be 0D tensor of {{.*}}, but got 'tensor<complex<f32>>'}}
  %0 = "stablehlo.rng"(%a, %b, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>}: (tensor<complex<f32>>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

// CHECK-LABEL: func @select
func.func @select(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @select_scalar_pred
func.func @select_scalar_pred(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @select_cast_compatible_types
func.func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<?x?xi32>, %arg2: tensor<2x3xi32>) -> tensor<?x?xi32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<?x?xi32>, tensor<2x3xi32>) -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @select_cast_compatible_types
func.func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @select_cast_compatible_types
func.func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<2x?xi32>, %arg2: tensor<?x3xi32>) -> tensor<?x?xi32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x?xi32>, tensor<?x3xi32>) -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @select_cast_compatible_types
func.func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<?x3xi32>, %arg2: tensor<2x?xi32>) -> tensor<?x?xi32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<?x3xi32>, tensor<2x?xi32>) -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @select_scalar_x_y
func.func @select_scalar_x_y(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @select_c1(%arg0: tensor<3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires the same shape for all operands}}
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

func.func @select_c2(%arg0: tensor<3xi1>, %arg1: tensor<2x4xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires compatible types for non-predicate operands}}
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x4xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

func.func @select_c2(%arg0: tensor<i1>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>) -> tensor<2x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<2x3xf32>' are incompatible with return type(s) of operation 'tensor<2x3xi32>'}}
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = "stablehlo.slice"(%arg0) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 2>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c2(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{the number of elements in start_indices (3) does not match the rank of the operand (2)}}
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 0, 0>,
    limit_indices = array<i64: 2, 4, 0>,
    strides = array<i64: 1, 2, 0>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c3(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{negative start index -1 in dimension 0}}
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: -1, 0>,
    limit_indices = array<i64: 2, 4>,
    strides = array<i64: 1, 2>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c3(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{limit index 5 is larger than dimension size 4 in dimension 1}}
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 0>,
    limit_indices = array<i64: 2, 5>,
    strides = array<i64: 1, 2>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c3(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{start index 3 is larger than limit index 2 in dimension 1}}
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 3>,
    limit_indices = array<i64: 2, 2>,
    strides = array<i64: 1, 2>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c4(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{stride must be positive but got 0 in dimension 0}}
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 0>,
    limit_indices = array<i64: 2, 4>,
    strides = array<i64: 0, 2>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @slice_dynamic_dim
func.func @slice_dynamic_dim(%arg0: tensor<3x?xi32>) -> tensor<1x?xi32> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 1>,
    limit_indices = array<i64: 2, 2>,
    strides = array<i64: 1, 1>
  } : (tensor<3x?xi32>) -> tensor<1x?xi32>
  func.return %0 : tensor<1x?xi32>
}

// -----

func.func @send_c1(%arg0: tensor<2x2xi64>, %arg1: !stablehlo.token) -> !stablehlo.token {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{channel_type should be DEVICE_TO_DEVICE when is_host_transfer is false}}
  %0 = "stablehlo.send"(%arg0, %arg1) {
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>
  } : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

// -----

func.func @send_c1(%arg0: tensor<2x2xi64>, %arg1: !stablehlo.token) -> !stablehlo.token {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{channel_type should be DEVICE_TO_HOST when is_host_transfer is true}}
  %0 = "stablehlo.send"(%arg0, %arg1) {
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
    is_host_transfer = true
  } : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

// -----

// CHECK-LABEL: func @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c2(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has mismatched number of slice sizes (1) and number of start indices (2)}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 4>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c2(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has mismatched number of start indices (1) and the rank of operand (2)}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1) {slice_sizes = array<i64: 1>} : (tensor<3x4xi32>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c3(%arg0: tensor<3x4xi32>, %arg1: tensor<i32>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{start indices must have same element type}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<3x4xi32>, tensor<i32>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c4(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has negative size index to dynamic slice: -1}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: -1, 4>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c4(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has slice size 10 greater than dimension size 4 in dimension 1 of operand}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 10>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_c5(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<2x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<1x4xi32>' are incompatible with return type(s) of operation 'tensor<2x4xi32>'}}
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x4xi32>
  func.return %0 : tensor<2x4xi32>
}

// -----

// CHECK-LABEL: func @dynamic_slice_dynamic_dim
func.func @dynamic_slice_dynamic_dim(%arg0: tensor<?x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = array<i64: 1, 4>} : (tensor<?x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @dynamic_update_slice
func.func @dynamic_update_slice(%operand: tensor<3x4xi64>, %update: tensor<1x4xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

func.func @dynamic_update_slice_c1(%operand: tensor<3x4xi64>, %update: tensor<1x4xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x5xi64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{op inferred type(s) 'tensor<3x4xi64>' are incompatible with return type(s) of operation 'tensor<3x5xi64>'}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<3x5xi64>
  func.return %0 : tensor<3x5xi64>
}

// -----

func.func @dynamic_update_slice_c3(%operand: tensor<3x4xi64>, %update: tensor<2xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{update rank does not match operand rank: 1 vs 2.}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<2xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

func.func @dynamic_update_slice_c4(%operand: tensor<3x4xi64>, %update: tensor<1x2xi64>, %start_indices0: tensor<i64>) -> tensor<3x4xi64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects number of start_indices to match operand rank: 1 vs 2.}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0) : (tensor<3x4xi64>, tensor<1x2xi64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

func.func @dynamic_update_slice_c5(%operand: tensor<11x3x4xi32>, %update: tensor<1x3x4xi32>, %start_indices0: tensor<i32>, %start_indices1: tensor<i64>, %start_indices2: tensor<i64>) -> tensor<11x3x4xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{start indices must have same element type}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1, %start_indices2) : (tensor<11x3x4xi32>, tensor<1x3x4xi32>, tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<11x3x4xi32>
  func.return %0 : tensor<11x3x4xi32>
}

// -----

func.func @dynamic_update_slice_c6(%operand: tensor<3x4xi64>, %update: tensor<1x5xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects size at dimension 1 of update to be in range [0, 4]. Got: 5.}}
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<1x5xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

// CHECK-LABEL: @dynamic_update_slice_dynamic_dim
func.func @dynamic_update_slice_dynamic_dim(%operand: tensor<?x4xi64>, %update: tensor<1x4xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<?x4xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

// CHECK-LABEL: func @dynamic_update_slice_dynamic_sizes
func.func @dynamic_update_slice_dynamic_sizes(%operand: tensor<?x4xi64>, %update: tensor<1x?xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<?x4xi64> {
  %0 = "stablehlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<?x4xi64>, tensor<1x?xi64>, tensor<i64>, tensor<i64>) -> tensor<?x4xi64>
  func.return %0 : tensor<?x4xi64>
}

// -----

// CHECK-LABEL: func @transpose
func.func @transpose(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 2>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0: tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_ranked(%arg0: tensor<?x?x?x?xi32>) ->  tensor<?x?x?x?xi32> {
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 2>} : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  func.return %0: tensor<?x?x?x?xi32>
}

// -----

func.func @transpose_missing_permutation(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  // expected-error@+1 {{requires attribute 'permutation'}}
  %0 = "stablehlo.transpose"(%arg0) {} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0: tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_bad_permutations_size(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{TransposeOp operand rank 4 does not match permutation size 1}}
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0: tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_bad_permutation(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{attribute permutation must be a permutation of [0, 1, 2, 3] but got 1, 0, 3, 9}}
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 9>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0: tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_operand_result_rank_mismatch(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{op inferred type(s) 'tensor<2x1x4x3xi32>' are incompatible with return type(s) of operation 'tensor<2xi32>'}}
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 2>} : (tensor<1x2x3x4xi32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// -----

func.func @transpose_operand_result_permutation_mismatch(%arg0: tensor<1x?x3x?xi32>) ->  tensor<?x2x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{op inferred type(s) 'tensor<?x1x?x3xi32>' are incompatible with return type(s) of operation 'tensor<?x2x?x?xi32>}}
  %0 = "stablehlo.transpose"(%arg0) {permutation = array<i64: 1, 0, 3, 2>} : (tensor<1x?x3x?xi32>) -> tensor<?x2x?x?xi32>
  func.return %0: tensor<?x2x?x?xi32>
}

// -----

// CHECK-LABEL: func @triangular_solve
func.func @triangular_solve(%arg0: tensor<10x5x4x4xf32>, %arg1: tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x5x4x4xf32>, tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32>
  func.return %0 : tensor<10x5x4x4xf32>
}

// -----

// CHECK-LABEL: func @triangular_solve_dynamic_dims_minor
func.func @triangular_solve_dynamic_dims_minor(%arg0: tensor<10x5x?x4xf32>, %arg1: tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x5x?x4xf32>, tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32>
  func.return %0 : tensor<10x5x4x4xf32>
}

// -----

// CHECK-LABEL: func @triangular_solve_dynamic_dims_shared
func.func @triangular_solve_dynamic_dims_shared(%arg0: tensor<10x5x4x?xf32>, %arg1: tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x5x4x?xf32>, tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32>
  func.return %0 : tensor<10x5x4x4xf32>
}

// -----

// CHECK-LABEL: func @triangular_solve_dynamic_dims_batch
func.func @triangular_solve_dynamic_dims_batch(%arg0: tensor<?x5x4x4xf32>, %arg1: tensor<10x?x4x4xf32>) -> tensor<10x5x4x4xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<?x5x4x4xf32>, tensor<10x?x4x4xf32>) -> tensor<10x5x4x4xf32>
  func.return %0 : tensor<10x5x4x4xf32>
}

// -----

func.func @triangular_solve_rank_less_than_2(%arg0: tensor<4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operand 'a' must have rank >= 2, but got 'tensor<4xf32>'}}
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

// -----

func.func @triangular_solve_unequal_minor_dims_a(%arg0: tensor<4x3xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{two minor dimensions of operand 'a' must be compatible, but got 'tensor<4x3xf32>'}}
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

// -----

func.func @triangular_solve_unequal_rank(%arg0: tensor<10x4x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operands must have equal rank, but got 'tensor<10x4x4xf32>' and 'tensor<4x3xf32>'}}
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

// -----

func.func @triangular_solve_mismatch_shared_dim(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{shared dimension of operands 'a' and 'b' must be compatible, but got 'tensor<4x4xf32>' and 'tensor<3x4xf32>'}}
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// -----

func.func @triangular_solve_mismatch_leading_dims(%arg0: tensor<10x5x4x4xf32>, %arg1: tensor<10x6x4x3xf32>) -> tensor<10x6x4x3xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{batch dimensions of the operands must be compatible, but got 'tensor<10x5x4x4xf32>' and 'tensor<10x6x4x3xf32>'}}
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x5x4x4xf32>, tensor<10x6x4x3xf32>) -> tensor<10x6x4x3xf32>
  func.return %0 : tensor<10x6x4x3xf32>
}

// -----

func.func @triangular_solve_mismatch_result_and_b_type(%arg0: tensor<4x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x4xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<4x3xf32>' are incompatible with return type(s) of operation 'tensor<4x4xf32>'}}
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @triangular_solve(%arg0: tensor<10x5x4x4xf32>, %arg1: tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Invalid transpose option value for triangular solve}}
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE_INVALID>, unit_diagonal = true} : (tensor<10x5x4x4xf32>, tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32>
  func.return %0 : tensor<10x5x4x4xf32>
}

// -----

// CHECK-LABEL: func @tuple
func.func @tuple(%arg0: tensor<1xi32>, %arg1: !stablehlo.token, %arg2: tuple<>) -> tuple<tensor<1xi32>, !stablehlo.token, tuple<>> {
  %0 = "stablehlo.tuple"(%arg0, %arg1, %arg2) : (tensor<1xi32>, !stablehlo.token, tuple<>) -> tuple<tensor<1xi32>, !stablehlo.token, tuple<>>
  func.return %0: tuple<tensor<1xi32>, !stablehlo.token, tuple<>>
}

// -----

// CHECK-LABEL: func @get_tuple_element
func.func @get_tuple_element(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  %0 = "stablehlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @get_tuple_element_c1(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  // expected-error@+1 {{op attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
  %0 = "stablehlo.get_tuple_element"(%arg0) {index = -1 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @get_tuple_element_c1(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{index 2 is out of bounds of operand with size 2}}
  %0 = "stablehlo.get_tuple_element"(%arg0) {index = 2 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @and_i32_type
func.func @and_i32_type(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = "stablehlo.and"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: func @or_i1_type
func.func @or_i1_type(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = "stablehlo.or"(%arg0, %arg1) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

func.func @or_invalid_f32_type(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{but got 'tensor<4xf32>'}}
  %0 = "stablehlo.or"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

func.func @floor_invalid_i32_type(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // expected-error-re@+1 {{must be ranked tensor of {{.*}}, but got 'tensor<4xi32>'}}
  %0 = "stablehlo.floor"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// Verifiers HLO constant op custom printing and parsing.
// CHECK-LABEL: func @constants
func.func @constants() -> () {
  // CHECK: stablehlo.constant dense<0> : tensor<i32>
  %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> (tensor<i32>)

  // CHECK: stablehlo.constant {extra_attr = 3 : i32} dense<0> : tensor<i32>
  %1 = "stablehlo.constant"() {extra_attr = 3 : i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
  func.return
}

// -----

func.func @constant_invalid() -> () {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.constant' op inferred type(s) 'tensor<i32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> (tensor<3xi32>)
  func.return
}

// -----

func.func @constant_invalid() -> () {
  // expected-error@+1 {{op result #0 must be statically shaped tensor}}
  %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<?xi32>
  func.return
}

// -----

func.func @constant_invalid() -> () {
  // expected-error@+1 {{elements literal type must have static shape}}
  %0 = "stablehlo.constant"() <{value = dense<1> : tensor<?xi32>}> : () -> tensor<?xi32>
  func.return
}

// -----

// CHECK-LABEL: func @sort
func.func @sort(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_c4(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{dimension attribute value must be in range [-2, 2), but found -3}}
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = -3 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_c4(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{dimension attribute value must be in range [-2, 2), but found 2}}
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 2 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_c5(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator block should have 4 arguments}}
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_c5(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator block argument #0 should be of type 'tensor<f32>' but got 'tensor<i32>'}}
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_c5(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator must return single output but got 2}}
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%7, %7) : (tensor<i1>, tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_c5(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator must return tensor<i1> but got 'tensor<i32>'}}
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    "stablehlo.return"(%arg2) : (tensor<i32>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_dynamism(%input0: tensor<?x16xf32>, %input1: tensor<16x16xi32>) {
  %0:2 = "stablehlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "stablehlo.compare"(%arg0, %arg1) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<?x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @reshape_invalid_shapes(%operand: tensor<2x4xf32>) -> tensor<3x3xf32> {
  // expected-error @+1 {{number of output elements (9) doesn't match expected number of elements (8)}}
  %0 = "stablehlo.reshape"(%operand) : (tensor<2x4xf32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}

// -----

// CHECK-LABEL: func @reverse
func.func @reverse(%operand: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "stablehlo.reverse"(%operand) {
    dimensions = array<i64: 0, 1>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @reverse_c2(%operand: tensor<3x2xi32>) -> tensor<3x2xi32> {
  // expected-error @+1 {{dimensions should be unique. Got: 0, 0}}
  %0 = "stablehlo.reverse"(%operand) {
    dimensions = array<i64: 0, 0>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @reverse_c3(%operand: tensor<?xi32>) -> tensor<?xi32> {
  // expected-error @+1 {{all dimensions should be non-negative. Got dimension: -1.}}
  %0 = "stablehlo.reverse"(%operand) {
    dimensions = array<i64: -1>
  } : (tensor<?xi32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

func.func @reverse_c3(%operand: tensor<3x2xi32>) -> tensor<3x2xi32> {
  // expected-error @+1 {{all dimensions should be non-negative. Got dimension: -1.}}
  %0 = "stablehlo.reverse"(%operand) {
    dimensions = array<i64: -1>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @reverse_c3(%operand: tensor<3x2xi32>) -> tensor<3x2xi32> {
  // expected-error @+1 {{all dimensions should be between [0, 2). Got dimension: 2.}}
  %0 = "stablehlo.reverse"(%operand) {
    dimensions = array<i64: 2>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

// CHECK-LABEL: func @dot_general
func.func @dot_general(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x5xf32>) -> tensor<2x4x5xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x3x4xf32>, tensor<2x3x5xf32>) -> tensor<2x4x5xf32>
  func.return %0 : tensor<2x4x5xf32>
}

// -----

// CHECK-LABEL: func @dot_general
func.func @dot_general(%arg0: tensor<1x?x1x?xf32>, %arg1: tensor<?x1x?x1x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2, 3],
      rhs_contracting_dimensions = [2, 3]
    >
  } : (tensor<1x?x1x?xf32>, tensor<?x1x?x1x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @dot_general_algorithm
func.func @dot_general_algorithm_tf32(%arg0: tensor<2x2x2xi64>, %arg1: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    algorithm = #stablehlo.dot_algorithm<lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
  }> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>  return %0 : tensor<2x2x2xi64>
}

// -----

func.func @dot_general_c1(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{lhs and rhs should have the same number of batching dimensions}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c2(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>{
  // expected-error @+1 {{lhs and rhs should have the same number of contracting dimensions}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c3(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{has duplicated dimension from lhs_batching_dimensions and lhs_contracting_dimensions: 0}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c4(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{has duplicated dimension from rhs_batching_dimensions and rhs_contracting_dimensions: 0}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c5(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{lhs_batching_dimensions value: -1 is out of range: [0, 3)}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [-1],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c5(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{lhs_batching_dimensions value: 3 is out of range: [0, 3)}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [3],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c6(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{lhs_contracting_dimensions value: -1 is out of range: [0, 3)}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [-1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c6(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{lhs_contracting_dimensions value: 3 is out of range: [0, 3)}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c7(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{rhs_batching_dimensions value: -1 is out of range: [0, 3)}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [-1],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c7(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{rhs_batching_dimensions value: 3 is out of range: [0, 3)}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [3],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c8(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{rhs_contracting_dimensions value: -1 is out of range: [0, 3)}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [-1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c8(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{rhs_contracting_dimensions value: 3 is out of range: [0, 3)}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [3]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c9(%arg0: tensor<2x?x?xf32>, %arg1: tensor<3x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{batching dimension sizes must match for lhs/rhs}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x?x?xf32>, tensor<3x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c10(%arg0: tensor<?x2x?xf32>, %arg1: tensor<?x3x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error @+1 {{contracting dimension sizes must match for lhs/rhs}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x2x?xf32>, tensor<?x3x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dot_general_c11(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x5xf32>) -> tensor<2x4x5xf32> {
  // expected-error@+1 {{expects precision config to be empty or have <= 2 elements}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x3x4xf32>, tensor<2x3x5xf32>) -> tensor<2x4x5xf32>
  func.return %0 : tensor<2x4x5xf32>
}

// -----

func.func @dot_general_c21(%arg0: tensor<2x2x2xi64>, %arg1: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {
  // expected-error @+1 {{must specify DEFAULT precision config when algorithm is set}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>],
    algorithm = #stablehlo.dot_algorithm<lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
  }> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>  return %0 : tensor<2x2x2xi64>
}

// -----

func.func @dot_general_c22(%arg0: tensor<2x2x2xi64>, %arg1: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {
  // expected-error@+3 {{lhs component count must be positive}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    algorithm = #stablehlo.dot_algorithm<lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = -1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
  }> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>  return %0 : tensor<2x2x2xi64>
}

// -----

func.func @dot_general_c23(%arg0: tensor<2x2x2xi64>, %arg1: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {
  // expected-error@+3 {{rhs component count must be positive}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    algorithm = #stablehlo.dot_algorithm<lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 0, num_primitive_operations = 1, allow_imprecise_accumulation = false>
  }> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>  return %0 : tensor<2x2x2xi64>
}

// -----

func.func @dot_general_c24(%arg0: tensor<2x2x2xi64>, %arg1: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {
  // expected-error@+3 {{num primitive operations must be positive}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    algorithm = #stablehlo.dot_algorithm<lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 0, allow_imprecise_accumulation = false>
  }> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>  return %0 : tensor<2x2x2xi64>
}

// -----

func.func @dot_general_i8(%arg0: tensor<2x2x2xi64>, %arg1: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {
  // expected-error @+3 {{dot algorithm not known to be supported on any hardware}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    algorithm = #stablehlo.dot_algorithm<lhs_precision_type = i8, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
  }> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>  return %0 : tensor<2x2x2xi64>
}

// -----

func.func @dot_general_i9(%arg0: tensor<2x2x2xi64>, %arg1: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {
  // expected-error @+3 {{dot algorithm not known to be supported on any hardware}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    algorithm = #stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = i8, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
  }> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>  return %0 : tensor<2x2x2xi64>
}

// -----

func.func @dot_general_i10(%arg0: tensor<2x2x2xi64>, %arg1: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {
  // expected-error @+3 {{dot algorithm not known to be supported on any hardware}}
  %0 = "stablehlo.dot_general"(%arg0, %arg1) <{
    dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
    algorithm = #stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = i8, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>
  }> : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>  return %0 : tensor<2x2x2xi64>
}

// -----

// CHECK-LABEL: func @dot_general_one_element_precision_config
func.func @dot_general_one_element_precision_config(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x5xf32>) -> tensor<2x4x5xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<2x3x4xf32>, tensor<2x3x5xf32>) -> tensor<2x4x5xf32>
  func.return %0 : tensor<2x4x5xf32>
}

// -----

func.func @dynamic_pad(
  %arg: tensor<4xf64>, %padding_value: tensor<f64>,
  %padding_low: tensor<1xi32>, %padding_high: tensor<1xi32>, %interior_padding: tensor<1xi32>
) {
  %0 = stablehlo.dynamic_pad %arg, %padding_value, %padding_low, %padding_high, %interior_padding
         : (tensor<4xf64>, tensor<f64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf64>
  func.return
}

// -----

func.func @dynamic_pad_c2(
  %arg: tensor<4xf64>, %padding_value: tensor<f64>,
  %padding_low: tensor<2xi32>, %padding_high: tensor<2xi32>, %interior_padding: tensor<2xi32>
) {
  // expected-error@+1 {{padding operands size (2) must match operand rank (1)}}
  %0 = stablehlo.dynamic_pad %arg, %padding_value, %padding_low, %padding_high, %interior_padding
         : (tensor<4xf64>, tensor<f64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?xf64>
  func.return
}

// -----

func.func @dynamic_pad_c3(
  %arg: tensor<4xf64>, %padding_value: tensor<f64>,
  %padding_low: tensor<1xi32>, %padding_high: tensor<1xi32>
) {
  %interior_padding = stablehlo.constant dense<-1> : tensor<1xi32>
  // expected-error@+1 {{interior_padding must be non-negative, but got -1}}
  %0 = stablehlo.dynamic_pad %arg, %padding_value, %padding_low, %padding_high, %interior_padding
         : (tensor<4xf64>, tensor<f64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf64>
  func.return
}

// -----

func.func @dynamic_pad_c4(%arg: tensor<4xf64>, %padding_value: tensor<f64>) {
  %padding = stablehlo.constant dense<1> : tensor<1xi32>
  // expected-error@+1 {{expected output dimension at index 0 to equal 9, but got 4}}
  %0 = stablehlo.dynamic_pad %arg, %padding_value, %padding, %padding, %padding
         : (tensor<4xf64>, tensor<f64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xf64>
  func.return
}

// -----

func.func @dynamic_reshape(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "stablehlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @dynamic_reshape_c1(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<?x?xf64> {
  // expected-error @+1 {{expects operand and result to have compatible element type}}
  %0 = "stablehlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @dynamic_reshape_c2(%arg0: tensor<11xf32>, %shape: tensor<2xindex>) -> tensor<2x5xf32> {
  // expected-error @+1 {{number of output elements (10) doesn't match expected number of elements}}
  %0 = "stablehlo.dynamic_reshape"(%arg0, %shape) : (tensor<11xf32>, tensor<2xindex>) -> tensor<2x5xf32>
  func.return %0 : tensor<2x5xf32>
}

// -----

func.func @dynamic_reshape_incompatible_shapes(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<?xf32> {
  // expected-error @+1 {{result should have a rank equal to the number of elements in output_shape}}
  %0 = "stablehlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @dynamic_reshape_output_shape_mismatching_size(%arg0: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-error@+2 {{output shape [2, 2] is incompatible with return type of operation 'tensor<1x4xf32>'}}
  %0 = stablehlo.constant dense<[2, 2]> : tensor<2xi64>
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}

// -----

func.func @dynamic_reshape_output_shape_matches_result(%arg0: tensor<4xf32>) -> tensor<1x4xf32> {
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}

// -----

func.func @dynamic_reshape_output_shape_compatible_with_result(%arg0: tensor<4xf32>) -> tensor<?x?xf32> {
  %0 = stablehlo.constant dense<[1, 4]> : tensor<2xi64>
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<4xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

func.func @dynamic_reshape_dynamic_output_shape(%arg0: tensor<?xf32>, %shape: tensor<?xindex>) -> tensor<1x4xf32> {
  // expected-error@+1 {{op operand #1 must be statically shaped}}
  %0 = "stablehlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<?xindex>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}

// -----

func.func @dynamic_reshape_input_count_mismatch_shape_count(%arg0: tensor<2x5xf32>) -> tensor<?x?x?xf32> {
  %0 = stablehlo.constant dense<[2, 3, 4]> : tensor<3xi32>
  // expected-error@+1 {{output_shape is incompatible with input type of operation: input has 10 elements, but output_shape has 24}}
  %1 = stablehlo.dynamic_reshape %arg0, %0 : (tensor<2x5xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

// -----

func.func @cbrt(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "stablehlo.cbrt"(%arg) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func @bitcast_convert
func.func @bitcast_convert(%arg: tensor<2xf32>) -> tensor<2x4xi8> {
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<2xf32>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}

// -----

// CHECK-LABEL: func @bitcast_convert
func.func @bitcast_convert(%arg: tensor<2x4xi8>) -> tensor<2xf32> {
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<2x4xi8>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @bitcast_convert
func.func @bitcast_convert(%arg: tensor<complex<f64>>) -> tensor<2xcomplex<f32>> {
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<complex<f64>>) -> tensor<2xcomplex<f32>>
  return %0 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @bitcast_convert
func.func @bitcast_convert(%arg: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @bitcast_convert
func.func @bitcast_convert(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @bitcast_convert_c1(%arg: tensor<2xf64>) -> tensor<3xi64> {
  // expected-error@+1 {{operand and result shapes must match except for the innermost dimension of the shape with the smaller element type. Got: 'tensor<2xf64>' and 'tensor<3xi64>'.}}
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<2xf64>) -> tensor<3xi64>
  return %0 : tensor<3xi64>
}

// -----

func.func @bitcast_convert_c1(%arg: tensor<f64>) -> tensor<f32> {
  // expected-error@+1 {{rank of smaller element type (0) should be 1 more than rank of larger element type (0), but 0 != 0 + 1.}}
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<f64>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @bitcast_convert_c1(%arg: tensor<2xf64>) -> tensor<4x2xf32> {
  // expected-error@+1 {{operand and result shapes must match except for the innermost dimension of the shape with the smaller element type. Got: 'tensor<2xf64>' and 'tensor<4x2xf32>'.}}
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<2xf64>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

func.func @bitcast_convert_c1(%arg: tensor<2xf64>) -> tensor<2x4xf32> {
  // expected-error@+1 {{requires compatible bit widths. Got: 'tensor<2xf64>' and 'tensor<2x4xf32>', but 32 * 4 != 64.}}
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<2xf64>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @bitcast_convert_c2(%arg: tensor<2x4xcomplex<f32>>) -> tensor<2x2xf64> {
  // expected-error@+1 {{cannot convert between real and complex types}}
  %0 = "stablehlo.bitcast_convert"(%arg) : (tensor<2x4xcomplex<f32>>) -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}

// -----

func.func @reduce_precision(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "stablehlo.reduce_precision"(%arg) {exponent_bits=2 : i32, mantissa_bits=3 : i32} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

func.func @reduce_precision_c2(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{op attribute 'exponent_bits' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive}}
  %0 = "stablehlo.reduce_precision"(%arg) {exponent_bits=0 : i32, mantissa_bits=3 : i32} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

func.func @reduce_precision_c3(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{op attribute 'mantissa_bits' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
  %0 = "stablehlo.reduce_precision"(%arg) {exponent_bits=1 : i32, mantissa_bits=-1 : i32} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<3x2x4x9xi32>, %start_indices : tensor<1x3x5x2xi32>) -> tensor<1x3x5x8xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [3],
      collapsed_slice_dims = [1, 2],
      operand_batching_dims = [0],
      start_indices_batching_dims = [1],
      start_index_map = [1, 2],
      index_vector_dim = 3
    >,
    slice_sizes = array<i64: 1, 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<3x2x4x9xi32>, tensor<1x3x5x2xi32>) -> tensor<1x3x5x8xi32>
  func.return %res : tensor<1x3x5x8xi32>
}

// -----

// CHECK: gather
func.func @gather(%operand : tensor<?x?x?x?x?x?x?x?xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<8x?x7x1x6x1x?xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0, 2, 3, 4, 5],
      collapsed_slice_dims = [0, 1, 3],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8, 1, 7, 1, 6, 1>,
    indices_are_sorted = false
  } : (tensor<?x?x?x?x?x?x?x?xi32>, tensor<1x5x2xi32>) -> tensor<8x?x7x1x6x1x?xi32>
  func.return %res : tensor<8x?x7x1x6x1x?xi32>
}

// -----

// CHECK: gather
func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<?x?x?xi32>) -> tensor<1x5x8xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<?x?x?xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<?x?x?x?xi32>, %start_indices : tensor<1x3x5x2xi32>) -> tensor<1x3x5x8xi32> {
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [3],
      collapsed_slice_dims = [1, 2],
      operand_batching_dims = [0],
      start_indices_batching_dims = [1],
      start_index_map = [1, 2],
      index_vector_dim = 3
    >,
    slice_sizes = array<i64: 1, 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<?x?x?x?xi32>, tensor<1x3x5x2xi32>) -> tensor<1x3x5x8xi32>
  func.return %res : tensor<1x3x5x8xi32>
}

// -----

func.func @gather_c1(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{offset_dims size (2) plus collapse_slice_dims size (2) plus operand_batching_dims size (0) is not equal to operand rank (3)}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c1(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{offset_dims size (2) plus collapse_slice_dims size (1) plus operand_batching_dims size (1) is not equal to operand rank (3)}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2],
      collapsed_slice_dims = [0],
      operand_batching_dims = [0],
      start_indices_batching_dims = [0],
      start_index_map = [1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c2(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects index_vector_dim to be in range [0, rank-of('start_indices')] i.e. [0, 3]. got: -1.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = -1
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c2(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects index_vector_dim to be in range [0, rank-of('start_indices')] i.e. [0, 3]. got: 4.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 4
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c3(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{start_index_map size (1) is not equal to size of index dimension (2) of start_indices (2)}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c3(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{start_index_map size (2) is not equal to size of index dimension (3) of start_indices (1)}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 3
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c4(%operand : tensor<16x11xi32>, %start_indices : tensor<5x2xi32>) -> tensor<5x8x6xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects offset_dims to be sorted, got: [2, 1]}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 1],
      collapsed_slice_dims = [],
      start_index_map = [0, 1],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 8, 6>,
    indices_are_sorted = false
  } : (tensor<16x11xi32>, tensor<5x2xi32>) -> tensor<5x8x6xi32>
  func.return %res : tensor<5x8x6xi32>
}

// -----

func.func @gather_c4(%operand : tensor<16x11xi32>, %start_indices : tensor<5x2xi32>) -> tensor<5x8x6xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects offset_dims to not repeat, got: [2, 2]}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 2],
      collapsed_slice_dims = [],
      start_index_map = [0, 1],
      index_vector_dim = 1
    >,
    slice_sizes = array<i64: 8, 6>,
    indices_are_sorted = false
  } : (tensor<16x11xi32>, tensor<5x2xi32>) -> tensor<5x8x6xi32>
  func.return %res : tensor<5x8x6xi32>
}

// -----

func.func @gather_c5(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of offset_dims to be in range [0, implied-result-rank) i.e. [0, 3). got: -1.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [-1],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c5(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of offset_dims to be in range [0, implied-result-rank) i.e. [0, 3). got: 3.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [3],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c6(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has duplicated dimension from collapsed_slice_dims and operand_batching_dims: 1}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [1, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c6(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x1xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has duplicated dimension from collapsed_slice_dims and operand_batching_dims: 0}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0],
      operand_batching_dims = [0],
      start_indices_batching_dims = [0],
      start_index_map = [1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x1xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c7(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects collapsed_slice_dims to be sorted, got: [1, 0]}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [1, 0],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c8(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of collapsed_slice_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: -1.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [-1, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c8(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of collapsed_slice_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: 17.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 17],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c9(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects that for each dim in collapsed_slice_dims, slice_sizes[dim] should be <= 1, but got 8}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 2],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @gather_c10(%operand : tensor<2x4x9xi32>, %start_indices : tensor<4x2xi32>) -> tensor<4x2x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects operand_batching_dims to be sorted, got: [1, 0]}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      operand_batching_dims = [1, 0],
      start_indices_batching_dims = [0, 1],
      start_index_map = [2],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<4x2xi32>) -> tensor<4x2x8xi32>
  func.return %res : tensor<4x2x8xi32>
}

// -----

func.func @gather_c11(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of operand_batching_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: -1.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [1],
      operand_batching_dims = [-1],
      start_indices_batching_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c11(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of operand_batching_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: 3.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [1],
      operand_batching_dims = [3],
      start_indices_batching_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c12(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects that for each dim in operand_batching_dims, slice_sizes[dim] should be <= 1, but got 2}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      operand_batching_dims = [0, 1],
      start_indices_batching_dims = [0, 1],
      start_index_map = [2],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 2, 1, 8>,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @gather_c13(%operand : tensor<2x4x9xi32>, %start_indices : tensor<4x2xi32>) -> tensor<4x2x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects start_indices_batching_dims to not repeat, got: [1, 0, 1]}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      operand_batching_dims = [0, 1],
      start_indices_batching_dims = [1, 0, 1],
      start_index_map = [2],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<4x2xi32>) -> tensor<4x2x8xi32>
  func.return %res : tensor<4x2x8xi32>
}

// -----

func.func @gather_c14(%operand : tensor<2x4x9xi32>, %start_indices : tensor<4x2xi32>) -> tensor<4x2x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of start_indices_batching_dims to be in range [0, rank-of('start_indices')) i.e. [0, 2). got: -1.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      operand_batching_dims = [0, 1],
      start_indices_batching_dims = [1, -1],
      start_index_map = [2],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<4x2xi32>) -> tensor<4x2x8xi32>
  func.return %res : tensor<4x2x8xi32>
}

// -----

func.func @gather_c14(%operand : tensor<2x4x9xi32>, %start_indices : tensor<4x2xi32>) -> tensor<4x2x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of start_indices_batching_dims to be in range [0, rank-of('start_indices')) i.e. [0, 2). got: 10.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      operand_batching_dims = [0, 1],
      start_indices_batching_dims = [1, 10],
      start_index_map = [2],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<4x2xi32>) -> tensor<4x2x8xi32>
  func.return %res : tensor<4x2x8xi32>
}

// -----

func.func @gather_c15(%operand : tensor<2x4x9xi32>, %start_indices : tensor<2x5x1xi32>) -> tensor<2x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects start_indices_batching_dims not to include index_vector_dim 2}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [1],
      operand_batching_dims = [0],
      start_indices_batching_dims = [0, 2],
      start_index_map = [1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<2x5x1xi32>) -> tensor<2x5x8xi32>
  func.return %res : tensor<2x5x8xi32>
}

// -----

func.func @gather_c16(%operand : tensor<2x4x9xi32>, %start_indices : tensor<2x5x1xi32>) -> tensor<2x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operand_batching_dims and start_indices_batching_dims should have the same size}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [1],
      operand_batching_dims = [0],
      start_indices_batching_dims = [0, 1],
      start_index_map = [1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<2x5x1xi32>) -> tensor<2x5x8xi32>
  func.return %res : tensor<2x5x8xi32>
}

// -----

func.func @gather_c17(%operand : tensor<2x4x9xi32>, %start_indices : tensor<2x5xi32>) -> tensor<2x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operand_batching_dims[1] and start_indices_batching_dims[1] must have compatible sizes, but got 4 and 5}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      operand_batching_dims = [0, 1],
      start_indices_batching_dims = [0, 1],
      start_index_map = [2],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<2x5xi32>) -> tensor<2x5x8xi32>
  func.return %res : tensor<2x5x8xi32>
}

// -----

func.func @gather_c18(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has duplicated dimension from start_index_map and operand_batching_dims: 0}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 0],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c18(%operand : tensor<2x4x9xi32>, %start_indices : tensor<2x5x2xi32>) -> tensor<2x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has duplicated dimension from start_index_map and operand_batching_dims: 0}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [1],
      operand_batching_dims = [0],
      start_indices_batching_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<2x5x2xi32>) -> tensor<2x5x8xi32>
  func.return %res : tensor<2x5x8xi32>
}

// -----

func.func @gather_c19(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of start_index_map to be in range [0, rank-of('operand')) i.e. [0, 3). got: -2.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [-2, -1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c19(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of start_index_map to be in range [0, rank-of('operand')) i.e. [0, 3). got: 3.}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 3],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c20(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{slice_sizes size (2) not equal to operand rank (3)}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather_c20(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{slice_sizes size (6) not equal to operand rank (3)}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8, 1, 2, 3>,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @gather_c21(%operand : tensor<?x?x2xi32>, %start_indices : tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{slice size (-1) is out of bounds for operand dimension (2) at index 2}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, -1>,
    indices_are_sorted = false
  } : (tensor<?x?x2xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @gather_c21(%operand : tensor<?x?x2xi32>, %start_indices : tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{slice size (8) is out of bounds for operand dimension (2) at index 2}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<?x?x2xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @gather_c22(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<1x5x8xi32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 1, 8>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<3xi32>
  func.return %res : tensor<3xi32>
}

// -----

func.func @gather_c22(%operand : tensor<2x4x9xi32>, %start_indices : tensor<4x5xi32>) -> tensor<3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<4x5x8xi32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      operand_batching_dims = [1],
      start_indices_batching_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0]
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 1, 8>
  } : (tensor<2x4x9xi32>, tensor<4x5xi32>) -> tensor<3xi32>
  func.return %res : tensor<3xi32>
}

// -----

func.func @gather_c22(%operand : tensor<?x?x?x?x?x?x?x?xi32>, %start_indices : tensor<?x?x?xi32>) -> tensor<3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<8x?x7x1x6x1x?xi32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %res = "stablehlo.gather"(%operand, %start_indices) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1, 3],
      index_vector_dim = 2,
      offset_dims = [0, 2, 3, 4, 5],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 1, 8, 1, 7, 1, 6, 1>
  } : (tensor<?x?x?x?x?x?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<3xi32>
  func.return %res : tensor<3xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>, %slice_sizes : tensor<3xi32>) -> tensor<1x5x8xi32> {
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?x?x?xi32> {
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?x?x?xi32> {
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c1(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>, %slice_sizes : tensor<3xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{offset_dims size (2) plus collapse_slice_dims size (2) plus operand_batching_dims size (0) is not equal to operand rank (3)}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1, 2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @dynamic_gather_c2(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects index_vector_dim to be in range [0, rank-of('start_indices')] i.e. [0, 3]. got: 4.}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 4,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  func.return %res : tensor<?xi32>
}

// -----

func.func @dynamic_gather_c3(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?x2xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{start_index_map size (1) is not equal to size of index dimension (2) of start_indices (2)}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0]
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?x2xi32>, tensor<3xi32>) -> tensor<?xi32>
  func.return %res : tensor<?xi32>
}

// -----

func.func @dynamic_gather_c4(%operand : tensor<2x4x9xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects offset_dims to be sorted, got: [2, 1]}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 2,
      offset_dims = [2, 1],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c5(%operand : tensor<2x4x9xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of offset_dims to be in range [0, implied-result-rank) i.e. [0, 3). got: -1.}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [-1],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c5(%operand : tensor<2x4x9xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of offset_dims to be in range [0, implied-result-rank) i.e. [0, 3). got: 3.}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [3],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c6(%operand : tensor<2x4x9xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has duplicated dimension from collapsed_slice_dims and operand_batching_dims: 1}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [1, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c7(%operand : tensor<2x4x9xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects collapsed_slice_dims to be sorted, got: [1, 0]}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [1, 0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c8(%operand : tensor<2x4x9xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of collapsed_slice_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: -1.}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [-1, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c8(%operand : tensor<2x4x9xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of collapsed_slice_dims to be in range [0, rank-of('operand')) i.e. [0, 3). got: 17.}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 17],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c9(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %slize_sizes = stablehlo.constant dense<[1,1,8]> : tensor<3xi32>
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects that for each dim in collapsed_slice_dims, slice_sizes[dim] should be <= 1, but got 8}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slize_sizes) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 2],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c12(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?xi32>) -> tensor<?x?x?xi32> {
  %slize_sizes = stablehlo.constant dense<[2,1,8]> : tensor<3xi32>
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects that for each dim in operand_batching_dims, slice_sizes[dim] should be <= 1, but got 2}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slize_sizes) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      operand_batching_dims = [0, 1],
      start_indices_batching_dims = [0, 1],
      start_index_map = [2],
      index_vector_dim = 2
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c18(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>, %slize_sizes : tensor<3xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{has duplicated dimension from start_index_map and operand_batching_dims: 0}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slize_sizes) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 0],
      index_vector_dim = 2
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @dynamic_gather_c19(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>, %slize_sizes : tensor<3xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of start_index_map to be in range [0, rank-of('operand')) i.e. [0, 3). got: -2.}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slize_sizes) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [-2, -1],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @dynamic_gather_c19(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>, %slize_sizes : tensor<3xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Expects each element of start_index_map to be in range [0, rank-of('operand')) i.e. [0, 3). got: 3.}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slize_sizes) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 3],
      index_vector_dim = 2
    >,
    slice_sizes = array<i64: 1, 1, 8>,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @dynamic_gather_c20(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<2xi32>) -> tensor<?x?x?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{slice_sizes size (2) not equal to operand rank (3)}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>, tensor<2xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c21(%operand : tensor<?x?x2xi32>, %start_indices : tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %slice_sizes = stablehlo.constant dense<[1,1,-1]> : tensor<3xi32>
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{slice size (-1) is out of bounds for operand dimension (2) at index 2}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    indices_are_sorted = false
  } : (tensor<?x?x2xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c21(%operand : tensor<?x?x2xi32>, %start_indices : tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %slice_sizes = stablehlo.constant dense<[1,1,8]> : tensor<3xi32>
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{slice size (8) is out of bounds for operand dimension (2) at index 2}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    indices_are_sorted = false
  } : (tensor<?x?x2xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// -----

func.func @dynamic_gather_c22(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>, %slice_sizes : tensor<3xi32>) -> tensor<3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<1x5x?xi32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<3xi32>
  func.return %res : tensor<3xi32>
}

// -----

func.func @dynamic_gather_c22(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>, %slice_sizes : tensor<3xi32>) -> tensor<3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<1x5x?xi32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<3xi32>
  func.return %res : tensor<3xi32>
}

// -----

func.func @dynamic_gather_c22(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<3xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<?x?x?xi32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<3xi32>
  func.return %res : tensor<3xi32>
}

// -----

func.func @dynamic_gather_c22(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<?xi32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<?x?x?xi32>' are incompatible with return type(s) of operation 'tensor<?xi32>'}}
  %res = "stablehlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<?xi32>
  func.return %res : tensor<?xi32>
}

// -----

func.func @get_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<i32> {
  %size = "stablehlo.get_dimension_size"(%I) {dimension = 2 : i64} : (tensor<1x128x512xf32>) -> tensor<i32>
  func.return %size : tensor<i32>
}

// -----

func.func @get_dimension_size_c1(%I: tensor<1x128x512xf32>) -> tensor<i32> {
  // expected-error@+1 {{op attribute 'dimension' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %size = "stablehlo.get_dimension_size"(%I) {dimension = -1 : i64} : (tensor<1x128x512xf32>) -> tensor<i32>
  func.return %size : tensor<i32>
}

// -----

func.func @get_dimension_size_c1(%I: tensor<1x128x512xf32>) -> tensor<i32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires dimension attribute in range [0, 3); found (3)}}
  %size = "stablehlo.get_dimension_size"(%I) {dimension = 3 : i64} : (tensor<1x128x512xf32>) -> tensor<i32>
  func.return %size : tensor<i32>
}

// -----

func.func @set_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> {
  %dim = stablehlo.constant dense<512> : tensor<1xi32>

  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{size operand should be of rank-0}}
  %result = "stablehlo.set_dimension_size"(%I, %dim) {dimension = 2 : i64} : (tensor<1x128x512xf32>, tensor<1xi32>) -> tensor<1x128x512xf32>
  func.return %result : tensor<1x128x512xf32>
}

// -----

func.func @set_dimension_size_negative_dimension(%I: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> {
  %dim = stablehlo.constant dense<512> : tensor<i32>
  // expected-error@+1 {{op attribute 'dimension' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %result = "stablehlo.set_dimension_size"(%I, %dim) {dimension =-1 : i64} : (tensor<1x128x512xf32>, tensor<i32>) -> tensor<1x128x512xf32>
  func.return %result : tensor<1x128x512xf32>
}

// -----

func.func @set_dimension_size_invalid_dimension(%I: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> {
  %dim = stablehlo.constant dense<512> : tensor<i32>
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires dimension attribute in range [0, 3); found (3)}}
  %result = "stablehlo.set_dimension_size"(%I, %dim) {dimension = 3 : i64} : (tensor<1x128x512xf32>, tensor<i32>) -> tensor<1x128x512xf32>
  func.return %result : tensor<1x128x512xf32>
}

// -----

// CHECK: func @custom_call_multiple_inputs_outputs
func.func @custom_call_multiple_inputs_outputs(%x: tensor<2xf32>, %token: !stablehlo.token) -> tensor<2xf32> {
  %0:3 = "stablehlo.custom_call"(%x, %token) {backend_config="", call_target_name = "foo", has_side_effect = false} : (tensor<2xf32>, !stablehlo.token) -> (tensor<2xf32>, tensor<2xf32>, !stablehlo.token)
  %1 = "stablehlo.add"(%0#0, %0#1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %1 : tensor<2xf32>
}

// -----

// CHECK: func @custom_call_multiple_inputs_outputs_with_layout
func.func @custom_call_multiple_inputs_outputs_with_layout(%x: tensor<2xf32>, %token: !stablehlo.token) -> tensor<f32> {
  %0:3 = "stablehlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>],
    result_layouts = [dense<> : tensor<0xindex>, dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>]
  } : (tensor<2xf32>, !stablehlo.token) -> (tensor<f32>, tensor<2xf32>, !stablehlo.token)
  func.return %0#0 : tensor<f32>
}

// -----

// CHECK: func @custom_call_tuple_output_with_layout
func.func @custom_call_tuple_output_with_layout(%x: tensor<2xf32>, %token: !stablehlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !stablehlo.token> {
  %0 = "stablehlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>],
    result_layouts = [dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>]
  } : (tensor<2xf32>, !stablehlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !stablehlo.token>
  func.return %0 : tuple<tensor<2xf32>, tensor<2xf32>, !stablehlo.token>
}

// -----

func.func @custom_call_only_operand_layout_constraints(%x: tensor<2xf32>, %token: !stablehlo.token) -> tensor<2xf32> {
  // expected-error@+1 {{Layout attributes should be specified for either both operands and results or none}}
  %0:3 = "stablehlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>]
  } : (tensor<2xf32>, !stablehlo.token) -> (tensor<2xf32>, tensor<2xf32>, !stablehlo.token)
  func.return %0#0 : tensor<2xf32>
}

// -----

func.func @custom_call_layout_mismatch_num_operands(%x: tensor<2xf32>, %token: !stablehlo.token) -> tensor<2xf32> {
  // expected-error@+1 {{Number of operands must match the number of operand layouts, 2 != 1}}
  %0:3 = "stablehlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>],
    result_layouts = [dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>]
  } : (tensor<2xf32>, !stablehlo.token) -> (tensor<2xf32>, tensor<2xf32>, !stablehlo.token)
  func.return %0#0 : tensor<2xf32>
}

// -----

func.func @custom_call_layout_mismatch_num_results() -> tensor<2xf32> {
  // expected-error@+1 {{Number of results must match the number of result layouts, 3 != 2}}
  %0:3 = "stablehlo.custom_call"() {
    call_target_name = "foo",
    operand_layouts = [],
    result_layouts = [dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>]
  } : () -> (tensor<2xf32>, tensor<2xf32>, !stablehlo.token)
  func.return %0#0 : tensor<2xf32>
}

// -----

func.func @custom_call_layout_mismatch_num_results_tuple(%x: tensor<2xf32>, %token: !stablehlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !stablehlo.token> {
  // expected-error@+1 {{Number of results must match the number of result layouts, 3 != 2}}
  %0 = "stablehlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>],
    result_layouts = [dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>]
  } : (tensor<2xf32>, !stablehlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !stablehlo.token>
  func.return %0 : tuple<tensor<2xf32>, tensor<2xf32>, !stablehlo.token>
}

// -----

func.func @custom_call_tuple_operand_input(%x: tuple<tensor<2xf32>>, %token: !stablehlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !stablehlo.token> {
  // expected-error@+1 {{Tuple types are not fully supported with layout constraints yet}}
  %0 = "stablehlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>],
    result_layouts = [dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>]
  } : (tuple<tensor<2xf32>>, !stablehlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !stablehlo.token>
  func.return %0 : tuple<tensor<2xf32>, tensor<2xf32>, !stablehlo.token>
}

// -----

func.func @custom_call_token_with_layout(%token: !stablehlo.token) {
  // expected-error@+1 {{Only tensor types can have non-empty layout: operand #0 of type '!stablehlo.token' has layout dense<[0, 1]> : tensor<2xindex>}}
  "stablehlo.custom_call"(%token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0, 1]> : tensor<2xindex>],
    result_layouts = []
  } : (!stablehlo.token) -> ()
  func.return
}

// -----

func.func @custom_call_mismatch_tensor_and_layout_rank(%arg: tensor<2x3xf32>) {
  // expected-error@+1 {{incorrect layout dense<[0, 1, 2]> : tensor<3xindex> for type 'tensor<2x3xf32>', layout must be a permutation of [0, 2)}}
  "stablehlo.custom_call"(%arg) {
    call_target_name = "foo",
    operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>],
    result_layouts = []
  } : (tensor<2x3xf32>) -> ()
  func.return
}

// -----

func.func @custom_call_mismatch_tensor_and_layout_permutation(%arg: tensor<1x2x3xf32>) {
  // expected-error@+1 {{incorrect layout dense<[0, 1, 3]> : tensor<3xindex> for type 'tensor<1x2x3xf32>', layout must be a permutation of [0, 3)}}
  "stablehlo.custom_call"(%arg) {
    call_target_name = "foo",
    operand_layouts = [dense<[0, 1, 3]> : tensor<3xindex>],
    result_layouts = []
  } : (tensor<1x2x3xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @custom_call_output_operand_alias
func.func @custom_call_output_operand_alias(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  // CHECK: stablehlo.custom_call @foo(%arg0, %arg1)
  // CHECK-SAME{LITERAL}: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = [1]>]}
  %0 = "stablehlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0],
                                 operand_index = 0,
                                 operand_tuple_indices = [1]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>
  func.return
}

// -----

func.func @custom_call_output_operand_alias_mismatch_operand_index(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  // expected-error@+1 {{expects operandIndex in the output_operand_alias attribute to be in range [0, 2); got: 2}}
  %0 = "stablehlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0],
                                 operand_index = 2,
                                 operand_tuple_indices = [1]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>
  func.return
}

// -----

func.func @custom_call_invalid_output_tuple_indices(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  // expected-error@+1 {{output_tuple_indices in the output_operand_alias attribute out of bounds}}
  %0 = "stablehlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [1],
                                 operand_index = 0,
                                 operand_tuple_indices = [1]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>
  func.return
}

// -----

func.func @custom_call_invalid_operand_tuple_indices(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  // expected-error@+1 {{operand_tuple_indices in the output_operand_alias attribute out of bounds}}
  %0 = "stablehlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0],
                                 operand_index = 0,
                                 operand_tuple_indices = [2]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>
  func.return
}

// -----

func.func @custom_call_output_operand_alias(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  // expected-error@+1 {{shapes mismatch in the output_operand_alias attribute: operand part has type 'tensor<2x3xf32>' and output part has type 'tensor<20x30xf32>'}}
  %0 = "stablehlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0],
                                 operand_index = 0,
                                 operand_tuple_indices = [1]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<20x30xf32>>
  func.return
}

// -----

// CHECK-LABEL: func @custom_call_unranked_types
func.func @custom_call_unranked_types(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: stablehlo.custom_call {{.*}} : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "stablehlo.custom_call"(%arg0) {call_target_name = "foo"} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @custom_call_with_dictionary_backend_config() {
  // CHECK: stablehlo.custom_call @foo() {api_version = 4 : i32, backend_config = {foo = 42 : i32}}
  "stablehlo.custom_call"() {api_version = 4 : i32, backend_config={foo = 42 : i32}, call_target_name = "foo"} : () -> ()
  func.return
}

// -----

func.func @custom_call_with_incompatible_backend_config() {
  // expected-error@+1 {{backend_config for api_version API_VERSION_TYPED_FFI must be a dictionary attribute}}
  "stablehlo.custom_call"() {api_version = 4 : i32, backend_config="bar=42", call_target_name = "foo"} : () -> ()
  func.return
}

// -----

func.func @custom_call_with_incompatible_backend_config() {
  // expected-error@+1 {{backend_config for api_version API_VERSION_STATUS_RETURNING_UNIFIED must be a string attribute}}
  "stablehlo.custom_call"() {api_version = 3 : i32, backend_config={bar = 42 : i32}, call_target_name = "foo"} : () -> ()
  func.return
}

// -----

// Test custom attribute printing/parsing.
// We really just need one op as holder, use module: this is the simplest top-level.

// CHECK: module
// CHECK-SAME: stablehlo.scatter = #stablehlo.scatter<>
module attributes{stablehlo.scatter = #stablehlo.scatter<>} {}

// -----

// CHECK: module
// CHECK-SAME: stablehlo.scatter = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 2], index_vector_dim = 1>
module attributes{
 stablehlo.scatter = #stablehlo.scatter<
  index_vector_dim = 1,
  scatter_dims_to_operand_dims = [0, 2],
  inserted_window_dims = [0, 1],
  update_window_dims = [1]
 >} {}

// -----

// CHECK: module
// CHECK-SAME: stablehlo.scatter = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1]>
module attributes{
 stablehlo.scatter = #stablehlo.scatter<
  inserted_window_dims = [0, 1],
  update_window_dims = [1]
 >} {}

// -----

module attributes{
 stablehlo.scatter = #stablehlo.scatter<
  index_vector_dim = 1,
  // expected-error@+2 {{duplicated `index_vector_dim` entry}}
  // expected-error@+1 {{failed parsing scatter dimension numbers}}
  index_vector_dim = 1,
 >} {}

// -----

// CHECK: module
// CHECK-SAME: stablehlo.gather = #stablehlo.gather<>
module attributes{stablehlo.gather = #stablehlo.gather<>} {}

// -----

// CHECK: module
// CHECK-SAME: stablehlo.gather = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>
module attributes{
 stablehlo.gather = #stablehlo.gather<
   collapsed_slice_dims = [0],
   index_vector_dim = 1,
   offset_dims = [1],
   start_index_map = [0],
 >} {}

// -----

module attributes{
 stablehlo.gather = #stablehlo.gather<
   collapsed_slice_dims = [0],
   // expected-error @+2 {{failed parsing gather dimension numbers}}
   // expected-error @+1 {{duplicated `collapsed_slice_dims` entry}}
   collapsed_slice_dims = [0],
 >} {}

// -----

// CHECK: module
// CHECK-SAME: stablehlo.dot = #stablehlo.dot<
// CHECK-SAME:       lhs_batching_dimensions = [0],
// CHECK-SAME:       rhs_batching_dimensions = [1],
// CHECK-SAME:       lhs_contracting_dimensions = [2],
// CHECK-SAME:       rhs_contracting_dimensions = [3]
// CHECK-SAME:     >
module attributes {
  stablehlo.dot = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [3]
  >} {}

// -----

// CHECK: module
// CHECK-SAME: stablehlo.dot = #stablehlo.dot<
// CHECK-SAME:       lhs_batching_dimensions = [0],
// CHECK-SAME:       rhs_batching_dimensions = [1],
// CHECK-SAME:       lhs_contracting_dimensions = [2],
// CHECK-SAME:       rhs_contracting_dimensions = [3]
// CHECK-SAME:     >
module attributes {
  stablehlo.dot = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [3],
  >} {}

// -----

// CHECK: module
// CHECK-SAME: stablehlo.dot = #stablehlo.dot<
// CHECK-SAME:       lhs_batching_dimensions = [0],
// CHECK-SAME:       rhs_batching_dimensions = [1],
// CHECK-SAME:       lhs_contracting_dimensions = [2],
// CHECK-SAME:       rhs_contracting_dimensions = [3]
// CHECK-SAME:     >
module attributes {
  stablehlo.dot = #stablehlo.dot<
      rhs_batching_dimensions = [1],
      rhs_contracting_dimensions = [3],
      lhs_contracting_dimensions = [2],
      lhs_batching_dimensions = [0],
  >} {}

// -----

module attributes {
  stablehlo.dot = #stablehlo.dot<
      rhs_batching_dimensions = [1],
      // expected-error@+2 {{duplicated `rhs_batching_dimensions` entry}}
      // expected-error@+1 {{failed parsing dot dimension numbers}}
      rhs_batching_dimensions = [3],
      lhs_contracting_dimensions = [2],
      lhs_batching_dimensions = [0],
  >} {}

// -----

module attributes {
  // expected-error@+3 {{expected '>'}}
  // expected-error@+3 {{failed parsing dot dimension numbers}}
  stablehlo.dot = #stablehlo.dot<
      rhs_batching_dimensions = [1]
      rhs_contracting_dimensions = [3]
      lhs_contracting_dimensions = [2]
      lhs_batching_dimensions = [0]
  >} {}


// -----

module attributes {
  // expected-error@+2 {{expected one of: `lhs_batching_dimensions`, `rhs_batching_dimensions`, `lhs_contracting_dimensions`, `rhs_contracting_dimensions`}}
  // expected-error@+1 {{failed parsing dot dimension numbers}}
  stablehlo.dot = #stablehlo.dot<foo = [0]>
} {}

// -----

module attributes {
  stablehlo.dot = #stablehlo.dot<
      rhs_batching_dimensions = [1],
      rhs_contracting_dimensions = [3],
      lhs_contracting_dimensions = [2],
      lhs_batching_dimensions = [0],
      // expected-error@+2 {{expected one of: `lhs_batching_dimensions`, `rhs_batching_dimensions`, `lhs_contracting_dimensions`, `rhs_contracting_dimensions`}}
      // expected-error@+1 {{failed parsing dot dimension numbers}}
      foo = [0]
  >} {}

// -----

// CHECK-LABEL: @batch_norm_training
func.func @batch_norm_training(%input: tensor<2x2x2x2xf64>, %scale: tensor<2xf64>, %offset: tensor<2xf64>) -> tensor<2x2x2x2xf64> {
  %0:3 = "stablehlo.batch_norm_training" (%input, %scale, %offset) {
    epsilon = 0.001 : f32,
    feature_index = 1 : i64
  } : (tensor<2x2x2x2xf64>, tensor<2xf64>, tensor<2xf64>) ->
      (tensor<2x2x2x2xf64>, tensor<2xf64>, tensor<2xf64>)
  func.return %0#0 : tensor<2x2x2x2xf64>
}

// -----

// CHECK-LABEL: @batch_norm_training_dynamic
func.func @batch_norm_training_dynamic(%input: tensor<?x?x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tensor<?x?x2x2xf32> {
  %0:3 = "stablehlo.batch_norm_training" (%input, %scale, %offset) {
    epsilon = 0.001 : f32,
    feature_index = 1 : i64
  } : (tensor<?x?x2x2xf32>, tensor<2xf32>, tensor<2xf32>) ->
      (tensor<?x?x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<?x?x2x2xf32>
}

// -----

func.func @batch_norm_training_c1(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{op attribute 'feature_index' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %0:3 = "stablehlo.batch_norm_training" (%input, %scale, %offset) {
    epsilon = 0.001 : f32,
    feature_index = -1 : i64
  } : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) ->
      (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @batch_norm_training_c1(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects featureIndex to be smaller than the rank of multi-dimensional operands; got featureIndex 4, and rank 4.}}
  %0:3 = "stablehlo.batch_norm_training" (%input, %scale, %offset) {
    epsilon = 0.001 : f32,
    feature_index = 4 : i64
  } : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) ->
      (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @batch_norm_training_c3_c4(%input: tensor<2x2x2x2xf32>, %scale: tensor<3xf32>, %offset: tensor<3xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects the size of single-dimensional operands to be compatible with feature count, but the size of single-dimensional operands is 3 and the feature count is 2.}}
  %0:3 = "stablehlo.batch_norm_training" (%input, %scale, %offset) {
    epsilon = 0.001 : f32,
    feature_index = 3 : i64
  } : (tensor<2x2x2x2xf32>, tensor<3xf32>, tensor<3xf32>) ->
      (tensor<2x2x2x2xf32>, tensor<3xf32>, tensor<3xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

// stablehlo.batch_norm_inference

// CHECK-LABEL: @batch_norm_inference
func.func @batch_norm_inference(%input: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>, %mean: tensor<256xf32>, %variance: tensor<256xf32>) -> (tensor<4x256xf32>) {
  %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {
    epsilon = 1.001000e-05 : f32,
    feature_index = 1 : i64
  } : (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256xf32>
  func.return %0 : tensor<4x256xf32>
}

// -----

// CHECK-LABEL: @batch_norm_inference_dynamic
func.func @batch_norm_inference_dynamic(%input: tensor<4x?xf32>, %scale: tensor<?xf32>, %offset: tensor<?xf32>, %mean: tensor<?xf32>, %variance: tensor<?xf32>) -> (tensor<4x?xf32>) {
  %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {
    epsilon = 1.001000e-05 : f32,
    feature_index = 1 : i64
  } : (tensor<4x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

// -----

func.func @batch_norm_inference_c1(%input: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>, %mean: tensor<256xf32>, %variance: tensor<256xf32>) -> (tensor<4x256xf32>) {
  // expected-error@+1 {{op attribute 'feature_index' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {
    epsilon = 1.001000e-05 : f32,
    feature_index = -1 : i64
  } : (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256xf32>
  func.return %0 : tensor<4x256xf32>
}

// -----

func.func @batch_norm_inference_c1(%input: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>, %mean: tensor<256xf32>, %variance: tensor<256xf32>) -> (tensor<4x256xf32>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects featureIndex to be smaller than the rank of multi-dimensional operands; got featureIndex 2, and rank 2.}}
  %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {
    epsilon = 1.001000e-05 : f32,
    feature_index = 2 : i64
  } : (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<4x256xf32>
  func.return %0 : tensor<4x256xf32>
}

// -----

func.func @error_batch_norm_inference_c3_c4_c5_c6(%input: tensor<4x256xf32>, %scale: tensor<25xf32>, %offset: tensor<25xf32>, %mean: tensor<25xf32>, %variance: tensor<25xf32>) -> (tensor<4x256xf32>) {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects the size of single-dimensional operands to be compatible with feature count, but the size of single-dimensional operands is 25 and the feature count is 256.}}
  %0 = "stablehlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {
    epsilon = 1.001000e-05 : f32,
    feature_index = 1 : i64
  } : (tensor<4x256xf32>, tensor<25xf32>, tensor<25xf32>, tensor<25xf32>, tensor<25xf32>) -> tensor<4x256xf32>
  func.return %0 : tensor<4x256xf32>
}

// -----

// CHECK-LABEL: @batch_norm_grad
func.func @batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @batch_norm_grad_dynamic(%input: tensor<?x2x2x2xf32>, %scale: tensor<?xf32>, %mean: tensor<?xf32>, %variance: tensor<?xf32>, %grad_output: tensor<?x2x2x2xf32>) -> tensor<?x2x2x2xf32> {
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {
    epsilon = 0.001 : f32, feature_index = 0 : i64
  } : (tensor<?x2x2x2xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x2x2x2xf32>) -> (tensor<?x2x2x2xf32>, tensor<?xf32>, tensor<?xf32>)
  func.return %0#0 : tensor<?x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad_c1(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{op attribute 'feature_index' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = -1 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad_c1(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects featureIndex to be smaller than the rank of multi-dimensional operands; got featureIndex 4, and rank 4.}}
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 4 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad_c3(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects multi-dimensional operands to have compatible shapes}}
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad_c4(%input: tensor<2x2x2x2xf32>, %scale: tensor<4xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects single-dimensional operands to have compatible shapes}}
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad_c4(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<4xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects single-dimensional operands to have compatible shapes}}
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<4xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad_c4(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<4xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects single-dimensional operands to have compatible shapes}}
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<4xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad_c5(%input: tensor<2x2x2x2xf32>, %scale: tensor<4xf32>, %mean: tensor<4xf32>, %variance: tensor<4xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{expects the size of single-dimensional operands to be compatible with feature count, but the size of single-dimensional operands is 4 and the feature count is 2.}}
  %0:3 = "stablehlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<4xf32>, tensor<4xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

// CHECK-LABEL: @fft
func.func @fft(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type FFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

// CHECK-LABEL: @ifft
func.func @ifft(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type IFFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

// CHECK-LABEL: @rfft
func.func @rfft(%arg0: tensor<3x9xf32>) -> tensor<3x5xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<3x9xf32>) -> tensor<3x5xcomplex<f32>>
  func.return %0 : tensor<3x5xcomplex<f32>>
}

// -----

// CHECK-LABEL: @irfft
func.func @irfft(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x16xf32> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 16>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x16xf32>
  func.return %0 : tensor<3x16xf32>
}

// -----

func.func @rfft_not_float32or64(%arg0: tensor<3x9xf16>) -> tensor<3x5xcomplex<f32>> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{RFFT requires f32 or f64 input type, but is given 'f16'.}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<3x9xf16>) -> tensor<3x5xcomplex<f32>>
  func.return %0 : tensor<3x5xcomplex<f32>>
}

// -----

func.func @fft_invalid_rank(%arg0: tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{rank must be between 1 and 3, but got 4.}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9, 9, 9, 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

func.func @fft_rank_mismatch(%arg0: tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{operand rank must not be less than fft rank of 3 for operand of type 'tensor<3x9xf32>'}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9, 9, 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

func.func @rfft_invalid_dim(%arg0: tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{RFFT requires innermost dimensions to be compatible with fft_length. Got: 3, 9 but wanted 9, 9.}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9, 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

func.func @irfft_invalid_dim(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{IRFFT requires non-final dimensions to be compatible with fft_length. Got: 3, 9 but wanted 9, 9, and 3 != 9.}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9, 9>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32>
  func.return %0 : tensor<3x9xf32>
}

// -----

func.func @irfft_invalid_dim(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{IRFFT requires innermost dimension to be compatible with fft_length[-1]/2+1. Got: 9 but fft_length is 9.}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32>
  func.return %0 : tensor<3x9xf32>
}

// -----

func.func @irfft_invalid_elt(%arg0: tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{FFT/IFFT/IRFFT take a complex tensor as input, but is given 'tensor<3x9xf32>'}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 16>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

func.func @irfft_invalid_ret_elt(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x16xcomplex<f32>> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<3x16xf32>' are incompatible with return type(s) of operation 'tensor<3x16xcomplex<f32>>'}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 16>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x16xcomplex<f32>>
  func.return %0 : tensor<3x16xcomplex<f32>>
}

// -----

func.func @rfft_invalid_ret_elt(%arg0: tensor<3x9xf32>) -> tensor<3x9xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<3x5xcomplex<f32>>' are incompatible with return type(s) of operation 'tensor<3x9xf32>'}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<3x9xf32>) -> tensor<3x9xf32>
  func.return %0 : tensor<3x9xf32>
}

// -----

// CHECK-LABEL: @rfft_dynamic
func.func @rfft_dynamic(%arg0: tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  func.return %0 : tensor<?x?xcomplex<f32>>
}

// -----

func.func @rfft_dynamic_incompatible_dims(%arg0: tensor<3x10xf32>) -> tensor<?x?xcomplex<f32>> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{RFFT requires innermost dimensions to be compatible with fft_length. Got: 3, 10 but wanted 9.}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 9>, fft_type = #stablehlo<fft_type RFFT> } : (tensor<3x10xf32>) -> tensor<?x?xcomplex<f32>>
  func.return %0 : tensor<?x?xcomplex<f32>>
}

// -----

// CHECK-LABEL: @irfft_dynamic
func.func @irfft_dynamic(%arg0: tensor<?x?xcomplex<f32>>) -> tensor<?x?xf32> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 16>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<?x?xcomplex<f32>>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @irfft_dynamic_incompatible_non_final_dims(%arg0: tensor<?x3x15xcomplex<f32>>) -> tensor<?x?x?xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{IRFFT requires non-final dimensions to be compatible with fft_length. Got: -9223372036854775808, 3, 15 but wanted 4, 16, and 3 != 4}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 4, 16>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<?x3x15xcomplex<f32>>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @irfft_dynamic_incompatible_final_dim(%arg0: tensor<?x8xcomplex<f32>>) -> tensor<?x?xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{IRFFT requires innermost dimension to be compatible with fft_length[-1]/2+1. Got: 8 but fft_length is 16.}}
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 16>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<?x8xcomplex<f32>>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @irfft_dynamic
func.func @irfft_dynamic(%arg0: tensor<?x?xcomplex<f32>>) -> tensor<?x?xf32> {
  %0 = "stablehlo.fft"(%arg0) { fft_length = array<i64: 16>, fft_type = #stablehlo<fft_type IRFFT> } : (tensor<?x?xcomplex<f32>>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @eltwise_static_and_dynamic_type(
//  CHECK-SAME: %[[A:.*]]: tensor<10x10xf32>, %[[B:.*]]: tensor<?x?xf32>) -> tensor<10x10xf32>
//       CHECK: %[[R:.*]] = stablehlo.add %[[A]], %[[B]] : (tensor<10x10xf32>, tensor<?x?xf32>) -> tensor<10x10xf32>
//       CHECK: return %[[R]] : tensor<10x10xf32>
func.func @eltwise_static_and_dynamic_type(%arg0: tensor<10x10xf32>, %arg1: tensor<?x?xf32>) -> tensor<10x10xf32> {
  %0 = stablehlo.add %arg0, %arg1 : (tensor<10x10xf32>, tensor<?x?xf32>) -> tensor<10x10xf32>
  func.return %0 : tensor<10x10xf32>
}

// -----

// CHECK-LABEL: func @convolution_operand_element_type_i4
func.func @convolution_operand_element_type_i4(%arg0: tensor<64x8x8x8xi4>, %arg1: tensor<4x4x8x32xi4>) -> tensor<64x3x3x32xi8> {
  // Note: This has been lowered and adapted from:
  // %0 = "tf.Conv2D"(%arg0, %arg1) {
  //        data_format = "NHWC",
  //        dilations = [1, 2, 2, 1],
  //        explicit_paddings = [0, 0, 0, 1, 0, 1, 0, 0],
  //        padding = "EXPLICIT",
  //        strides = [1, 1, 1, 1]} :
  //      (tensor<64x8x8x8xf32>, tensor<4x4x8x32xf32>) -> tensor<64x3x3x32xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [2, 2]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64} :
       (tensor<64x8x8x8xi4>, tensor<4x4x8x32xi4>) -> tensor<64x3x3x32xi8>
  func.return %0 : tensor<64x3x3x32xi8>
}

// -----

// CHECK: func @convolution_quantized_conv2d
// CHECK: stablehlo.convolution
// CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
func.func @convolution_quantized_conv2d(%arg0: tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>> {
  %0 = stablehlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} :
       (tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0 : tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

// CHECK-LABEL: func @quantized_clamp
func.func @quantized_clamp(%arg0: tensor<1x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<1x!quant.uniform<ui8:f32, 34.0:16>> {
  %0 = "stablehlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1x!quant.uniform<ui8:f32, 34.0:16>>, tensor<1x!quant.uniform<ui8:f32, 34.0:16>>, tensor<1x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<1x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %0: tensor<1x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL: func @quantized_dot_i8
func.func @quantized_dot_i8(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<2x2x!quant.uniform<i8:f32, 10.0:50>> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<2x2x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

// CHECK-LABEL: func @quantized_dot_i8_per_axis
func.func @quantized_dot_i8_per_axis(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>>) -> tensor<2x2x!quant.uniform<i8:f32, 10.0:50>> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>>) -> tensor<2x2x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

// CHECK-LABEL: func @quantized_dot_i4
func.func @quantized_dot_i4(%arg0: tensor<2x2x!quant.uniform<i4:f32, 2.0:1>>, %arg1: tensor<2x2x!quant.uniform<i4:f32, 5.0:2>>) -> tensor<2x2x!quant.uniform<i4:f32, 10.0:5>> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i4:f32, 2.0:1>>, tensor<2x2x!quant.uniform<i4:f32, 5.0:2>>) -> tensor<2x2x!quant.uniform<i4:f32, 10.0:5>>
  func.return %0: tensor<2x2x!quant.uniform<i4:f32, 10.0:5>>
}

// -----

// CHECK-LABEL: func @quantized_dot_general
func.func @quantized_dot_general(%arg0: tensor<2x16x32x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x32x32x!quant.uniform<i8:f32, 5.0:0>>) -> tensor<2x16x32x!quant.uniform<i8:f32, 10.0:50>> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}
    : (tensor<2x16x32x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x32x32x!quant.uniform<i8:f32, 5.0:0>>) -> tensor<2x16x32x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0 : tensor<2x16x32x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

// CHECK-LABEL: func @uniform_quantize
func.func @uniform_quantize(%arg: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
  %0 = stablehlo.uniform_quantize %arg : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %0 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK: func @uniform_requantize
func.func @uniform_requantize(%arg: tensor<16x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<16x16x!quant.uniform<i8:f32, 34.0:16>> {
  %0 = stablehlo.uniform_quantize %arg : (tensor<16x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>
  func.return %0 : tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL: func @uniform_dequantize
func.func @uniform_dequantize(%arg: tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32> {
  %0 = stablehlo.uniform_dequantize %arg : (tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func @quantized_constants
func.func @quantized_constants() -> (tensor<2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x!quant.uniform<ui8:f32, 34.0:16>>, tensor<2x!quant.uniform<i8:f32, 2.0:15>>) {
  %0 = stablehlo.constant() {value = dense<[1, 2]> : tensor<2xi8>} : () -> tensor<2x!quant.uniform<i8:f32, 2.000000e+00:15>>
  %1 = stablehlo.constant dense<[10.0, 12.0]> : tensor<2xf32>
  %2 = stablehlo.constant dense<[3.0, 100.0]> : tensor<2xf32>
  %3 = stablehlo.uniform_quantize %2 : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 2.0:15>>
  %4 = stablehlo.uniform_quantize %1 : (tensor<2xf32>) -> tensor<2x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %0, %4, %3 : tensor<2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x!quant.uniform<ui8:f32, 34.0:16>>, tensor<2x!quant.uniform<i8:f32, 2.0:15>>
  // CHECK: stablehlo.constant() <{value = dense<[1, 2]> : tensor<2xi8>}> : () -> tensor<2x!quant.uniform<i8:f32, 2.000000e+00:15>>
  // CHECK-NEXT: stablehlo.constant dense<[1.000000e+01, 1.200000e+01]> : tensor<2xf32>
  // CHECK-NEXT: stablehlo.constant dense<[3.000000e+00, 1.000000e+02]> : tensor<2xf32>
}

// -----

func.func @quantized_constants_invalid_storage_type() -> () {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.constant' op inferred type(s) 'tensor<2xui8>' are incompatible with return type(s) of operation 'tensor<2x!quant.uniform<i8:f32, 2.000000e+00:15>>}}
  %0 = "stablehlo.constant"() {value = dense<[1, 2]> : tensor<2xui8>} : () -> tensor<2x!quant.uniform<i8:f32, 2.0:15>>
  func.return
}

// -----

func.func @dot_i4xi4_i8(%arg0: tensor<1x2xi4>, %arg1: tensor<2x1xi4>) -> tensor<1x1xi8> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<1x2xi4>, tensor<2x1xi4>) -> tensor<1x1xi8>
  func.return %0: tensor<1x1xi8>
}

// -----

// CHECK-LABEL: func @dot_i8xi8_i16
func.func @dot_i8xi8_i16(%arg0: tensor<1x2xi8>, %arg1: tensor<2x1xi8>) -> tensor<1x1xi16> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<1x2xi8>, tensor<2x1xi8>) -> tensor<1x1xi16>
  func.return %0: tensor<1x1xi16>
}

// -----

// CHECK-LABEL: func @einsum_i4xi4_i8
func.func @einsum_i4xi4_i8(%arg0: tensor<1x2xi4>, %arg1: tensor<2x1xi4>) -> tensor<1x1xi8> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ab,bc->ac"} : (tensor<1x2xi4>, tensor<2x1xi4>) -> tensor<1x1xi8>
  func.return %0: tensor<1x1xi8>
}

// -----

// CHECK-LABEL: func @einsum_i8xi8_i16
func.func @einsum_i8xi8_i16(%arg0: tensor<1x2xi8>, %arg1: tensor<2x1xi8>) -> tensor<1x1xi16> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ab,bc->ac"} : (tensor<1x2xi8>, tensor<2x1xi8>) -> tensor<1x1xi16>
  func.return %0: tensor<1x1xi16>
}

// -----

// CHECK-LABEL: func @pad
func.func @pad(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x7xf16> {
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1, 2>,
    edge_padding_high = array<i64: 1, 1, 0>,
    interior_padding = array<i64: 0, 0, 1>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x7xf16>
  func.return %0 : tensor<2x4x7xf16>
}


// -----

func.func @pad_c2(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x7xf16> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{edge_padding_low length (2) must match operand rank (3)}}
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1>,
    edge_padding_high = array<i64: 1, 1>,
    interior_padding = array<i64: 0, 0>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x7xf16>
  func.return %0 : tensor<2x4x7xf16>
}

// -----

func.func @pad_c3(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x3xf16> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Interior padding cannot be negative: -1}}
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1, 2>,
    edge_padding_high = array<i64: 1, 1, 0>,
    interior_padding = array<i64: 0, 0, -1>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x3xf16>
  func.return %0 : tensor<2x4x3xf16>
}

// -----

func.func @pad_c4(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x7xf16> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Padding result in negative size for dimension 2}}
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1, -4>,
    edge_padding_high = array<i64: 1, 1, 0>,
    interior_padding = array<i64: 0, 0, 0>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x7xf16>
  func.return %0 : tensor<2x4x7xf16>
}

// -----

func.func @pad_c4(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<8x8x8xf16> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.pad' op inferred type(s) 'tensor<2x4x7xf16>' are incompatible with return type(s) of operation 'tensor<8x8x8xf16>'}}
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 0, 1, 2>,
    edge_padding_high = array<i64: 1, 1, 0>,
    interior_padding = array<i64: 0, 0, 1>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<8x8x8xf16>
  func.return %0 : tensor<8x8x8xf16>
}

// -----

// CHECK-LABEL: func @pad_dynamic
func.func @pad_dynamic(%arg0: tensor<?x48x48x32xf32>) -> tensor<?x48x48x48xf32> {
  %0 = "stablehlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "stablehlo.pad"(%arg0, %0) {
    edge_padding_low = array<i64: 0, 0, 0, 0>,
    edge_padding_high = array<i64: 0, 0, 0, 16>,
    interior_padding = array<i64: 0, 0, 0, 0>
  } : (tensor<?x48x48x32xf32>, tensor<f32>) -> tensor<?x48x48x48xf32>
  func.return %1 : tensor<?x48x48x48xf32>
}

// -----

func.func @pad_i3(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x7xf16> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{edge_padding_low length (1) must match operand rank (3)}}
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_low = array<i64: 1>,
    edge_padding_high = array<i64: 1>,
    interior_padding = array<i64: 1>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x7xf16>
  func.return %0 : tensor<2x4x7xf16>
}

// -----

func.func @is_compatible_dynamism_mix(%arg0: tensor<?xf32>, %arg1: tensor<1xf32>) {
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "stablehlo.add"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<1xf32>
  %2 = "stablehlo.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
  %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<1xf32>) -> tensor<1xf32>
  %4 = "stablehlo.add"(%arg1, %arg0) : (tensor<1xf32>, tensor<?xf32>) -> tensor<?xf32>
  %5 = "stablehlo.add"(%arg1, %arg0) : (tensor<1xf32>, tensor<?xf32>) -> tensor<1xf32>
  %6 = "stablehlo.add"(%arg1, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<?xf32>
  %7 = "stablehlo.add"(%arg1, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return
}

// -----

func.func @is_compatible_dynamism_ranked_mismatch(%arg0: tensor<?xf32>) {
  // expected-error@+1 {{op requires the same shape for all operands and results}}
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  func.return
}

// -----

func.func @is_compatible_dynamism_dim_mismatch(%arg0: tensor<1x?xf32>) {
  // expected-error@+1 {{op requires the same shape for all operands and results}}
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<1x?xf32>, tensor<1x?xf32>) -> tensor<2x2xf32>
  func.return
}

// -----

func.func @is_compatible_quant_mix_non_quant(%arg0: tensor<1xf32>, %arg1: tensor<1x!quant.uniform<i8:f32, 1.0:17>>) {
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "stablehlo.add"(%arg1, %arg1) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:f32, 1.0:17>>
  %2 = "stablehlo.add"(%arg1, %arg1) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:f32, 1.0:17>>
  %3 = "stablehlo.add"(%arg1, %arg1) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:f32, 2.0:17>>
  %4 = "stablehlo.add"(%arg1, %arg1) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:f32, 1.0:18>>

  func.return
}


// -----

func.func @add_c4(%arg0: tensor<1x!quant.uniform<i8:f32, 1.0:17>>) {
  // expected-error@+1 {{mismatched operands and result quantization expressed types}}
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:bf16, 1.0:17>>
  func.return
}

// -----

func.func @add_c3(%arg0: tensor<1x!quant.uniform<i8:f32, 1.0:17>>) {
  // expected-error@+1 {{mismatched operands and result quantization storage types}}
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i4:f32, 1.0:1>>
  func.return
}

// -----

func.func @add_c2(%arg0: tensor<1x!quant.uniform<i8:f32, 1.0:17>>) {
  // expected-error@+1 {{all operands and results to be either quantized or non-quantized}}
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1xf32>
  func.return
}

// -----

func.func @add_c5(%arg0: tensor<1x!quant.uniform<i8:f32:0, {1.0:17}>>) {
  // expected-error@+1 {{result is not per_axis quantized but lhs or rhs are}}
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<1x!quant.uniform<i8:f32:0, {1.0:17}>>, tensor<1x!quant.uniform<i8:f32:0, {1.0:17}>>) -> tensor<1x!quant.uniform<i8:f32, 1.0:17>>
  func.return
}

// -----

func.func @add_c6(%arg0: tensor<1x2x!quant.uniform<i8:f32:0, {1.0:17}>>) {
  // expected-error@+1 {{quantization_dimension of lhs and result are not same}}
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<1x2x!quant.uniform<i8:f32:0, {1.0:17}>>, tensor<1x2x!quant.uniform<i8:f32:0, {1.0:17}>>) -> tensor<1x2x!quant.uniform<i8:f32:1, {1.0:17, 1.0:17}>>
  func.return
}

// -----

func.func @add_c7(%arg0: tensor<1x2x!quant.uniform<i8:f32:0, {1.0:17}>>, %arg1: tensor<1x2x!quant.uniform<i8:f32:1, {1.0:17, 1.0:17}>>) {
  // expected-error@+1 {{quantization_dimension of rhs and result are not same}}
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<1x2x!quant.uniform<i8:f32:0, {1.0:17}>>, tensor<1x2x!quant.uniform<i8:f32:1, {1.0:17, 1.0:17}>>) -> tensor<1x2x!quant.uniform<i8:f32:0, {1.0:17}>>
  func.return
}

// -----

func.func @is_compatible_quant_signedness_mismatch(%arg0: tensor<1x!quant.uniform<i8:f32, 1.0:17>>) {
  // expected-error@+2 {{op failed to infer returned types}}
  // expected-error@+1 {{op inferred type(s) 'tensor<1x!quant.uniform<i8:f32, 1.000000e+00:17>>' are incompatible with return type(s) of operation 'tensor<1x!quant.uniform<u8:f32, 1.000000e+00:17>>'}}
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<u8:f32, 1.0:17>>
  func.return
}

// -----

// CHECK-LABEL: is_compatible_dynamism_bounds
func.func @is_compatible_dynamism_bounds_mismatch(
  %arg0: tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>,
  %arg1: tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>) {
  %0 = "stablehlo.add"(%arg0, %arg1) : (
    tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>,
    tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>) -> tensor<3xf32>
  func.return
}

// -----

func.func @is_compatible_dynamism_bounds_mismatch(
  %arg0: tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>,
  %arg1: tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>) {
  // expected-error@+2 {{op failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.add' op inferred type(s) 'tensor<?xf32, #stablehlo.bounds<4>>' are incompatible with return type(s) of operation 'tensor<5xf32>'}}
  %0 = "stablehlo.add"(%arg0, %arg1) : (
    tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>,
    tensor<?xf32, #stablehlo.type_extensions<bounds = [4]>>) -> tensor<5xf32>
  func.return
}

// -----

// CHECK-LABEL: scatter_update_scalar
func.func @scatter_update_scalar(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                            %arg2: tensor<1xi32>) -> tensor<3xi32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "stablehlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: scatter_variadic
func.func @scatter_variadic(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                            %arg2: tensor<1xi32>) -> tensor<3xi32> {
  %0, %1 = "stablehlo.scatter"(%arg0, %arg0, %arg1, %arg2, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>):
    "stablehlo.return"(%arg3, %arg5) : (tensor<i32>, tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<3xi32>, tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<3xi32>, tensor<3xi32>)
  func.return %0 : tensor<3xi32>
}

// -----


#SV = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

func.func @is_compatible_sparse_mix_non_sparse(%arg0: tensor<1xf32>, %arg1: tensor<1xf32, #SV>) {
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "stablehlo.add"(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32, #SV>
  %2 = "stablehlo.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32, #SV>) -> tensor<1xf32, #SV>
  %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32, #SV>) -> tensor<1xf32, #SV>
  %4 = "stablehlo.add"(%arg1, %arg0) : (tensor<1xf32, #SV>, tensor<1xf32>) -> tensor<1xf32>
  %5 = "stablehlo.add"(%arg1, %arg0) : (tensor<1xf32, #SV>, tensor<1xf32>) -> tensor<1xf32>
  %6 = "stablehlo.add"(%arg1, %arg1) : (tensor<1xf32, #SV>, tensor<1xf32, #SV>) -> tensor<1xf32, #SV>
  %7 = "stablehlo.add"(%arg1, %arg1) : (tensor<1xf32, #SV>, tensor<1xf32, #SV>) -> tensor<1xf32, #SV>
  func.return
}

// CHECK-LABEL: func @abs
func.func @abs(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = "stablehlo.abs"(%arg0) {} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @abs_complex
func.func @abs_complex(%arg0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xf32> {
  %0 = "stablehlo.abs"(%arg0) {} : (tensor<1x2xcomplex<f32>>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// -----

func.func @abs_c2(%arg0: tensor<1x2xf32>) -> tensor<1x2xf64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.abs' op inferred type(s) 'tensor<1x2xf32>' are incompatible with return type(s) of operation 'tensor<1x2xf64>'}}
  %0 = "stablehlo.abs"(%arg0) {} : (tensor<1x2xf32>) -> tensor<1x2xf64>
  func.return %0 : tensor<1x2xf64>
}

// -----

func.func @abs_c2(%arg0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xf64> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'stablehlo.abs' op inferred type(s) 'tensor<1x2xf32>' are incompatible with return type(s) of operation 'tensor<1x2xf64>'}}
  %0 = "stablehlo.abs"(%arg0) {} : (tensor<1x2xcomplex<f32>>) -> tensor<1x2xf64>
  func.return %0 : tensor<1x2xf64>
}

// -----

// CHECK-LABEL: func @round_even
func.func @round_even(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "stablehlo.round_nearest_even"(%arg0) {} : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @complex
func.func @complex(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xcomplex<f32>> {
  %0 = "stablehlo.complex"(%arg0, %arg1) {} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xcomplex<f32>>
  func.return %0 : tensor<10x10xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @is_finite
func.func @is_finite(%arg0: tensor<3xf32>) -> tensor<3xi1> {
  %0 = "stablehlo.is_finite"(%arg0) {} : (tensor<3xf32>) -> tensor<3xi1>
  func.return %0 : tensor<3xi1>
}

// -----

func.func @is_finite_int_input(%arg0: tensor<3xi32>) -> tensor<3xi1> {
  // expected-error-re@+1 {{operand #0 must be ranked tensor of {{.*}}, but got 'tensor<3xi32>'}}
  %0 = "stablehlo.is_finite"(%arg0) {} : (tensor<3xi32>) -> tensor<3xi1>
  func.return %0 : tensor<3xi1>
}

// -----

func.func @is_finite_mismatch_return_element_type(%arg0: tensor<3xf32>) -> tensor<3xi10> {
  // expected-error-re@+1 {{result #0 must be ranked tensor of {{.*}}, but got 'tensor<3xi10>'}}
  %0 = "stablehlo.is_finite"(%arg0) {} : (tensor<3xf32>) -> tensor<3xi10>
  func.return %0 : tensor<3xi10>
}

// -----

func.func @is_finite_mismatch_return_shape(%arg0: tensor<3xf32>) -> tensor<4xi1> {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %0 = "stablehlo.is_finite"(%arg0) {} : (tensor<3xf32>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

func.func @negative_dimension_attr(%arg0: tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, -1]>>, %arg1: tensor<i32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{op attribute 'dimension' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %result = "stablehlo.set_dimension_size"(%arg0, %arg1) {dimension = -1 : i64} : (tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, -1]>>, tensor<i32>) -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}

// -----

func.func @invalid_dimension_attr(%arg0: tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, -1]>>, %arg1: tensor<i32>) -> tensor<?x?xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{requires dimension attribute in range [0, 2); found (2)}}
  %result = "stablehlo.set_dimension_size"(%arg0, %arg1) {dimension = 2 : i64} : (tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, -1]>>, tensor<i32>) -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @convert
func.func @convert(%arg0: tensor<f64>) -> tensor<bf16> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<f64>) -> tensor<bf16>
  func.return %0 : tensor<bf16>
}

// -----

// CHECK-LABEL: func @convert_f4e2m1fn
func.func @convert_f4e2m1fn(%arg0: tensor<f16>) -> tensor<f4E2M1FN> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<f16>) -> tensor<f4E2M1FN>
  func.return %0 : tensor<f4E2M1FN>
}

// -----

// CHECK-LABEL: func @convert_f6e2m3fn
func.func @convert_f6e2m3fn(%arg0: tensor<f16>) -> tensor<f6E2M3FN> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<f16>) -> tensor<f6E2M3FN>
  func.return %0 : tensor<f6E2M3FN>
}

// -----

// CHECK-LABEL: func @convert_f6e3m2fn
func.func @convert_f6e3m2fn(%arg0: tensor<f16>) -> tensor<f6E3M2FN> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<f16>) -> tensor<f6E3M2FN>
  func.return %0 : tensor<f6E3M2FN>
}

// -----

// CHECK-LABEL: func @convert_f8e3m4
func.func @convert_f8e3m4(%arg0: tensor<f16>) -> tensor<f8E3M4> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<f16>) -> tensor<f8E3M4>
  func.return %0 : tensor<f8E3M4>
}

// -----

// CHECK-LABEL: func @convert_f8e4m3
func.func @convert_f8e4m3(%arg0: tensor<f16>) -> tensor<f8E4M3> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<f16>) -> tensor<f8E4M3>
  func.return %0 : tensor<f8E4M3>
}

// -----

// CHECK-LABEL: func @convert_f8e4m3fn
func.func @convert_f8e4m3fn(%arg0: tensor<f16>) -> tensor<f8E4M3FN> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<f16>) -> tensor<f8E4M3FN>
  func.return %0 : tensor<f8E4M3FN>
}

// -----

// CHECK-LABEL: func @convert_f8e5m2
func.func @convert_f8e5m2(%arg0: tensor<f16>) -> tensor<f8E5M2> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<f16>) -> tensor<f8E5M2>
  func.return %0 : tensor<f8E5M2>
}

// -----

// CHECK-LABEL: func @f8e4m3fnuz
func.func @f8e4m3fnuz(%arg0: tensor<f16>) -> tensor<f8E4M3FNUZ> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<f16>) -> tensor<f8E4M3FNUZ>
  func.return %0 : tensor<f8E4M3FNUZ>
}

// -----

// CHECK-LABEL: func @f8e5m2fnuz
func.func @f8e5m2fnuz(%arg0: tensor<f16>) -> tensor<f8E5M2FNUZ> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<f16>) -> tensor<f8E5M2FNUZ>
  func.return %0 : tensor<f8E5M2FNUZ>
}

// -----

// CHECK-LABEL: func @f8e8m0fnu
func.func @f8e8m0fnu(%arg0: tensor<f16>) -> tensor<f8E8M0FNU> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<f16>) -> tensor<f8E8M0FNU>
  func.return %0 : tensor<f8E8M0FNU>
}

// -----

func.func @dynamic_iota_static() -> tensor<4xf32> {
  %0 = stablehlo.constant dense<[4]> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// -----

func.func @dynamic_iota_dynamic() -> tensor<?xf32> {
  %0 = stablehlo.constant dense<[4]> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// -----

func.func @dynamic_iota_invalid_iota_dimension_negative() -> tensor<?xf32> {
  // expected-error@+2 {{op attribute 'iota_dimension' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  %0 = stablehlo.constant dense<[4]> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = -1 : (tensor<1xi64>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// -----

func.func @dynamic_iota_invalid_iota_dimension_too_big() -> tensor<?xf32> {
  %0 = stablehlo.constant dense<[4]> : tensor<1xi64>
  // expected-error@+1 {{iota dimension cannot go beyond the output rank}}
  %1 = stablehlo.dynamic_iota %0, dim = 2 : (tensor<1xi64>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// -----

func.func @dynamic_iota_output_shape_negative_size() -> tensor<4xf32> {
  // expected-error@+2 {{output shape [-1] is incompatible with return type of operation 'tensor<4xf32>'}}
  %0 = stablehlo.constant dense<[-1]> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// -----

func.func @dynamic_iota_output_shape_mismatching_size() -> tensor<4xf32> {
  // expected-error@+2 {{output shape [1] is incompatible with return type of operation 'tensor<4xf32>'}}
  %0 = stablehlo.constant dense<[1]> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// -----

func.func @dynamic_iota_output_shape_matches_result() -> tensor<4xf32> {
  %0 = stablehlo.constant dense<[4]> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// -----

func.func @dynamic_iota_output_shape_compatible_with_result() -> tensor<?xf32> {
  %0 = stablehlo.constant dense<[4]> : tensor<1xi64>
  %1 = stablehlo.dynamic_iota %0, dim = 0 : (tensor<1xi64>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// -----

func.func @first(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  func.return %arg0 : tensor<f32>
}

func.func @composite_generic(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  %0 = "stablehlo.composite"(%arg0, %arg1) {
    name = "stablehlo.first",
    decomposition = @first,
    version = 1 : i32,
    composite_attributes = {
      an_attribute = "foo"
    }
  } : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return
}

// -----

func.func @foo() { func.return }
func.func @composite_c1() {
  // expected-error@+1 {{name must be a valid namespaced op name}}
  stablehlo.composite "foo" { decomposition = @foo } : () -> ()
  func.return
}

// -----

func.func @foo() { func.return }
func.func @composite_c1() {
  // expected-error@+1 {{name must be a valid namespaced op name}}
  stablehlo.composite "." { decomposition = @foo } : () -> ()
  func.return
}

// -----

func.func @foo() { func.return }
func.func @composite_c1() {
  // expected-error@+1 {{name must be a valid namespaced op name}}
  stablehlo.composite "foo." { decomposition = @foo } : () -> ()
  func.return
}

// -----

func.func @foo() { func.return }
func.func @composite_c1() {
  // expected-error@+1 {{name must be a valid namespaced op name}}
  stablehlo.composite ".foo" { decomposition = @foo } : () -> ()
  func.return
}

// -----

func.func @foo() { func.return }
func.func @composite_c1() {
  // expected-error@+1 {{name must be a valid namespaced op name}}
  stablehlo.composite "0.foo" { decomposition = @foo } : () -> ()
  func.return
}

// -----

func.func @foo() { func.return }
func.func @composite_c1() {
  // expected-error@+1 {{name must be a valid namespaced op name}}
  stablehlo.composite "foo.%" { decomposition = @foo } : () -> ()
  func.return
}

// -----

func.func @foo() { func.return }
func.func @composite_c1() {
  // expected-error@+1 {{name must be a valid namespaced op name}}
  stablehlo.composite "foo.foo.%" { decomposition = @foo } : () -> ()
  func.return
}

// -----

func.func @foo() { func.return }
func.func @composite_c1() {
  // valid name
  stablehlo.composite "f00._.$" { decomposition = @foo } : () -> ()
  func.return
}

// -----

func.func @composite_c2(%arg0: tensor<f32>) {
  // expected-error@+1 {{'nonexistent' does not reference a valid function}}
  %0 = stablehlo.composite "stablehlo.nonexistent" %arg0 {
    decomposition = @nonexistent
  } : (tensor<f32>) -> tensor<f32>
  func.return
}

// -----

func.func @foo() -> !stablehlo.token {
  %0 = stablehlo.create_token : !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @composite_c3(%arg0: tensor<f32>) {
  // expected-error@+1 {{has 1 operand(s), but decomposition has 0}}
  %0 = stablehlo.composite "stablehlo.identity" %arg0 {
    decomposition = @foo
  } : (tensor<f32>) -> !stablehlo.token
  func.return
}

// -----

func.func @foo(%arg0: tensor<f64>) -> !stablehlo.token {
  %0 = stablehlo.create_token : !stablehlo.token
  func.return %0 : !stablehlo.token
}

func.func @composite_c3(%arg0: tensor<f32>) {
  // expected-error@+1 {{operand at index 0 has type 'tensor<f32>', but decomposition has type 'tensor<f64>'}}
  %0 = stablehlo.composite "stablehlo.identity" %arg0 {
    decomposition = @foo
  } : (tensor<f32>) -> !stablehlo.token
  func.return
}

// -----

func.func @foo(%arg0: !stablehlo.token) {
  func.return
}

func.func @composite_c4(%arg0: !stablehlo.token) {
  // expected-error@+1 {{has 1 result(s), but decomposition has 0}}
  %0 = stablehlo.composite "stablehlo.identity" %arg0 {
    decomposition = @foo
  } : (!stablehlo.token) -> tensor<f32>
  func.return
}

// -----

func.func @foo(%arg0: !stablehlo.token) -> tensor<f64> {
  %0 = stablehlo.constant dense<0.> : tensor<f64>
  func.return %0 : tensor<f64>
}

func.func @composite_c4(%arg0: !stablehlo.token) {
  // expected-error@+1 {{result at index 0 has type 'tensor<f32>', but decomposition has type 'tensor<f64>'}}
  %0 = stablehlo.composite "stablehlo.identity" %arg0 {
    decomposition = @foo
  } : (!stablehlo.token) -> tensor<f32>
  func.return
}
