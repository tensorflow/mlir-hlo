// RUN: mlir-hlo-opt %s -chlo-legalize-to-hlo -hlo-legalize-to-lhlo \
// RUN: -std-bufferize -tensor-bufferize -finalizing-bufferize \
// RUN: --canonicalize -buffer-hoisting -buffer-deallocation \
// RUN: -copy-removal -canonicalize -cse -lhlo-legalize-to-linalg \
// RUN: -lhlo-fuse-linalg -convert-linalg-to-loops -canonicalize -cse \
// RUN: -convert-linalg-to-llvm -lower-affine -convert-scf-to-std \
// RUN: -convert-std-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s --dump-input=always

func @main() -> () {
  call @trivial_broadcast_wrapper() : () -> ()
  call @broadcast_in_X_dim_wrapper() : () -> ()
  call @broadcast_in_Y_dim_wrapper() : () -> ()
  call @broadcast_in_X_dim_transpose_wrapper() : () -> ()
  call @broadcast_in_Y_dim_transpose_wrapper() : () -> ()
  call @broadcast_scalar_1d_wrapper() : () -> ()
  call @broadcast_scalar_2d_wrapper() : () -> ()
  call @broadcast_to_the_same_shape() : () -> ()
  call @broadcast_1d_to_2d() : () -> ()
  call @broadcast_1d_to_2d_with_transpose() : () -> ()
  return
}

func private @print_memref_i8(memref<*xi8>) attributes { llvm.emit_c_interface }
func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @trivial_broadcast_wrapper() {
  %input_buf = memref.alloc() : memref<3xf32>

  %c1f32 = constant 1.0 : f32
  %c2f32 = constant 2.0 : f32
  %c3f32 = constant 3.0 : f32

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  memref.store %c1f32, %input_buf[%c0] : memref<3xf32>
  memref.store %c2f32, %input_buf[%c1] : memref<3xf32>
  memref.store %c3f32, %input_buf[%c2] : memref<3xf32>
  %input = memref.tensor_load %input_buf : memref<3xf32>

  // Test BroadcastInDimOp.
  %output = "mhlo.broadcast_in_dim"(%input) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<3xf32>) -> tensor<3x4xf32>

  %output_buf = memref.buffer_cast %output : memref<3x4xf32>

  %unranked_output = memref.cast %output_buf : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1,   1,   1,   1]
  // CHECK-NEXT: [2,   2,   2,   2]
  // CHECK-NEXT: [3,   3,   3,   3]

  // Test DynamicBroadcastInDimOp.
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %shape = tensor.from_elements %c3, %c4 : tensor<2xindex>
  %dyn_output = "mhlo.dynamic_broadcast_in_dim"(%input, %shape) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<3xf32>, tensor<2xindex>) -> tensor<3x4xf32>

  %dyn_output_buf = memref.buffer_cast %dyn_output : memref<3x4xf32>

  %unranked_dyn_output = memref.cast %dyn_output_buf
    : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_dyn_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1,   1,   1,   1]
  // CHECK-NEXT: [2,   2,   2,   2]
  // CHECK-NEXT: [3,   3,   3,   3]
  return
}

func @broadcast_in_X_dim_wrapper() {
  %input_buf = memref.alloc() : memref<1x4xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  memref.store %c1f32, %input_buf[%c0, %c0] : memref<1x4xf32>
  %c2f32 = constant 2.0 : f32
  %c1 = constant 1 : index
  memref.store %c2f32, %input_buf[%c0, %c1] : memref<1x4xf32>
  %c3f32 = constant 3.0 : f32
  %c2 = constant 2 : index
  memref.store %c3f32, %input_buf[%c0, %c2] : memref<1x4xf32>
  %c4f32 = constant 4.0 : f32
  %c3 = constant 3 : index
  memref.store %c4f32, %input_buf[%c0, %c3] : memref<1x4xf32>
  %input = memref.tensor_load %input_buf : memref<1x4xf32>

  // Test BroadcastInDimOp.
  %output = "mhlo.broadcast_in_dim"(%input) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<1x4xf32>) -> tensor<3x4xf32>

  %output_buf = memref.buffer_cast %output : memref<3x4xf32>

  %unranked_output = memref.cast %output_buf : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1,   2,   3,   4]
  // CHECK-NEXT: [1,   2,   3,   4]
  // CHECK-NEXT: [1,   2,   3,   4]

  // Test DynamicBroadcastInDimOp.
  %c4 = constant 4 : index
  %shape = tensor.from_elements %c3, %c4 : tensor<2xindex>
  %dyn_output = "mhlo.dynamic_broadcast_in_dim"(%input, %shape) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<1x4xf32>, tensor<2xindex>) -> tensor<3x4xf32>

  %dyn_output_buf = memref.buffer_cast %dyn_output : memref<3x4xf32>

  %unranked_dyn_output = memref.cast %dyn_output_buf
    : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_dyn_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1,   2,   3,   4]
  // CHECK-NEXT: [1,   2,   3,   4]
  // CHECK-NEXT: [1,   2,   3,   4]
  return
}

func @broadcast_in_Y_dim_wrapper() {
  %input_buf = memref.alloc() : memref<3x1xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  memref.store %c1f32, %input_buf[%c0, %c0] : memref<3x1xf32>
  %c2f32 = constant 2.0 : f32
  %c1 = constant 1 : index
  memref.store %c2f32, %input_buf[%c1, %c0] : memref<3x1xf32>
  %c3f32 = constant 3.0 : f32
  %c2 = constant 2 : index
  memref.store %c3f32, %input_buf[%c2, %c0] : memref<3x1xf32>
  %input = memref.tensor_load %input_buf : memref<3x1xf32>

  // Test BroadcastInDimOp.
  %output = "mhlo.broadcast_in_dim"(%input) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<3x1xf32>) -> tensor<3x4xf32>

  %output_buf = memref.buffer_cast %output : memref<3x4xf32>

  %unranked_output = memref.cast %output_buf : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1,   1,   1,   1]
  // CHECK-NEXT: [2,   2,   2,   2]
  // CHECK-NEXT: [3,   3,   3,   3]

  // Test DynamicBroadcastInDimOp.
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %shape = tensor.from_elements %c3, %c4 : tensor<2xindex>
  %dyn_output = "mhlo.dynamic_broadcast_in_dim"(%input, %shape) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<3x1xf32>, tensor<2xindex>) -> tensor<3x4xf32>

  %dyn_output_buf = memref.buffer_cast %dyn_output : memref<3x4xf32>

  %unranked_dyn_output = memref.cast %dyn_output_buf
    : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_dyn_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1,   1,   1,   1]
  // CHECK-NEXT: [2,   2,   2,   2]
  // CHECK-NEXT: [3,   3,   3,   3]
  return
}

func @broadcast_in_X_dim_transpose_wrapper() {
  %input_buf = memref.alloc() : memref<4x1xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  memref.store %c1f32, %input_buf[%c0, %c0] : memref<4x1xf32>
  %c2f32 = constant 2.0 : f32
  %c1 = constant 1 : index
  memref.store %c2f32, %input_buf[%c1, %c0] : memref<4x1xf32>
  %c3f32 = constant 3.0 : f32
  %c2 = constant 2 : index
  memref.store %c3f32, %input_buf[%c2, %c0] : memref<4x1xf32>
  %c4f32 = constant 4.0 : f32
  %c3 = constant 3 : index
  memref.store %c4f32, %input_buf[%c3, %c0] : memref<4x1xf32>
  %input = memref.tensor_load %input_buf : memref<4x1xf32>

  // Test BroadcastInDimOp.
  %output = "mhlo.broadcast_in_dim"(%input) {
    broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<4x1xf32>) -> tensor<3x4xf32>

  %output_buf = memref.buffer_cast %output : memref<3x4xf32>

  %unranked_output = memref.cast %output_buf : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1,   2,   3,   4]
  // CHECK-NEXT: [1,   2,   3,   4]
  // CHECK-NEXT: [1,   2,   3,   4]

  // Test DynamicBroadcastInDimOp.
  %c4 = constant 4 : index
  %shape = tensor.from_elements %c3, %c4 : tensor<2xindex>
  %dyn_output = "mhlo.dynamic_broadcast_in_dim"(%input, %shape) {
    broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<4x1xf32>, tensor<2xindex>) -> tensor<3x4xf32>

  %dyn_output_buf = memref.buffer_cast %dyn_output : memref<3x4xf32>

  %unranked_dyn_output = memref.cast %dyn_output_buf
    : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_dyn_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1,   2,   3,   4]
  // CHECK-NEXT: [1,   2,   3,   4]
  // CHECK-NEXT: [1,   2,   3,   4]
  return
}

func @broadcast_in_Y_dim_transpose_wrapper() {
  %input_buf = memref.alloc() : memref<1x3xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  memref.store %c1f32, %input_buf[%c0, %c0] : memref<1x3xf32>
  %c2f32 = constant 2.0 : f32
  %c1 = constant 1 : index
  memref.store %c2f32, %input_buf[%c0, %c1] : memref<1x3xf32>
  %c3f32 = constant 3.0 : f32
  %c2 = constant 2 : index
  memref.store %c3f32, %input_buf[%c0, %c2] : memref<1x3xf32>
  %input = memref.tensor_load %input_buf : memref<1x3xf32>

  // Test BroadcastInDimOp.
  %output = "mhlo.broadcast_in_dim"(%input) {
    broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<1x3xf32>) -> tensor<3x4xf32>

  %output_buf = memref.buffer_cast %output : memref<3x4xf32>

  %unranked_output = memref.cast %output_buf : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT-NEXT: [1,   1,   1,   1]
  // CHECK-NEXT-NEXT: [2,   2,   2,   2]
  // CHECK-NEXT-NEXT: [3,   3,   3,   3]

  // Test DynamicBroadcastInDimOp.
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %shape = tensor.from_elements %c3, %c4 : tensor<2xindex>
  %dyn_output = "mhlo.dynamic_broadcast_in_dim"(%input, %shape) {
    broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<1x3xf32>, tensor<2xindex>) -> tensor<3x4xf32>

  %dyn_output_buf = memref.buffer_cast %dyn_output : memref<3x4xf32>

  %unranked_dyn_output = memref.cast %dyn_output_buf
    : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_dyn_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT-NEXT: [1,   1,   1,   1]
  // CHECK-NEXT-NEXT: [2,   2,   2,   2]
  // CHECK-NEXT-NEXT: [3,   3,   3,   3]
  return
}

func @broadcast_scalar_1d_wrapper() {
  %input_buf = memref.alloc() : memref<1xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  memref.store %c1f32, %input_buf[%c0] : memref<1xf32>
  %input = memref.tensor_load %input_buf : memref<1xf32>

  // Test BroadcastInDimOp.
  %output = "mhlo.broadcast_in_dim"(%input) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<1xf32>) -> tensor<3x4xf32>

  %output_buf = memref.buffer_cast %output : memref<3x4xf32>

  %unranked_output = memref.cast %output_buf : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1, 1, 1, 1]
  // CHECK-NEXT: [1, 1, 1, 1]
  // CHECK-NEXT: [1, 1, 1, 1]

  // Test DynamicBroadcastInDimOp.
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %shape = tensor.from_elements %c3, %c4 : tensor<2xindex>
  %dyn_output = "mhlo.dynamic_broadcast_in_dim"(%input, %shape) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<1xf32>, tensor<2xindex>) -> tensor<3x4xf32>

  %dyn_output_buf = memref.buffer_cast %dyn_output : memref<3x4xf32>

  %unranked_dyn_output = memref.cast %dyn_output_buf
    : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_dyn_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1, 1, 1, 1]
  // CHECK-NEXT: [1, 1, 1, 1]
  // CHECK-NEXT: [1, 1, 1, 1]
  return
}

func @broadcast_scalar_2d_wrapper() {
  %input_buf = memref.alloc() : memref<1x1xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  memref.store %c1f32, %input_buf[%c0, %c0] : memref<1x1xf32>
  %input = memref.tensor_load %input_buf : memref<1x1xf32>

  // Test BroadcastInDimOp.
  %output = "mhlo.broadcast_in_dim"(%input) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<1x1xf32>) -> tensor<3x4xf32>

  %output_buf = memref.buffer_cast %output : memref<3x4xf32>

  %unranked_output = memref.cast %output_buf : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1, 1, 1, 1]
  // CHECK-NEXT: [1, 1, 1, 1]
  // CHECK-NEXT: [1, 1, 1, 1]

  // Test DynamicBroadcastInDimOp.
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %shape = tensor.from_elements %c3, %c4 : tensor<2xindex>
  %dyn_output = "mhlo.dynamic_broadcast_in_dim"(%input, %shape) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<1x1xf32>, tensor<2xindex>) -> tensor<3x4xf32>

  %dyn_output_buf = memref.buffer_cast %dyn_output : memref<3x4xf32>

  %unranked_dyn_output = memref.cast %dyn_output_buf
    : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_dyn_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
  // CHECK-NEXT: [1, 1, 1, 1]
  // CHECK-NEXT: [1, 1, 1, 1]
  // CHECK-NEXT: [1, 1, 1, 1]
  return
}

func @broadcast_to_the_same_shape() {
  %input_buf = memref.alloc() : memref<2x3xf32>

  %c1f32 = constant 1.0 : f32
  %c2f32 = constant 2.0 : f32
  %c3f32 = constant 3.0 : f32

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  memref.store %c1f32, %input_buf[%c0, %c0] : memref<2x3xf32>
  memref.store %c1f32, %input_buf[%c1, %c0] : memref<2x3xf32>
  memref.store %c2f32, %input_buf[%c0, %c1] : memref<2x3xf32>
  memref.store %c2f32, %input_buf[%c1, %c1] : memref<2x3xf32>
  memref.store %c3f32, %input_buf[%c0, %c2] : memref<2x3xf32>
  memref.store %c3f32, %input_buf[%c1, %c2] : memref<2x3xf32>
  %input = memref.tensor_load %input_buf : memref<2x3xf32>

  // Test BroadcastInDimOp.
  %output = "mhlo.broadcast_in_dim"(%input) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<2x3xf32>) -> tensor<2x3xf32>

  %output_buf = memref.buffer_cast %output : memref<2x3xf32>

  %unraked_output = memref.cast %output_buf : memref<2x3xf32> to memref<*xf32>
  call @print_memref_f32(%unraked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK-NEXT: [1,   2,   3]
  // CHECK-NEXT: [1,   2,   3]

  // Test DynamicBroadcastInDimOp.
  %shape = tensor.from_elements %c2, %c3 : tensor<2xindex>
  %dyn_output = "mhlo.dynamic_broadcast_in_dim"(%input, %shape) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<2x3xf32>, tensor<2xindex>) -> tensor<2x3xf32>

  %dyn_output_buf = memref.buffer_cast %dyn_output : memref<2x3xf32>

  %unranked_dyn_output = memref.cast %dyn_output_buf
    : memref<2x3xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_dyn_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK-NEXT: [1,   2,   3]
  // CHECK-NEXT: [1,   2,   3]
  return
}

func @broadcast_1d_to_2d() {
  %input_buf = memref.alloc() : memref<3xf32>

  %c1f32 = constant 1.0 : f32
  %c2f32 = constant 2.0 : f32
  %c3f32 = constant 3.0 : f32

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  memref.store %c1f32, %input_buf[%c0] : memref<3xf32>
  memref.store %c2f32, %input_buf[%c1] : memref<3xf32>
  memref.store %c3f32, %input_buf[%c2] : memref<3xf32>
  %input = memref.tensor_load %input_buf : memref<3xf32>

  // Test BroadcastInDimOp.
  %output = "mhlo.broadcast_in_dim"(%input) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<3xf32>) -> tensor<3x3xf32>

  %output_buf = memref.buffer_cast %output : memref<3x3xf32>

  %unraked_output = memref.cast %output_buf : memref<3x3xf32> to memref<*xf32>
  call @print_memref_f32(%unraked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1]
  // CHECK-NEXT: [1,   1,   1]
  // CHECK-NEXT: [2,   2,   2]
  // CHECK-NEXT: [3,   3,   3]

  // Test DynamicBroadcastInDimOp.
  %c3 = constant 3 : index
  %c4 = constant 3 : index
  %shape = tensor.from_elements %c3, %c4 : tensor<2xindex>
  %dyn_output = "mhlo.dynamic_broadcast_in_dim"(%input, %shape) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<3xf32>, tensor<2xindex>) -> tensor<3x3xf32>

  %dyn_output_buf = memref.buffer_cast %dyn_output : memref<3x3xf32>

  %unranked_dyn_output = memref.cast %dyn_output_buf
    : memref<3x3xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_dyn_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1]
  // CHECK-NEXT: [1,   1,   1]
  // CHECK-NEXT: [2,   2,   2]
  // CHECK-NEXT: [3,   3,   3]
  return
}

func @broadcast_1d_to_2d_with_transpose() {
  %input_buf = memref.alloc() : memref<3xf32>

  %c1f32 = constant 1.0 : f32
  %c2f32 = constant 2.0 : f32
  %c3f32 = constant 3.0 : f32

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  memref.store %c1f32, %input_buf[%c0] : memref<3xf32>
  memref.store %c2f32, %input_buf[%c1] : memref<3xf32>
  memref.store %c3f32, %input_buf[%c2] : memref<3xf32>
  %input = memref.tensor_load %input_buf : memref<3xf32>

  // Test BroadcastInDimOp.
  %output = "mhlo.broadcast_in_dim"(%input) {
    broadcast_dimensions = dense<1> : tensor<1xi64>
  } : (tensor<3xf32>) -> tensor<3x3xf32>

  %output_buf = memref.buffer_cast %output : memref<3x3xf32>

  %unraked_output = memref.cast %output_buf : memref<3x3xf32> to memref<*xf32>
  call @print_memref_f32(%unraked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1]
  // CHECK-NEXT: [1,   2,   3]
  // CHECK-NEXT: [1,   2,   3]
  // CHECK-NEXT: [1,   2,   3]

  // Test DynamicBroadcastInDimOp.
  %c3 = constant 3 : index
  %shape = tensor.from_elements %c3, %c3 : tensor<2xindex>
  %dyn_output = "mhlo.dynamic_broadcast_in_dim"(%input, %shape) {
    broadcast_dimensions = dense<1> : tensor<1xi64>
  } : (tensor<3xf32>, tensor<2xindex>) -> tensor<3x3xf32>

  %dyn_output_buf = memref.buffer_cast %dyn_output : memref<3x3xf32>

  %unranked_dyn_output = memref.cast %dyn_output_buf
    : memref<3x3xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_dyn_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1]
  // CHECK-NEXT: [1,   2,   3]
  // CHECK-NEXT: [1,   2,   3]
  // CHECK-NEXT: [1,   2,   3]
  return
}
