// RUN: mlir-hlo-opt %s -chlo-legalize-to-hlo -hlo-legalize-to-lhlo -buffer-hoisting -buffer-deallocation -copy-removal -canonicalize -cse -lhlo-legalize-to-linalg -lhlo-fuse-linalg -convert-linalg-to-loops -canonicalize -cse -convert-linalg-to-llvm -convert-std-to-llvm | mlir-cpu-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | FileCheck %s

func @main() -> () {
  call @trivial_broadcast_wrapper() : () -> ()
  call @broadcast_in_X_dim_wrapper() : () -> ()
  call @broadcast_in_Y_dim_wrapper() : () -> ()
  call @broadcast_in_X_dim_transpose_wrapper() : () -> ()
  call @broadcast_in_Y_dim_transpose_wrapper() : () -> ()
  call @broadcast_scalar_1d_wrapper() : () -> ()
  call @broadcast_scalar_2d_wrapper() : () -> ()
  return
}

func @print_memref_i8(memref<*xi8>) attributes { llvm.emit_c_interface }
func @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @trivial_broadcast_wrapper() {
  %input = alloc() : memref<3xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  store %c1f32, %input[%c0] : memref<3xf32>
  %c2f32 = constant 2.0 : f32
  %c1 = constant 1 : index
  store %c2f32, %input[%c1] : memref<3xf32>
  %c3f32 = constant 3.0 : f32
  %c2 = constant 2 : index
  store %c3f32, %input[%c2] : memref<3xf32>
  %input_tensor = tensor_load %input : memref<3xf32>

  %output_tensor = "mhlo.broadcast_in_dim"(%input_tensor) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<3xf32>) -> tensor<3x4xf32>

  %output = alloc() : memref<3x4xf32>
  tensor_store %output_tensor, %output : memref<3x4xf32>

  %cast_for_print = memref_cast %output : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%cast_for_print) : (memref<*xf32>) -> ()
  return
}
// CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
// CHECK: [1,   1,   1,   1]
// CHECK: [2,   2,   2,   2]
// CHECK: [3,   3,   3,   3]

func @broadcast_in_X_dim_wrapper() {
  %input = alloc() : memref<1x4xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  store %c1f32, %input[%c0, %c0] : memref<1x4xf32>
  %c2f32 = constant 2.0 : f32
  %c1 = constant 1 : index
  store %c2f32, %input[%c0, %c1] : memref<1x4xf32>
  %c3f32 = constant 3.0 : f32
  %c2 = constant 2 : index
  store %c3f32, %input[%c0, %c2] : memref<1x4xf32>
  %c4f32 = constant 4.0 : f32
  %c3 = constant 3 : index
  store %c4f32, %input[%c0, %c3] : memref<1x4xf32>
  %input_tensor = tensor_load %input : memref<1x4xf32>

  %output_tensor = "mhlo.broadcast_in_dim"(%input_tensor) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<1x4xf32>) -> tensor<3x4xf32>

  %output = alloc() : memref<3x4xf32>
  tensor_store %output_tensor, %output : memref<3x4xf32>

  %cast_for_print = memref_cast %output : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%cast_for_print) : (memref<*xf32>) -> ()
  return
}
// CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
// CHECK: [1,   2,   3,   4]
// CHECK: [1,   2,   3,   4]
// CHECK: [1,   2,   3,   4]

func @broadcast_in_Y_dim_wrapper() {
  %input = alloc() : memref<3x1xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  store %c1f32, %input[%c0, %c0] : memref<3x1xf32>
  %c2f32 = constant 2.0 : f32
  %c1 = constant 1 : index
  store %c2f32, %input[%c1, %c0] : memref<3x1xf32>
  %c3f32 = constant 3.0 : f32
  %c2 = constant 2 : index
  store %c3f32, %input[%c2, %c0] : memref<3x1xf32>
  %input_tensor = tensor_load %input : memref<3x1xf32>

  %output_tensor = "mhlo.broadcast_in_dim"(%input_tensor) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<3x1xf32>) -> tensor<3x4xf32>

  %output = alloc() : memref<3x4xf32>
  tensor_store %output_tensor, %output : memref<3x4xf32>

  %cast_for_print = memref_cast %output : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%cast_for_print) : (memref<*xf32>) -> ()
  return
}
// CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
// CHECK: [1,   1,   1,   1]
// CHECK: [2,   2,   2,   2]
// CHECK: [3,   3,   3,   3]

func @broadcast_in_X_dim_transpose_wrapper() {
  %input = alloc() : memref<4x1xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  store %c1f32, %input[%c0, %c0] : memref<4x1xf32>
  %c2f32 = constant 2.0 : f32
  %c1 = constant 1 : index
  store %c2f32, %input[%c1, %c0] : memref<4x1xf32>
  %c3f32 = constant 3.0 : f32
  %c2 = constant 2 : index
  store %c3f32, %input[%c2, %c0] : memref<4x1xf32>
  %c4f32 = constant 4.0 : f32
  %c3 = constant 3 : index
  store %c4f32, %input[%c3, %c0] : memref<4x1xf32>
  %input_tensor = tensor_load %input : memref<4x1xf32>

  %output_tensor = "mhlo.broadcast_in_dim"(%input_tensor) {
    broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<4x1xf32>) -> tensor<3x4xf32>

  %output = alloc() : memref<3x4xf32>
  tensor_store %output_tensor, %output : memref<3x4xf32>

  %cast_for_print = memref_cast %output : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%cast_for_print) : (memref<*xf32>) -> ()
  return
}
// CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
// CHECK: [1,   2,   3,   4]
// CHECK: [1,   2,   3,   4]
// CHECK: [1,   2,   3,   4]

func @broadcast_in_Y_dim_transpose_wrapper() {
  %input = alloc() : memref<1x3xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  store %c1f32, %input[%c0, %c0] : memref<1x3xf32>
  %c2f32 = constant 2.0 : f32
  %c1 = constant 1 : index
  store %c2f32, %input[%c0, %c1] : memref<1x3xf32>
  %c3f32 = constant 3.0 : f32
  %c2 = constant 2 : index
  store %c3f32, %input[%c0, %c2] : memref<1x3xf32>
  %input_tensor = tensor_load %input : memref<1x3xf32>

  %output_tensor = "mhlo.broadcast_in_dim"(%input_tensor) {
    broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<1x3xf32>) -> tensor<3x4xf32>

  %output = alloc() : memref<3x4xf32>
  tensor_store %output_tensor, %output : memref<3x4xf32>

  %cast_for_print = memref_cast %output : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%cast_for_print) : (memref<*xf32>) -> ()
  return
}
// CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
// CHECK: [1,   1,   1,   1]
// CHECK: [2,   2,   2,   2]
// CHECK: [3,   3,   3,   3]

func @broadcast_scalar_1d_wrapper() {
  %input = alloc() : memref<1xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  store %c1f32, %input[%c0] : memref<1xf32>
  %input_tensor = tensor_load %input : memref<1xf32>

  %output_tensor = "mhlo.broadcast_in_dim"(%input_tensor) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<1xf32>) -> tensor<3x4xf32>

  %output = alloc() : memref<3x4xf32>
  tensor_store %output_tensor, %output : memref<3x4xf32>

  %cast_for_print = memref_cast %output : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%cast_for_print) : (memref<*xf32>) -> ()
  return
}
// CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
// CHECK: [1, 1, 1, 1]
// CHECK: [1, 1, 1, 1]
// CHECK: [1, 1, 1, 1]

func @broadcast_scalar_2d_wrapper() {
  %input = alloc() : memref<1x1xf32>
  %c1f32 = constant 1.0 : f32
  %c0 = constant 0 : index
  store %c1f32, %input[%c0, %c0] : memref<1x1xf32>
  %input_tensor = tensor_load %input : memref<1x1xf32>

  %output_tensor = "mhlo.broadcast_in_dim"(%input_tensor) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<1x1xf32>) -> tensor<3x4xf32>

  %output = alloc() : memref<3x4xf32>
  tensor_store %output_tensor, %output : memref<3x4xf32>

  %cast_for_print = memref_cast %output : memref<3x4xf32> to memref<*xf32>
  call @print_memref_f32(%cast_for_print) : (memref<*xf32>) -> ()
  return
}
// CHECK: rank = 2 offset = 0 sizes = [3, 4] strides = [4, 1]
// CHECK: [1, 1, 1, 1]
// CHECK: [1, 1, 1, 1]
// CHECK: [1, 1, 1, 1]

