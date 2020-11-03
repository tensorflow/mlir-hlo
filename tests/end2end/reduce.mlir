// RUN: mlir-hlo-opt %s -chlo-legalize-to-hlo \
// RUN: -hlo-legalize-to-lhlo -buffer-hoisting \
// RUN: -buffer-deallocation -copy-removal -canonicalize -cse \
// RUN: -lhlo-legalize-to-linalg -lhlo-fuse-linalg -convert-linalg-to-loops \
// RUN: -lower-affine -convert-scf-to-std -canonicalize -cse \
// RUN: -test-lhlo-legalize-to-llvm | mlir-cpu-runner -e main \
// RUN: -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s

func @main() -> () {
  call @reduce_add() : () -> ()
  call @reduce_max() : () -> ()
  return
}

func @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @reduce_add() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // Initialize input.
  %input = alloc() : memref<2x3xf32>
  %dim_x = dim %input, %c0 : memref<2x3xf32>
  %dim_y = dim %input, %c1 : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%dim_x, %dim_y) step (%c1, %c1) {
    %i_i64 = index_cast %i : index to i64
    %i_f32 = sitofp %i_i64 : i64 to f32
    store %i_f32, %input[%i, %j] : memref<2x3xf32>
  }
  %unranked_input = memref_cast %input : memref<2x3xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_input) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK: [0,   0,   0]
  // CHECK: [1,   1,   1]

  %in = tensor_load %input : memref<2x3xf32>
  %init = mhlo.constant dense<0.000000e+00> : tensor<f32>

  %reduce = "mhlo.reduce"(%in, %init) ( {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>}
      : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>

  %output = alloc() : memref<2xf32>
  tensor_store %reduce, %output : memref<2xf32>
  %unranked_output = memref_cast %output : memref<2xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 1 offset = 0 sizes = [2] strides = [1]
  // CHECK: [0,  3]
  return
}

func @reduce_max() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // Initialize input.
  %input = alloc() : memref<2x3xf32>
  %dim_x = dim %input, %c0 : memref<2x3xf32>
  %dim_y = dim %input, %c1 : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%dim_x, %dim_y) step (%c1, %c1) {
    %i_i64 = index_cast %i : index to i64
    %i_f32 = sitofp %i_i64 : i64 to f32
    store %i_f32, %input[%i, %j] : memref<2x3xf32>
  }
  %unranked_input = memref_cast %input : memref<2x3xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_input) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK: [0,   0,   0]
  // CHECK: [1,   1,   1]

  %in = tensor_load %input : memref<2x3xf32>
  %init = mhlo.constant dense<0xff800000> : tensor<f32>

  %reduce = "mhlo.reduce"(%in, %init) ( {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.maximum %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>}
      : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>

  %output = alloc() : memref<2xf32>
  tensor_store %reduce, %output : memref<2xf32>
  %unranked_output = memref_cast %output : memref<2xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 1 offset = 0 sizes = [2] strides = [1]
  // CHECK: [0,  1]
  return
}
