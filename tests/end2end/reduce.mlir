// RUN: mlir-hlo-opt %s -chlo-legalize-to-hlo -hlo-legalize-to-memref \
// RUN: -hlo-legalize-to-linalg -arith-bufferize -tensor-bufferize \
// RUN: -linalg-bufferize  -finalizing-bufferize \
// RUN: -canonicalize -buffer-hoisting \
// RUN: -buffer-deallocation -canonicalize -cse \
// RUN: -convert-linalg-to-loops -canonicalize -cse \
// RUN: -convert-linalg-to-llvm -lower-affine -convert-scf-to-cf \
// RUN: -arith-expand -memref-expand \
// RUN: -convert-memref-to-llvm -convert-func-to-llvm \
// RUN: -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN: -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @main() -> () {
  func.call @reduce_add() : () -> ()
  func.call @reduce_max() : () -> ()
  func.return
}

func.func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func.func @reduce_add() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Initialize input.
  %input = memref.alloc() : memref<2x3xf32>
  %dim_x = memref.dim %input, %c0 : memref<2x3xf32>
  %dim_y = memref.dim %input, %c1 : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%dim_x, %dim_y) step (%c1, %c1) {
    %i_i64 = arith.index_cast %i : index to i64
    %i_f32 = arith.sitofp %i_i64 : i64 to f32
    memref.store %i_f32, %input[%i, %j] : memref<2x3xf32>
  }
  %unranked_input = memref.cast %input : memref<2x3xf32> to memref<*xf32>
  func.call @print_memref_f32(%unranked_input) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK: [0,   0,   0]
  // CHECK: [1,   1,   1]

  %in = bufferization.to_tensor %input : memref<2x3xf32>
  %init = mhlo.constant dense<0.000000e+00> : tensor<f32>

  %reduce = "mhlo.reduce"(%in, %init) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>}
      : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>

  %output = bufferization.to_memref %reduce : memref<2xf32>
  %unranked_output = memref.cast %output : memref<2xf32> to memref<*xf32>
  func.call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 1 offset = 0 sizes = [2] strides = [1]
  // CHECK: [0,  3]
  func.return
}

func.func @reduce_max() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Initialize input.
  %input = memref.alloc() : memref<2x3xf32>
  %dim_x = memref.dim %input, %c0 : memref<2x3xf32>
  %dim_y = memref.dim %input, %c1 : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%dim_x, %dim_y) step (%c1, %c1) {
    %i_i64 = arith.index_cast %i : index to i64
    %i_f32 = arith.sitofp %i_i64 : i64 to f32
    memref.store %i_f32, %input[%i, %j] : memref<2x3xf32>
  }
  %unranked_input = memref.cast %input : memref<2x3xf32> to memref<*xf32>
  func.call @print_memref_f32(%unranked_input) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK: [0,   0,   0]
  // CHECK: [1,   1,   1]

  %in = bufferization.to_tensor %input : memref<2x3xf32>
  %init = mhlo.constant dense<0xff800000> : tensor<f32>

  %reduce = "mhlo.reduce"(%in, %init) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.maximum %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>}
      : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>

  %output = bufferization.to_memref %reduce : memref<2xf32>
  %unranked_output = memref.cast %output : memref<2xf32> to memref<*xf32>
  func.call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 1 offset = 0 sizes = [2] strides = [1]
  // CHECK: [0,  1]
  func.return
}
