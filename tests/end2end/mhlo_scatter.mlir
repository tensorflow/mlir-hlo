// RUN: mlir-hlo-opt %s \
// RUN: --hlo-canonicalize-scatter --legalize-mhlo-to-thlo \
// RUN: --hlo-legalize-to-linalg  --gml-tile-by-one \
// RUN: --gml-st-rewrite-forall-ops --scalarize -cse --canonicalize |\
// RUN: mlir-hlo-opt \
// RUN: --hlo-one-shot-bufferize --canonicalize -cse \
// RUN: --convert-bufferization-to-memref --convert-linalg-to-loops \
// RUN: --buffer-results-to-out-params --convert-scf-to-cf \
// RUN: --generic-host-to-llvm -cse --canonicalize |\
// RUN: mlir-cpu-runner \
// RUN: -e main -entry-point-result=void \
// RUN: --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @scatter(%indices: tensor<5x2xi64>, %updates: tensor<5x2x1xf32>,
                   %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = "mhlo.scatter"(%init, %indices, %updates) ({
    ^bb0(%out: tensor<f32>,  %in: tensor<f32>):
      %sum = mhlo.add %in, %out : tensor<f32>
      "mhlo.return"(%sum) : (tensor<f32>) -> ()
    }) {
      scatter_dimension_numbers = #mhlo.scatter<
        update_window_dims = [1, 2],
        inserted_window_dims = [],
        scatter_dims_to_operand_dims = [0, 1],
        index_vector_dim = 1,
      >,
      unique_indices = false,
      indices_are_sorted = false
    } : (tensor<3x3xf32>, tensor<5x2xi64>, tensor<5x2x1xf32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}

func.func @main() {
  %updates = arith.constant dense<[
    [[1.0], [1.0]],
    [[2.0], [2.0]],
    [[3.0], [3.0]],
    [[4.0], [4.0]],
    [[5.0], [5.0]]
  ]> : tensor<5x2x1xf32>

  %indices = arith.constant dense<[
    [0, 0],
    [1, 0],
    [1, 1],
    [2, 2],
    [3, 3]
  ]> : tensor<5x2xi64>

  %init = arith.constant dense<0.0> : tensor<3x3xf32>

  %result = func.call @scatter(%indices, %updates, %init)
      : (tensor<5x2xi64>, tensor<5x2x1xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>

  // CHECK: rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1]
  // CHECK-NEXT: [1, 0, 0]
  // CHECK-NEXT: [3, 3, 0]
  // CHECK-NEXT: [2, 3, 0]
  %result_unranked = tensor.cast %result : tensor<3x3xf32> to tensor<*xf32>
  func.call @printMemrefF32(%result_unranked) : (tensor<*xf32>) -> ()

  func.return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
