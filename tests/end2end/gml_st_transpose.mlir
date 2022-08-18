// RUN: mlir-hlo-opt --split-input-file %s \
// RUN:   --gml-st-pipeline="tile-sizes=[4,4] fuse lower-to-loops" \
// RUN:   --convert-scf-to-cf \
// RUN:   --generic-host-to-llvm | \
// RUN: mlir-cpu-runner \
// RUN:   -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-hlo-opt --split-input-file %s \
// RUN:   --gml-st-pipeline="tile-sizes=[1,1] fuse lower-to-loops" \
// RUN:   --convert-scf-to-cf \
// RUN:   --generic-host-to-llvm | \
// RUN: mlir-cpu-runner \
// RUN:   -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @abs(%arg0: tensor<5x2xf32>) -> tensor<2x5xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x2xf32>) -> tensor<2x5xf32>
  %1 = mhlo.abs %0 : tensor<2x5xf32>
  func.return %1 : tensor<2x5xf32>
}

func.func @main() {
  // CHECK: [1.1, 0.1, 0.3, 2.1, 0]
  // CHECK-NEXT: [1.2, 0.2, 2.2, 0, 0.3]
  %abs_test = arith.constant dense<[[-1.1, 1.2], [0.1, -0.2], [0.3, -2.2], [2.1, 0.0], [-0.0, 0.3]]> : tensor<5x2xf32>
  %abs_res = func.call @abs(%abs_test) : (tensor<5x2xf32>) -> tensor<2x5xf32>
  %abs_res_unranked = tensor.cast %abs_res : tensor<2x5xf32> to tensor<*xf32>
  func.call @printMemrefF32(%abs_res_unranked) : (tensor<*xf32>) -> ()

  func.return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
