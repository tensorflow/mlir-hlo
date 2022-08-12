// RUN: mlir-hlo-opt %s \
// RUN:   --gml-st-pipeline="tile-sizes=1,1 fuse lower-to-loops" \
// RUN:   --convert-scf-to-cf \
// RUN:   --generic-host-to-llvm | \
// RUN: mlir-cpu-runner \
// RUN:   -e main --entry-point-result=void \
// RUN:   --shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @concat(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %concat = "mhlo.concatenate"(%a, %b, %c) { dimension = 1 }
      : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %concat : tensor<?x?xf32>
}

func.func @main() {
  %test_arg_a = arith.constant dense<[[1.11, 1.12], [1.21, 1.22]]> : tensor<2x2xf32>
  %test_arg_a_ = tensor.cast %test_arg_a : tensor<2x2xf32> to tensor<?x?xf32>
  %test_arg_b = arith.constant dense<[[2.11, 212., 2.13], [2.21, 2.22, 2.23]]> : tensor<2x3xf32>
  %test_arg_b_ = tensor.cast %test_arg_b : tensor<2x3xf32> to tensor<?x?xf32>
  %test_arg_c = arith.constant dense<[[], []]> : tensor<2x0xf32>
  %test_arg_c_ = tensor.cast %test_arg_c : tensor<2x0xf32> to tensor<?x?xf32>
  %test_concat = func.call @concat(%test_arg_a_, %test_arg_b_, %test_arg_c_)
      : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK: rank = 2
  // CHECK: offset = 0
  // CHECK: sizes = [2, 5]
  // CHECK: strides = [5, 1]
  // CHECK: data =
  // CHECK: 1.11, 1.12, 2.11, 212, 2.13
  // CHECK: 1.21, 1.22, 2.21, 2.22, 2.23
  %test_concat_unranked = tensor.cast %test_concat
      : tensor<?x?xf32> to tensor<*xf32>
  func.call @printMemrefF32(%test_concat_unranked) : (tensor<*xf32>) -> ()
  func.return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
