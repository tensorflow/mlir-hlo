// RUN: mlir-hlo-opt %s --legalize-mhlo-to-thlo \
// RUN: --gml-tile-by-one --gml-st-rewrite-forall-ops --scalarize \
// RUN: --empty-tensor-to-alloc-tensor --hlo-one-shot-bufferize \
// RUN: --convert-scf-to-cf --generic-host-to-llvm | \
// RUN: mlir-cpu-runner \
// RUN: -e main -entry-point-result=void \
// RUN: --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @reverse(%arg : tensor<?x2xf32>) -> tensor<?x2xf32> {
  %0 = "mhlo.reverse"(%arg)
      {dimensions = dense<[0, 1]> : tensor<2xi64>}
      : (tensor<?x2xf32>) -> tensor<?x2xf32>
  func.return %0 : tensor<?x2xf32>
}

func.func @main() {
  %test_arg = arith.constant dense<[[1.1, 2.2],[3.3, 4.4],[5.5, 6.6]]>
      : tensor<3x2xf32>
  %test_arg_ = tensor.cast %test_arg : tensor<3x2xf32> to tensor<?x2xf32>

  %test_res = func.call @reverse(%test_arg_)
      : (tensor<?x2xf32>) -> tensor<?x2xf32>

  %test_res_unranked = tensor.cast %test_res
      : tensor<?x2xf32> to tensor<*xf32>
  func.call @printMemrefF32(%test_res_unranked) : (tensor<*xf32>) -> ()

  func.return
}

//      CHECK: 6.6, 5.5
// CHECK-NEXT: 4.4, 3.3
// CHECK-NEXT: 2.2, 1.1

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
