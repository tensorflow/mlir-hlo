// RUN: mlir-hlo-opt %s \
// RUN: --hlo-canonicalize-scatter --legalize-mhlo-to-thlo \
// RUN: --hlo-legalize-to-linalg \
// RUN: --gml-tiling="tile-sizes=1 distribute=false op-name=linalg.generic" \
// RUN: --scalarize -cse --canonicalize |\
// RUN: mlir-hlo-opt \
// RUN: --empty-tensor-to-alloc-tensor \
// RUN: --hlo-one-shot-bufferize --canonicalize -cse \
// RUN: --convert-bufferization-to-memref \
// RUN: --gml-st-to-scf --buffer-results-to-out-params --convert-scf-to-cf \
// RUN: --generic-host-to-llvm -cse --canonicalize |\
// RUN: mlir-cpu-runner \
// RUN: -e main -entry-point-result=void \
// RUN: --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

func.func @abs(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %0 = mhlo.abs %arg0 : tensor<5xf32>
  func.return %0 : tensor<5xf32>
}

func.func @neg(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = mhlo.negate %arg0 : tensor<8xf32>
  func.return %0 : tensor<8xf32>
}

func.func @main() {

  // CHECK: 1, 1, 0, 0, 0.1
  %abs_test = arith.constant dense<[-1.0, 1.0, 0.0, -0.0, 0.1]> : tensor<5xf32>
  %abs_res = func.call @abs(%abs_test) : (tensor<5xf32>) -> tensor<5xf32>
  %abs_res_unranked = tensor.cast %abs_res : tensor<5xf32> to tensor<*xf32>
  func.call @printMemrefF32(%abs_res_unranked) : (tensor<*xf32>) -> ()

  // CHECK: 1, -1, -0, 0, 0.1, -0.1, -3, 3
  %neg_test = arith.constant dense<[-1.0, 1.0, 0.0, -0.0, -0.1, 0.1, 3.0, -3.0]>
      : tensor<8xf32>
  %neg_res = func.call @neg(%neg_test) : (tensor<8xf32>) -> tensor<8xf32>
  %neg_res_unranked = tensor.cast %neg_res : tensor<8xf32> to tensor<*xf32>
  func.call @printMemrefF32(%neg_res_unranked) : (tensor<*xf32>) -> ()

  func.return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
