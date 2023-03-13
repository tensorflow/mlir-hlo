// RUN: mlir-hlo-opt %s \
// RUN: --legalize-mhlo-to-thlo="enable-experimental" \
// RUN: --gml-tile-by-one  --gml-st-rewrite-forall-ops --canonicalize --cse | \
// RUN: mlir-hlo-opt --empty-tensor-to-alloc-tensor  --hlo-one-shot-bufferize \
// RUN: --canonicalize --cse --convert-bufferization-to-memref | \
// RUN: mlir-hlo-opt --thlo-legalize-sort --cse --canonicalize | \
// RUN: mlir-hlo-opt --generic-host-to-llvm --cse --canonicalize | \
// RUN: mlir-cpu-runner \
// RUN: -e main -entry-point-result=void \
// RUN: --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @sort(%input0: tensor<2x5xf32>, %input1: tensor<2x5xi32>)
    -> (tensor<2x5xf32>, tensor<2x5xi32>) {
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<2x5xf32>, tensor<2x5xi32>) -> (tensor<2x5xf32>, tensor<2x5xi32>)
  func.return %0#0, %0#1 : tensor<2x5xf32>, tensor<2x5xi32>
}

func.func @main() {
  %input0 = arith.constant dense<[
    [4.0, 2.0, 1.0, 5.0,  3.0],
    [6.0, 9.0, 8.0, 7.0, 10.0]
  ]> : tensor<2x5xf32>

  %input1 = arith.constant dense<[
    [1, 2, 3, 4,  5],
    [6, 7, 8, 9, 10]
  ]> : tensor<2x5xi32>

  %results:2 = func.call @sort(%input0, %input1)
      : (tensor<2x5xf32>, tensor<2x5xi32>) -> (tensor<2x5xf32>, tensor<2x5xi32>)

  // CHECK: rank = 2 offset = 0 sizes = [2, 5] strides = [5, 1]
  // CHECK-NEXT: [1, 2, 3, 4, 5]
  // CHECK-NEXT: [6, 7, 8, 9, 10]
  %result0_unranked = tensor.cast %results#0 : tensor<2x5xf32> to tensor<*xf32>
  func.call @printMemrefF32(%result0_unranked) : (tensor<*xf32>) -> ()

  // CHECK: rank = 2 offset = 0 sizes = [2, 5] strides = [5, 1]
  // CHECK-NEXT: [3, 2, 5, 1, 4]
  // CHECK-NEXT: [6, 9, 8, 7, 10]
  %result1_unranked = tensor.cast %results#1 : tensor<2x5xi32> to tensor<*xi32>
  func.call @printMemrefI32(%result1_unranked) : (tensor<*xi32>) -> ()

  func.return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
func.func private @printMemrefI32(%ptr : tensor<*xi32>)
