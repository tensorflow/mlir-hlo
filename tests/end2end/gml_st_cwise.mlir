// RUN: mlir-hlo-opt --split-input-file %s \
// RUN:  --gml-st-pipeline="tile-sizes=256 lower-to-loops" \
// RUN:  --convert-scf-to-cf \
// RUN:  --generic-host-to-llvm \
// RUN: | FileCheck %s

func.func @abs(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
  %0 = mhlo.abs %arg0 : tensor<2048xf32>
  return %0 : tensor<2048xf32>
}
// CHECK-LABEL: llvm.func @abs
