// RUN: mlir-hlo-opt %s --mhlo-legalize-trigonometric-to-approximation \
// RUN: --finalize-memref-to-llvm --convert-math-to-llvm --convert-func-to-llvm \
// RUN: -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner -e main -entry-point-result=void --shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext |\
// RUN: FileCheck %s

func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }

// Helper function to print scalar values.
func.func @print_f32(%arg : f32) {
  %mem = memref.alloca() : memref<1xf32>
  %c0 = arith.constant 0 : index
  memref.store %arg, %mem[%c0] : memref<1xf32>
  %mem_unranked = memref.cast %mem : memref<1xf32> to memref<*xf32>
  func.call @printMemrefF32(%mem_unranked) : (memref<*xf32>) -> ()
  func.return
}

func.func @tanh_f32(%arg : f32) {
  %res = math.tanh %arg : f32
  func.call @print_f32(%res) : (f32) -> ()
  func.return
}

func.func @main() {
  // Some constants to use as arguments.
  %cf_n50_0 = arith.constant -50.0 : f32
  %cf_n5_0 = arith.constant -5.0 : f32
  %cf_n3_0 = arith.constant -3.0 : f32
  %cf_n2_0 = arith.constant -2.0 : f32
  %cf_n1_0 = arith.constant -1.0 : f32
  %cf_n0_5 = arith.constant -0.5 : f32
  %cf_n0_1 = arith.constant -0.1 : f32
  %cf_0_0 = arith.constant 0.0 : f32
  %cf_0_1 = arith.constant 0.1 : f32
  %cf_0_5 = arith.constant 0.5 : f32
  %cf_1_0 = arith.constant 1.0 : f32
  %cf_2_0 = arith.constant 2.0 : f32
  %cf_3_0 = arith.constant 3.0 : f32
  %cf_5_0 = arith.constant 5.0 : f32
  %cf_50_0 = arith.constant 50.0 : f32

  // Tanh.
  func.call @tanh_f32(%cf_n50_0) : (f32) -> ()
  // CHECK: -1
  func.call @tanh_f32(%cf_n5_0) : (f32) -> ()
  // CHECK: -0.999{{.*}}
  func.call @tanh_f32(%cf_n3_0) : (f32) -> ()
  // CHECK: -0.995{{.*}}
  func.call @tanh_f32(%cf_n2_0) : (f32) -> ()
  // CHECK: -0.964{{.*}}
  func.call @tanh_f32(%cf_n1_0) : (f32) -> ()
  // CHECK: -0.761{{.*}}
  func.call @tanh_f32(%cf_n0_5) : (f32) -> ()
  // CHECK: -0.462{{.*}}
  func.call @tanh_f32(%cf_n0_1) : (f32) -> ()
  // CHECK: -0.099{{.*}}
  func.call @tanh_f32(%cf_0_0) : (f32) -> ()
  // CHECK: 0
  func.call @tanh_f32(%cf_0_1) : (f32) -> ()
  // CHECK: 0.099{{.*}}
  func.call @tanh_f32(%cf_0_5) : (f32) -> ()
  // CHECK: 0.462{{.*}}
  func.call @tanh_f32(%cf_1_0) : (f32) -> ()
  // CHECK: 0.761{{.*}}
  func.call @tanh_f32(%cf_2_0) : (f32) -> ()
  // CHECK: 0.964{{.*}}
  func.call @tanh_f32(%cf_3_0) : (f32) -> ()
  // CHECK: 0.995{{.*}}
  func.call @tanh_f32(%cf_5_0) : (f32) -> ()
  // CHECK: 0.999{{.*}}
  func.call @tanh_f32(%cf_50_0) : (f32) -> ()
  // CHECK: 1

  func.return
}
