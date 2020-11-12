// RUN: mlir-hlo-opt %s --mhlo-legalize-trigonometric-to-approximation --convert-std-to-llvm | mlir-cpu-runner -e main -entry-point-result=void --shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | FileCheck %s

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

// Helper function to print scalar values.
func @print_f32(%arg : f32) {
  %mem = alloca() : memref<1xf32>
  %c0 = constant 0 : index
  store %arg, %mem[%c0] : memref<1xf32>
  %mem_unranked = memref_cast %mem : memref<1xf32> to memref<*xf32>
  call @print_memref_f32(%mem_unranked) : (memref<*xf32>) -> ()
  return
}

// Compute and print trigonometric function.
func @atan_f32(%arg : f32) {
  %res = atan %arg : f32
  call @print_f32(%res) : (f32) -> ()
  return
}

func @atan2_f32(%arg0 : f32, %arg1 : f32) {
  %res = atan2 %arg0, %arg1 : f32
  call @print_f32(%res) : (f32) -> ()
  return
}

func @tanh_f32(%arg : f32) {
  %res = tanh %arg : f32
  call @print_f32(%res) : (f32) -> ()
  return
}

func @main() {
  // Some constants to use as arguments.
  %cf_n50_0 = constant -50.0 : f32
  %cf_n5_0 = constant -5.0 : f32
  %cf_n3_0 = constant -3.0 : f32
  %cf_n2_0 = constant -2.0 : f32
  %cf_n1_0 = constant -1.0 : f32
  %cf_n0_5 = constant -0.5 : f32
  %cf_n0_1 = constant -0.1 : f32
  %cf_0_0 = constant 0.0 : f32
  %cf_0_1 = constant 0.1 : f32
  %cf_0_5 = constant 0.5 : f32
  %cf_1_0 = constant 1.0 : f32
  %cf_2_0 = constant 2.0 : f32
  %cf_3_0 = constant 3.0 : f32
  %cf_5_0 = constant 5.0 : f32
  %cf_50_0 = constant 50.0 : f32

  // Tanh.
  call @tanh_f32(%cf_n50_0) : (f32) -> ()
  // CHECK: -1
  call @tanh_f32(%cf_n5_0) : (f32) -> ()
  // CHECK: -0.999{{.*}}
  call @tanh_f32(%cf_n3_0) : (f32) -> ()
  // CHECK: -0.995{{.*}}
  call @tanh_f32(%cf_n2_0) : (f32) -> ()
  // CHECK: -0.964{{.*}}
  call @tanh_f32(%cf_n1_0) : (f32) -> ()
  // CHECK: -0.761{{.*}}
  call @tanh_f32(%cf_n0_5) : (f32) -> ()
  // CHECK: -0.462{{.*}}
  call @tanh_f32(%cf_n0_1) : (f32) -> ()
  // CHECK: -0.099{{.*}}
  call @tanh_f32(%cf_0_0) : (f32) -> ()
  // CHECK: 0
  call @tanh_f32(%cf_0_1) : (f32) -> ()
  // CHECK: 0.099{{.*}}
  call @tanh_f32(%cf_0_5) : (f32) -> ()
  // CHECK: 0.462{{.*}}
  call @tanh_f32(%cf_1_0) : (f32) -> ()
  // CHECK: 0.761{{.*}}
  call @tanh_f32(%cf_2_0) : (f32) -> ()
  // CHECK: 0.964{{.*}}
  call @tanh_f32(%cf_3_0) : (f32) -> ()
  // CHECK: 0.995{{.*}}
  call @tanh_f32(%cf_5_0) : (f32) -> ()
  // CHECK: 0.999{{.*}}
  call @tanh_f32(%cf_50_0) : (f32) -> ()
  // CHECK: 1

  // Atan2 with divisor 1.
  call @atan2_f32(%cf_n50_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: -1.550{{.*}}
  call @atan2_f32(%cf_n5_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: -1.373{{.*}}
  call @atan2_f32(%cf_n3_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: -1.249{{.*}}
  call @atan2_f32(%cf_n2_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: -1.107{{.*}}
  call @atan2_f32(%cf_n1_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: -0.785{{.*}}
  call @atan2_f32(%cf_n0_5, %cf_1_0) : (f32, f32) -> ()
  // CHECK: -0.463{{.*}}
  call @atan2_f32(%cf_n0_1, %cf_1_0) : (f32, f32) -> ()
  // CHECK: -0.099{{.*}}
  call @atan2_f32(%cf_0_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: 0
  call @atan2_f32(%cf_0_1, %cf_1_0) : (f32, f32) -> ()
  // CHECK: 0.099{{.*}}
  call @atan2_f32(%cf_0_5, %cf_1_0) : (f32, f32) -> ()
  // CHECK: 0.463{{.*}}
  call @atan2_f32(%cf_1_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: 0.785{{.*}}
  call @atan2_f32(%cf_2_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: 1.107{{.*}}
  call @atan2_f32(%cf_3_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: 1.249{{.*}}
  call @atan2_f32(%cf_5_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: 1.373{{.*}}
  call @atan2_f32(%cf_50_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: 1.550{{.*}}

  // Atan2 all four quadrants.
  call @atan2_f32(%cf_n1_0, %cf_n1_0) : (f32, f32) -> ()
  // CHECK: -2.356{{.*}}
  call @atan2_f32(%cf_n1_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: -0.785{{.*}}
  call @atan2_f32(%cf_1_0, %cf_n1_0) : (f32, f32) -> ()
  // CHECK: 2.356{{.*}}
  call @atan2_f32(%cf_1_0, %cf_1_0) : (f32, f32) -> ()
  // CHECK: 0.785{{.*}}

  // Atan.
  call @atan_f32(%cf_n50_0) : (f32) -> ()
  // CHECK: -1.550{{.*}}
  call @atan_f32(%cf_n5_0) : (f32) -> ()
  // CHECK: -1.373{{.*}}
  call @atan_f32(%cf_n3_0) : (f32) -> ()
  // CHECK: -1.249{{.*}}
  call @atan_f32(%cf_n2_0) : (f32) -> ()
  // CHECK: -1.107{{.*}}
  call @atan_f32(%cf_n1_0) : (f32) -> ()
  // CHECK: -0.785{{.*}}
  call @atan_f32(%cf_n0_5) : (f32) -> ()
  // CHECK: -0.463{{.*}}
  call @atan_f32(%cf_n0_1) : (f32) -> ()
  // CHECK: -0.099{{.*}}
  call @atan_f32(%cf_0_0) : (f32) -> ()
  // CHECK: 0
  call @atan_f32(%cf_0_1) : (f32) -> ()
  // CHECK: 0.099{{.*}}
  call @atan_f32(%cf_0_5) : (f32) -> ()
  // CHECK: 0.463{{.*}}
  call @atan_f32(%cf_1_0) : (f32) -> ()
  // CHECK: 0.785{{.*}}
  call @atan_f32(%cf_2_0) : (f32) -> ()
  // CHECK: 1.107{{.*}}
  call @atan_f32(%cf_3_0) : (f32) -> ()
  // CHECK: 1.249{{.*}}
  call @atan_f32(%cf_5_0) : (f32) -> ()
  // CHECK: 1.373{{.*}}
  call @atan_f32(%cf_50_0) : (f32) -> ()
  // CHECK: 1.550{{.*}}

  return
}
