// RUN: mlir-hlo-opt %s -mhlo-test-chlo-legalize-to-hlo -hlo-legalize-to-lhlo=results-escape-function=true -buffer-placement -lhlo-copy-removal -canonicalize -cse -lhlo-legalize-to-linalg -lhlo-fuse-linalg -convert-linalg-to-loops -convert-scf-to-std -canonicalize -cse -test-lhlo-legalize-to-llvm | mlir-cpu-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | FileCheck %s

func @main() -> () {
  call @reshape_with_static_shape_size_matrix_to_1D() : () -> ()
  call @reshape_with_static_shape_size_matrix_to_3D() : () -> ()
  call @reshape_with_dynamic_shape_size_matrix_to_1D() : () -> ()
  call @reshape_with_dynamic_shape_size_matrix_to_3D() : () -> ()
  return
}

func @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

func @reshape_with_static_shape_size_matrix_to_1D() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // Initialize input.
  %input = alloc() : memref<2x3xf32>
  %dim_x = dim %input, %c0 : memref<2x3xf32>
  %dim_y = dim %input, %c1 : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%dim_x, %dim_y) step (%c1, %c1) {
    %i_i64 = index_cast %i : index to i64
    %i_f32 = sitofp %i_i64 : i64 to f32
    store %i_f32, %input[%i, %j] : memref<2x3xf32>
  }
  %unranked_input = memref_cast %input : memref<2x3xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_input) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK: [0,   0,   0]
  // CHECK: [1,   1,   1]

  // Initialize shape.
  %shape = alloc() : memref<1xi64>
  %num_elements = muli %dim_x, %dim_y : index
  %num_elements_i64 = index_cast %num_elements : index to i64
  store %num_elements_i64, %shape[%c0] : memref<1xi64>

  // 1. Ranked input, ranked output.
  %output_1 = lmhlo.reshape_memref_cast %input(%shape)
                 : (memref<2x3xf32>, memref<1xi64>) -> memref<6xf32>
  %unranked_output_1 = memref_cast %output_1 : memref<6xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output_1) : (memref<*xf32>) -> ()
  // CHECK: rank = 1 offset = 0 sizes = [6] strides = [1]
  // CHECK: [0,  0,  0,  1,  1,  1]

  // 2. Ranked input, unranked output.
  %output_2 = lmhlo.reshape_memref_cast %input(%shape)
                 : (memref<2x3xf32>, memref<1xi64>) -> memref<*xf32>
  call @print_memref_f32(%output_2) : (memref<*xf32>) -> ()
  // CHECK: rank = 1 offset = 0 sizes = [6] strides = [1]
  // CHECK: [0,  0,  0,  1,  1,  1]

  // 3. Unranked input, ranked output.
  %output_3 = lmhlo.reshape_memref_cast %unranked_input(%shape)
                 : (memref<*xf32>, memref<1xi64>) -> memref<?xf32>
  %unranked_output_3 = memref_cast %output_3 : memref<?xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_output_3) : (memref<*xf32>) -> ()
  // CHECK: rank = 1 offset = 0 sizes = [6] strides = [1]
  // CHECK: [0,  0,  0,  1,  1,  1]

  // 4. Unranked input, unranked output.
  %output_4 = lmhlo.reshape_memref_cast %unranked_input(%shape)
                 : (memref<*xf32>, memref<1xi64>) -> memref<*xf32>
  call @print_memref_f32(%output_4) : (memref<*xf32>) -> ()
  // CHECK: rank = 1 offset = 0 sizes = [6] strides = [1]
  // CHECK: [0,  0,  0,  1,  1,  1]
  return
}

func @reshape_with_static_shape_size_matrix_to_3D() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index

  // Initialize input.
  %input = alloc() : memref<2x3xf32>
  %dim_x = dim %input, %c0 : memref<2x3xf32>
  %dim_y = dim %input, %c1 : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%dim_x, %dim_y) step (%c1, %c1) {
    %i_i64 = index_cast %i : index to i64
    %i_f32 = sitofp %i_i64 : i64 to f32
    store %i_f32, %input[%i, %j] : memref<2x3xf32>
  }
  %unranked_input = memref_cast %input : memref<2x3xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_input) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK: [0,   0,   0]
  // CHECK: [1,   1,   1]

  // Initialize shape.
  %shape = alloc() : memref<3xi64>
  %c1_i64 = constant 1 : i64
  %c2_i64 = constant 2 : i64
  %c3_i64 = constant 3 : i64
  store %c3_i64, %shape[%c0] : memref<3xi64>
  store %c1_i64, %shape[%c1] : memref<3xi64>
  store %c2_i64, %shape[%c2] : memref<3xi64>

  // Static shape input and shape, dynamic output.
  %unranked_output = lmhlo.reshape_memref_cast %input(%shape)
                 : (memref<2x3xf32>, memref<3xi64>) -> memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 3 offset = 0 sizes = [3, 1, 2] strides = [2, 2, 1]
  // CHECK: {{\[}}{{\[}}[0,    0]],
  // CHECK:       {{\[}}[0,    1]],
  // CHECK:       {{\[}}[1,    1]]]
  return
}

func @reshape_with_dynamic_shape_size_matrix_to_1D() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  // Initialize input.
  %input = alloc() : memref<2x3xf32>
  %dim_x = dim %input, %c0 : memref<2x3xf32>
  %dim_y = dim %input, %c1 : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%dim_x, %dim_y) step (%c1, %c1) {
    %i_i64 = index_cast %i : index to i64
    %i_f32 = sitofp %i_i64 : i64 to f32
    store %i_f32, %input[%i, %j] : memref<2x3xf32>
  }
  %unranked_input = memref_cast %input : memref<2x3xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_input) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK: [0,   0,   0]
  // CHECK: [1,   1,   1]

  // Initialize shape.
  %shape = alloc(%c1) : memref<?xi64>
  %num_elements = muli %dim_x, %dim_y : index
  %num_elements_i64 = index_cast %num_elements : index to i64
  store %num_elements_i64, %shape[%c0] : memref<?xi64>

  // 1. Ranked input, unranked output.
  %output_2 = lmhlo.reshape_memref_cast %input(%shape)
                 : (memref<2x3xf32>, memref<?xi64>) -> memref<*xf32>
  call @print_memref_f32(%output_2) : (memref<*xf32>) -> ()
  // CHECK: rank = 1 offset = 0 sizes = [6] strides = [1]
  // CHECK: [0,  0,  0,  1,  1,  1]

  // 2. Unranked input, unranked output.
  %output_4 = lmhlo.reshape_memref_cast %unranked_input(%shape)
                 : (memref<*xf32>, memref<?xi64>) -> memref<*xf32>
  call @print_memref_f32(%output_4) : (memref<*xf32>) -> ()
  // CHECK: rank = 1 offset = 0 sizes = [6] strides = [1]
  // CHECK: [0,  0,  0,  1,  1,  1]
  return
}

func @reshape_with_dynamic_shape_size_matrix_to_3D() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index

  // Initialize input.
  %input = alloc() : memref<2x3xf32>
  %dim_x = dim %input, %c0 : memref<2x3xf32>
  %dim_y = dim %input, %c1 : memref<2x3xf32>
  scf.parallel (%i, %j) = (%c0, %c0) to (%dim_x, %dim_y) step (%c1, %c1) {
    %i_i64 = index_cast %i : index to i64
    %i_f32 = sitofp %i_i64 : i64 to f32
    store %i_f32, %input[%i, %j] : memref<2x3xf32>
  }
  %unranked_input = memref_cast %input : memref<2x3xf32> to memref<*xf32>
  call @print_memref_f32(%unranked_input) : (memref<*xf32>) -> ()
  // CHECK: rank = 2 offset = 0 sizes = [2, 3] strides = [3, 1]
  // CHECK: [0,   0,   0]
  // CHECK: [1,   1,   1]

  // Initialize shape.
  %shape = alloc(%c3) : memref<?xi64>
  %c1_i64 = constant 1 : i64
  %c2_i64 = constant 2 : i64
  %c3_i64 = constant 3 : i64
  store %c3_i64, %shape[%c0] : memref<?xi64>
  store %c1_i64, %shape[%c1] : memref<?xi64>
  store %c2_i64, %shape[%c2] : memref<?xi64>

  // Static shape input, dynamic output and shape.
  %unranked_output = lmhlo.reshape_memref_cast %input(%shape)
                 : (memref<2x3xf32>, memref<?xi64>) -> memref<*xf32>
  call @print_memref_f32(%unranked_output) : (memref<*xf32>) -> ()
  // CHECK: rank = 3 offset = 0 sizes = [3, 1, 2] strides = [2, 2, 1]
  // CHECK: {{\[}}{{\[}}[0,    0]],
  // CHECK:       {{\[}}[0,    1]],
  // CHECK:       {{\[}}[1,    1]]]
  return
}
