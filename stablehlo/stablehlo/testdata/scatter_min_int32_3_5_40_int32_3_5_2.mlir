// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5x40xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3x5x40xi32>, tensor<3x5x2xi32>)
    %1 = call @expected() : () -> tensor<3x5x40xi32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i32>
      stablehlo.return %3 : tensor<i32>
    }) : (tensor<3x5x40xi32>, tensor<2x1xi64>, tensor<3x5x2xi32>) -> tensor<3x5x40xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<3x5x40xi32>, tensor<3x5x40xi32>) -> ()
    return %2 : tensor<3x5x40xi32>
  }
  func.func private @inputs() -> (tensor<3x5x40xi32> {mhlo.layout_mode = "default"}, tensor<3x5x2xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x01000000FEFFFFFFFEFFFFFF0500000000000000FCFFFFFFFEFFFFFFFDFFFFFFFEFFFFFF0100000000000000FEFFFFFFFCFFFFFFFFFFFFFF00000000FDFFFFFFFEFFFFFF03000000FDFFFFFFF8FFFFFF05000000FDFFFFFFFFFFFFFF0700000000000000FEFFFFFFFEFFFFFF01000000FEFFFFFF06000000010000000000000000000000FEFFFFFF00000000000000000400000003000000FBFFFFFFFDFFFFFF00000000FEFFFFFFFFFFFFFF0000000001000000F9FFFFFF00000000000000000000000001000000FFFFFFFF000000000200000000000000FEFFFFFF0300000000000000FDFFFFFFFCFFFFFF0200000001000000FEFFFFFF0100000004000000010000000000000004000000FFFFFFFF0100000000000000FEFFFFFF0000000004000000000000000200000001000000000000000200000001000000FBFFFFFF000000000100000000000000FEFFFFFF020000000000000003000000FEFFFFFFFDFFFFFFFFFFFFFFFDFFFFFF0000000000000000FDFFFFFF04000000FDFFFFFFFCFFFFFF0000000002000000000000000100000000000000FEFFFFFFFEFFFFFF0400000000000000FFFFFFFF04000000FEFFFFFFFFFFFFFFFCFFFFFF0000000000000000FDFFFFFFFBFFFFFF01000000FEFFFFFF0000000001000000FCFFFFFF0300000000000000FEFFFFFF02000000FEFFFFFFFDFFFFFF00000000FAFFFFFFFFFFFFFFFFFFFFFF0200000002000000FEFFFFFF0300000003000000FDFFFFFF00000000FCFFFFFF0300000002000000FEFFFFFF02000000FCFFFFFF0300000002000000000000000100000002000000FAFFFFFF02000000FFFFFFFF09000000FFFFFFFF00000000FFFFFFFFF9FFFFFF05000000FDFFFFFF00000000040000000000000005000000FFFFFFFFFDFFFFFF02000000FEFFFFFFFBFFFFFF00000000FEFFFFFF0000000000000000000000000200000002000000FBFFFFFF0400000004000000FDFFFFFFFDFFFFFF0100000004000000FCFFFFFFF8FFFFFFFFFFFFFF00000000FDFFFFFF0100000000000000FEFFFFFF0000000001000000FCFFFFFF0000000000000000FEFFFFFFFFFFFFFFFFFFFFFF02000000FEFFFFFF05000000FFFFFFFF000000000200000004000000FDFFFFFFFEFFFFFFFDFFFFFFFEFFFFFF0200000004000000FEFFFFFFFEFFFFFFFCFFFFFF0100000006000000FEFFFFFFFCFFFFFF00000000FFFFFFFF00000000FDFFFFFF0000000002000000FFFFFFFFFAFFFFFF01000000FEFFFFFFFFFFFFFF04000000FDFFFFFFFFFFFFFFFFFFFFFFFDFFFFFF010000000000000001000000FDFFFFFF0100000007000000FCFFFFFFFEFFFFFFFDFFFFFF03000000FFFFFFFFFDFFFFFFFEFFFFFF000000000000000000000000FFFFFFFFFEFFFFFFFFFFFFFF0400000004000000FDFFFFFFFCFFFFFFFFFFFFFF0100000000000000FFFFFFFF000000000000000001000000F9FFFFFFFFFFFFFF02000000020000000400000000000000FDFFFFFF0000000000000000FFFFFFFF00000000FFFFFFFFFBFFFFFF0000000002000000FDFFFFFF000000000200000004000000FDFFFFFF06000000FFFFFFFF020000000200000000000000FFFFFFFF01000000040000000400000001000000FFFFFFFF0000000001000000000000000000000001000000040000000100000002000000FDFFFFFF00000000FCFFFFFFFEFFFFFF02000000FEFFFFFFFFFFFFFFFCFFFFFFFEFFFFFF01000000FFFFFFFF0200000000000000030000000100000000000000FBFFFFFFFDFFFFFFFDFFFFFF0300000003000000000000000300000007000000FBFFFFFF000000000500000005000000FEFFFFFF000000000100000000000000FEFFFFFF00000000020000000100000000000000010000000000000002000000030000000200000000000000FBFFFFFF01000000010000000100000001000000FEFFFFFF0100000000000000FBFFFFFFFDFFFFFFFEFFFFFFFFFFFFFF0100000000000000FEFFFFFF040000000400000002000000FEFFFFFFFFFFFFFF00000000FEFFFFFFFEFFFFFFFBFFFFFFFFFFFFFF0200000002000000FCFFFFFF000000000200000002000000FEFFFFFFFFFFFFFFFDFFFFFFFAFFFFFF03000000FDFFFFFF03000000020000000000000007000000FDFFFFFFFFFFFFFF02000000FCFFFFFFFEFFFFFFFFFFFFFF020000000000000007000000040000000000000006000000010000000100000003000000FDFFFFFF00000000FFFFFFFF01000000010000000300000002000000FEFFFFFF000000000200000002000000FFFFFFFF0600000002000000000000000200000000000000FCFFFFFFFEFFFFFFFFFFFFFFFBFFFFFFFEFFFFFF0100000001000000000000000000000000000000FAFFFFFF0000000005000000000000000200000000000000FFFFFFFFFFFFFFFF00000000FFFFFFFFF8FFFFFFFFFFFFFF01000000010000000000000002000000FEFFFFFF05000000FCFFFFFFFEFFFFFF0000000001000000030000000000000000000000FDFFFFFFFDFFFFFF01000000FAFFFFFFFBFFFFFF02000000FFFFFFFF0200000000000000020000000000000000000000010000000100000002000000FEFFFFFF01000000000000000300000002000000FBFFFFFF0000000000000000010000000000000003000000FFFFFFFFFFFFFFFF000000000000000002000000FEFFFFFF03000000FDFFFFFFFBFFFFFF0300000000000000FFFFFFFF0300000000000000FDFFFFFF02000000FCFFFFFF00000000FEFFFFFF000000000100000002000000FCFFFFFF01000000FAFFFFFF0300000000000000FDFFFFFF01000000F9FFFFFF05000000FCFFFFFFFDFFFFFFFCFFFFFF05000000FFFFFFFF04000000FEFFFFFFFEFFFFFF01000000FDFFFFFF00000000FEFFFFFF04000000040000000000000002000000FEFFFFFF0200000001000000FEFFFFFF0300000002000000FBFFFFFFFFFFFFFF02000000FFFFFFFF0000000003000000FFFFFFFF00000000FFFFFFFFFCFFFFFFFEFFFFFF02000000FCFFFFFFFBFFFFFFFCFFFFFF02000000FDFFFFFFFFFFFFFF0000000000000000FDFFFFFF07000000F8FFFFFFFFFFFFFF00000000FEFFFFFF00000000020000000000000000000000FFFFFFFF020000000200000000000000FDFFFFFFFBFFFFFFFFFFFFFF020000000000000001000000FFFFFFFF0000000000000000FFFFFFFF0000000002000000FFFFFFFF00000000FBFFFFFFFEFFFFFFFEFFFFFF0400000002000000FAFFFFFF01000000000000000100000004000000000000000400000002000000020000000500000002000000FFFFFFFF020000000300000000000000"> : tensor<3x5x40xi32>
    %c_0 = stablehlo.constant dense<[[[1, -3], [3, -3], [4, -5], [0, -5], [5, -2]], [[0, -2], [-1, -1], [3, 0], [-2, 3], [1, 2]], [[0, 0], [0, 0], [-5, -1], [-2, 2], [-1, 0]]]> : tensor<3x5x2xi32>
    return %c, %c_0 : tensor<3x5x40xi32>, tensor<3x5x2xi32>
  }
  func.func private @expected() -> (tensor<3x5x40xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x01000000FDFFFFFFFEFFFFFF0500000000000000FCFFFFFFFEFFFFFFFDFFFFFFFEFFFFFF0100000000000000FEFFFFFFFCFFFFFFFFFFFFFF00000000FDFFFFFFFEFFFFFF03000000FDFFFFFFF8FFFFFF05000000FDFFFFFFFFFFFFFF0700000000000000FEFFFFFFFEFFFFFF01000000FEFFFFFF06000000010000000000000000000000FEFFFFFF00000000000000000400000003000000FBFFFFFFFDFFFFFF00000000FDFFFFFFFFFFFFFF0000000001000000F9FFFFFF00000000000000000000000001000000FFFFFFFF000000000200000000000000FEFFFFFF0300000000000000FDFFFFFFFCFFFFFF0200000001000000FEFFFFFF0100000004000000010000000000000004000000FFFFFFFF0100000000000000FEFFFFFF0000000004000000000000000200000001000000000000000200000001000000FBFFFFFF00000000FBFFFFFF00000000FEFFFFFF020000000000000003000000FEFFFFFFFDFFFFFFFFFFFFFFFDFFFFFF0000000000000000FDFFFFFF04000000FDFFFFFFFCFFFFFF0000000002000000000000000100000000000000FEFFFFFFFEFFFFFF0400000000000000FFFFFFFF04000000FEFFFFFFFFFFFFFFFCFFFFFF0000000000000000FDFFFFFFFBFFFFFF01000000FEFFFFFF0000000001000000FCFFFFFF03000000FBFFFFFFFEFFFFFF02000000FEFFFFFFFDFFFFFF00000000FAFFFFFFFFFFFFFFFFFFFFFF0200000002000000FEFFFFFF0300000003000000FDFFFFFF00000000FCFFFFFF0300000002000000FEFFFFFF02000000FCFFFFFF0300000002000000000000000100000002000000FAFFFFFF02000000FFFFFFFF09000000FFFFFFFF00000000FFFFFFFFF9FFFFFF05000000FDFFFFFF000000000400000000000000FEFFFFFFFFFFFFFFFDFFFFFF02000000FEFFFFFFFBFFFFFF00000000FEFFFFFF0000000000000000000000000200000002000000FBFFFFFF0400000004000000FDFFFFFFFDFFFFFF0100000004000000FCFFFFFFF8FFFFFFFFFFFFFF00000000FDFFFFFF0100000000000000FEFFFFFF0000000001000000FCFFFFFF0000000000000000FEFFFFFFFFFFFFFFFFFFFFFF02000000FEFFFFFF05000000FFFFFFFFFEFFFFFF0200000004000000FDFFFFFFFEFFFFFFFDFFFFFFFEFFFFFF0200000004000000FEFFFFFFFEFFFFFFFCFFFFFF0100000006000000FEFFFFFFFCFFFFFF00000000FFFFFFFF00000000FDFFFFFF0000000002000000FFFFFFFFFAFFFFFF01000000FEFFFFFFFFFFFFFF04000000FDFFFFFFFFFFFFFFFFFFFFFFFDFFFFFF010000000000000001000000FDFFFFFF0100000007000000FCFFFFFFFEFFFFFFFDFFFFFF03000000FFFFFFFFFDFFFFFFFEFFFFFF000000000000000000000000FFFFFFFFFEFFFFFFFFFFFFFF0400000004000000FDFFFFFFFCFFFFFFFFFFFFFF0100000000000000FFFFFFFF000000000000000001000000F9FFFFFFFFFFFFFF02000000020000000400000000000000FDFFFFFF0000000000000000FFFFFFFF00000000FFFFFFFFFBFFFFFF0000000002000000FDFFFFFF000000000200000000000000FDFFFFFF06000000FFFFFFFF020000000200000000000000FFFFFFFF01000000040000000400000001000000FFFFFFFF0000000001000000000000000000000001000000040000000100000002000000FDFFFFFF00000000FCFFFFFFFEFFFFFF02000000FEFFFFFFFFFFFFFFFCFFFFFFFEFFFFFF01000000FFFFFFFF0200000000000000030000000100000000000000FBFFFFFFFDFFFFFFFDFFFFFFFEFFFFFF03000000000000000300000007000000FBFFFFFF000000000500000005000000FEFFFFFF000000000100000000000000FEFFFFFF00000000020000000100000000000000010000000000000002000000030000000200000000000000FBFFFFFF01000000010000000100000001000000FEFFFFFF0100000000000000FBFFFFFFFDFFFFFFFEFFFFFFFFFFFFFF0100000000000000FEFFFFFF040000000100000002000000FEFFFFFFFFFFFFFF00000000FEFFFFFFFEFFFFFFFBFFFFFFFFFFFFFF0200000002000000FCFFFFFF000000000200000002000000FEFFFFFFFFFFFFFFFDFFFFFFFAFFFFFF03000000FDFFFFFF03000000020000000000000007000000FDFFFFFFFFFFFFFF02000000FCFFFFFFFEFFFFFFFFFFFFFF020000000000000007000000040000000000000006000000010000000100000003000000FDFFFFFF00000000FFFFFFFF01000000010000000300000002000000FEFFFFFF000000000200000002000000FFFFFFFF0600000002000000000000000200000000000000FCFFFFFFFEFFFFFFFFFFFFFFFBFFFFFFFEFFFFFF0100000001000000000000000000000000000000FAFFFFFF0000000005000000000000000200000000000000FFFFFFFFFFFFFFFF00000000FFFFFFFFF8FFFFFFFFFFFFFF01000000000000000000000002000000FEFFFFFF05000000FCFFFFFFFEFFFFFF0000000001000000030000000000000000000000FDFFFFFFFDFFFFFF01000000FAFFFFFFFBFFFFFF02000000FFFFFFFF0200000000000000020000000000000000000000010000000100000002000000FEFFFFFF01000000000000000300000002000000FBFFFFFF0000000000000000010000000000000003000000FFFFFFFFFFFFFFFFFBFFFFFF0000000002000000FEFFFFFF03000000FDFFFFFFFBFFFFFF0300000000000000FFFFFFFF0300000000000000FDFFFFFF02000000FCFFFFFF00000000FEFFFFFF000000000100000002000000FCFFFFFF01000000FAFFFFFF0300000000000000FDFFFFFF01000000F9FFFFFF05000000FCFFFFFFFDFFFFFFFCFFFFFF05000000FFFFFFFF04000000FEFFFFFFFEFFFFFF01000000FDFFFFFF00000000FEFFFFFF04000000040000000000000002000000FEFFFFFF0200000001000000FEFFFFFF0300000002000000FBFFFFFFFFFFFFFF02000000FFFFFFFF0000000003000000FFFFFFFF00000000FFFFFFFFFCFFFFFFFEFFFFFF02000000FCFFFFFFFBFFFFFFFCFFFFFF02000000FDFFFFFFFFFFFFFF0000000000000000FDFFFFFF07000000F8FFFFFFFFFFFFFF00000000FEFFFFFF000000000200000000000000FFFFFFFFFFFFFFFF020000000200000000000000FDFFFFFFFBFFFFFFFFFFFFFF020000000000000001000000FFFFFFFF0000000000000000FFFFFFFF0000000002000000FFFFFFFF00000000FBFFFFFFFEFFFFFFFEFFFFFF0400000002000000FAFFFFFF01000000000000000100000004000000000000000400000002000000020000000500000002000000FFFFFFFF020000000300000000000000"> : tensor<3x5x40xi32>
    return %c : tensor<3x5x40xi32>
  }
}