// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xf16>, tensor<5x2x2x7xf16>)
    %1 = call @expected() : () -> tensor<5x6x7xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<5x6x7xf16>, tensor<2x2x1xi64>, tensor<5x2x2x7xf16>) -> tensor<5x6x7xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> ()
    return %2 : tensor<5x6x7xf16>
  }
  func.func private @inputs() -> (tensor<5x6x7xf16> {mhlo.layout_mode = "default"}, tensor<5x2x2x7xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xAC3CC83DB13C0CBCD9BC2F3870BF8E3AB1C12B3F5EC10D455243B5403A3E56BB4648B547AD4145BCC2BD88BF21BEB1C28FC8E23B0A3FBABBD5B70FC3A3C2353FA83FF6C0D634AA445FC360BF8DB954BD83BBE6C538BF713387B8844964BFD4BFB8BD7DACB239D2B71E40F9C3BF394A47D4C356C065BE554647C28842AA3CDABC68BEB0C1D13CB5BFA2418A46EABA70BD3635794403BC09BC24BC5845E1BFE8B99B49D73DFB41223902453B40B8383D3EF6C34D20CC412E40F83C983D8B381C379D3AB43A16C44937D63814BF1BC046BAD53F0DBB1345A5C4FD422240FC243EC5083E84430DC5F83DD3C0B5BA74B83A450941B0B95DC5804654C3AA431C398B3A443BB7BD3FC59AC49745C340D33DBA3D2840CB3FA045BEB533C083C4AA39BBC6B2384A3F623E28C43D40663E09C5A5B4AEBDEA41EFB9C23FE13FB0C1FF34A2C416309AC449C2BF431C45ACBE48C43EC4E5BFDF4681C070BCAD3D22BCC3C474C4FD3E723CD3B86CB1EBB4D3454442712CFE3B8A42AFC2CABCC6C4A03724412E448EC3808F7141DDBC9C42EB3B763C04BD9337C1224F412546E5C18FB4144191B1E1459E3D"> : tensor<5x6x7xf16>
    %cst_0 = stablehlo.constant dense<"0x063E2CC0AF45053FEBC2533CAF3223C5D247AE3C27459C4311A648C1D23C493A974093C45F46B2416F2CF1BC54BCBFC17FC623456944AD343C4347C4B244413D8F4681B37B37C03BB04199427336FC446AB630C342BC1B43DEC569410DC112415DAFEA41E3BB0DC34F3B4340E3B5CB4025C5522AE542D44212399D3ECB41713E1D458C40E8C23B3D4B41E7C8963CEE3E524639B6CE402CB437BC323AD6BD8443053CADB848C468A4AE41BEC0D4C0DE2C5D3DDC4169C1ADC49BB8503F26C3D4442C388D4401424740E7A830397CC2F74131B96EC253B5F44422414AC282BF8A3F333F0437FCC31BC3FCB151428C3CF443103FFA2D4CB888BF1CC4C7406F4262C0AABE8935503CFF422E3AA1BCABC00FBD30C78D43893C8541"> : tensor<5x2x2x7xf16>
    return %cst, %cst_0 : tensor<5x6x7xf16>, tensor<5x2x2x7xf16>
  }
  func.func private @expected() -> (tensor<5x6x7xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x594120B9DB46F239ACC46A3E9ABE51C4FA44EC41F0406E48464398B4864134B06C4944429B481F3F7BBD3CC23AC138C6CECB1F462C4664B9D5B70FC3A3C2353FA83FF6C0D634AA445FC360BF8DB954BD83BBE6C5403F0BC421442C4AB64462C0B2BB303B1C439F41EC40FC3B14356443FAC48A3D77C78548AAC5CD45343CFA3E2DC15EC63C408832E6407848EABA70BD3635794403BC09BC24BC5845E1BFE8B99B49D73DFB41223960B054400A44F944B2C2A63ECC4566435B465843C5C1023FF2427CC8E1C16040ED4651C098352EBC3C3BD8AE3B4318BB8044EE3D43C442C5083E84430DC5F83DD3C0B5BA74B83A450941B0B95DC5804654C3AA43F54236BE06BE69BDD0C3B0BEC54197C00B3B8542FCBDC74626463144383BBFC05C3915C650C1CE44933B5FC7253F8E46F0C0DFC298C2D844EFB9C23FE13FB0C1FF34A2C416309AC449C2BF431C45ACBE48C43EC490B14F473EC6AAC4EE3C404040C3A0B70643D23C90BC1BC06BC41B485A463EC056B93B4387C09A4000C472B99033D4417CC98D43B6432D3E9C42EB3B763C04BD9337C1224F412546E5C18FB4144191B1E1459E3D"> : tensor<5x6x7xf16>
    return %cst : tensor<5x6x7xf16>
  }
}
