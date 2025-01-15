// RUN: stablehlo-opt --stablehlo-complex-math-expander %s | stablehlo-translate --interpret
// This file is generated, see build_tools/math/README.md for more information.
module @exponential_complex64 {
  func.func private @samples() -> tensor<169xcomplex<f32>> {
    %0 = stablehlo.constant dense<"0x000080FF000080FFFFFF7FFF000080FFFEFF7FFF000080FF0000C0BF000080FF0000E09F000080FF01000080000080FF00000000000080FF01000000000080FF0000E01F000080FF0000C03F000080FFFEFF7F7F000080FFFFFF7F7F000080FF0000807F000080FF000080FFFFFF7FFFFFFF7FFFFFFF7FFFFEFF7FFFFFFF7FFF0000C0BFFFFF7FFF0000E09FFFFF7FFF01000080FFFF7FFF00000000FFFF7FFF01000000FFFF7FFF0000E01FFFFF7FFF0000C03FFFFF7FFFFEFF7F7FFFFF7FFFFFFF7F7FFFFF7FFF0000807FFFFF7FFF000080FFFEFF7FFFFFFF7FFFFEFF7FFFFEFF7FFFFEFF7FFF0000C0BFFEFF7FFF0000E09FFEFF7FFF01000080FEFF7FFF00000000FEFF7FFF01000000FEFF7FFF0000E01FFEFF7FFF0000C03FFEFF7FFFFEFF7F7FFEFF7FFFFFFF7F7FFEFF7FFF0000807FFEFF7FFF000080FF0000C0BFFFFF7FFF0000C0BFFEFF7FFF0000C0BF0000C0BF0000C0BF0000E09F0000C0BF010000800000C0BF000000000000C0BF010000000000C0BF0000E01F0000C0BF0000C03F0000C0BFFEFF7F7F0000C0BFFFFF7F7F0000C0BF0000807F0000C0BF000080FF0000E09FFFFF7FFF0000E09FFEFF7FFF0000E09F0000C0BF0000E09F0000E09F0000E09F010000800000E09F000000000000E09F010000000000E09F0000E01F0000E09F0000C03F0000E09FFEFF7F7F0000E09FFFFF7F7F0000E09F0000807F0000E09F000080FF01000080FFFF7FFF01000080FEFF7FFF010000800000C0BF010000800000E09F010000800100008001000080000000000100008001000000010000800000E01F010000800000C03F01000080FEFF7F7F01000080FFFF7F7F010000800000807F01000080000080FF00000000FFFF7FFF00000000FEFF7FFF000000000000C0BF000000000000E09F000000000100008000000000000000000000000001000000000000000000E01F000000000000C03F00000000FEFF7F7F00000000FFFF7F7F000000000000807F00000000000080FF01000000FFFF7FFF01000000FEFF7FFF010000000000C0BF010000000000E09F010000000100008001000000000000000100000001000000010000000000E01F010000000000C03F01000000FEFF7F7F01000000FFFF7F7F010000000000807F01000000000080FF0000E01FFFFF7FFF0000E01FFEFF7FFF0000E01F0000C0BF0000E01F0000E09F0000E01F010000800000E01F000000000000E01F010000000000E01F0000E01F0000E01F0000C03F0000E01FFEFF7F7F0000E01FFFFF7F7F0000E01F0000807F0000E01F000080FF0000C03FFFFF7FFF0000C03FFEFF7FFF0000C03F0000C0BF0000C03F0000E09F0000C03F010000800000C03F000000000000C03F010000000000C03F0000E01F0000C03F0000C03F0000C03FFEFF7F7F0000C03FFFFF7F7F0000C03F0000807F0000C03F000080FFFEFF7F7FFFFF7FFFFEFF7F7FFEFF7FFFFEFF7F7F0000C0BFFEFF7F7F0000E09FFEFF7F7F01000080FEFF7F7F00000000FEFF7F7F01000000FEFF7F7F0000E01FFEFF7F7F0000C03FFEFF7F7FFEFF7F7FFEFF7F7FFFFF7F7FFEFF7F7F0000807FFEFF7F7F000080FFFFFF7F7FFFFF7FFFFFFF7F7FFEFF7FFFFFFF7F7F0000C0BFFFFF7F7F0000E09FFFFF7F7F01000080FFFF7F7F00000000FFFF7F7F01000000FFFF7F7F0000E01FFFFF7F7F0000C03FFFFF7F7FFEFF7F7FFFFF7F7FFFFF7F7FFFFF7F7F0000807FFFFF7F7F000080FF0000807FFFFF7FFF0000807FFEFF7FFF0000807F0000C0BF0000807F0000E09F0000807F010000800000807F000000000000807F010000000000807F0000E01F0000807F0000C03F0000807FFEFF7F7F0000807FFFFF7F7F0000807F0000807F0000807F"> : tensor<169xcomplex<f32>>
    return %0 : tensor<169xcomplex<f32>>
  }
  func.func private @expected() -> tensor<169xcomplex<f32>> {
    %0 = stablehlo.constant dense<"0x0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000000000000000000000000000000000000000000000001BE7423E727BEE3D965F5A3FB399053F965F5A3FB399053F965F5A3FB399053F965F5A3FB399053F965F5A3FB399053FA0AB744059B015400000807F0000807F0000807F0000807F0000807F0000807F00000000000000000000008000000000000000800000000081242FBEABBB123ED13B44BF2C67243FD13B44BF2C67243FD13B44BF2C67243FD13B44BF2C67243FD13B44BF2C67243F54DD5BC067333840000080FF0000807F000080FF0000807F000080FF0000807F0000000000000000000000000000008000000000000000809D4C813CB5E963BEAADE903DD55B7FBFAADE903DD55B7FBFAADE903DD55B7FBFAADE903DD55B7FBFAADE903DD55B7FBFBA50A23E070E8FC00000807F000080FF0000807F000080FF0000807F000080FF0000000000000000000000000000008000000000000000803C7C643EB4ECC79E0000803F0000E09F0000803F0000E09F0000803F0000E09F0000803F0000E09F0000803F0000E09FFF698F407FF9FAA00000807F000080FF0000807F000080FF0000807F000080FF0000000000000000000000000000008000000000000000803C7C643E000000800000803F010000800000803F010000800000803F010000800000803F010000800000803F01000080FF698F40040000800000807F000080FF0000807F000080FF0000807F000080FF0000000000000000000000000000000000000000000000003C7C643E000000000000803F000000000000803F000000000000803F000000000000803F000000000000803F00000000FF698F40000000000000807F000000000000807F000000000000807F000000000000000000000000000000000000000000000000000000003C7C643E000000000000803F010000000000803F010000000000803F010000000000803F010000000000803F01000000FF698F40040000000000807F0000807F0000807F0000807F0000807F0000807F0000000000000000000000000000000000000000000000003C7C643EB4ECC71E0000803F0000E01F0000803F0000E01F0000803F0000E01F0000803F0000E01F0000803F0000E01FFF698F407FF9FA200000807F0000807F0000807F0000807F0000807F0000807F0000000000000000000000000000000000000000000000009D4C813CB5E9633EAADE903DD55B7F3FAADE903DD55B7F3FAADE903DD55B7F3FAADE903DD55B7F3FAADE903DD55B7F3FBA50A23E070E8F400000807F0000807F0000807F0000807F0000807F0000807F00000000000000000000008000000080000000800000008081242FBEABBB12BED13B44BF2C6724BFD13B44BF2C6724BFD13B44BF2C6724BFD13B44BF2C6724BFD13B44BF2C6724BF54DD5BC0673338C0000080FF000080FF000080FF000080FF000080FF000080FF0000000000000000000000000000008000000000000000801BE7423E727BEEBD965F5A3FB39905BF965F5A3FB39905BF965F5A3FB39905BF965F5A3FB39905BF965F5A3FB39905BFA0AB744059B015C00000807F000080FF0000807F000080FF0000807F000080FF0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F0000C07F"> : tensor<169xcomplex<f32>>
    return %0 : tensor<169xcomplex<f32>>
  }
  func.func public @main() {
    %0 = call @samples() : () -> tensor<169xcomplex<f32>>
    %1 = "stablehlo.exponential"(%0) : (tensor<169xcomplex<f32>>) -> tensor<169xcomplex<f32>>
    %2 = call @expected() : () -> tensor<169xcomplex<f32>>
    check.expect_close %1, %2, max_ulp_difference = 3 : tensor<169xcomplex<f32>>, tensor<169xcomplex<f32>>
    func.return
  }
}