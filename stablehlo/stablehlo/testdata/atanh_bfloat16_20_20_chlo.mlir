// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xbf16>
    %1 = call @expected() : () -> tensor<20x20xbf16>
    %2 = chlo.atanh %0 : tensor<20x20xbf16> -> tensor<20x20xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
    return %2 : tensor<20x20xbf16>
  }
  func.func private @inputs() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xDDBFD8BF9CBE8D400140A83F78C09CC0FDC02BBF17BFDBBED4405DC018BD4C40B5BE5E4069BF463F3D40233E8CC0A53E0E40B5C044C01640294066C08BC0A0408C3F86405D400AC040BEBE40DFBEC93F3BBD1F3EBE3F6340B83FDB3E083F0B40AF3F55C034405B4043C033C0CFC084C02A3F21C02340E6BFA1BF8DBFB6C0C4BF853F14C08EC040BE25C080402141E8BEADBF51C08ABFDB3F71BFF6BE98407BBE7AC006C0383F67BFC5BFF2BFA63FC8C03F40F93FA9BF12404DBF29C064BF6E3F5B400C401F3D13C04240F4BFB340BFBF68BF3E40803F014070BF10BF2DC02DC059BF5B40D5BFDDC075C03940BDBFA1BF18409ABECE40733FF13F8ABFA640D3C0A1BF3B4090C0E1BE9940A23E353FF23EF0BFAFBF0640AB3F0F401F409A3FA9400BC0AA40264089BF93C030C0343C3E403C40774099C0ACC008404BC0A140023ECC3F39402A4082BFBD3FD03F9A3F99C048C0AE3FB63F414038BE68BF48405ABEE6BF04C0DFBF3F3FBE3F02C0D9C0A23D54C0034042BF4E3FBB3FCABE6F3FAABFA6C0D7C0763F98BF853F1E3F3C3F5F4093400DC1BDC0DDBF94BF32C093C01840403F44C0BBC08AC0113E3BBE9F3FA2408DBE074079C0153F81BF253F1A408FC056BE84BF95C0593ED1BE8F40EB3EB33F30BF8FC063400340EF3F9DBFA23FF0BFA54097C00A407ABE463FA54086406ABFE23FA13FA03FB44081C05FC098C0A1400DBF6D4091C0BBBF9CC0BABD7540A13FA240AF3F8340083F154077BFF1BF4DBF1140A3C0FDBF93C018C02A40BD40173FABC0A94080BFD7BDD63C473F19C0AD40214091BF79C0E13E7EC087C0104095C0ACBE5DC0B33D67C010C0C83E8B3F7C3F91BF81BFEB3F94BF6FC077BF41C03FC0D5BF413F8FBF36C0B1BFA73C4A4088BF94BF83C05A40ADBF9440AEBEB34029C003C02A40A9BF4B40224075BF743F8CBFDDBF243F35BD773F2BC0B7BB123F12C0DDBE9A4094BF81BE37C08FBF92409CC0BB3F873F8D3EFE40BF408640BFC0B43EDC3F1A3EE13B053E38405F3E06408BBF31C049C05CC090BF144076BE8640013F24C01BC09E3FA6BF36C067BFEC3D393F993EED3F4E401540C03FB7BF6D401040623FD03F90C03AC0B3BEA9BF93C0D63F"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
  func.func private @expected() -> (tensor<20x20xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xC07FC07FA1BEC07FC07FC07FC07FC07FC07F4EBF2DBFEABEC07FC07F18BDC07FBDBEC07FC4BF843FC07F243EC07FAB3EC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F42BEC07FEEBEC07F3BBD203EC07FC07FC07FEA3E183FC07FC07FC07FC07FC07FC07FC07FC07FC07F4D3FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F42BEC07FC07FC07FFABEC07FC07FC07FC07FE0BF06BFC07F80BEC07FC07F683FBEBFC07FC07FC07FC07FC07FC07FC07FC07F8DBFC07FB7BFD43FC07FC07F1F3DC07FC07FC07FC07FC07FC0BFC07F807FC07FDBBF23BFC07FC07FA0BFC07FC07FC07FC07FC07FC07FC07FC07F9FBEC07FEA3FC07FC07FC07FC07FC07FC07FC07FF2BEC07FA83E623F043FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F343CC07FC07FC07FC07FC07FC07FC07FC07F033EC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F3ABEC0BFC07F5EBEC07FC07FC07F763FC07FC07FC07FA23DC07FC07F7EBF8E3FC07FD5BED83FC07FC07FC07FFB3FC07FC07F383F703FC07FC07FC07FC07FC07FC07FC07FC07FC07F783FC07FC07FC07F123E3DBEC07FC07F91BEC07FC07F2A3FC07F443FC07FC07F59BEC07FC07F5C3EDEBEC07FFE3EC07F58BFC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F7FBE843FC07FC07FC6BFC07FC07FC07FC07FC07FC07FC07FC07F1FBFC07FC07FC07FC07FBABDC07FC07FC07FC07FC07F183FC07F01C0C07F8DBFC07FC07FC07FC07FC07FC07FC07F2D3FC07FC07F80FFD8BDD63C853FC07FC07FC07FC07FC07FF23EC07FC07FC07FC07FB3BEC07FB43DC07FC07FD43EC07F1B40C07FC07FC07FC07FC07F01C0C07FC07FC07F7B3FC07FC07FC07FA73CC07FC07FC07FC07FC07FC07FC07FB6BEC07FC07FC07FC07FC07FC07FC07FF4BFEF3FC07FC07F423F35BD0140C07FB7BB263FC07FEDBEC07FC07F84BEC07FC07FC07FC07FC07FC07F913EC07FC07FC07FC07FBC3EC07F1B3EE13B063EC07F633EC07FC07FC07FC07FC07FC07FC07F7BBEC07F0E3FC07FC07FC07FC07FC07FBEBFED3D6A3F9E3EC07FC07FC07FC07FC07FC07FC07FB23FC07FC07FC07FBBBEC07FC07FC07F"> : tensor<20x20xbf16>
    return %cst : tensor<20x20xbf16>
  }
}