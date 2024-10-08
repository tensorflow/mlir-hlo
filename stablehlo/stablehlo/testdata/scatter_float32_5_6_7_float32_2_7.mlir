// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xf32>, tensor<2x7xf32>)
    %1 = call @expected() : () -> tensor<5x6x7xf32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      stablehlo.return %arg1 : tensor<f32>
    }) : (tensor<5x6x7xf32>, tensor<2x2xi64>, tensor<2x7xf32>) -> tensor<5x6x7xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<5x6x7xf32>, tensor<5x6x7xf32>) -> ()
    return %2 : tensor<5x6x7xf32>
  }
  func.func private @inputs() -> (tensor<5x6x7xf32> {mhlo.layout_mode = "default"}, tensor<2x7xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x9543D63DA1C397BFF696093FB2DDFB3FF96CE9BF21E6FEBF21F3F0BF8D8427C01E52784028448A4040D5ED3D893AB040BE491640B21BAC3E75E8E93F7B4784BE38726DC0AC8490C0051CF73FCEA473C06CA330C02971563F4339C03F352C1F404C9551407B6046403F49773FFCC179C0C01337C0D1F34D4034DB1BC0BD763440546C9E408DEDEC3F65E2BA3FB8D605C0067E193F2B158EC06445B13D055D014098AC20C041F0764070939E3EF41BD33FA0D522BE1E3E1340E4DF2040C63EAA403EA6143F7FC30940E45F39C0E3299BBFD7101E40FF564740C00FA54049F2B040BC7387C09D67B940986291C0239DE1BF4C8D163EBD23194002F48CC0CDD4143E62E0A63F298D47BF4AEAB13F2477DABFB6F49E3EBC9AE83F86CE713F95438A40C50F313DF3C41FBF421104BF7C2870BF0173CD3FECB967BD8C7FD2BF6CA657C0C8F2863F6118C63F055A7440151CFE3F4FEA763FE5653540E45DA33FF24B3B402500A13F4243FB3F0E6423BFFDEA0F3FC797B2BF569C644071B069C02F6F8EC054868840EB3AC33F2B829CC040ED47C03B37E13F5492B13F87BA0040F027833D1177E0BF64DA683F8FA77C40CC0B78BE1EEEAABE41BFF03FAE9F04BF28C7D8BF61BA064076BC8C3E20D35B4057972F40E68A353F08071DC0F653B0BF90D9AC3FCF9893C00D45CFC076EF38BF4BA176C0A001963F4EF791C07CF1B4BF4220FD40EAAEB0C0B27D46409A9EF63F3C661BC0322CDB3E43F990C092145FC0BD44D3BFE4A3EABC9337AEC0878EC83ECE2380C054361540549AD23F188630400556ABBF5AC9B1C094FDDD3FC9A50C3EF1F950C0B628DDBFD613A2C00945C7C057912E4047E6173FCACB9CC0E43C3C40061B83408F2B31C0054BA43F54CC70407330244060BCA7402D2322BFBB22D43F1D18773FFD68BFBFD9EEAB3F9B7C603F4C5E233FBF6426C03DF37D40981EB24044F0364044921F3F7923433F1D1E023F66978E409DACE93FF5129BC070E40BBEBAD7243FCDF871BF9AE0C8C0A62928C0CB560DC093FE0540379AD6BFCC946040696E41C009EC9D3F18239F40CC0ED240C6C08040FBFC17C0B6FF84401DD846C03A1A33C0E3D2CE3F202CC0BF6AD3C23D10323540FE6F31407553B440555CB1BE03F88BC0BAD037BFA3544840248FACBF18A03E40CCA53A405EF18E3F"> : tensor<5x6x7xf32>
    %cst_0 = stablehlo.constant dense<[[-1.12188053, -2.36665463, 4.968260e+00, 2.31045961, -2.5612936, 0.756939291, -1.96624672], [-1.5773772, 1.34981406, 0.856585264, -3.598180e+00, -1.84722638, -5.25752354, -1.41762674]]> : tensor<2x7xf32>
    return %cst, %cst_0 : tensor<5x6x7xf32>, tensor<2x7xf32>
  }
  func.func private @expected() -> (tensor<5x6x7xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x9543D63DA1C397BFF696093FB2DDFB3FF96CE9BF21E6FEBF21F3F0BFC8998FBF457717C0FCFB9E4092DE13403CEC23C0C6C6413FF9ADFBBF75E8E93F7B4784BE38726DC0AC8490C0051CF73FCEA473C06CA330C02971563F4339C03F352C1F404C9551407B6046403F49773FFCC179C0C01337C0D1F34D4034DB1BC0BD763440546C9E408DEDEC3F65E2BA3FB8D605C0067E193F2B158EC06445B13D055D014098AC20C041F0764070939E3EF41BD33FA0D522BE1E3E1340E4DF2040C63EAA403EA6143F7FC30940E45F39C0E3299BBFD7101E40FF564740C00FA54049F2B040BC7387C09D67B940986291C0239DE1BF4C8D163EBD23194002F48CC0CDD4143E62E0A63F298D47BF4AEAB13F2477DABFB6F49E3EBC9AE83F86CE713F95438A40C50F313DF3C41FBF421104BF7C2870BF0173CD3FECB967BD8C7FD2BF6CA657C0C8F2863F6118C63F055A7440151CFE3F4FEA763FE5653540E45DA33FF24B3B402500A13F4243FB3F0E6423BFFDEA0F3FC797B2BF569C644071B069C02F6F8EC054868840EB3AC33F2B829CC040ED47C03B37E13F5492B13F87BA0040F027833D1177E0BF7FE7C9BFB5C6AC3F2C495B3F954866C0EA71ECBFA23DA8C0CB74B5BF61BA064076BC8C3E20D35B4057972F40E68A353F08071DC0F653B0BF90D9AC3FCF9893C00D45CFC076EF38BF4BA176C0A001963F4EF791C07CF1B4BF4220FD40EAAEB0C0B27D46409A9EF63F3C661BC0322CDB3E43F990C092145FC0BD44D3BFE4A3EABC9337AEC0878EC83ECE2380C054361540549AD23F188630400556ABBF5AC9B1C094FDDD3FC9A50C3EF1F950C0B628DDBFD613A2C00945C7C057912E4047E6173FCACB9CC0E43C3C40061B83408F2B31C0054BA43F54CC70407330244060BCA7402D2322BFBB22D43F1D18773FFD68BFBFD9EEAB3F9B7C603F4C5E233FBF6426C03DF37D40981EB24044F0364044921F3F7923433F1D1E023F66978E409DACE93FF5129BC070E40BBEBAD7243FCDF871BF9AE0C8C0A62928C0CB560DC093FE0540379AD6BFCC946040696E41C009EC9D3F18239F40CC0ED240C6C08040FBFC17C0B6FF84401DD846C03A1A33C0E3D2CE3F202CC0BF6AD3C23D10323540FE6F31407553B440555CB1BE03F88BC0BAD037BFA3544840248FACBF18A03E40CCA53A405EF18E3F"> : tensor<5x6x7xf32>
    return %cst : tensor<5x6x7xf32>
  }
}
