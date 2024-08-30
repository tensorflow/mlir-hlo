// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xf32>
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = chlo.atan %0 : tensor<20x20xf32> -> tensor<20x20xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    return %2 : tensor<20x20xf32>
  }
  func.func private @inputs() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x8D3201C09333CBBE9AA2553FBF3BA7C02D561540E5C318C0E4F1BCC0A8EA4640FE1417C07675D8BF29101BC04403163F289E4D3F9DF37A3F55EE313FE60F11C08F2769C0922743BF27044ABFDC926F3E684FBDBF20778DBFA76ECA3E219B00C1260DD9BFEE178C3FCB9CBD3F10EB01406B4B52C0A0CDB0BFAD528840C49E1F3F1BFC1ABF90E4D23F40E918C0CA9CEDBE976440C0380D12C0D49A3A403C13C7BF16F810BFE73C693FE5AAFFBFA0AF53C0152E85C09FD62FC0343FE7BFA5A65E3F81C2793FD36FD63F9E92BABF230E44C06A1252C0B1115EC0E1DBA43F69F9F3404B876BC08E767940680486BF25381EC07F232F3F4CC85C40624B913F9A8C9E3FF75880404CBF353F19D956C018CF39BF66DCCE3F05828240C7942C3E964EAF4054CC1040F1430E40B6DBEBBF02481FBEFC6A5D40D2B649401267E1BF837CC1C01080B9BEF0252D40785FA1BE6A2B70C0404519C084F06BC0845B4740C2FB5FC00BB2303F4D1DB9BF5AF942C00F7BE93F6F8AF5BFE2B9D7BFCD438BC008D94ABF064EC43F9D6B2FC024C6ADBEF651D1BF3AF58CC035CA06C18306163F26393640649B0E3E7F17DE4053E35AC0ABF25F4061C7473FFD405C3F3F4ADE3F517D1D40FF0DBE3EB8A0A43FE3D5804018B42BC0EACC8CC08617753EDF8BA93E09E077401AF0BABF5ECDC93E3EEB01BFB8112CC0CA7237BE8113403E0FE320BF2513A13F150240BEE908D9BFA2D10AC01CEA95BE4C579F3F4B852D40778DC2BF871509C08CA3E9BFF5E47C40C1278E3F5ACBA0402B1387402614A4BFC0C83640C0D385C0C45D6C400897A53F8C412BC09080CC3EECBC02C06AF6814036AE5CC0EA91763F4A669BBEE3960E3F015E3540F58D7D4066205B406578B23FE0E9D73FE1D9A2BFCB775DBFC53F8D3FF6B811408C7CF3C0531CD540B07FBE3F55EA873FED85AC3F5DC860BDABB0C13F0601F0BE99AEB3C025DA803E411F1E4099CD86BF441A05BE2FBD70C0E3C2B4BF136FEE4055FD254091466C40933517BEA15274C0C32732401E5649C0C15D154085F9683EA3B4963F366500401BEC8AC0D453033E128A0EC01593EC3F7A3F9C40DFACF8401F260AC0B139393F0D601440EBE17CC08022503FF206AE3FEF4FA4BFA2A2F1BFE4DBD4BE3A2BAF3FC2221540047972C05CD09D3FB68CA4BF6A302D40CBDC5EC070D1AEBF263645C0FCE85EC02C9C393FF8970840291CB1C0C2382F400460B13EE1B82F3F1EC738BFB9468D40150AB8BFD02C23C03ADE6940C7C29CC0BE990EC0D252AB406124814037023840B2207440F8DD4AC0426C0140DFD91440F3391940243A0DC0667807C0AEB0C8BFBE7E5D4083595D4002F5353F16B339407F990CC0DF1D8ABF7E5E073F83660940695CBB3EB6AF0EC004F934C0AA67FBBF95BCEABF15E317C0AC9CE43FFD9736BF83EE9D406376F63F7A53234015B495C05FA3E5BEFC9CB440E1458A4014A5A33F5CAE30405FD97CC0A2EA8BC0D86AA5BEC1BC3AC039D0F93CDB50ABC024038140F245FA3F8923DBBF1778A13F19E40F405A9F0F402CC1B2BFD5F5283F51D8CFBFFAE624406437A13FDBB8A940594E223F62C0EEBFA996013F799394BE65F748C0525A28C047F8593F80CB303F6E8BD5C0407DC1BFB10561BF5EC740C06BE4303EE62B96C02462C73EF31E81C053DDF93FB92804BEA1345AC0CAF154400518F23FF90DC440E4F35FBE0E9A9E3F6603F5BE63C2023F9CDD80C0B291CB3FF2355540D36EFFBFB50CBC3F677DA33F605152C08A52E13F19CFB6406504263F48D7AEBE92600A40BE69E9BF513115C0DCF191404A2B0DBF20F29ABF948A60BFEDA20E40A9414C40D62E994024E41F404D698040B3A5A540B0206F3F82DA9D403C51533F1B661CC03F53EF3FEBEF09C07E022B40C8CB2E4056CE8EC000EA8A3F0606A7C055337640F21F43BF253D97BF58C98CC0FF08A0BEC9C2D8BFC9940B3EAFDB2B40A29436C0D1319040AE8933C0649A93C074542F3FFD818140032605BF188090BEB84F6D408F8A033F55125E3F9BE216C099807FC004022E4010C4004075B342C066F8F5BF4B89F83E1BED8EBF3759F4BF9CAFA6BFDEBABDC042B982405025773F58BCCABF2FFF0040C48F86BED13704BFC6BA6C3EEF1180BF03E9A1BF12857440E4C19BC0DCE02FC026B1C44049A2383EDD3421C0BD9913C02775063EA2BFFABE12F183C03D7BC43F1504A5BD71EBA1C01837BFBF92946DBF4DEA9F3FF7EBA5403A6396408C2B5A40"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
  func.func private @expected() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xC2308EBF9270C1BEF107323F9BDCB0BF9F3C953FD14796BF6196B3BFBB37A13FEDC595BFEDB484BF43F596BF83B1073F483B2D3F3F83463FFF7D1B3FA1E093BF62C5A6BF49BE26BFF8062BBF37576B3E7AEC79BF01D855BF5CC6C03EFE37B9BF25DC84BF209A543FFB1C7A3F25798E3F3C3FA3BFC3B471BFC98CAB3F66BA0E3FFA5C0BBF9B3C833FF75296BFA677DEBED0F49FBFB53294BF56C59E3F26D77FBF1CE803BFF3283D3F03A68DBF437BA3BFF5E0AABFA7619CBF3D5A88BF2241373F4BE7453FE22D843F5A3078BF7CACA0BF9035A3BF5225A5BF001D693F5A5EB83FC019A7BFACEAA83F9FF04EBF73DC97BF8999193F86F2A43F8B3D593F6840643F46C9A93F240C1E3F0401A4BF52BA20BFCF22823FB848AA3F74F92A3E9EF3B13F94CA933F80F3923FEF6A89BFD7031EBEAE0BA53F67BCA13F72F486BF1715B4BFF1F7B1BE54BE9B3FAE539CBE32BAA7BF566E96BF3228A7BFDC4CA13FE46FA5BF40A81A3F1F4077BFE876A0BF5FDF883F7E8D8BBF358484BFF426ACBFF3892BBF26347E3F91489CBF1B87A7BEA1CF82BFE87CACBF4DF0B9BFEDB3073F2AD49D3F15B20D3EC3BEB63FBDA6A4BF846EA53F7FA4293F22E2353F1930863FB3A7973FA4FBB53E76F0683F6AE6A93FC2649BBFFF74ACBFF590703E34BAA33E2AB8A83F1C6C78BFD23AC03ED972F0BE8E7B9BBFD68535BEB2DE3D3E63A30FBFC239663FDDCD3DBE0DDB84BF64C491BF88D691BEE2DF643F35D59B3FD4267DBF132791BFB9E888BFC755A93F9976563FBCEAAF3F7549AB3F518668BF9EF39D3FF604ABBF2637A73F64A9693FC0489BBFFF8FC23EB2CA8EBFED28AA3F7BEEA4BF4F42443F11E096BEFE18023FD6A39D3F136AA93F5AB0A43F7AD8723FB290843F699767BF509436BF20A6553F8317943FE455B8BF01FCB53F95AA7A3FD3BC503F0AB76E3FB38E60BD03A17C3F7D6EE0BEAC80B2BF27767C3E72D5973FFDAF4FBF495C04BE7CCDA7BF8C6374BF36FCB73F2AF5993FFA33A73FD81F16BEF144A8BFD3EA9C3FAFAAA1BFF93E953F4613653E2EDF5D3F6FDF8D3F5215ACBF4D9D023E110B93BF8E94893F6A34AF3F41AEB83F058891BF5C58203FD3EF943F6A55A9BF13C12E3FE5C66F3F84B368BF26B58ABF8DBAC9BE3E93703FA62C953FB007A8BF6EAB633F61E168BFD8C09B3F5E44A5BFA75470BF31E5A0BF3A46A5BFF598203F00FA903F912EB2BF9B3C9C3FFBBFAA3E2FFF193F180D20BFDA8CAC3F5F8D76BFDF3799BFE9DEA63F7E49AFBF531093BF226DB13F9EF8A93FAD379E3F863EA83F1CF2A1BF78478E3FF015953FFB6A963F7A9992BFC49190BFD06380BFBA0EA53FFC08A53FD72F1E3F58949E3F896292BF12CA52BFBC0AF93E0144913F879CB33EB01793BF6D8D9DBF5CC88CBF5E2989BF5B0496BFA7BA873FFE9B1EBF1C79AF3FC4BF8B3F2C42993F591BAEBF08DED7BED79DB23FACF3AB3F2332683FEA939C3F6254A9BF4648ACBF5AFF9FBE79CC9EBF69BCF93CDE6CB1BFEBF0A93F818C8C3F0F6585BFC887663F457E933F8667933FDD0973BFFC57153F516882BFB4AC993FCD55663F5935B13F2BA7103FDD118ABF42ECEF3EBA9A90BE4499A1BFD88F9ABFA690343F7DB91A3FC905B6BFBB817CBF2A9938BF7608A0BFDE292F3E3A30AEBFFE21BE3E5CF7A9BFC9768C3FBB6E03BE258BA4BFEEB0A33FD6CE8A3F375AB43F2C7B5CBE054B643FE285E4BE8FC8F13E35E8A9BFC636813F36BCA33FF6998DBF1021793F0014683F3E40A3BF70EF863F79E1B23F6248133FE17BA8BEA69C913F5EDB88BF2C3195BF096FAD3FD50201BF495E61BFA65338BF6713933F1B32A23FA6B3AE3FBB53983F18CDA93FB5A2B03F7857403FF475AF3F9AA8303F145897BF97328A3FD77491BF46399B3FE5229C3F84D8ACBF1E86533F02D5B0BF3C82A83F76B926BF5C515EBF4B74ACBFEC1B9BBEF0C884BFE9B90A3E696E9B3F38E89DBFE01BAD3F2E3B9DBF17BCADBFE0BA193F380EAA3F088FF5BE5AD68CBE1458A73FAC05F33E99EC363F94B695BF61A5A9BF01F39B3F1A058E3F4B69A0BFF6A48BBFC261E73ECC2657BF064C8BBF697A6ABFB5ACB3BF3B55AA3FAF8E443F25FA80BF7E1C8E3FF89583BE8417F4BE7EA4683EC82149BFCADE66BF6A4BA83F2220AFBF0C649CBF1A6BB43FD6AB363E13B098BF55B194BF5AB1053E522BE9BE219BAABF1C4F7E3F04A9A4BD5716B0BF621C7BBF4A833FBFF252653FCEACB03FD339AE3FB489A43F"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
}