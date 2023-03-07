// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3x9x10xf32>, tensor<3x3x4x5xf32>)
    %1 = call @expected() : () -> tensor<2x3x14x15xf32>
    %2 = stablehlo.convolution(%0#0, %0#1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {lhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x3x9x10xf32>, tensor<3x3x4x5xf32>) -> tensor<2x3x14x15xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3x14x15xf32>, tensor<2x3x14x15xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3x9x10xf32>, tensor<3x3x4x5xf32>) {
    %0 = stablehlo.constant dense<"0x3F5BA23F27D05840394C02405401C9407C5926C02A656AC0977F6940E6B2C03DD7DFF43A7A8F25BF7B17C7BF816F3CC0161D16C05D70C8408C1B42C0BAA6813F76161F406F1D40401AF8C23F9BBA86C090387ABD1A6B11BEB1C28640B9A94DC00F7E5C3ECCAC30BF847D883E0D586040F7C8923FB5522D40F28A2A40C9C683C0CB871F3FD35F943E12D18C40374F5440910D92C0B619E04076C39FBF82696D406D58023F2A1A7C40F5EA34C03384B73D21013D4090BF8740BD321DC0161CA8BFF87490406166DEBF9BC0B8BDE69FA83F37E3CDBE7560893F98594DC0B82F6ABFB7CA9B403E4423C1E2ED59C0C71471BF5C65A540F847503DB35B72409E36994070B4F0C04969A5C0C1DAA0406C77233EA75DF9BF894A5140480F98C0C5B36040DA2A13C08F963840951D273E82464B3F5E97273FFD2FEEBE8A0831C08B55853F84F706BF238DDD3F4650B5C0C9F992C0FA244D3FDAF85AC0EC82F03E6B8086C018A32E40767620C012A140C0661309403083C03FC911A540EE16483F701F76C0A5DE79C0721C9C40B7B248409F6672BF5214EABFF81A3BC00ED39BC0988745C0FB3F73BF569664C0D725BC4005A8C63FB6F0163FDD7C1EC09A6DBD3E5F172E40DED5ACBFA82FAD3F5682443EC2091640926D94C0199F964073E8793F7E70A6BE0B1334C0C4B61DBFF5C9A63FDEAD3F3F76F19B40A0DDC73F7E91AE3FE5C2B1BF16B62AC0A84C5C40063FD93ED4906840FCAF9D4009F84EC0BD58594023064D40D55BBB4032E546C01BD9043F68AB314043596ABEE34F81402FAD44C08C50E2BE9F9F4C3F84CC7540F623DEBFAFBF9FC0F7ABE33F8272293F32CEECBE071E8CC0784D87C0EF50683D994C6EBF60C775C00DCAA43BD767453E15EAA9BF82550340E4C35EC0648F72C0F119A03FAB4C7AC08DB046C05BD78E400C37163F03F30ABFABA993C08DCF41C07BD58E4014F56A40A8C662400BA089C0528F9E4021BC673FDCCA5840EF491A3FE8878A3FBB44423F59C945BF3381A8BE64D390C0DC466B3FED28F83EAA2E89C0F5649B40710BD5BF34C12B40E84994BF5E8A3B3CEE7680BE059961C057AF853E055CDB3FE5035AC08BB5C7BFA7DFB540289F3F40CCC5383F379F8B40FFCB2ABFBBA0883EE00496406C3FD5BF4B0F94C0737D0FC0EE47F3BE55B4D93DF16FCBC0201DDB3F9E9872C0F1A20140548CEE4053E2B53F0ACD1D40486D36400F40143FB5FA1E3F6B728340F15E0FBED6B90CC0A3A6D73E8451D9BF70EF0FC0D4C58E3D9E10C7BFADA2063F6A8F18BEF4D5B9BF9FF8C7BD9B2BD33D5D6B8B40062DCF3FFEC26CBFE3B69340774143BFC98AA63F56641CC0AC7F94402AED39C058AD16C04D69B3404B4420C0AD7F8AC0A6B82DC0E1DA58C0DE571B4050188040D184A63FF01E9E40B6AD23BF1C7B963EAF101F401E72D4BF842A7E3FBBB0A7BF8A01FC3E7A8F843F58AA38BF0580E4BFF2DF5CC01FD25C3E3B824140F4C3E4BEDAEA8D40CE318F40B39164406A3903409254423FECE86740F9064540B6ACACBFAAA24C41DF553740F5762340C384223EAB9B1EC08AB205C04BA76E40EEC2653ECB5228C0156785C0E10481BF6F4A60BF1CE605C02693AF3F022022C0B311053F9FD1BBBD699ED9BF3782BD3F715073C098DB803FE24807412EDB303FE860723F640287BE0D46343FC5F57CC012BCCB3FC3052D4098DA83404E15D43E998AB53FE084EEC0CE36143DCD846A408B951FBFD2D29E3E1CD1834054CA3B40005939C09CA915C0B67982403345073F4A45F33F491DD3BFCDEF0A4072146840BFEDCD3F93FB15407FA508C09CD9803FFC7B03404A29933FF26BED3FB94CBB3FD990C1C04C9B21C0D08106BF7A5089402EC01741059F01C0BE90C8BFE03A6B3E3A36953F19FBC23F2930453F6A101C3FA14A974026F09ABE615D99BDB7920C3F99D0EC40EDD182C03E861BC0493C0F408841F2BF33AB2C40172429408218873F8C5E4DC01E0C6ABDC8CB9140201B033E6753A3BD2EBE12C09C1A4540511BCABE44E953C07D3546C0A66217C01C60814022E596BF42596EBF64839AC0E07DE83F632B043FED3F62BD4F0B1340FD86104049A28140C0A7AF4021FEF7BED0D46040B2F7274031F59DBFD298F83F025605C099439B3DA98F993F739EEA3F48010D4084B1263F4C7200BFF7DD7B3F313130C0D48D953ED96D6BC090CB48BE671C9740C021AB3FEBFF713E500921C0322B08C0B68431C06EE48AC0EC73A63E759D05C0E9B184C0F7C104C08CCC6E40DC57C73F6AF5D44085598F40DEF44740B3148340DE2646C06B67F6BF1BECEB3F998FBA3FB07C974091C9A13ED8541540114A6DBF34532CBE2D3DB740710B4A40FDCA85C027605240F0DB7FBD3F805CC0DA5938C090D2BEBF86799CBF8D149C40A03C1FC08B18B2C0DB2BE3BFFE7BB9C002EF90C0673A80C0BEFEB13F3F377E40684151C029814F40A5F792403F001CC0598E653F3C8129C09BDAB9BD3784A7C0EA9AD53F710A9C3FF1350940D47082C0E57E9CC07E5AD0BFB8DE43BFD2D7B23FEB4F993EAB5149C0F691523F3467073DD91E573E78F5C3BFE721933F1137E0BECFE3C2BD4592623F7D00B740B46EC13FA16127C00581043E0B9BC03FDC6E95BD4A5C1F40CC592840C8264E40375A044067CA23BD360938C067A6283E35909540A2930C3FE88489BF697B023F5CF4CA40937ABABE634AAD4080E07C40389F3CC0FACB164053250E409A17E53F3D7D9B3FE3920FC0E5ABE93F6F1F4A4073619B3F8564C63F6CEC84BFE7B495C00F80A83FC3C86540109BD340657A433E3894564032D62A3F2DD3F3BE9EF24740971139BDCFBE6540F8B77540356DD8407B4892C07209823EF8E17D3D85F0293E90A780C0F5018F3E8E15CE3F43869F40ACFF693F7C43DBBDC819CBC016EA02405630C33F0DC7EEBFCD525EBE0D4A834079FFA7BE7DB98E3DC7C695C09E110A3FEC231B40340B9CBE0740B63F2D9F23404F822FBEA378E73D02D7A0C0F8CDE3BDB9632C402F3205C06C1B0C402F813C3FC6BAE63FB1BF42C0"> : tensor<2x3x9x10xf32>
    %1 = stablehlo.constant dense<"0xF433F4BF96FAFF3F1B4D4AC04306B740402C003F86533540C40886C08C1DC84038ABB53F721424BF2BBD3E3FD8FD073FE24491BF9E6CD240AFB2813E369F3E3F90730EC0A9542540417C0240243FB43F5D72E9BE39C353C021CAE5BDA3352A3FA8AE0DC0B5B390C050923F40112A6D404A03ADC0B51312BE4AA07F40A25F4BBE620285C0CC050A3FAB675D4040C51040A7BF97C0F8BEB03F030EAA4061A5EE3FF28F48BFE9343040C54E41C0B6D80B40E7D9913F0B043EBF4437B7BED9C3964028D4954093477CC0A67E40C02D2A863D3BDA9E40DFFC60BFB174A2408AE16E40C520584034A791BF53026B400B3B5C40DCDFAAC0F7534B40EBA81040D76F0E41C19108C0D02200402A6B72BF86D117C0F2049CC0AA34AEBF72983D40A2E569C042BC034176F9903EB8E538C00ADB0B4067E309407C4CB0C004DD5B3FB4871D4030728EC000DC02C03259E83D116E93C028FF873E1E8BF2BFE1AA7840ED802540C904B8405C396A3E6B3D6EC0D13F34400D00FE3D53106A409CC53B40E9DCF1BE06180B40D16F51BF7A0ABEC0B1BECE3FD31C243F0E1E2F40C1C12E408D606BBF91B6813F782CEDBFEE55F5BF6BEF62C04D7082C0D8244E4091C2403FB0C1623FC4BCDA405D4607BF0B236AC0822D8FBFED750EBF5BC48CBE761977BF980F19C009A73DC03EEE17BF017B34BCEC7D843DD697BBBD689315C051FE67C060C49ABEA064EE3F9419993F0D9168BEFCB18F403D351F4084544F4016A2A54028CE61BFD1CC61C06B44EE4092AB2B40261FAEBFBE9CB2BFC7C8D740A7F7B63FC5F4CF3FE6C253C0ABBB533FEDF08C3F7C3881BFA62A673ED351F240E2A8B0BD024F6CBE0587CBBF2D5281BE78C767BDD98ECCC03FEF4AC02E31CAC0BBCB87C0A4A701C1DE0F7FBF80BBA9BFCAACF7BE59C5A8BF3937A6BFA40C134020E89ABF80AA48BF48725CC0F09286C0C979C0C070A89FC0394522C09349363F2C79A53F0E9D8FBF6F034CBFBFBCB7BFE363AA3BE4B4AB3E"> : tensor<3x3x4x5xf32>
    return %0, %1 : tensor<2x3x9x10xf32>, tensor<3x3x4x5xf32>
  }
  func.func private @expected() -> tensor<2x3x14x15xf32> {
    %0 = stablehlo.constant dense<"0x28CC3DC2B69F66C10691B6C1A2918542702ECDC1AE1205C2BA4BE4C1669C03C2813B2F422F903142A6CAECC12E714542ED4C654200C34DC084E450C14434714131F411C06CB2B3C1DC5F1C425D21D44176D46CC135FA1A4210A18B3F72C51EC2B8A6D5C23D4EBB4161F294423149AA41063EA6C1C55DA2418CFB09C1C6E69940B9389D4234FB6041360D1040A29D28415D9E16C2EBFEBBBF03265AC278279A4159E74142EE0C1B4282C27DC2F420244296460DC2B50C2140A9B01942741B6E402B1F9C4110C35E422EBB8842B2886042E3281CC213A0D241AEC1E341DD74C7C154E068C1B2F1A042047EC8C19D3A38424D3DDEC013B65541178D12429B761640185331422741F441414EBAC1AE2D954019237C424B036AC294A97F41E14394425AF5A3C1B86E09C1B940EC4110B324C11B110A41BE07DAC128E1E1C060191042AAF92141AE56364220E662BFA1A8174289C669C15813F9C0583E1EC0DC47B84172624B41BC6979C017A6194176D0B6C1B06486BF8FBA964110D538C158E49342AC5541C2B62E51426F9B76C1D2C1A5C1FC0790C1FE15AA41D22D06425EC12A42266104C297B72D421AAB61C2FEC24841C12190420EAC21C2A0A80AC2FA6E5A42EECDBA4151B50B42C34055C276984C41D0CEFBBFE65CC7C25AEC2C42497730426297D1C1542EF3C14A3D7E426E89A0C1D7C2A2C13896AC3FB7A4A941D619A34058BB01C293307541B6C7B1C07419D2C2C602174273083541682A22C2779C5841A8187642C26D17C2975D0742168BF041CF5296C26D02A2C2E64895C1AC01CDC1C51F3342C8B611408CD898C16C5533C238A8404104CD20C2A2ED1B42DE9452413013DA40975987420A12B4C28D465EC2C9C6A0C137E127C240E55CC0824864423C0AEDC177F82FC2F7455342BA1C23C2447B9B4229A8A2C1B42E54429F5F5D422B6E25C2A5E36042FA8B3AC20A8EDDC1FF7389423F358FC268908CC1B085E7418CBD9FC1708F823FA0AC8D3F6E99E8C1A0DA42BFCA9ED8419DB614C223589042F3F174C1A61A3DC22F2B134220505AC2422F784101F3AC41AC0A8641A66F5A401FB4F5C0E4DE94C100A9B83DEC2F9C4128466DC2D2F613C10686CC418830F73F00E44E4270B2A241D4A421C26EE98442EEF85B42F4968640B4FD1C419FB19B417CAF7D4287D3FBC0370462C1180ED1405256E9C298D28C40EB8B8042FC6A4FC2FC0428C2D030A4C19A098BC1AC02134262FF5642DFF2B24167DD0C42AC81FDC1C6B7D841F213F1C09C25E240240F2BC2274089C2D936D3418C32B0C1ABB23040E680BFC1005222C11C775F42BAC98642D28E18C218BC01C2560A2DC1D2A3EF40FE923DC164E78D401077A441B8D134422AD3A7427C9EC041E4A4B9C109CF0A4270F90EC2580850C1B01147C199320641229DBD41C47907429FCA9541DE4EBF41EBA60AC28136C2C1EE0B5CC1617A0E41B72AB2C1816F64C2A3AC4742FCD83BC190C6B9C072D3054250C34B40420429C2948B2542F6184F4004395CC2CED24542BC0214C1F2B4A1C1FB850E426818E34199DF2E422F584BC12F1AD041588F3340FCCF73C2A2300742367A7542C16906C2B4C833C2273F4141945818C1631D1942407CCCBF887295C10CEA00C2109D51C1ECF6B4C02D1237C1C0F428C0F4CF2D40526F73C1D902A5C1637007C2F1F3A4C0D0F29B4114FA59C1060D1B41F6FD2C417932ABC11ED61A42BF1CDB41988A1642CB2F2D42E994A2C10CBD99C267968D424EAF0DC18461D841A73CDD411EA244C1EA89A0428C25994174EC6A3F0F0B2EC275DCA0BF3392CA41E65FA4C1C73DD240AE9F9E4281A81BC2BBDA2442B6AE3842F4E395C2B0F8BFC1C01E9BC054BD60C2057325C26B7DC4C02095EA41DCBA5B413FA6BFC1F4ED2E42BF17B0C12AE1D4C180B244C1022886C2D9A563C2238B91429CF105C2A187E8414E898BC02C24A2C160F600C2574229C22A020542CB439442EB80A6411CCAAD41755308C2097B51C220987341C7B66D413594D2414E6B234286760EC25270E9C1FA2180424627124100B9EF428CD435C2416C8DC2BFEB15C2D6E492C2FB3F7B42F408DD414DFF93C2A4E381C1C44C71C2BFA39542BE9F13C10B5EB0C2A232C8C1DD53B5C13909A8C1A0EB8342F549624280BCC8C0390920C10245C9C074EAA53FD4F402C0EE6D3D41784444406CBA05BF504EEABFC23EF341E608D7401C239C42CB652C423079A0C2E9A9DAC18ACC59C2CE8B144228AB3542916FE141265B363FBB1ECAC17405CEC16B3ABBC181AB34C2B8BA41C1DAA50B428A42D1C106376DC1B19CA7C234E3E0C184D30D410563B941F6173BC1147D1DC2CE375CC0DF50D241FDB60EC24B004CC25A924EC160759E3FEDE997414AC84F429D80E6418AE34542B0089AC173AC6DC19EB6D641288E30C014BF2A42D83FC540AA2EB5C04C4AE241C40768C2707017C224EEBE41FC445441DCCBF4C08C9B72C2F07A0CC21AE25DC2262C0D41F2C3BB42B965D74172AA33C260DF14C1ACDFE23F647E4DC2F60A62C2BAFD87419CFE99C066DBB141C8F8A9C1E4E13AC1481787C2DF5805C2A76CA0C0A8E81D4034C337414AF4714262FB6A4250CFDC3FAE9A9641B48E99BF9BDB82C101BDD441D55C30C18FD544C23C3F96C26CE191C0922ED841A32853C1C93265C2313F96C1C4A59AC1EE382042A9F889424F008EC1EEB408C09063A3BFC3DDA541753A62411C9A8DC17EF688C13D116BC19AFBB5C1BB9313429629BCC04498DD4058AF86C15A5AFCC14450F940100B67429BB34D4168E7D8C19F611BC2AC0F9FC22136A2C1F6888EC0005B69BD94E21A41BC820FC200958BC2C1D8AFC2473967C210B7D24106C027C22C4F0BC186E9EF416E1A7CC18C2AA4414077853DB32D97C188C4E1C090619041D8B3E1414D1881427411CAC186F32D4153C5EDC1641A0AC06EDC3D42DE074FC1C9952DC2483A2242DA3996C1BAE13CC2AA9B034212269842E3BA5FC0B7BC10C22215A2C0D02516405D8B02C052270F42D81A9FC06C4AE9C1F15B2F42819CE6418C5936C182931D42F6B18E40BCEC534112F35EC232C0CAC1100671C14C2397410EE556C11A4D9041C8B670C2747D0142C8D62FC22AD9A7C2F483F8C085E789416C7BD641694F744260422DC0E2F2AB42CB1521C2601DEE3F430001428C45C6C0D1216D42F4110B4203B56AC2D242B641BBB5ED416CFDE2C1CADB8642E4853742F5F1A8416696CAC127DA77C24208544094042EC2467FA8C100EA88422445BF4132E444424BDA8C4160514FC1CF0418C29FF5B3411C64853FB6A2D5C1C428FA41722BA54144657F4200378C3E9C69E1C170F2C0416DF4D34162B9A241D9E6D6C1EA9FC1C178CC4FC1703BBCC06BAA4D42C05908C2546AE1C122AB76414BE109C23EB38A40D8EBC440847FA64051054D414A86FF419CE67CC1984087419B9EDF400B129DC1C669A8C14ECA80C1002DC0C187BD84C23B3EBCC267849E41008C9DC2906479C0DCD10342DB7308C21E29C3C2A0AE39BF8DB201C2046A02C20EF2FAC280AB624020CF04C10CF60542B3B1D7C10A22C8BFF5A08B42D52272C207A2B141CEF69C41F2F32941CE7D4DC0C8843E41E69586BF7E7A8DC1E8726342542D99C1845D3F41EC269D4010A1B5C1BFED1EC176201EC17E8C85C058324542EAB43CC111E090C16092F641C40B4141127B24C10E68DAC037F025C1C0E048422A1C81C125F12FC2C4170FC239232DC1958EA2C12ECD3F42F02C9C41A8C88F4144A3144080D195C04603BF40CA509CC1F9829BC13A694D423EA07841B0D9424067CA4142D6FF26C2CDA1984160F5FDBF02F5D6C16B171142A4128542271F02C2D8FF914019BA0E41F33B49C11BA7194160B7CCC1DA3B074286941F42A46922C2170E4F40E0B91F4182DD37C14B3D2D42991C72423E16E0C199288FC1FBAC9DC1AEF321408AD0B1C17BA88E41817459406197C941866F9BC0DAA27E42A8C99B4079E1BDC14548F2C14E641E42A3B75A42DE16F341B66245C0C6EFA74118D39E42B375AC417455FD41024F3142E7D411C26A400D4291AFEDC1DC1F94405D9D6DC28E5816C1B02C5C4191D73442862480418A8C99C1E185B742BD67C24004626C42A6428E4152F282C17914C1418E2E234253568542211DE9C1A0E06C42308C2EC16E620EC2C25CF94162729242D8AE74411DA60BC17DD21542F193A042E0D90FC13016E9C1C8E7ED4157F789C1F0AD5A3FB6421AC1F4676BC0420128C203649742B918A041E8885E414436B34063FF32C1598323423D6087C22EA89441DE6AB5403D7514C26DFEB5C047A93D42261E13C264748E41ACC31C428D2CFAC175B5F741ED07434122272A42C083FD41E8A68FC1FF32F4401A30A8C261FAD3413644ACC1667C6DC240183EC185D472424539F941865D9DC163216B4133A2A341B46B04C2C653E14115F861C2BEE722426E2E84C2986228414B39F841B81636C2E29BA4411F8FE1C18DE20A4214114E4238064441149C06420EFA9C41861F8641FA4585C2EA6D5E41BCA036C1089014C154007D42A6C46E4113C379C1669FDF416A3BEF40FBB17E422A1F28C23EE9144215594FC2A460844268AC36C2FF9AADC18206814218C88CC1BBE7B141D03582C08AC985C1FE32C640556F4542CA1A0941CD5C40C20045E8C0C288AB4064A6DB4106DB58C12A6137420010BBC098FE76C17A0107C264F12CC16400A4C018F49DC0F7AE624277746CC24C5AFD414D595242A6333AC2F4920743206A23C16A84A042D47FD3C2EC644F42A17822C1941D81428ACF99C15D40ECC1E637CC401CAC72C160AB86C025E687C192D60042024A8D418CC9434251D6BB4035030842503B3AC2AD2B2C42C924B041C3628C41440619C12566C4416389CFC04EBFBAC118F4BAC14307374290A545C27D6E30C218E3C3C12CA14FC20655B2C15D2B97429DFA80C2A5FC2F427BB108C148474140AAAA01C2887E24C2BEF65DC1FB31BA4207BF64C19CDD614288E664C001B3054244D954402C830CC2EBEF8DC2108EB0C1A2CA09429CAA92C08CA5B0BF34324D41C4B601C155312EC2989133C1B79AA2424AC5B7C1A2B502C261CE1CC29CE8A240038869429EED54424017C941AC7BB7C27351AD4189DBA8C1C78449C19AA88841C238B2C1EF9E58426D9205C24EDB6DC17435D94109C0F3C07E9DF541ED898DC27FF70CC2204EB5C027734C40E09520C205CF62C2551683C2AEB40CC1503F6C4047494F41CD09B0421CA22442FA5F66C18E3AF3418980BEC2A82B7C42800E4842F29DB6C1161218C14F199F41BD58D64274D4AB42EE5CB4C1457D2E412775B840BECF3CC1E8C9244044C896416E936F420966B73F3E0F7541A60975C1A6630D429633884190DE90C058E0FAC192F7D43F4824DDC09FF605C23DF5B0C18A2D914237C312C1C12029C265C9B5C11B5BECC1112B38C146FB13426AB1A3C14064F9410086953E18496A415023C2C04B8B984169E01842720C95C266D682C11E8F8E42CFE019C2B212CB415973DC409204CDC163140840543A19C20FF0FEC18CE303C2F47C19C1B67577428436AE411E7FC142FDF1B041269EA2C15E8E58C20A9774424A208AC26A1A6A42BC5A5B4138874D41E54A94419591F14106C27A42BA2E5AC28008CEC1C66830C15E0AD5C26B7F29C1DE96FFC03B6084C2326BD5C0E2028AC2A9C02242852600C2361B8EC25B59E64160AB3C41A701654000075C427442C5C16CAA3DC2A875F4C1DA8FFAC08DCBD44106AAB5428C3DEC41C020D141B83BAA4091EED1414B7D4B42BC77C8C123C030C1B46C4C4114F6C8C087081542420801C2AA9C2342CC02CCC1E52F17C1731435C12B489C4199B80CC28CCAEAC090A30341DC7795C04F7385414EE99C4158761B422D2E2FC1DE6E2C429303724120A89E3E7EB1A2C1408F48C21C6A9AC2AF37ACC2278119C20460EA3F7085AB41376A1BC2E008B3C180CBA93FCEFBDEC1EBC71CC28C6CECC1EC8684406419DFC02E9D65C15F3A054111C507C2E484944064DDEA41C32097C125969F4126646E429ABE15C27883363F8EAB2141CF8449414F1723C1464D2B4188F2E240B700BD41B1B653C2529DA84061AE29C1760D1F422960A44201A093423CAAA4C0D5658F41341CD6C11754EF41F06786404BF0A841003786BD608FD8C0648BDAC17E2727C17E73B8C1405A7D40D4DB9840E414A24047730C41A6AE0942163B38C2C0D89EC08CD33542FB911C426C13004252424042AFA5BCC1982EE0BF355BBDC1DE498041B24059C1D3DC66423AC2DA41EA06CB40CF2F5DC2EE240BC22E9C81C22DB026C26EB18A4132260741E6CE33C080E26B3FA4DFD44146B15B420AFA9EC102DE3BC26A0AF7C1FAE70FC358699FC166299FC28E79AAC27660FCC2EA9E14C1101499C217160BC28E0A33C2F10B31C162E9F5C14912BEC13A74D14161B866C2153D43C1BC90CF4158F681417684E541905E5BC1D91CA8C0B134C7C11C893FC1C1745EC2244D27C265EAEBC187E4A1419A02CCC199DD63C2751D27C27947AF4106BE394131140A41782933405C91F0C10B25EBC1979CA5C1E0B1003F72FD13C258417FC2735C4CC204D18EC224390EC2A2AFC0C19D72B5C1A958004274DACEC03A29B64136C185C1D160264287FC8BC270577DC0F16F22C2C9795042928E78C1E8ED2A40637714C28AF4634219B837419AE9CA428CA666C1DE6EFD41D84505C0ACFA54C19EEE78C130F716C00FA00AC1A69A91424850F33FE0741A42740784412726E8416FE57E42C9277A426E61EC407CA67842EAA037C291DD26C2A3782DC2143933C1040981C1DD7722C140586DC0ECBC5E4246C22E42CE0931424064903FDD376742DBB9C241E3758BC1B8A553C2E52275C2592D3BC19E16844208E1E4C1E6BD36412273D941027AACC28D2451C102C18DC2D70203C2F0962DC220D0AA40158BF5C1C58EEEC1CAB51AC21464BA409C06E4C093E21A41F1BFFB410EC7C8406E0799C1EB5E0241F6BC80412B9573C1CA1ACBC102992241A9B4C141D09EA43F535DC041AF58B041FA178A4232460142546723C2F66567C20002713EEFD41E413AD7BF42741327429C8FE1419C2BD44141B58742"> : tensor<2x3x14x15xf32>
    return %0 : tensor<2x3x14x15xf32>
  }
}
