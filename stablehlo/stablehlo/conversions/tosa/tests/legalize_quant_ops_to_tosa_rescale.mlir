// RUN: stablehlo-opt %s --stablehlo-quant-legalize-to-tosa-rescale --split-input-file -verify-each | FileCheck %s

// -----
// CHECK-LABEL: @add
func.func @add(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  // CHECK-DAG: %[[SHIFT0:.+]] = "tosa.const"() <{values = dense<50>
  // CHECK-DAG: %[[SHIFT1:.+]] = "tosa.const"() <{values = dense<11>
  // CHECK-DAG: %[[SHIFT2:.+]] = "tosa.const"() <{values = dense<13>
  // CHECK-DAG: %[[MUL0:.+]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[MUL1:.+]] = "tosa.const"() <{values = dense<1431655765> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP0:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP1:.+]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0, %[[MUL1]], %[[SHIFT2]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1, %[[MUL0]], %[[SHIFT1]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: %[[V2:.+]] = stablehlo.add %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]], %[[MUL0]], %[[SHIFT0]], %[[ZP0]], %[[ZP1]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}

// -----
// CHECK-LABEL: @sub
func.func @sub(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  // CHECK-DAG: %[[SHIFT0:.+]] = "tosa.const"() <{values = dense<50>
  // CHECK-DAG: %[[SHIFT1:.+]] = "tosa.const"() <{values = dense<11>
  // CHECK-DAG: %[[SHIFT2:.+]] = "tosa.const"() <{values = dense<13>
  // CHECK-DAG: %[[MUL0:.+]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[MUL1:.+]] = "tosa.const"() <{values = dense<1431655765> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP0:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP1:.+]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0, %[[MUL1]], %[[SHIFT2]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1, %[[MUL0]], %[[SHIFT1]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: %[[V2:.+]] = stablehlo.subtract %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]], %[[MUL0]], %[[SHIFT0]], %[[ZP0]], %[[ZP1]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>>
  %0 = "stablehlo.subtract"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}

// -----
// CHECK-LABEL: @mul
func.func @mul(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>> {
  // CHECK-DAG: %[[SHIFT0:.+]] = "tosa.const"() <{values = dense<30>
  // CHECK-DAG: %[[SHIFT1:.+]] = "tosa.const"() <{values = dense<37>
  // CHECK-DAG: %[[MUL0:.+]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[MUL1:.+]] = "tosa.const"() <{values = dense<1717986918> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP0:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP1:.+]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0, %[[MUL0]], %[[SHIFT0]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1, %[[MUL0]], %[[SHIFT0]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: %[[V2:.+]] = stablehlo.multiply %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]], %[[MUL1]], %[[SHIFT1]], %[[ZP0]], %[[ZP1]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-1>>
  %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-1>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-1>>
}

// -----
// CHECK-LABEL: @div
func.func @div(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>> {
  // CHECK-DAG: %[[SHIFT0:.+]] = "tosa.const"() <{values = dense<30>
  // CHECK-DAG: %[[SHIFT1:.+]] = "tosa.const"() <{values = dense<37>
  // CHECK-DAG: %[[MUL0:.+]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[MUL1:.+]] = "tosa.const"() <{values = dense<1717986918> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP0:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP1:.+]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[ZP2:.+]] = "tosa.const"() <{values = dense<-2> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[ZP3:.+]] = "tosa.const"() <{values = dense<-3> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0, %[[MUL0]], %[[SHIFT0]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1, %[[MUL0]], %[[SHIFT0]], %[[ZP2]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: %[[V2:.+]] = stablehlo.divide %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]], %[[MUL1]], %[[SHIFT1]], %[[ZP0]], %[[ZP3]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-3>>
  %0 = "stablehlo.divide"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
}

// -----
// CHECK-LABEL: @max
func.func @max(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>> {
  // CHECK-DAG: %[[SHIFT0:.+]] = "tosa.const"() <{values = dense<12>
  // CHECK-DAG: %[[SHIFT1:.+]] = "tosa.const"() <{values = dense<10>
  // CHECK-DAG: %[[SHIFT2:.+]] = "tosa.const"() <{values = dense<51>
  // CHECK-DAG: %[[MUL0:.+]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[MUL1:.+]] = "tosa.const"() <{values = dense<1431655765> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP0:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP1:.+]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[ZP2:.+]] = "tosa.const"() <{values = dense<-2> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[ZP3:.+]] = "tosa.const"() <{values = dense<-3> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0, %[[MUL1]], %[[SHIFT0]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1, %[[MUL0]], %[[SHIFT1]], %[[ZP2]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: %[[V2:.+]] = stablehlo.maximum %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]], %[[MUL0]], %[[SHIFT2]], %[[ZP0]], %[[ZP3]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-3>>
  %0 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
}

// -----
// CHECK-LABEL: @min
func.func @min(%arg0 : tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>,
               %arg1 : tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>) -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>> {
  // CHECK-DAG: %[[SHIFT0:.+]] = "tosa.const"() <{values = dense<12>
  // CHECK-DAG: %[[SHIFT1:.+]] = "tosa.const"() <{values = dense<10>
  // CHECK-DAG: %[[SHIFT2:.+]] = "tosa.const"() <{values = dense<51>
  // CHECK-DAG: %[[MUL0:.+]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[MUL1:.+]] = "tosa.const"() <{values = dense<1431655765> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP0:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP1:.+]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[ZP2:.+]] = "tosa.const"() <{values = dense<-2> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[ZP3:.+]] = "tosa.const"() <{values = dense<-3> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[V0:.+]] = tosa.rescale %arg0, %[[MUL1]], %[[SHIFT0]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK-DAG: %[[V1:.+]] = tosa.rescale %arg1, %[[MUL0]], %[[SHIFT1]], %[[ZP2]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: %[[V2:.+]] = stablehlo.minimum %[[V0]], %[[V1]] : tensor<2x2xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V2]], %[[MUL0]], %[[SHIFT2]], %[[ZP0]], %[[ZP3]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: return %[[V3]] : tensor<2x2x!quant.uniform<i8:f32, 1.500000e-01:-3>>
  %0 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 0.025:-1>>, tensor<2x2x!quant.uniform<i8:f32, 0.075:-2>>)
            -> tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 1.5e-01:-3>>
}

// -----
// CHECK-LABEL: @abs
func.func @abs(%arg0 : tensor<20x20x!quant.uniform<i8:f32, 0.025:-1>>) -> tensor<20x20x!quant.uniform<i8:f32, 1.5e-01:-128>> {
  // CHECK-DAG: %[[SHIFT0:.+]] = "tosa.const"() <{values = dense<30>
  // CHECK-DAG: %[[SHIFT1:.+]] = "tosa.const"() <{values = dense<33>
  // CHECK-DAG: %[[MUL0:.+]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[MUL1:.+]] = "tosa.const"() <{values = dense<1431655765> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP0:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP1:.+]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[ZP2:.+]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: %[[V0:.+]] = tosa.rescale %arg0, %[[MUL0]], %[[SHIFT0]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: %[[V1:.+]] = stablehlo.abs %[[V0]] : tensor<20x20xi32>
  // CHECK: %[[V3:.+]] = tosa.rescale %[[V1]], %[[MUL1]], %[[SHIFT1]], %[[ZP0]], %[[ZP2]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: return %[[V3]] : tensor<20x20x!quant.uniform<i8:f32, 1.500000e-01:-128>>
  %0 = "stablehlo.abs"(%arg0) : (tensor<20x20x!quant.uniform<i8:f32, 0.025:-1>>) -> tensor<20x20x!quant.uniform<i8:f32, 1.5e-01:-128>>
  return %0 : tensor<20x20x!quant.uniform<i8:f32, 1.5e-01:-128>>
}

// -----
// CHECK-LABEL: @compareGE
func.func @compareGE(%arg0 : tensor<20x20x!quant.uniform<i8:f32, 0.025:-1>>,
                   %arg1 : tensor<20x20x!quant.uniform<i8:f32, 0.075:-2>>) -> tensor<20x20xi1> {
  // CHECK-DAG: %[[SHIFT0:.+]] = "tosa.const"() <{values = dense<12>
  // CHECK-DAG: %[[SHIFT1:.+]] = "tosa.const"() <{values = dense<10>
  // CHECK-DAG: %[[MUL0:.+]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[MUL1:.+]] = "tosa.const"() <{values = dense<1431655765> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP0:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP1:.+]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[ZP2:.+]] = "tosa.const"() <{values = dense<-2> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: %[[V0:.+]] = tosa.rescale %arg0, %[[MUL1]], %[[SHIFT0]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: %[[V1:.+]] = tosa.rescale %arg1, %[[MUL0]], %[[SHIFT1]], %[[ZP2]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: %[[V2:.+]] = stablehlo.compare GE, %[[V0]], %[[V1]], TOTALORDER :
  // CHECK: return %[[V2]]
  %0 = stablehlo.compare GE, %arg0, %arg1, TOTALORDER : (tensor<20x20x!quant.uniform<i8:f32, 0.025:-1>>, tensor<20x20x!quant.uniform<i8:f32, 0.075:-2>>) -> tensor<20x20xi1>
  return %0 : tensor<20x20xi1>
}

// -----
// CHECK-LABEL: @compareLT
func.func @compareLT(%arg0 : tensor<20x20x!quant.uniform<i16:f32, 0.025:0>>,
                   %arg1 : tensor<20x20x!quant.uniform<i16:f32, 0.075:0>>) -> tensor<20x20xi1> {
  // CHECK-DAG: %[[SHIFT0:.+]] = "tosa.const"() <{values = dense<17>
  // CHECK-DAG: %[[SHIFT1:.+]] = "tosa.const"() <{values = dense<15>
  // CHECK-DAG: %[[MUL0:.+]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[MUL1:.+]] = "tosa.const"() <{values = dense<1431655765> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP0:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZP1:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
  // CHECK: %[[V0:.+]] = tosa.rescale %arg0, %[[MUL1]], %[[SHIFT0]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: %[[V1:.+]] = tosa.rescale %arg1, %[[MUL0]], %[[SHIFT1]], %[[ZP1]], %[[ZP0]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true}
  // CHECK: %[[V2:.+]] = stablehlo.compare LT, %[[V0]], %[[V1]], TOTALORDER :
  // CHECK: return %[[V2]]
  %0 = stablehlo.compare LT, %arg0, %arg1, TOTALORDER : (tensor<20x20x!quant.uniform<i16:f32, 0.025:0>>, tensor<20x20x!quant.uniform<i16:f32, 0.075:0>>) -> tensor<20x20xi1>
  return %0 : tensor<20x20xi1>
}
