/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef STABLEHLO_INTEGRATIONS_C_STABLEHLO_ATTRIBUTES_H
#define STABLEHLO_INTEGRATIONS_C_STABLEHLO_ATTRIBUTES_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// ScatterDimensionNumbers
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloScatterDimensionNumbersGet(
    MlirContext ctx,                                                  //
    intptr_t nUpdateWindowDims, const int64_t *updateWindowDims,      //
    intptr_t nInsertedWindowDims, const int64_t *insertedWindowDims,  //
    intptr_t nInputBatchingDims, const int64_t *inputBatchingDims,    //
    intptr_t nScatterIndicesBatchingDims,                             //
    const int64_t *scatterIndicesBatchingDims,                        //
    intptr_t nScatteredDimsToOperandDims,                             //
    const int64_t *scatteredDimsToOperandDims,                        //
    int64_t indexVectorDim);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAScatterDimensionNumbers(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloScatterDimensionNumbersGetUpdateWindowDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloScatterDimensionNumbersGetUpdateWindowDimsElem(MlirAttribute attr,
                                                        intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloScatterDimensionNumbersGetInsertedWindowDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloScatterDimensionNumbersGetInsertedWindowDimsElem(MlirAttribute attr,
                                                          intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloScatterDimensionNumbersGetInputBatchingDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloScatterDimensionNumbersGetInputBatchingDimsElem(MlirAttribute attr,
                                                         intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize(
    MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem(
    MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(
    MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(
    MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
stablehloDimensionNumbersGetIndexVectorDim(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// GatherDimensionNumbers
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloGatherDimensionNumbersGet(
    MlirContext ctx, intptr_t nOffsetDims, const int64_t *offsetDims,
    intptr_t nCollapsedSliceDims, const int64_t *collapsedSliceDims,
    intptr_t nOperandBatchingDims, const int64_t *operandBatchingDims,
    intptr_t nStartIndicesBatchingDims, const int64_t *startIndicesBatchingDims,
    intptr_t nStartIndexMap, const int64_t *startIndexMap,
    int64_t indexVectorDim);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAGatherDimensionNumbers(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloGatherDimensionNumbersGetOffsetDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t stablehloGatherDimensionNumbersGetOffsetDimsElem(
    MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(MlirAttribute attr,
                                                         intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(MlirAttribute attr,
                                                          intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(
    MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(
    MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloGatherDimensionNumbersGetStartIndexMapSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t stablehloGatherDimensionNumbersGetStartIndexMapElem(
    MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
stablehloGatherDimensionNumbersGetIndexVectorDim(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// DotAlgorithm
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloDotAlgorithmGet(
    MlirContext ctx, MlirType lhsPrecisionType, MlirType rhsPrecisionType,
    MlirType accumulationType, int64_t lhsComponentCount,
    int64_t rhsComponentCount, int64_t numPrimitiveOperations,
    bool allowImpreciseAccumulation);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsADotAlgorithm(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirType
stablehloDotAlgorithmGetLhsPrecisionType(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirType
stablehloDotAlgorithmGetRhsPrecisionType(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirType
stablehloDotAlgorithmGetAccumulationType(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t
stablehloDotAlgorithmGetLhsComponentCount(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t
stablehloDotAlgorithmGetRhsComponentCount(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t
stablehloDotAlgorithmGetNumPrimitiveOperations(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool stablehloDotAlgorithmGetAllowImpreciseAccumulation(
    MlirAttribute attr);

//===----------------------------------------------------------------------===//
// DotDimensionNumbers
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloDotDimensionNumbersGet(
    MlirContext ctx,                                                        //
    intptr_t nLhsBatchingDimensions, const int64_t *lhsBatchingDimensions,  //
    intptr_t nRhsBatchingDimensions, const int64_t *rhsBatchingDimensions,  //
    intptr_t nLhsContractingDimensions,                                     //
    const int64_t *lhsContractingDimensions,                                //
    intptr_t nRhsContractingDimensions,                                     //
    const int64_t *rhsContractingDimensions);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsADotDimensionNumbers(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(MlirAttribute attr,
                                                         intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(MlirAttribute attr,
                                                         intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(MlirAttribute attr,
                                                            intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t
stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(MlirAttribute attr,
                                                            intptr_t pos);

//===----------------------------------------------------------------------===//
// ConvDimensionNumbers
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloConvDimensionNumbersGet(
    MlirContext ctx, int64_t inputBatchDimension, int64_t inputFeatureDimension,
    intptr_t nInputSpatialDimensions, const int64_t *inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    intptr_t nKernelSpatialDimensions, const int64_t *kernelSpatialDimensions,
    int64_t outputBatchDimension, int64_t outputFeatureDimension,
    intptr_t nOutputSpatialDimensions, const int64_t *outputSpatialDimensions);

// Returns true of the given attribute is a ConvDimensionNumbers attribute.
MLIR_CAPI_EXPORTED bool stablehloAttributeIsAConvDimensionNumbers(
    MlirAttribute attr);

// Returns the properties of ConvDimensionNumbers attributes.
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetInputBatchDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetInputFeatureDimension(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(MlirAttribute attr,
                                                           intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetKernelInputFeatureDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(
    MlirAttribute attr);
MLIR_CAPI_EXPORTED intptr_t
stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(MlirAttribute attr,
                                                            intptr_t pos);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetOutputBatchDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetOutputFeatureDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED intptr_t
stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(MlirAttribute attr,
                                                            intptr_t pos);

//===----------------------------------------------------------------------===//
// OutputOperandAlias
//===----------------------------------------------------------------------===//

// Creates a new OutputOperandAlias attribute with the given parameters. The
// pairs of consecutive intptr_t / int64_t* arguments are interpreted as sized
// arrays.
MLIR_CAPI_EXPORTED MlirAttribute stablehloOutputOperandAliasGet(
    MlirContext ctx, intptr_t nOutputTupleIndices,
    const int64_t *outputTupleIndices, int64_t operandIndex,
    intptr_t nOperandTupleIndices, const int64_t *operandTupleIndices);

// Returns true of the given attribute is a OutputOperandAlias attribute.
MLIR_CAPI_EXPORTED bool stablehloAttributeIsAOutputOperandAlias(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloOutputOperandAliasGetOutputTupleIndicesSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t stablehloOutputOperandAliasGetOutputTupleIndicesElem(
    MlirAttribute attr, intptr_t pos);

MLIR_CAPI_EXPORTED int64_t
stablehloOutputOperandAliasGetOperandIndex(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloOutputOperandAliasGetOperandTupleIndicesSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloOutputOperandAliasGetOperandTupleIndicesElem(MlirAttribute attr,
                                                      intptr_t pos);

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
stablehloComparisonDirectionAttrGet(MlirContext ctx, MlirStringRef value);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAComparisonDirectionAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloComparisonDirectionAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
stablehloComparisonTypeAttrGet(MlirContext ctx, MlirStringRef value);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAComparisonTypeAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloComparisonTypeAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// PrecisionAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloPrecisionAttrGet(MlirContext ctx,
                                                           MlirStringRef value);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAPrecisionAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloPrecisionAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// FftTypeAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloFftTypeAttrGet(MlirContext ctx,
                                                         MlirStringRef value);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAFftTypeAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloFftTypeAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// TransposeAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloTransposeAttrGet(MlirContext ctx,
                                                           MlirStringRef value);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsATransposeAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloTransposeAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// RngDistributionAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
stablehloRngDistributionAttrGet(MlirContext ctx, MlirStringRef value);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsARngDistributionAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloRngDistributionAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// RngAlgorithmAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
stablehloRngAlgorithmAttrGet(MlirContext ctx, MlirStringRef value);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsARngAlgorithmAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloRngAlgorithmAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// ChannelHandle
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloChannelHandleGet(MlirContext ctx,
                                                           int64_t handle,
                                                           int64_t type);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsChannelHandle(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t stablehloChannelHandleGetHandle(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t stablehloChannelHandleGetType(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// TypeExtensions
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloTypeExtensionsGet(
    MlirContext ctx, intptr_t nBounds, const int64_t *bounds);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsTypeExtensions(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloTypeExtensionsGetBoundsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloTypeExtensionsGetBoundsElem(MlirAttribute attr, intptr_t pos);

// ===---------------------------------------------------------------------===//
// ResultAccuracyModeAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
stablehloResultAccuracyModeAttrGet(MlirContext ctx, MlirStringRef value);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAResultAccuracyModeAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloResultAccuracyModeAttrGetValue(MlirAttribute attr);

// ===---------------------------------------------------------------------===//
// ResultAccuracyAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
stablehloResultAccuracyAttrGet(MlirContext ctx, double atol, double rtol,
                               int64_t ulps, MlirStringRef value);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsAResultAccuracyAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED double stablehloResultAccuracyAttrGetAtol(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED double stablehloResultAccuracyAttrGetRtol(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t
stablehloResultAccuracyAttrGetUlps(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
stablehloResultAccuracyAttrGetMode(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // STABLEHLO_INTEGRATIONS_C_STABLEHLO_ATTRIBUTES_H
