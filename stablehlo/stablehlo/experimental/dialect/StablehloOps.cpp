/* Copyright 2023 The StableHLO Authors.

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

#include "stablehlo/experimental/dialect/StablehloOps.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace stablehlo {
namespace experimental {

LogicalResult DynamicReduceWindowOpAdaptor::verify() {
  // Before checking the constraints inherited from ReduceWindowOp,
  // make sure that the operands and the attributes of the underlying custom
  // call make sense.
  if (op_->getNumOperands() != 2 * op_->getNumResults() + 5)
    return op_.emitError("expects size(operands) = 2 * size(results) + 5");
  if (op_->getNumResults() == 0)
    return op_.emitError("expects size(results) > 0");
  for (const auto& attr : op_->getAttrs()) {
    // api_version and backend_config have default values.
    // call_target_name should be "stablehlo.dynamic_reduce_window".
    // called_computations carries the body.
    if (attr.getName() != "api_version" && attr.getName() != "backend_config" &&
        attr.getName() != "call_target_name" &&
        attr.getName() != "called_computations")
      return op_.emitError()
             << attr.getName() << " is not a supported attribute";
  }
  if (!op_.getBackendConfig().empty())
    return op_.emitError() << "expects an empty backend_config";
  if (op_.getCallTargetName() != "stablehlo.dynamic_reduce_window")
    return op_.emitError() << "expects @stablehlo.dynamic_reduce_window";

  // Unpack operands and attributes of the underlying custom call into
  // operation-specific inputs.
  auto numInputs = getInputs().size();
  auto inputs = op_.getInputs().slice(0, numInputs);
  auto initValues = op_.getInputs().slice(numInputs, numInputs);
  auto windowDimensions = op_.getInputs()[op_.getInputs().size() - 5];
  auto windowStrides = op_.getInputs()[op_.getInputs().size() - 4];
  auto baseDilations = op_.getInputs()[op_.getInputs().size() - 3];
  auto windowDilations = op_.getInputs()[op_.getInputs().size() - 2];
  auto padding = op_.getInputs()[op_.getInputs().size() - 1];
  auto results = op_.getResults();

  // reduce_window_c1
  // This constraint hold automatically thanks to the checks that we have
  // performed above.

  // reduce_window_i1
  SmallVector<ShapedType> inputTypes;
  for (auto [index, input] : llvm::enumerate(inputs)) {
    auto inputType = input.getType().dyn_cast<ShapedType>();
    inputTypes.push_back(inputType);
    if (!inputType)
      return op_.emitError()
             << "expects inputs (e.g. operand #" << index << ") to be tensors";
  }

  // reduce_window_i2
  SmallVector<ShapedType> initValueTypes;
  for (auto [index, initValue] : llvm::enumerate(initValues)) {
    auto initValueType = initValue.getType().dyn_cast<ShapedType>();
    initValueTypes.push_back(initValueType);
    if (!initValueType || !initValueType.hasRank() ||
        initValueType.getRank() != 0)
      return op_.emitError() << "expects init_values (e.g. operand #"
                             << numInputs + index << ") "
                             << "to be 0-dimensional tensors";
  }

  // reduce_window_i3...reduce_window_i7
  auto checkRank = [&](StringRef name, int64_t index, Value dynamicAttr,
                       int64_t expectedRank) -> LogicalResult {
    auto type = dynamicAttr.getType().dyn_cast<ShapedType>();
    if (!type || !type.hasRank() || type.getRank() != expectedRank ||
        !type.getElementType().isIntOrIndex()) {
      if (index < 0) index += op_->getNumOperands();
      return op_.emitError()
             << "expects " << name << " (operand #" << index << ") "
             << "to be a " << expectedRank << "-dimensional tensor "
             << "of integer or index type";
    }
    return success();
  };
  if (failed(checkRank("window_dimensions", -5, windowDimensions, 1)) ||
      failed(checkRank("window_strides", -4, windowStrides, 1)) ||
      failed(checkRank("base_dilations", -3, baseDilations, 1)) ||
      failed(checkRank("window_dilations", -2, windowDilations, 1)) ||
      failed(checkRank("padding", -1, padding, 2)))
    return failure();

  // reduce_window_i7
  auto paddingType = getPadding().getType().dyn_cast<ShapedType>();
  if (!paddingType || !paddingType.hasRank() || paddingType.getRank() != 2 ||
      paddingType.getDimSize(1) != 2 ||
      !paddingType.getElementType().isIntOrIndex())
    return op_.emitError()
           << "expects padding_type (operand #" << op_.getNumOperands() - 1
           << ") to be a 2-dimensional tensor of integer or index type";

  // reduce_window_c2
  std::optional<ArrayRef<int64_t>> inputShape;
  for (auto inputType : inputTypes) {
    if (!inputType.hasRank()) continue;
    if (!inputShape) inputShape = inputType.getShape();
    if (failed(verifyCompatibleShape(inputType.getShape(), *inputShape)))
      return op_.emitError() << "expects all inputs (operands 0.." << numInputs
                             << ") to have compatible shapes";
  }

  // reduce_window_c3
  for (auto [inputType, initValueType] :
       llvm::zip(inputTypes, initValueTypes)) {
    if (inputType.getElementType() != initValueType.getElementType())
      return op_.emitError() << "expects inputs (operands 0.." << numInputs
                             << ") and init_values (operands " << numInputs
                             << ".." << numInputs * 2 << ") to have pairwise "
                             << "the same element types";
  }

  // reduce_window_c4...reduce_window_c12
  // In this range, we only verify the constraints with even numbers.
  // Verifying the constraints with odd numbers would require knowing the
  // actual values of window_dimensions, window_strides, etc.
  // While we certainly can try to check whether they are constants and
  // verify them in that case, that seems like too much at this point.
  auto checkShape = [&](StringRef name, int64_t index, Value dynamicAttr,
                        ArrayRef<int64_t> expectedShape) -> LogicalResult {
    auto type = dynamicAttr.getType().cast<ShapedType>();
    if (type.getShape() != expectedShape) {
      if (index < 0) index += op_->getNumOperands();
      return op_.emitError()
             << "expects " << name << " (operand #" << index << ") "
             << "to have shape [" << expectedShape << "]";
    }
    return success();
  };
  if (inputShape) {
    auto inputRank = static_cast<int64_t>(inputShape->size());
    if (failed(checkShape("window_dimensions", -5, windowDimensions,
                          {inputRank})) ||
        failed(checkShape("window_strides", -4, windowStrides, {inputRank})) ||
        failed(checkShape("base_dilations", -3, baseDilations, {inputRank})) ||
        failed(
            checkShape("window_dilations", -2, windowDilations, {inputRank})) ||
        failed(checkShape("padding", -1, padding, {inputRank, 2})))
      return failure();
  }

  // reduce_window_c13
  if (op_.getCalledComputations().size() != 1)
    return op_.emitError() << "expects called_computations to have 1 element";
  auto bodyAttr = op_.getCalledComputations()[0].cast<FlatSymbolRefAttr>();
  auto bodyFunc =
      op_->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(bodyAttr);
  if (!bodyFunc)
    return op_.emitError() << "expects called_computations to refer to "
                           << "a function that exists within a parent module";

  // reduce_window_c13
  SmallVector<Type> expectedBodyInputs;
  llvm::append_range(expectedBodyInputs, initValueTypes);
  llvm::append_range(expectedBodyInputs, initValueTypes);
  SmallVector<Type> expectedBodyOutputs;
  llvm::append_range(expectedBodyOutputs, initValueTypes);
  auto expectedBodyType = FunctionType::get(
      op_.getContext(), expectedBodyInputs, expectedBodyOutputs);
  if (bodyFunc.getFunctionType() != expectedBodyType)
    return op_.emitError() << "expects body to have type " << expectedBodyType;

  // reduce_window_c14
  SmallVector<ShapedType> resultTypes;
  std::optional<ArrayRef<int64_t>> resultShape;
  for (auto result : results) {
    auto resultType = result.getType().dyn_cast<ShapedType>();
    resultTypes.push_back(resultType);
    if (!resultType) return op_.emitError() << "expects results to be tensors";

    if (!resultType.hasRank()) continue;
    if (!resultShape) resultShape = resultType.getShape();
    if (failed(verifyCompatibleShape(resultType.getShape(), *resultShape)))
      return op_.emitError() << "expects all results to have compatible shapes";
  }

  // reduce_window_c15
  // Verifying this constraint would require knowing the actual values of
  // window_dimensions, window_strides, etc.
  // While we certainly can try to check whether they are constants and
  // verify them in that case, that seems like too much at this point.

  // reduce_window_c16
  for (auto [resultType, initValueType] :
       llvm::zip(resultTypes, initValueTypes)) {
    if (resultType.getElementType() != initValueType.getElementType())
      return op_.emitError() << "expects results and init_values (operands "
                             << numInputs << ".." << numInputs * 2 << ") "
                             << "to have pairwise the same element types";
  }

  return success();
}

ValueRange DynamicReduceWindowOpAdaptor::getInputs() {
  auto numInputs = (op_.getInputs().size() - 5) / 2;
  return op_.getInputs().slice(0, numInputs);
}

ValueRange DynamicReduceWindowOpAdaptor::getInitValues() {
  auto numInputs = (op_.getInputs().size() - 5) / 2;
  return op_.getInputs().slice(numInputs, numInputs);
}

TypedValue<ShapedType> DynamicReduceWindowOpAdaptor::getWindowDimensions() {
  return op_.getInputs()[op_.getInputs().size() - 5]
      .cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> DynamicReduceWindowOpAdaptor::getWindowStrides() {
  return op_.getInputs()[op_.getInputs().size() - 4]
      .cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> DynamicReduceWindowOpAdaptor::getBaseDilations() {
  return op_.getInputs()[op_.getInputs().size() - 3]
      .cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> DynamicReduceWindowOpAdaptor::getWindowDilations() {
  return op_.getInputs()[op_.getInputs().size() - 2]
      .cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> DynamicReduceWindowOpAdaptor::getPadding() {
  return op_.getInputs()[op_.getInputs().size() - 1]
      .cast<TypedValue<ShapedType>>();
}

Region& DynamicReduceWindowOpAdaptor::getBody() {
  auto bodyAttr = op_.getCalledComputations()[0].cast<FlatSymbolRefAttr>();
  auto bodyFunc =
      op_->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(bodyAttr);
  return bodyFunc.getBody();
}

ValueRange DynamicReduceWindowOpAdaptor::getResults() {
  return op_.getResults();
}

std::optional<DynamicReduceWindowOpAdaptor> getDynamicReduceWindowOp(
    CustomCallOp op) {
  if (op.getCallTargetName() != "stablehlo.dynamic_reduce_window") return {};
  return DynamicReduceWindowOpAdaptor(op);
}

LogicalResult DynamicRngBitGeneratorOpAdaptor::verify() {
  // Before checking the constraints inherited from RngBitGeneratorOp,
  // make sure that the operands and the attributes of the underlying custom
  // call make sense.
  if (op_->getNumOperands() != 2)
    return op_.emitError("expects size(operands) = 2");
  if (op_->getNumResults() != 2)
    return op_.emitError("expects size(results) = 2");
  for (const auto& attr : op_->getAttrs()) {
    // api_version and backend_config have default values.
    // call_target_name should be "stablehlo.dynamic_rng_bit_generator".
    // rng_algorithm comes from the operation.
    if (attr.getName() != "api_version" && attr.getName() != "backend_config" &&
        attr.getName() != "call_target_name" &&
        attr.getName() != "rng_algorithm")
      return op_.emitError()
             << attr.getName() << " is not a supported attribute";
  }
  if (!op_.getBackendConfig().empty())
    return op_.emitError() << "expects an empty backend_config";
  if (op_.getCallTargetName() != "stablehlo.dynamic_rng_bit_generator")
    return op_.emitError() << "expects @stablehlo.dynamic_rng_bit_generator";
  if (!op_->hasAttr("rng_algorithm"))
    return op_.emitError() << "expects an rng_algorithm";

  // Unpack operands and attributes of the underlying custom call into
  // operation-specific inputs.
  auto rngAlgorithmAttr = op_->getAttr("rng_algorithm");
  auto initialState = op_.getInputs()[0];
  auto outputShape = op_.getInputs()[1];
  auto outputState = op_.getResults()[0];
  auto output = op_.getResults()[1];

  // dynamic_rng_bit_generator_i1
  if (!rngAlgorithmAttr.isa<RngAlgorithmAttr>())
    return op_.emitError()
           << "expects a #stablehlo<rng_algorithm ...> rng_algorithm";

  // dynamic_rng_bit_generator_i2
  // TODO(#643): Clarify supported types for RngBitGeneratorOp.
  auto initialStateType = initialState.getType().dyn_cast<ShapedType>();
  if (!initialStateType || !initialStateType.getElementType().isIntOrFloat())
    return op_.emitError()
           << "expects initial_state (operand #0) "
           << "to be a tensor of integer or floating-point type";

  // dynamic_rng_bit_generator_i3
  auto outputShapeType = outputShape.getType().dyn_cast<ShapedType>();
  if (!outputShapeType || !outputShapeType.hasRank() ||
      outputShapeType.getRank() != 1 ||
      !outputShapeType.getElementType().isIntOrIndex())
    return op_.emitError()
           << "expects output_shape (operand #1) "
           << "to be a 1-dimensional tensor of integer or index type";

  // dynamic_rng_bit_generator_o1
  // TODO(#643): Clarify supported types for RngBitGeneratorOp.
  auto outputStateType = outputState.getType().dyn_cast<ShapedType>();
  if (!outputStateType || !outputStateType.getElementType().isIntOrFloat())
    return op_.emitError()
           << "expects output_state (result #0) "
           << "to be a tensor of integer or floating-point type";

  // dynamic_rng_bit_generator_o2
  auto outputType = output.getType().dyn_cast<ShapedType>();
  if (!outputType || !outputType.getElementType().isIntOrFloat())
    return op_.emitError()
           << "expects output (result #1) "
           << "to be a tensor of integer or floating-point type";

  // dynamic_rng_bit_generator_c1
  if (!hlo::isCompatibleForHloTypeInference(initialStateType, outputStateType))
    return op_.emitError()
           << "expects initial_state (operand #0) and output_state (result #0) "
           << "to have compatible shapes";

  // dynamic_rng_bit_generator_c2
  // TODO(#486): Verify rng_algorithm in RngBitGeneratorOp.

  // dynamic_rng_bit_generator_c3
  if (!hlo::isCompatibleForHloTypeInference(outputShape, outputType))
    return op_.emitError() << "expects output (result #1) to have shape  "
                           << "compatible with output_shape (operand #2)";

  return success();
}

RngAlgorithm DynamicRngBitGeneratorOpAdaptor::getRngAlgorithm() {
  return op_->getAttr("rng_algorithm").cast<RngAlgorithmAttr>().getValue();
}

TypedValue<ShapedType> DynamicRngBitGeneratorOpAdaptor::getInitialState() {
  return op_.getInputs()[0].cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> DynamicRngBitGeneratorOpAdaptor::getOutputShape() {
  return op_.getInputs()[1].cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> DynamicRngBitGeneratorOpAdaptor::getOutputState() {
  return op_.getResults()[0].cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> DynamicRngBitGeneratorOpAdaptor::getOutput() {
  return op_.getResults()[1].cast<TypedValue<ShapedType>>();
}

std::optional<DynamicRngBitGeneratorOpAdaptor> getDynamicRngBitGeneratorOp(
    CustomCallOp op) {
  if (op.getCallTargetName() != "stablehlo.dynamic_rng_bit_generator")
    return {};
  return DynamicRngBitGeneratorOpAdaptor(op);
}

LogicalResult DynamicTopKOpAdaptor::verify() {
  if (op_->getNumOperands() != 2)
    return op_.emitError("expects size(operands) = 2");
  if (op_->getNumResults() != 2)
    return op_.emitError("expects size(results) = 2");
  for (const auto& attr : op_->getAttrs()) {
    // api_version and backend_config have default values.
    // call_target_name should be "stablehlo.dynamic_top_k".
    if (attr.getName() != "api_version" && attr.getName() != "backend_config" &&
        attr.getName() != "call_target_name")
      return op_.emitError()
             << attr.getName() << " is not a supported attribute";
  }
  if (!op_.getBackendConfig().empty())
    return op_.emitError() << "expects an empty backend_config";
  if (op_.getCallTargetName() != "stablehlo.dynamic_top_k")
    return op_.emitError() << "expects @stablehlo.dynamic_top_k";

  auto operand = op_.getInputs()[0];
  auto k = op_.getInputs()[1];
  auto values = op_.getResults()[0];
  auto indices = op_.getResults()[1];

  // dynamic_top_k_i1
  auto operandType = operand.getType().dyn_cast<ShapedType>();
  if (!operandType || !operandType.hasRank() || operandType.getRank() < 1 ||
      !operandType.getElementType().isIntOrFloat())
    return op_.emitError()
           << "expects operand #0 "
           << "to be a tensor of integer or floating-point type "
           << "of rank at least 1";

  // dynamic_top_k_i2
  auto kType = k.getType().dyn_cast<ShapedType>();
  if (!kType || !kType.hasRank() || kType.getRank() != 0 ||
      !kType.getElementType().isIntOrIndex())
    return op_.emitError()
           << "expects k (operand #1) "
           << "to be a 0-dimensional tensor of integer or index type";

  // dynamic_top_k_o1
  auto valuesType = values.getType().dyn_cast<ShapedType>();
  if (!valuesType || !valuesType.hasRank() || valuesType.getRank() < 1 ||
      !valuesType.getElementType().isIntOrFloat())
    return op_.emitError()
           << "expects values (result #0) "
           << "to be a tensor of integer or floating-point type "
           << "of rank at least 1";

  // dynamic_top_k_o2
  auto indicesType = indices.getType().dyn_cast<ShapedType>();
  if (!indicesType || !indicesType.hasRank() || indicesType.getRank() < 1 ||
      !indicesType.getElementType().isSignlessInteger(32))
    return op_.emitError() << "expects indices (result #1) "
                           << "to be a tensor of si32 of rank at least 1";

  // dynamic_top_k_c1
  auto operandLastDim = operandType.getRank() - 1;
  SmallVector<int64_t> expectedValuesShape(operandType.getShape());
  expectedValuesShape[operandLastDim] =
      valuesType.getDimSize(valuesType.getRank() - 1);
  if (failed(verifyCompatibleShape(expectedValuesShape, valuesType.getShape())))
    return op_.emitError() << "expects the values shape to match the operand "
                              "shape in all but the last dimension";

  // dynamic_top_k_c2
  if (valuesType.getElementType() != operandType.getElementType())
    return op_.emitError()
           << "expects the values element type to be the same as the operand "
           << "element type";

  // dynamic_top_k_c3
  if (!operandType.isDynamicDim(operandLastDim) &&
      !valuesType.isDynamicDim(operandLastDim) &&
      operandType.getDimSize(operandLastDim) <
          valuesType.getDimSize(operandLastDim))
    return op_.emitError() << "expects the values last dimension to have size "
                              "at least as large "
                           << "as operand last dimension";

  // dynamic_top_k_c4
  if (failed(
          verifyCompatibleShape(indicesType.getShape(), valuesType.getShape())))
    return op_.emitError()
           << "expects the indices shape to match the values shape";

  return success();
}

TypedValue<ShapedType> DynamicTopKOpAdaptor::getOperand() {
  return op_.getInputs()[0].cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> DynamicTopKOpAdaptor::getK() {
  return op_.getInputs()[1].cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> DynamicTopKOpAdaptor::getValues() {
  return op_.getResults()[0].cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> DynamicTopKOpAdaptor::getIndices() {
  return op_.getResults()[1].cast<TypedValue<ShapedType>>();
}

std::optional<DynamicTopKOpAdaptor> getDynamicTopKOp(CustomCallOp op) {
  if (op.getCallTargetName() != "stablehlo.dynamic_top_k") return {};
  return DynamicTopKOpAdaptor(op);
}

LogicalResult TopKOpAdaptor::verify() {
  if (op_->getNumOperands() != 1)
    return op_.emitError("expects size(operands) = 1");
  if (op_->getNumResults() != 2)
    return op_.emitError("expects size(results) = 2");
  if (!op_.getBackendConfig().empty())
    return op_.emitError() << "expects an empty backend_config";
  if (op_.getCallTargetName() != "mhlo.topk")
    return op_.emitError() << "expects @mhlo.topk";

  auto operand = op_.getInputs()[0];
  auto values = op_.getResults()[0];
  auto indices = op_.getResults()[1];
  DictionaryAttr topkAttributes =
      op_->getAttrOfType<DictionaryAttr>("mhlo.attributes");
  if (!topkAttributes) {
    return op_.emitError()
           << "mhlo.attributes missing or not a dictionary attribute";
  }

  IntegerAttr k_attr = topkAttributes.get("k").dyn_cast_or_null<IntegerAttr>();
  if (!k_attr) {
    return op_.emitError() << "mhlo.attributes.k not present or not an integer";
  }
  int64_t k = k_attr.getInt();

  // mhlo.topk_c5
  if (k < 0) return op_.emitError() << "expects k >= 0";

  // mhlo.topk_i1
  auto operandType = operand.getType().dyn_cast<ShapedType>();
  if (!operandType || !operandType.hasRank() || operandType.getRank() < 1 ||
      !operandType.getElementType().isIntOrFloat())
    return op_.emitError()
           << "expects operand #0 "
           << "to be a tensor of integer or floating-point type "
           << "of rank at least 1";

  // mhlo.topk_o1
  auto valuesType = values.getType().dyn_cast<ShapedType>();
  if (!valuesType || !valuesType.hasRank() || valuesType.getRank() < 1 ||
      !valuesType.getElementType().isIntOrFloat())
    return op_.emitError()
           << "expects values (result #0) "
           << "to be a tensor of integer or floating-point type "
           << "of rank at least 1";

  // mhlo.topk_o2
  auto indicesType = indices.getType().dyn_cast<ShapedType>();
  if (!indicesType || !indicesType.hasRank() || indicesType.getRank() < 1 ||
      !indicesType.getElementType().isSignlessInteger(32))
    return op_.emitError() << "expects indices (result #1) "
                           << "to be a tensor of si32 of rank at least 1";

  // mhlo.topk_c1 && mhlo.topk_c2
  auto operandLastDim = operandType.getRank() - 1;
  SmallVector<int64_t> expectedValuesShape(operandType.getShape());
  expectedValuesShape[operandLastDim] = k;
  if (failed(verifyCompatibleShape(expectedValuesShape, valuesType.getShape())))
    return op_.emitError() << "expects the values shape to match the operand "
                              "shape in all but the last dimension, and "
                              "that the last dimension of the values shape "
                              "has a size k";

  // mhlo.topk_c3
  if (valuesType.getElementType() != operandType.getElementType())
    return op_.emitError()
           << "expects the values element type to be the same as the operand "
           << "element type";

  // mhlo.topk_c4
  if (failed(
          verifyCompatibleShape(indicesType.getShape(), valuesType.getShape())))
    return op_.emitError()
           << "expects the indices shape to match the values shape";

  return success();
}

TypedValue<ShapedType> TopKOpAdaptor::getOperand() {
  return op_.getInputs()[0].cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> TopKOpAdaptor::getValues() {
  return op_.getResults()[0].cast<TypedValue<ShapedType>>();
}

TypedValue<ShapedType> TopKOpAdaptor::getIndices() {
  return op_.getResults()[1].cast<TypedValue<ShapedType>>();
}

int64_t TopKOpAdaptor::getK() {
  DictionaryAttr topkAttributes =
      op_->getAttrOfType<DictionaryAttr>("mhlo.attributes");
  return topkAttributes.get("k").cast<mlir::IntegerAttr>().getInt();
}

bool TopKOpAdaptor::getLargest() {
  DictionaryAttr topkAttributes =
      op_->getAttrOfType<DictionaryAttr>("mhlo.attributes");
  IntegerAttr largest =
      topkAttributes.get("largest").dyn_cast_or_null<mlir::IntegerAttr>();

  return (!largest) ? true : largest.getInt();
}

std::optional<TopKOpAdaptor> getTopKOp(CustomCallOp op) {
  if (op.getCallTargetName() != "mhlo.topk") return {};
  return TopKOpAdaptor(op);
}

}  // namespace experimental
}  // namespace stablehlo
}  // namespace mlir
