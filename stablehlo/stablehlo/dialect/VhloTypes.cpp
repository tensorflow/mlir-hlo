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

#include "stablehlo/dialect/VhloTypes.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/VhloTypes.h"

namespace mlir {
namespace vhlo {
namespace {
Type convertBuiltinIntegerType(IntegerType type) {
  if (!type.isSignless() && !type.isUnsigned()) return {};

  if (type.getWidth() == 1 && type.isSignless()) {  // Predicate
    return BooleanV1Type::get(type.getContext());
  }

  // Has valid signedness, check for valid widths
  // NOTE: Signless builtin types correspond to signed VHLO types.
  bool isSignless = type.isSignless();
  auto ctx = type.getContext();
  switch (type.getWidth()) {
    case 2:
      return isSignless ? cast<Type>(IntegerSI2V1Type::get(ctx))
                        : cast<Type>(IntegerUI2V1Type::get(ctx));
    case 4:
      return isSignless ? cast<Type>(IntegerSI4V1Type::get(ctx))
                        : cast<Type>(IntegerUI4V1Type::get(ctx));
    case 8:
      return isSignless ? cast<Type>(IntegerSI8V1Type::get(ctx))
                        : cast<Type>(IntegerUI8V1Type::get(ctx));
    case 16:
      return isSignless ? cast<Type>(IntegerSI16V1Type::get(ctx))
                        : cast<Type>(IntegerUI16V1Type::get(ctx));
    case 32:
      return isSignless ? cast<Type>(IntegerSI32V1Type::get(ctx))
                        : cast<Type>(IntegerUI32V1Type::get(ctx));
    case 64:
      return isSignless ? cast<Type>(IntegerSI64V1Type::get(ctx))
                        : cast<Type>(IntegerUI64V1Type::get(ctx));
  }
  return {};
}
}  // namespace

void VhloTypeConverter::addBuiltinToVhloConversions() {
  addConversion([&](BFloat16Type type) {
    return FloatBF16V1Type::get(type.getContext());
  });
  addConversion([&](ComplexType type) {
    return ComplexV1Type::get(type.getContext(),
                              convertType(type.getElementType()));
  });
  addConversion(
      [&](Float16Type type) { return FloatF16V1Type::get(type.getContext()); });
  addConversion(
      [&](Float32Type type) { return FloatF32V1Type::get(type.getContext()); });
  addConversion(
      [&](Float64Type type) { return FloatF64V1Type::get(type.getContext()); });
  addConversion([&](Float4E2M1FNType type) {
    return FloatF4E2M1FNV1Type::get(type.getContext());
  });
  addConversion([&](Float6E2M3FNType type) {
    return FloatF6E2M3FNV1Type::get(type.getContext());
  });
  addConversion([&](Float6E3M2FNType type) {
    return FloatF6E3M2FNV1Type::get(type.getContext());
  });
  addConversion([&](Float8E3M4Type type) {
    return FloatF8E3M4V1Type::get(type.getContext());
  });
  addConversion([&](Float8E4M3Type type) {
    return FloatF8E4M3V1Type::get(type.getContext());
  });
  addConversion([&](Float8E4M3FNType type) {
    return FloatF8E4M3FNV1Type::get(type.getContext());
  });
  addConversion([&](Float8E5M2Type type) {
    return FloatF8E5M2V1Type::get(type.getContext());
  });
  addConversion([&](Float8E4M3FNUZType type) {
    return FloatF8E4M3FNUZV1Type::get(type.getContext());
  });
  addConversion([&](Float8E4M3B11FNUZType type) {
    return FloatF8E4M3B11FNUZV1Type::get(type.getContext());
  });
  addConversion([&](Float8E5M2FNUZType type) {
    return FloatF8E5M2FNUZV1Type::get(type.getContext());
  });
  addConversion([&](Float8E8M0FNUType type) {
    return FloatF8E8M0FNUV1Type::get(type.getContext());
  });
  addConversion([&](FloatTF32Type type) {
    return FloatTF32V1Type::get(type.getContext());
  });
  addConversion([&](FunctionType type) -> Type {
    SmallVector<Type> convertedInputs;
    SmallVector<Type> convertedResults;
    if (failed(convertTypes(type.getInputs(), convertedInputs))) return {};
    if (failed(convertTypes(type.getResults(), convertedResults))) return {};
    return FunctionV1Type::get(type.getContext(), convertedInputs,
                               convertedResults);
  });
  addConversion(
      [&](IndexType type) { return IndexV1Type::get(type.getContext()); });
  addConversion(
      [&](IntegerType type) { return convertBuiltinIntegerType(type); });
  addConversion(
      [&](NoneType type) { return NoneV1Type::get(type.getContext()); });
  addConversion([&](RankedTensorType type) -> Type {
    auto encoding = type.getEncoding();
    auto convertedEncoding = encoding ? convertEncoding(encoding) : encoding;
    auto convertedElementType = convertType(type.getElementType());
    if ((encoding && !convertedEncoding) || !convertedElementType) return {};
    return RankedTensorV1Type::get(type.getContext(), type.getShape(),
                                   convertedElementType, convertedEncoding);
  });
  addConversion([&](TupleType type) -> Type {
    SmallVector<Type> convertedTypes;
    if (failed(convertTypes(type.getTypes(), convertedTypes))) return {};
    return vhlo::TupleV1Type::get(type.getContext(), convertedTypes);
  });
  addConversion([&](quant::UniformQuantizedType type) -> Type {
    Type convertedStorageType = convertType(type.getStorageType());
    Type convertedExpressedType = convertType(type.getExpressedType());
    if (!convertedStorageType || !convertedExpressedType) return {};
    return vhlo::UniformQuantizedV1Type::get(
        type.getContext(), type.getFlags(), convertedStorageType,
        convertedExpressedType, APFloat(type.getScale()), type.getZeroPoint(),
        type.getStorageTypeMin(), type.getStorageTypeMax());
  });
  addConversion([&](quant::UniformQuantizedPerAxisType type) -> Type {
    Type convertedStorageType = convertType(type.getStorageType());
    Type convertedExpressedType = convertType(type.getExpressedType());
    if (!convertedStorageType || !convertedExpressedType) return {};
    auto scales = llvm::map_to_vector(
        type.getScales(), [](double scale) { return APFloat(scale); });
    return vhlo::UniformQuantizedPerAxisV1Type::get(
        type.getContext(), type.getFlags(), convertedStorageType,
        convertedExpressedType, type.getQuantizedDimension(), scales,
        type.getZeroPoints(), type.getStorageTypeMin(),
        type.getStorageTypeMax());
  });
  addConversion([&](UnrankedTensorType type) -> Type {
    auto convertedElementType = convertType(type.getElementType());
    if (!convertedElementType) return {};
    return UnrankedTensorV1Type::get(type.getContext(), convertedElementType);
  });
  addConversion([&](shape::WitnessType type) -> Type {
    return vhlo::WitnessV1Type::get(type.getContext());
  });
}

void VhloTypeConverter::addVhloToBuiltinConversions() {
  addConversion([&](BooleanV1Type type) {
    return IntegerType::get(type.getContext(), 1);
  });
  addConversion([&](ComplexV1Type type) {
    return ComplexType::get(convertType(type.getElementType()));
  });
  addConversion([&](FloatBF16V1Type type) {
    return BFloat16Type::get(type.getContext());
  });
  addConversion(
      [&](FloatF16V1Type type) { return Float16Type::get(type.getContext()); });
  addConversion(
      [&](FloatF32V1Type type) { return Float32Type::get(type.getContext()); });
  addConversion(
      [&](FloatF64V1Type type) { return Float64Type::get(type.getContext()); });
  addConversion([&](FloatF4E2M1FNV1Type type) {
    return Float4E2M1FNType::get(type.getContext());
  });
  addConversion([&](FloatF6E2M3FNV1Type type) {
    return Float6E2M3FNType::get(type.getContext());
  });
  addConversion([&](FloatF6E3M2FNV1Type type) {
    return Float6E3M2FNType::get(type.getContext());
  });
  addConversion([&](FloatF8E3M4V1Type type) {
    return Float8E3M4Type::get(type.getContext());
  });
  addConversion([&](FloatF8E4M3V1Type type) {
    return Float8E4M3Type::get(type.getContext());
  });
  addConversion([&](FloatF8E4M3FNV1Type type) {
    return Float8E4M3FNType::get(type.getContext());
  });
  addConversion([&](FloatF8E5M2V1Type type) {
    return Float8E5M2Type::get(type.getContext());
  });
  addConversion([&](FloatF8E4M3FNUZV1Type type) {
    return Float8E4M3FNUZType::get(type.getContext());
  });
  addConversion([&](FloatF8E4M3B11FNUZV1Type type) {
    return Float8E4M3B11FNUZType::get(type.getContext());
  });
  addConversion([&](FloatF8E5M2FNUZV1Type type) {
    return Float8E5M2FNUZType::get(type.getContext());
  });
  addConversion([&](FloatF8E8M0FNUV1Type type) {
    return Float8E8M0FNUType::get(type.getContext());
  });
  addConversion([&](FloatTF32V1Type type) {
    return FloatTF32Type::get(type.getContext());
  });
  addConversion([&](FunctionV1Type type) -> Type {
    SmallVector<Type> convertedInputs;
    SmallVector<Type> convertedOutputs;
    if (failed(convertTypes(type.getInputs(), convertedInputs))) return {};
    if (failed(convertTypes(type.getOutputs(), convertedOutputs))) return {};
    return FunctionType::get(type.getContext(), convertedInputs,
                             convertedOutputs);
  });
  addConversion(
      [&](IndexV1Type type) { return IndexType::get(type.getContext()); });
  addConversion([&](IntegerSI2V1Type type) {
    return IntegerType::get(type.getContext(), 2);
  });
  addConversion([&](IntegerSI4V1Type type) {
    return IntegerType::get(type.getContext(), 4);
  });
  addConversion([&](IntegerSI8V1Type type) {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](IntegerSI16V1Type type) {
    return IntegerType::get(type.getContext(), 16);
  });
  addConversion([&](IntegerSI32V1Type type) {
    return IntegerType::get(type.getContext(), 32);
  });
  addConversion([&](IntegerSI64V1Type type) {
    return IntegerType::get(type.getContext(), 64);
  });
  addConversion([&](IntegerUI2V1Type type) {
    return IntegerType::get(type.getContext(), 2, IntegerType::Unsigned);
  });
  addConversion([&](IntegerUI4V1Type type) {
    return IntegerType::get(type.getContext(), 4, IntegerType::Unsigned);
  });
  addConversion([&](IntegerUI8V1Type type) {
    return IntegerType::get(type.getContext(), 8, IntegerType::Unsigned);
  });
  addConversion([&](IntegerUI16V1Type type) {
    return IntegerType::get(type.getContext(), 16, IntegerType::Unsigned);
  });
  addConversion([&](IntegerUI32V1Type type) {
    return IntegerType::get(type.getContext(), 32, IntegerType::Unsigned);
  });
  addConversion([&](IntegerUI64V1Type type) {
    return IntegerType::get(type.getContext(), 64, IntegerType::Unsigned);
  });
  addConversion(
      [&](NoneV1Type type) { return NoneType::get(type.getContext()); });
  addConversion([&](RankedTensorV1Type type) -> Type {
    auto encoding = type.getEncoding();
    auto convertedEncoding = encoding ? convertEncoding(encoding) : encoding;
    auto convertedElementType = convertType(type.getElementType());
    if ((encoding && !convertedEncoding) || !convertedElementType) return {};
    return RankedTensorType::get(type.getShape(), convertedElementType,
                                 convertedEncoding);
  });
  addConversion([&](TupleV1Type type) -> Type {
    SmallVector<Type> convertedTypes;
    if (failed(convertTypes(type.getTypes(), convertedTypes))) return {};
    return TupleType::get(type.getContext(), convertedTypes);
  });
  addConversion([&](UniformQuantizedV1Type type) -> Type {
    Type convertedStorageType = convertType(type.getStorageType());
    Type convertedExpressedType = convertType(type.getExpressedType());
    if (!convertedStorageType || !convertedExpressedType) return {};
    return quant::UniformQuantizedType::get(
        type.getFlags(), convertedStorageType, convertedExpressedType,
        type.getScale().convertToDouble(), type.getZeroPoint(),
        type.getStorageTypeMin(), type.getStorageTypeMax());
  });
  addConversion([&](UniformQuantizedPerAxisV1Type type) -> Type {
    Type convertedStorageType = convertType(type.getStorageType());
    Type convertedExpressedType = convertType(type.getExpressedType());
    if (!convertedStorageType || !convertedExpressedType) return {};
    auto scales = llvm::map_to_vector(
        type.getScales(),
        [](const APFloat& scale) { return scale.convertToDouble(); });
    return quant::UniformQuantizedPerAxisType::get(
        type.getFlags(), convertedStorageType, convertedExpressedType, scales,
        type.getZeroPoints(), type.getQuantizedDimension(),
        type.getStorageTypeMin(), type.getStorageTypeMax());
  });
  addConversion([&](UnrankedTensorV1Type type) -> Type {
    auto convertedElementType = convertType(type.getElementType());
    if (!convertedElementType) return {};
    return UnrankedTensorType::get(convertedElementType);
  });
  addConversion([&](WitnessV1Type type) -> Type {
    return shape::WitnessType::get(type.getContext());
  });
}

namespace {
Value materializeIllegalCast(OpBuilder& builder, Type type, ValueRange inputs,
                             Location loc) {
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
      ->getResult(0);
}
}  // namespace

void VhloTypeConverter::addUnrealizedMaterializations() {
  addTargetMaterialization(materializeIllegalCast);
  addSourceMaterialization(materializeIllegalCast);
}

namespace {
// Helper functions for VHLO verifiers
template <typename TypeOrAttr>
bool isFromVhlo(TypeOrAttr t) {
  // Requires string comparison to avoid cyclic dependency with VhloOps.h.
  return t.getDialect().getNamespace() == "vhlo";
}

template <typename TypeOrAttr>
bool allFromVhlo(ArrayRef<TypeOrAttr> range) {
  return llvm::all_of(range, isFromVhlo<TypeOrAttr>);
}
}  // namespace

// Helper functions for VHLO type printers and parsers.
void printEncoding(AsmPrinter& os, Attribute encoding) {
  if (!encoding) return;
  os << ", " << encoding;
}

ParseResult parseEncoding(AsmParser& parser, Attribute& encoding) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }
  if (failed(parser.parseAttribute(encoding))) return failure();
  return success();
}

void printShape(AsmPrinter& os, ArrayRef<int64_t> dimSizes) {
  if (dimSizes.empty()) return;
  for (int64_t dimSize : dimSizes) {
    os << hlo::dimSizeToString(dimSize) << 'x';
  }
}

ParseResult parseShape(AsmParser& parser, SmallVector<int64_t>& dimSizes) {
  if (failed(parser.parseDimensionList(dimSizes))) {
    return failure();
  }
  return success();
}

// Print types in parentheses: (!vhlo.type, !vhlo.type)
static void printTypeArray(AsmPrinter& os, ArrayRef<Type> typeArray) {
  if (typeArray.empty()) os << "()";
  os << typeArray;
}

// Parse types in parentheses: (!vhlo.type, !vhlo.type)
ParseResult parseTypeArray(AsmParser& parser, SmallVector<Type>& typeArray) {
  if (succeeded(parser.parseOptionalLParen()) &&
      succeeded(parser.parseOptionalRParen())) {
    return success();
  }

  auto parseEle = [&]() { return parser.parseType(typeArray.emplace_back()); };
  if (failed(parser.parseCommaSeparatedList(parseEle))) {
    return failure();
  }
  return success();
}

// RankedTensorV1Type implement ShapedTypeInterface
mlir::ShapedType RankedTensorV1Type::cloneWith(
  std::optional<llvm::ArrayRef<int64_t>> values, Type elementType) const {
ArrayRef<int64_t> shape = values.value_or(getShape());
return RankedTensorV1Type::get(getContext(), shape, elementType,
                               getEncoding());
}

bool RankedTensorV1Type::hasRank() const { return true; }

}  // namespace vhlo
}  // namespace mlir

// Include order matters
#include "stablehlo/dialect/VhloTypeInterfaces.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "stablehlo/dialect/VhloTypeDefs.cpp.inc"

namespace mlir {
namespace vhlo {

LogicalResult printVhloType(Type type, AsmPrinter& printer) {
  return generatedTypePrinter(type, printer);
}

OptionalParseResult parseVhloType(mlir::AsmParser& parser,
                                  llvm::StringRef* mnemonic, mlir::Type& type) {
  return generatedTypeParser(parser, mnemonic, type);
}

namespace {
template <typename... Types>
void registerVhloTypes(MLIRContext* context) {
  (mlir::detail::TypeUniquer::registerType<Types>(context), ...);
}
}  // namespace

void registerVhloTypes(MLIRContext* context) {
  registerVhloTypes<
#define GET_TYPEDEF_LIST
#include "stablehlo/dialect/VhloTypeDefs.cpp.inc"
      >(context);
}

}  // namespace vhlo
}  // namespace mlir
