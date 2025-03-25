/* Copyright 2019 The IREE Authors
   Copyright 2023 OpenXLA Authors. All Rights Reserved.

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

// Implements logic for lowering StableHLO pointwise ops to Linalg dialect.
// These patterns are separated out to their own file to save on the compilation
// times, given that we instantiate a large number of class templates here.

#include <algorithm>
#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/LegalizeToLinalgUtils.h"
#include "stablehlo/conversions/linalg/transforms/MapStablehloToScalarOp.h"
#include "stablehlo/conversions/linalg/transforms/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo {
namespace {
int64_t getRank(Value v) { return cast<ShapedType>(v.getType()).getRank(); }

int64_t getMaxRank(ValueRange operands) {
  int64_t maxRank = 0;
  for (Value operand : operands) {
    maxRank = std::max(maxRank, getRank(operand));
  }
  return maxRank;
}

bool isScalar(Value v) { return getRank(v) == 0; }

/// Inserts block arguments in places where scalar inputs have a nullptr.
SmallVector<Value> interleaveScalarAndBlockArgs(ValueRange scalarInputs,
                                                ValueRange blockArgs) {
  SmallVector<Value> result;
  auto argsIter = blockArgs.begin();
  for (Value scalarInput : scalarInputs) {
    if (scalarInput) {
      result.push_back(scalarInput);
    } else {
      result.push_back(*argsIter);
      ++argsIter;
    }
  }
  return result;
}

struct PointwiseConversionInfo {
  int64_t maxOperandRank = 0;
  ShapedType resultType;
};

/// Checks the preconditions for conversion of pointwise HLO ops to linalg.
/// Returns the max operand rank and the result type on success.
FailureOr<PointwiseConversionInfo> checkOperandsAndResults(
    Operation *op, ValueRange operands, const TypeConverter &typeConverter,
    ConversionPatternRewriter &rewriter) {
  int64_t maxRank = getMaxRank(operands);

  // Apply only if all operands are scalar or have the same rank. Some ops,
  // like `stablehlo.select`, support implicit broadcasting of scalars.
  if (!llvm::all_of(operands, [&](Value v) {
        int64_t r = getRank(v);
        return r == 0 || r == maxRank;
      })) {
    return rewriter.notifyMatchFailure(
        op, "Operands must be of same rank or scalar.");
  }

  // Find result type, if on tensors.
  auto resultTy = cast_or_null<RankedTensorType>(
      typeConverter.convertType(op->getResultTypes().front()));

  // Check result type compatibility.
  if (!resultTy || resultTy.getRank() != maxRank ||
      !(resultTy.getElementType().isSignlessIntOrFloat() ||
        isa<ComplexType>(resultTy.getElementType()))) {
    return rewriter.notifyMatchFailure(
        op, "mismatched operand/result types or iterator count");
  }

  // All-scalar pointwise ops inside of linalg ops are processes by
  // ScalarHloToArithmeticPattern.
  if (maxRank == 0 && isInBodyOfLinalgOps(op)) return failure();

  return PointwiseConversionInfo{maxRank, resultTy};
}

/// If `input` is a splat constant value, materialize the scalar splat
/// value. Otherwise, return nullopt.
std::optional<Value> materializeSplatScalarConstant(RewriterBase &rewriter,
                                                    Location loc, Value input) {
  SplatElementsAttr attr;
  Type elementType = mlir::getElementTypeOrSelf(input.getType());
  if (!matchPattern(input, m_Constant(&attr))) return {};
  if (isa<IntegerType, FloatType, IndexType>(elementType)) {
    return rewriter
        .create<arith::ConstantOp>(loc, elementType,
                                   attr.getSplatValue<TypedAttr>())
        .getResult();
  }
  if (isa<ComplexType>(elementType)) {
    return rewriter
        .create<complex::ConstantOp>(loc, elementType,
                                     attr.getSplatValue<ArrayAttr>())
        .getResult();
  }
  return {};
}

/// Converts a HLO operation to a linalg.map op that contains the corresponding
/// scalar operations.
template <typename OpTy>
struct PointwiseToLinalgMapConverter : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  virtual FailureOr<Operation *> createLinalgOp(
      OpTy &op, ConversionPatternRewriter &rewriter,
      ArrayRef<Value> mappedInputs, ArrayRef<Value> scalarVals,
      Value emptyTensor, int64_t maxRank) const {
    Operation *mapOp = rewriter.create<linalg::MapOp>(
        op.getLoc(), mappedInputs, emptyTensor,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value innerResult = mlir::stablehlo::StablehloOpToStdScalarOp::mapOp(
              op, getElementTypeOrSelf(emptyTensor),
              interleaveScalarAndBlockArgs(scalarVals, args), &b);

          b.create<linalg::YieldOp>(loc, innerResult);
        },
        linalg::getPrunedAttributeList(op));
    return mapOp;
  }

  LogicalResult matchAndRewrite(
      OpTy op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto conversionInfo = checkOperandsAndResults(
        op, adaptor.getOperands(), *this->typeConverter, rewriter);
    if (failed(conversionInfo)) {
      return failure();
    }

    int64_t maxRank = conversionInfo->maxOperandRank;
    ShapedType resultTy = conversionInfo->resultType;
    Location loc = op.getLoc();

    // Find input/output values and types.
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, resultTy, op, adaptor.getOperands());

    // Mapped inputs are cast to the same shape as the init tensor.
    // Values from scalar inputs are extracted and used directly in the block.
    SmallVector<Value> mappedInputs;
    SmallVector<Value> scalarInputs;
    for (Value input : adaptor.getOperands()) {
      if (std::optional<Value> splatVal =
              materializeSplatScalarConstant(rewriter, loc, input)) {
        scalarInputs.push_back(*splatVal);
      } else if (getRank(input) == maxRank) {
        mappedInputs.push_back(coerceTensorShape(
            rewriter, loc, cast<TypedValue<ShapedType>>(input),
            cast<ShapedType>(emptyTensor.getType())));
        scalarInputs.push_back(nullptr);
      } else {
        scalarInputs.push_back(rewriter.create<tensor::ExtractOp>(loc, input));
      }
    }

    auto mapOp = createLinalgOp(op, rewriter, mappedInputs, scalarInputs,
                                emptyTensor, maxRank);
    if (failed(mapOp)) return failure();

    rewriter.replaceOp(op, (*mapOp)->getResults());
    return success();
  }
};

/// Converts a HLO operation to a linalg.generic op that contains the
/// corresponding scalar operations.
template <typename OpTy>
struct PointwiseToLinalgConverter : PointwiseToLinalgMapConverter<OpTy> {
  using PointwiseToLinalgMapConverter<OpTy>::PointwiseToLinalgMapConverter;
  using OpAdaptor = typename OpTy::Adaptor;

  FailureOr<Operation *> createLinalgOp(OpTy &op,
                                        ConversionPatternRewriter &rewriter,
                                        ArrayRef<Value> mappedInputs,
                                        ArrayRef<Value> scalarVals,
                                        Value emptyTensor,
                                        int64_t maxRank) const override {
    // Create indexing maps.
    AffineMap scalarMap = AffineMap::get(maxRank, 0, rewriter.getContext());
    AffineMap idMap = rewriter.getMultiDimIdentityMap(maxRank);
    SmallVector<AffineMap> maps;
    for (Value v : mappedInputs)
      maps.push_back(isScalar(v) ? scalarMap : idMap);
    maps.push_back(idMap);
    bool failed = false;
    Operation *linalgOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), emptyTensor.getType(), mappedInputs, emptyTensor, maps,
        getNParallelLoopsAttrs(maxRank),
        [&](OpBuilder &nestedBuilder, Location /*nested_loc*/,
            ValueRange args) {
          Type innerResultTy = getElementTypeOrSelf(emptyTensor);
          auto argvec =
              llvm::to_vector<2>(args.take_front(mappedInputs.size()));
          Value semiring = preSparsify(op, argvec, innerResultTy, &rewriter);
          Value innerResult = mlir::stablehlo::StablehloOpToStdScalarOp::mapOp(
              op, innerResultTy, interleaveScalarAndBlockArgs(scalarVals, args),
              &rewriter);
          if (!innerResult) {
            failed = true;
          } else {
            innerResult = postSparsify(op, semiring, innerResult, &rewriter);
            nestedBuilder.create<linalg::YieldOp>(op.getLoc(), innerResult);
          }
        },
        linalg::getPrunedAttributeList(op));
    if (failed) return failure();
    return linalgOp;
  }
};
}  // namespace

namespace detail {
void populatePointwiseStablehloToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns, bool enablePrimitiveOps) {
  if (enablePrimitiveOps) {
    patterns->add<
        PointwiseToLinalgMapConverter<mlir::stablehlo::AbsOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::AddOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::AndOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::Atan2Op>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::BitcastConvertOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::CbrtOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::CeilOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::ClampOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::ClzOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::CompareOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::ComplexOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::ConvertOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::CosineOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::DivOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::ExpOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::Expm1Op>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::FloorOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::ImagOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::IsFiniteOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::Log1pOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::LogOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::LogisticOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::MaxOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::MinOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::MulOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::NegOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::NotOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::OrOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::PopulationCountOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::PowOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::RealOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::ReducePrecisionOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::RemOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::RoundNearestEvenOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::RoundOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::RsqrtOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::SelectOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::ShiftLeftOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::ShiftRightArithmeticOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::ShiftRightLogicalOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::SignOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::SineOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::SqrtOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::SubtractOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::TanhOp>,
        PointwiseToLinalgMapConverter<mlir::stablehlo::XorOp>>(typeConverter,
                                                               context);
    return;
  }

  patterns
      ->add<PointwiseToLinalgConverter<mlir::stablehlo::AbsOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::AddOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::AndOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::Atan2Op>,
            PointwiseToLinalgConverter<mlir::stablehlo::BitcastConvertOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::CbrtOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::CeilOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::ClampOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::ClzOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::CompareOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::ComplexOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::ConvertOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::CosineOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::DivOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::ExpOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::Expm1Op>,
            PointwiseToLinalgConverter<mlir::stablehlo::FloorOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::ImagOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::IsFiniteOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::Log1pOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::LogOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::LogisticOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::MaxOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::MinOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::MulOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::NegOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::NotOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::OrOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::PopulationCountOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::PowOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::RealOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::ReducePrecisionOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::RemOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::RoundNearestEvenOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::RoundOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::RsqrtOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::SelectOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::ShiftLeftOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::ShiftRightArithmeticOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::ShiftRightLogicalOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::SignOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::SineOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::SqrtOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::SubtractOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::TanhOp>,
            PointwiseToLinalgConverter<mlir::stablehlo::XorOp>>(typeConverter,
                                                                context);
}
}  // namespace detail
}  // namespace mlir::stablehlo
