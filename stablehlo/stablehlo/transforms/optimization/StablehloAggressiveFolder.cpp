/* Copyright 2024 The StableHLO Authors.
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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/optimization/Passes.h"

#define DEBUG_TYPE "stablehlo-optimization"

namespace mlir::stablehlo {

#define GEN_PASS_DEF_STABLEHLOAGGRESSIVEFOLDERPASS
#include "stablehlo/transforms/optimization/Passes.h.inc"

namespace {

static constexpr StablehloAggressiveFolderPassOptions kDefaultOptions;

// DenseElementsAttr can be constructed from ArrayRef<APInt> but not from
// ArrayRef<APSInt>. This helper bridges the gap.
DenseIntElementsAttr getTensorAttr(ShapedType type, ArrayRef<APSInt> values) {
  SmallVector<APInt> supportedValues(values);
  return DenseIntElementsAttr::get(type, supportedValues);
}

APSInt getAPSInt(Type type, uint64_t value) {
  unsigned numBits;
  bool isUnsigned;
  if (auto integerType = dyn_cast<IntegerType>(type)) {
    numBits = integerType.getWidth();
    // Signless types are treated as signed, per StableHLO convention.
    isUnsigned = integerType.isUnsignedInteger();
  } else {
    llvm::report_fatal_error("expected integer type");
  }
  return APSInt(
      {/*numBits=*/numBits, value, /*isSigned=*/false, /*implicitTrunc=*/true},
      /*isUnsigned=*/isUnsigned);
}

LogicalResult validateStaticShapeResult(PatternRewriter& rewriter,
                                        Operation* op, ShapedType resultType) {
  if (!resultType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "unable to fold dynamically shaped result type to constant");
  return success();
}

/// Binary constant folder that used a generic folder function to handle both
/// ints and floats.
template <typename Fn>
static TypedAttr foldBinaryOpIntOrFloat(TypedAttr lhs, TypedAttr rhs,
                                        Fn&& folder) {
  Attribute operands[2] = {lhs, rhs};
  Type elemTy = getElementTypeOrSelf(lhs);

  Attribute res;
  if (isa<IntegerType>(elemTy))
    res = constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(operands,
                                                                       folder);
  if (isa<FloatType>(elemTy))
    res = constFoldBinaryOp<FloatAttr, FloatAttr::ValueType, void>(operands,
                                                                   folder);
  if (res) return cast<TypedAttr>(res);

  return nullptr;
}

template <class AttrElementT, class TargetAttrElementT, class CalculationT,
          typename OpType>
LogicalResult evalConvertHelper(PatternRewriter& rewriter, OpType op,
                                DenseIntOrFPElementsAttr elements, Type resType,
                                CalculationT&& calculate) {
  auto result = constFoldCastOp<AttrElementT, TargetAttrElementT,
                                typename AttrElementT::ValueType,
                                typename TargetAttrElementT::ValueType, void>(
      elements, resType, calculate);

  if (!result)
    return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
      diag << "cast of " << elements.getElementType() << " to " << resType
           << " failed";
    });

  rewriter.replaceOpWithNewOp<ConstantOp>(op, result);
  return success();
}

template <typename OpType>
LogicalResult evalConvert(PatternRewriter& rewriter, OpType op,
                          DenseIntOrFPElementsAttr elements,
                          RankedTensorType resultType) {
  auto oldType = getElementTypeOrSelf(elements);
  auto newType = getElementTypeOrSelf(resultType);
  size_t newBitWidth = newType.getIntOrFloatBitWidth();

  bool isOldTypeUnsigned = oldType.isInteger(1) || oldType.isUnsignedInteger();
  bool isNewTypeUnsigned = newType.isInteger(1) || newType.isUnsignedInteger();

  if (isa<FloatType>(oldType)) {
    if (auto newFloatType = dyn_cast<FloatType>(newType)) {
      // Float -> Float
      const auto& targetSemantics = newFloatType.getFloatSemantics();
      return evalConvertHelper<FloatAttr, FloatAttr>(
          rewriter, op, elements, resultType,
          [&targetSemantics](const APFloat& operand, bool& castStatus) {
            bool losesInfo;
            APFloat newValue = operand;
            castStatus = APFloat::opInvalidOp !=
                         newValue.convert(targetSemantics,
                                          llvm::RoundingMode::NearestTiesToEven,
                                          &losesInfo);
            return newValue;
          });
    }

    // Float -> Int
    return evalConvertHelper<FloatAttr, IntegerAttr>(
        rewriter, op, elements, resultType,
        [&newBitWidth, &isNewTypeUnsigned](const APFloat& operand,
                                           bool& castStatus) {
          APSInt api(newBitWidth, isNewTypeUnsigned);
          if (operand.isInfinity() || operand.isNegZero()) {
            castStatus = false;
            return api;
          }
          bool ignored;
          castStatus =
              APFloat::opInvalidOp !=
              operand.convertToInteger(api, APFloat::rmTowardZero, &ignored);
          return api;
        });
  }

  if (auto newFloatType = dyn_cast<FloatType>(newType)) {
    // Int -> Float
    return evalConvertHelper<IntegerAttr, FloatAttr>(
        rewriter, op, elements, resultType,
        [&newFloatType, &isOldTypeUnsigned](const APInt& operand,
                                            bool& /*castStatus*/) {
          APFloat apf(newFloatType.getFloatSemantics(),
                      APInt::getZero(newFloatType.getWidth()));
          apf.convertFromAPInt(operand, !isOldTypeUnsigned,
                               APFloat::rmNearestTiesToEven);
          return apf;
        });
  }

  // Int -> Int
  return evalConvertHelper<IntegerAttr, IntegerAttr>(
      rewriter, op, elements, resultType,
      [&newBitWidth, &isOldTypeUnsigned](const APInt& operand,
                                         bool& /*castStatus*/) {
        return APSInt(operand, isOldTypeUnsigned).extOrTrunc(newBitWidth);
      });
}

// The patterns below implement partial evaluation of shape computations which
// is a critical part of implementing type refinement for ops like
// dynamic_broadcast_in_dim, dynamic_iota and dynamic_reshape whose shape
// depends on the value of their shape operands.

template <typename OpType, typename FuncType>
LogicalResult evalElementwise(PatternRewriter& rewriter, OpType op,
                              FuncType fn) {
  auto resultType = op.getType();
  if (failed(validateStaticShapeResult(rewriter, op, resultType)))
    return failure();

  if (!isa<IntegerType>(resultType.getElementType()))
    return rewriter.notifyMatchFailure(op,
                                       "expected integer result tensor type");

  SmallVector<APSInt> result;
  if constexpr (OpType::template hasTrait<OpTrait::OneOperand>()) {
    SmallVector<APSInt> operand;
    if (failed(hlo::matchInts(op.getOperand(), operand)))
      return rewriter.notifyMatchFailure(op, "expected constant operand");
    for (const auto& operandEl : operand) {
      result.push_back(fn(operandEl));
    }
  } else if constexpr (OpType::template hasTrait<
                           OpTrait::NOperands<2>::Impl>()) {
    SmallVector<APSInt> lhs, rhs;
    if (failed(hlo::matchInts(op.getLhs(), lhs)) ||
        failed(hlo::matchInts(op.getRhs(), rhs)))
      return rewriter.notifyMatchFailure(op, "expected constant operands");
    for (auto [lhsEl, rhsEl] : llvm::zip(lhs, rhs)) {
      result.push_back(fn(lhsEl, rhsEl));
    }
  } else if constexpr (OpType::template hasTrait<
                           OpTrait::NOperands<3>::Impl>()) {
    SmallVector<APSInt> x, y, z;
    if (failed(hlo::matchInts(op->getOperand(0), x)) ||
        failed(hlo::matchInts(op->getOperand(1), y)) ||
        failed(hlo::matchInts(op->getOperand(2), z)))
      return rewriter.notifyMatchFailure(op, "expected constant operands");
    for (auto [xEl, yEl, zEl] : llvm::zip(x, y, z)) {
      result.push_back(fn(xEl, yEl, zEl));
    }
  } else {
    llvm::report_fatal_error("unsupported number of operands");
  }

  rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                          getTensorAttr(resultType, result));
  return success();
}

template <typename OpType>
struct FoldOpRewritePattern : OpRewritePattern<OpType> {
  FoldOpRewritePattern(MLIRContext* context,
                       const StablehloAggressiveFolderPassOptions& options,
                       PatternBenefit benefit = 1,
                       ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern<OpType>(context, benefit, generatedNames),
        options(options) {}

  // Prevent `options` from binding to a temporary.
  FoldOpRewritePattern(MLIRContext* context,
                       StablehloAggressiveFolderPassOptions&& options,
                       PatternBenefit benefit = 1,
                       ArrayRef<StringRef> generatedNames = {}) = delete;

  const StablehloAggressiveFolderPassOptions& options;
};

struct FoldAddOpPattern final : FoldOpRewritePattern<mlir::stablehlo::AddOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::AddOp op,
                                PatternRewriter& rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // Pattern: add(cst,cst) -> cst
    TypedAttr lhsAttr, rhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));
    matchPattern(rhs, m_Constant(&rhsAttr));

    if (TypedAttr res;
        lhsAttr && rhsAttr &&
        (res = foldBinaryOpIntOrFloat(lhsAttr, rhsAttr, std::plus<>{}))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
      return success();
    }

    return failure();
  }
};

// A base class to use for patterns that may be used for integer shape math,
// but also may be used for general folding of floats.
template <typename OpType>
struct ShapeOpRewritePattern : public FoldOpRewritePattern<OpType> {
  using FoldOpRewritePattern<OpType>::FoldOpRewritePattern;
  using FoldOpRewritePattern<OpType>::matchAndRewrite;
  using FoldOpRewritePattern<OpType>::options;

  LogicalResult validateShapeFoldDtype(PatternRewriter& rewriter, OpType op,
                                       ShapedType resultType) const {
    if (resultType.getElementType().isInteger()) return success();
    if (options.optimizeFloat && isa<FloatType>(resultType.getElementType()))
      return success();
    return rewriter.notifyMatchFailure(op, "skipping fold of shape op dtype");
  }
};

struct EvalAddOpShapePattern : public FoldOpRewritePattern<AddOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt lhs, APSInt rhs) { return lhs + rhs; });
  }
};

struct EvalAndOpPattern : public FoldOpRewritePattern<AndOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(AndOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (!resultType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(op, "expected boolean element type");

    return evalElementwise(rewriter, op, [&](APSInt lhsInt, APSInt rhsInt) {
      return getAPSInt(resultType.getElementType(), lhsInt != 0 && rhsInt != 0);
    });
  }
};

// Pattern: broadcast_in_dim(splat, _) -> constant(splat)
struct FoldBroadcastInDimSplatPattern final
    : FoldOpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    TypedValue<RankedTensorType> operand = op.getOperand();

    if (SplatElementsAttr cstAttr;
        matchPattern(operand, m_Constant(&cstAttr))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, SplatElementsAttr::get(op.getType(),
                                     cstAttr.getSplatValue<Attribute>()));
      return success();
    }
    return failure();
  }
};

struct EvalBroadcastInDimOpPattern
    : public FoldOpRewritePattern<BroadcastInDimOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)))
      return failure();

    auto operandType = op.getOperand().getType();
    if (operandType.getRank() != 0)
      return rewriter.notifyMatchFailure(op, "expected 0-dimensional type");

    SmallVector<APSInt> operand;
    if (failed(hlo::matchInts(op.getOperand(), operand)))
      return rewriter.notifyMatchFailure(op, "expected constant operands");
    auto scalar = operand[0];

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, getTensorAttr(op.getType(), scalar));
    return success();
  }
};

struct EvalClampOpPattern : public FoldOpRewritePattern<ClampOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(ClampOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt min, APSInt operand, APSInt max) {
                             if (operand < min) return min;
                             if (max < operand) return max;
                             return operand;
                           });
  }
};

struct EvalCompareOpPattern : public FoldOpRewritePattern<CompareOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    auto kind = op.getCompareType();
    return evalElementwise(rewriter, op, [&](APInt lhs, APInt rhs) {
      bool result = false;
      switch (op.getComparisonDirection()) {
        case ComparisonDirection::EQ:
          result = lhs == rhs;
          break;
        case ComparisonDirection::NE:
          result = lhs != rhs;
          break;
        case ComparisonDirection::GE:
          result = kind == ComparisonType::SIGNED ? lhs.sge(rhs) : lhs.uge(rhs);
          break;
        case ComparisonDirection::GT:
          result = kind == ComparisonType::SIGNED ? lhs.sgt(rhs) : lhs.ugt(rhs);
          break;
        case ComparisonDirection::LE:
          result = kind == ComparisonType::SIGNED ? lhs.sle(rhs) : lhs.ule(rhs);
          break;
        case ComparisonDirection::LT:
          result = kind == ComparisonType::SIGNED ? lhs.slt(rhs) : lhs.ult(rhs);
          break;
      }
      return getAPSInt(resultType.getElementType(), result);
    });
  }
};

//////////////////////////////////
// ConcatenateOp
/////////////////////////////////

struct FoldConcatenateOpPattern final
    : FoldOpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    RankedTensorType type = op.getType();
    if (!type.hasStaticShape()) return failure();

    size_t numElems = type.getNumElements();
    if (numElems > static_cast<size_t>(options.foldOpElementLimit))
      return failure();

    // Fold concatenate when all inputs are constants.
    OperandRange inputs = op.getInputs();
    SmallVector<DenseElementsAttr> constants(inputs.size());
    for (auto [input, constant] : llvm::zip_equal(inputs, constants)) {
      if (!matchPattern(input, m_Constant(&constant))) return failure();
    }

    uint64_t dim = op.getDimension();
    ArrayRef<int64_t> shape = type.getShape();
    int64_t topSize = std::accumulate(shape.begin(), shape.begin() + dim,
                                      int64_t{1}, std::multiplies<>{});

    SmallVector<Attribute> newElems;
    newElems.reserve(numElems);

    for (int64_t i = 0; i != topSize; ++i) {
      for (ElementsAttr attr : constants) {
        size_t bottomSize = attr.getNumElements() / topSize;
        auto begin = attr.value_begin<Attribute>() + (i * bottomSize);
        newElems.append(begin, begin + bottomSize);
      }
    }

    assert(newElems.size() == numElems);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), newElems));
    return success();
  }

  int64_t foldOpElementLimit;
};

struct EvalConcatenateOpPattern : public FoldOpRewritePattern<ConcatenateOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)))
      return failure();

    if (op.getDimension() != 0)
      return rewriter.notifyMatchFailure(op, "expected dimension = 0");

    SmallVector<APSInt> result;
    for (Value operand : op->getOperands()) {
      if (failed(hlo::matchInts(operand, result)))
        return rewriter.notifyMatchFailure(op, "expected constant operands");
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                            getTensorAttr(resultType, result));
    return success();
  }
};

struct EvalConvertOpPattern : public ShapeOpRewritePattern<ConvertOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter& rewriter) const override {
    auto operand = op.getOperand();
    RankedTensorType resultType = op.getType();

    if (failed(validateStaticShapeResult(rewriter, op, resultType)) ||
        failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    auto operandElemType = getElementTypeOrSelf(operand.getType());
    auto resultElemType = getElementTypeOrSelf(resultType);
    if (!options.optimizeFloat &&
        (isa<FloatType>(operandElemType) || isa<FloatType>(resultElemType)))
      return rewriter.notifyMatchFailure(op, "skipping fold of float convert");

    DenseIntOrFPElementsAttr elements;
    if (!matchPattern(operand, m_Constant(&elements)))
      return rewriter.notifyMatchFailure(
          op, "expected constant integer or float operand");

    return evalConvert(rewriter, op, elements, resultType);
  }
};

struct EvalDivOpPattern : public FoldOpRewritePattern<DivOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt lhs, APSInt rhs) { return lhs / rhs; });
  }
};

struct EvalGetDimensionSizeOpPattern
    : public FoldOpRewritePattern<GetDimensionSizeOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(GetDimensionSizeOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)))
      return failure();

    auto operandType = op.getOperand().getType();
    if (operandType.isDynamicDim(op.getDimension()))
      return rewriter.notifyMatchFailure(op, "expected static dimension");

    auto result = operandType.getDimSize(op.getDimension());
    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseIntElementsAttr::get<int32_t>(resultType, result));
    return success();
  }
};

struct EvalMaxOpPattern : public FoldOpRewritePattern<MaxOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(MaxOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op, [&](APSInt lhs, APSInt rhs) {
      return lhs >= rhs ? lhs : rhs;
    });
  }
};

struct EvalMinOpPattern : public FoldOpRewritePattern<MinOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(MinOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op, [&](APSInt lhs, APSInt rhs) {
      return lhs <= rhs ? lhs : rhs;
    });
  }
};

struct FoldMulOpPattern final : FoldOpRewritePattern<mlir::stablehlo::MulOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::MulOp op,
                                PatternRewriter& rewriter) const override {
    TypedAttr lhsAttr;
    matchPattern(op.getLhs(), m_Constant(&lhsAttr));

    TypedAttr rhsAttr;
    matchPattern(op.getRhs(), m_Constant(&rhsAttr));

    if (TypedAttr res;
        lhsAttr && rhsAttr &&
        (res = foldBinaryOpIntOrFloat(lhsAttr, rhsAttr, std::multiplies<>{}))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
      return success();
    }

    return failure();
  }
};

struct EvalMulOpPattern : public FoldOpRewritePattern<MulOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt lhs, APSInt rhs) { return lhs * rhs; });
  }
};

struct EvalOrOpPattern : public FoldOpRewritePattern<OrOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(OrOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (!resultType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(op, "expected boolean element type");

    return evalElementwise(rewriter, op, [&](APSInt lhsInt, APSInt rhsInt) {
      return getAPSInt(resultType.getElementType(), lhsInt != 0 || rhsInt != 0);
    });
  }
};

struct EvalRemOpPattern : public FoldOpRewritePattern<RemOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(RemOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt lhs, APSInt rhs) { return lhs % rhs; });
  }
};

struct EvalReshapeOpPattern : public ShapeOpRewritePattern<ReshapeOp> {
  using ShapeOpRewritePattern::ShapeOpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)) ||
        failed(validateShapeFoldDtype(rewriter, op, resultType)))
      return failure();

    // Pattern: reshape(cst, shape) -> cst
    DenseIntOrFPElementsAttr attr;
    if (!matchPattern(op.getOperand(), m_Constant(&attr)))
      return rewriter.notifyMatchFailure(op, "expected constant operand");
    rewriter.replaceOpWithNewOp<ConstantOp>(op, attr.reshape(resultType));
    return success();
  }
};

struct EvalSelectOpPattern : public FoldOpRewritePattern<SelectOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)))
      return failure();

    SmallVector<APSInt> pred, onTrue, onFalse;
    if (failed(hlo::matchInts(op.getPred(), pred)) ||
        failed(hlo::matchInts(op.getOnTrue(), onTrue)) ||
        failed(hlo::matchInts(op.getOnFalse(), onFalse)))
      return rewriter.notifyMatchFailure(op, "expected constant operands");

    SmallVector<APSInt> result;
    for (auto [predEl, onTrueEl, onFalseEl] :
         llvm::zip(pred, onTrue, onFalse)) {
      result.push_back(predEl != 0 ? onTrueEl : onFalseEl);
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, getTensorAttr(op.getType(), result));
    return success();
  }
};

struct EvalSignOpPattern : public FoldOpRewritePattern<SignOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(SignOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (!isa<IntegerType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(op,
                                         "expected integer result tensor type");
    return evalElementwise(rewriter, op, [&](APSInt operand) {
      int64_t result;
      if (operand.isNegative())
        result = -1;
      else if (operand.isZero())
        result = 0;
      else
        result = 1;
      return getAPSInt(resultType.getElementType(), result);
    });
  }
};

template <typename RangeType>
DenseElementsAttr sliceType(SliceOp& op, const RangeType& data) {
  using ElementType = std::decay_t<decltype(*std::begin(data))>;

  RankedTensorType operandType = op.getOperand().getType();
  RankedTensorType resultType = op.getResult().getType();

  const auto dimOffsets = computeStrides(operandType.getShape());
  auto startIndices = op.getStartIndices();
  auto limitIndices = op.getLimitIndices();
  auto strides = op.getStrides();

  const SmallVector<int64_t> startIndex(startIndices);
  const SmallVector<int64_t> endIndex(limitIndices);

  SmallVector<ElementType> result;
  result.reserve(resultType.getNumElements());

  SmallVector<int64_t> srcIndex(startIndex);
  for (int64_t i = 0; i < resultType.getNumElements(); ++i) {
    auto srcLinearIndex = linearize(srcIndex, dimOffsets);
    result.push_back(data[srcLinearIndex]);
    for (int64_t dim = srcIndex.size() - 1; dim >= 0; --dim) {
      srcIndex[dim] += strides[dim];
      if (srcIndex[dim] >= endIndex[dim])
        srcIndex[dim] = startIndex[dim];
      else
        break;
    }
  }

  return DenseElementsAttr::get(op.getResult().getType(),
                                ArrayRef<ElementType>(result));
}

struct EvalSliceOpPattern : public FoldOpRewritePattern<SliceOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)))
      return failure();

    auto operand = op.getOperand();
    RankedTensorType operandType = operand.getType();
    if (!operandType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "expected operand with static ranked tensor type");

    ElementsAttr els;
    if (!matchPattern(operand, m_Constant(&els)))
      return rewriter.notifyMatchFailure(
          op, "expected constant integer or float operand");

    DenseElementsAttr resAttr;
    if (auto data = els.tryGetValues<APInt>())
      resAttr = sliceType(op, *data);
    else if (auto data = els.tryGetValues<APFloat>())
      resAttr = sliceType(op, *data);
    else
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "unsupported element type");

    rewriter.replaceOpWithNewOp<ConstantOp>(op, resAttr);
    return success();
  }
};

struct FoldSubtractOpPattern final
    : FoldOpRewritePattern<mlir::stablehlo::SubtractOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SubtractOp op,
                                PatternRewriter& rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    TypedAttr lhsAttr, rhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));
    matchPattern(rhs, m_Constant(&rhsAttr));

    if (TypedAttr res;
        lhsAttr && rhsAttr &&
        (res = foldBinaryOpIntOrFloat(lhsAttr, rhsAttr, std::minus<>{}))) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
      return success();
    }

    return failure();
  }
};

struct EvalSubtractOpPattern : public FoldOpRewritePattern<SubtractOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(SubtractOp op,
                                PatternRewriter& rewriter) const override {
    return evalElementwise(rewriter, op,
                           [&](APSInt lhs, APSInt rhs) { return lhs - rhs; });
  }
};

struct FoldSqrtOpPattern
    : public FoldOpRewritePattern<mlir::stablehlo::SqrtOp> {
  using FoldOpRewritePattern<mlir::stablehlo::SqrtOp>::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SqrtOp op,
                                PatternRewriter& rewriter) const final {
    TypedAttr lhsAttr;
    matchPattern(op.getOperand(), m_Constant(&lhsAttr));

    if (!lhsAttr)
      return rewriter.notifyMatchFailure(op, "operand not constant");

    if (auto res = constFoldUnaryOp<FloatAttr, FloatAttr::ValueType, void>(
            lhsAttr, foldSqrt)) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), llvm::cast<ElementsAttr>(res));
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unable to fold sqrt");
  }

  static std::optional<APFloat> foldSqrt(const APFloat& a) {
    if (a.getSizeInBits(a.getSemantics()) == 64)
      return APFloat(std::sqrt(a.convertToDouble()));

    if (a.getSizeInBits(a.getSemantics()) == 32)
      return APFloat(sqrtf(a.convertToFloat()));
    return {};
  }
};

struct EvalIotaOpPattern : public FoldOpRewritePattern<IotaOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(IotaOp op,
                                PatternRewriter& rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "EvalIotaOpPattern folding: " << op << '\n');
    auto resultType = cast<RankedTensorType>(op.getType());
    size_t numElems = resultType.getNumElements();
    if (numElems > static_cast<size_t>(options.foldOpElementLimit))
      return rewriter.notifyMatchFailure(op, "too many elements to fold");

    auto elementType = resultType.getElementType();

    if (!elementType.isInteger())
      return rewriter.notifyMatchFailure(op, "expected integer result type");

    auto outputSize = resultType.getNumElements();
    auto resultBitWidth = elementType.getIntOrFloatBitWidth();
    int64_t dimension = op.getIotaDimension();

    llvm::SmallVector<APInt> values;
    values.reserve(outputSize);

    if (outputSize == 0) {
      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, DenseIntElementsAttr::get(resultType, values));
      return success();
    }

    int64_t sequences = 1;
    int64_t sequenceMax = resultType.getDimSize(dimension);
    int64_t elementRepetitions = 1;
    for (int64_t i = 0; i < resultType.getRank(); i++) {
      sequences *= i < dimension ? resultType.getDimSize(i) : 1;
      elementRepetitions *= i > dimension ? resultType.getDimSize(i) : 1;
    }

    for (int64_t i = 0; i < sequences; ++i) {
      for (int64_t value = 0; value < sequenceMax; ++value) {
        for (int64_t k = 0; k < elementRepetitions; ++k) {
          values.push_back(APInt(resultBitWidth, value));
        }
      }
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, DenseIntElementsAttr::get(resultType, values));
    return success();
  }

  int64_t foldOpElementLimit;
};

template <typename RangeType>
DenseElementsAttr transposeType(TransposeOp& op, const RangeType& data) {
  using ElementType = std::decay_t<decltype(*std::begin(data))>;

  RankedTensorType operandType = op.getOperand().getType();
  RankedTensorType resultType = op.getResult().getType();

  const auto operandStrides = computeStrides(operandType.getShape());
  const auto resultStrides = computeStrides(resultType.getShape());
  const auto inversePermutation = invertPermutationVector(op.getPermutation());

  SmallVector<ElementType> result;
  result.reserve(resultType.getNumElements());

  for (int64_t i = 0; i < resultType.getNumElements(); ++i) {
    auto dstDimIndex = delinearize(i, resultStrides);
    auto srcDimIndex = applyPermutation(dstDimIndex, inversePermutation);
    auto srcLinearIndex = linearize(srcDimIndex, operandStrides);
    result.push_back(data[srcLinearIndex]);
  }

  return DenseElementsAttr::get(resultType, ArrayRef<ElementType>(result));
}

// transpose(constant) => constant with permuted dimensions
// This covers ranked tensor types with 0 dimensions(zero elements) and 0
// rank(scalar), as well as splat values.
struct EvalTransposeOpPattern : public FoldOpRewritePattern<TransposeOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter& rewriter) const override {
    auto resultType = op.getType();
    if (failed(validateStaticShapeResult(rewriter, op, resultType)))
      return failure();

    ElementsAttr els;
    if (!matchPattern(op.getOperand(), m_Constant(&els)))
      return rewriter.notifyMatchFailure(
          op, "expected constant integer or float operand");

    DenseElementsAttr resAttr;
    if (auto data = els.tryGetValues<APInt>())
      resAttr = transposeType(op, *data);
    else if (auto data = els.tryGetValues<APFloat>())
      resAttr = transposeType(op, *data);
    else
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "unsupported element type");

    rewriter.replaceOpWithNewOp<ConstantOp>(op, resAttr);
    return success();
  }
};

struct LowerBoolSplatConstantsIntoReduceOpRegion
    : public FoldOpRewritePattern<ReduceOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    Block& body = op.getBody().front();

    if (body.getOperations().size() != 2)
      return rewriter.notifyMatchFailure(op, "Incompatible op count in body.");
    if (!isa<AndOp, OrOp>(body.front()))
      return rewriter.notifyMatchFailure(op, "Only match AND and OR ops.");

    SmallVector<DenseElementsAttr, 4> bodyArgConstantAttrs;

    for (auto [inputValue, bodyArg] :
         llvm::zip_equal(op.getOperands(), body.getArguments())) {
      auto inputConstantOp = inputValue.getDefiningOp<ConstantOp>();
      if (!inputConstantOp)
        return rewriter.notifyMatchFailure(op, "Input must be a constant.");

      auto inputConstantAttr =
          dyn_cast_or_null<DenseElementsAttr>(inputConstantOp.getValue());
      if (!inputConstantAttr)
        return rewriter.notifyMatchFailure(op,
                                           "Input must be a splat constant.");

      auto bodyArgShapedType = dyn_cast<ShapedType>(bodyArg.getType());
      if (!bodyArgShapedType)
        return rewriter.notifyMatchFailure(
            op, "Could not get the shape of the body argument.");

      bodyArgConstantAttrs.push_back(DenseElementsAttr::get(
          bodyArgShapedType, inputConstantAttr.getSplatValue<Attribute>()));
    }

    for (BlockArgument bodyArg : body.getArguments()) {
      rewriter.replaceAllUsesWith(
          bodyArg, rewriter.create<ConstantOp>(
                       body.front().getLoc(), bodyArg.getType(),
                       bodyArgConstantAttrs[bodyArg.getArgNumber()]));
    }

    return success();
  }
};

struct FoldReduceOpReducingZeroDims : public FoldOpRewritePattern<ReduceOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    // Fail to match if the reduce op operates on any dimensions.
    if (!op.getDimensions().empty())
      return rewriter.notifyMatchFailure(
          op, "The reduce op reduces a nonzero number of dimensions.");

    // Check that input and output types match.
    for (auto [in, out] : llvm::zip_equal(op.getInputs(), op.getResults())) {
      if (in.getType() != out.getType())
        return rewriter.notifyMatchFailure(
            op, "Input and output types do not match.");
    }

    rewriter.replaceOp(op, op.getInputs());
    return success();
  }
};

struct FoldReduceOpToConstantInitializer
    : public FoldOpRewritePattern<ReduceOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    Block& body = op.getBody().front();
    if (body.getOperations().size() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "Body must contain exactly one op.");

    auto returnOp = dyn_cast<ReturnOp>(body.back());
    if (!returnOp)
      return rewriter.notifyMatchFailure(op, "Body must end with a return op.");

    SmallVector<DenseElementsAttr> resultAttrs;
    for (auto [bodyResult, opResult] :
         llvm::zip_equal(returnOp.getResults(), op.getResults())) {
      auto* sourceOfBlockResult = bodyResult.getDefiningOp();
      if (!sourceOfBlockResult ||
          !sourceOfBlockResult->hasTrait<OpTrait::ConstantLike>())
        return rewriter.notifyMatchFailure(op,
                                           "Body result must be a constant.");

      DenseElementsAttr constantAttr;
      if (!matchPattern(sourceOfBlockResult, m_Constant(&constantAttr)))
        return rewriter.notifyMatchFailure(
            op, "Could not extract constant attribute from body result.");

      auto resultShapedType = dyn_cast<ShapedType>(opResult.getType());
      if (!resultShapedType)
        return rewriter.notifyMatchFailure(
            op, "Could not get the shape of the reduce op's result.");

      resultAttrs.push_back(DenseElementsAttr::get(
          resultShapedType, {constantAttr.getSplatValue<Attribute>()}));
    }

    SmallVector<Value> resultValues;
    for (auto resultAttr : resultAttrs) {
      resultValues.push_back(rewriter.create<ConstantOp>(
          op.getLoc(), resultAttr.getType(), resultAttr));
    }

    rewriter.replaceOp(op, resultValues);
    return success();
  }
};

struct FoldReduceOpWithRedundantResults
    : public FoldOpRewritePattern<ReduceOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    Block& body = op.getBody().front();
    auto returnOp = dyn_cast<ReturnOp>(body.back());
    if (!returnOp)
      return rewriter.notifyMatchFailure(op, "Body must end with a return op.");

    Region* returnOpParentRegion = returnOp->getParentRegion();

    for (auto [reduceOpResult, returnOpResult] :
         llvm::zip_equal(op.getResults(), returnOp.getResults())) {
      if (returnOpResult.getParentRegion() == returnOpParentRegion ||
          returnOpResult.getType() != reduceOpResult.getType()) {
        return rewriter.notifyMatchFailure(
            op, "The reduce op's result isn't redundant.");
      }
    }
    rewriter.replaceOp(op, returnOp.getResults());
    return success();
  }
};

struct FoldWhileOpPattern : public FoldOpRewritePattern<WhileOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter& rewriter) const override {
    // It is, unfortunately, possible for code to depend on the very existence
    // of a side effect even if that side effect is unreachable. We'd ideally
    // like to fix this, but that's not as simple as it sounds. For now, we need
    // to make sure we don't DCE code with side effects in case something else
    // depends on it.
    if (op->use_empty() && !isOpTriviallyDead(op))
      return rewriter.notifyMatchFailure(
          op, "Keeping dead while op due to known or potential side effects.");

    auto condReturnOp = dyn_cast<ReturnOp>(op.getCond().front().back());
    if (!condReturnOp)
      return rewriter.notifyMatchFailure(
          op, "Condition region is missing a return statement.");

    DenseIntElementsAttr condValue;
    if (!matchPattern(condReturnOp.getOperand(0), m_Constant(&condValue)))
      return rewriter.notifyMatchFailure(
          op, "Condition block does not return a constant.");
    if (condValue.getSplatValue<BoolAttr>().getValue())
      return rewriter.notifyMatchFailure(
          op, "Condition value is not a splat of the bool `false`.");

    // Replace uses of the op's result, but don't remove the op itself; let
    // dedicated DCE logic handle that step if appropriate. (This is because of
    // the aforementioned issue where ops with side effects might need to remain
    // in the IR even if unreachable.)
    rewriter.replaceAllOpUsesWith(op, op.getOperand());
    return success();
  }
};

bool hasNoDeclaredSideEffects(Operation* op) {
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Return false if the op has memory effects of its own.
    if (!memInterface.hasNoEffect()) return false;
    // The op has no direct memory effects. Return true if it has no recursive
    // memory effects, either.
    if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) return true;
  } else {
    // The op doesn't implement the memory-effect interface. This function is
    // only interested in explicitly declared side effects, so we treat it as
    // having none and move on to checking its regions in case they have any.
  }

  // The op doesn't declare any side effects of its own, but its regions could
  // still contain ops that do declare side effects. Recursively check them.
  for (Region& region : op->getRegions()) {
    for (Operation& op : region.getOps()) {
      if (!hasNoDeclaredSideEffects(&op)) return false;
    }
  }
  return true;
}

struct RemoveDeadWhileOpWithNoSideEffects
    : public FoldOpRewritePattern<WhileOp> {
  using FoldOpRewritePattern::FoldOpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter& rewriter) const override {
    if (!op->use_empty()) {
      return rewriter.notifyMatchFailure(op, "The op's result is in use.");
    }

    if (options.assumeNoUndeclaredSideEffects) {
      if (!hasNoDeclaredSideEffects(op)) {
        return rewriter.notifyMatchFailure(
            op,
            "The op, or another op within its region, explicitly declares side "
            "effects.");
      }
    } else {
      if (!isMemoryEffectFree(op)) {
        return rewriter.notifyMatchFailure(
            op, "Not removing the op due to potential side effects.");
      }
    }

    // Neither this op nor any in its regions have any declared side effects (or
    // any potential side effects if `assumeNoUndeclaredSideEffects` is false),
    // and the op's result is unused. Erase the op.
    rewriter.eraseOp(op);
    return success();
  }
};

struct StablehloAggressiveFolderPass
    : public impl::StablehloAggressiveFolderPassBase<
          StablehloAggressiveFolderPass> {
  explicit StablehloAggressiveFolderPass(
      StablehloAggressiveFolderPassOptions options,
      GreedyRewriteConfig rewriteConfig = {})
      : StablehloAggressiveFolderPassBase(options),
        rewriteConfig(rewriteConfig) {}

  explicit StablehloAggressiveFolderPass()
      : StablehloAggressiveFolderPassBase() {}

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    StablehloAggressiveFolderPassOptions options{
        /*assumeNoUndeclaredSideEffects=*/assumeNoUndeclaredSideEffects,
        /*foldOpElementLimit=*/foldOpElementLimit,
        /*optimizeFloat=*/optimizeFloat,
    };

    populateStablehloAggressiveFolderPatterns(context, &patterns, options);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     rewriteConfig)))
      signalPassFailure();
  }

 private:
  GreedyRewriteConfig rewriteConfig;
};

}  // namespace

void populateStablehloAggressiveFolderPatterns(
    MLIRContext* context, RewritePatternSet* patterns,
    const StablehloAggressiveFolderPassOptions& options,
    PatternBenefit benefit) {
  populateStablehloShapeFolderPatterns(context, patterns, options, benefit);

  patterns->add<EvalIotaOpPattern,                          //
                EvalTransposeOpPattern,                     //
                FoldReduceOpReducingZeroDims,               //
                FoldReduceOpToConstantInitializer,          //
                FoldReduceOpWithRedundantResults,           //
                FoldWhileOpPattern,                         //
                LowerBoolSplatConstantsIntoReduceOpRegion,  //
                RemoveDeadWhileOpWithNoSideEffects>(context, options, benefit);

  // TODO: Consolidate FoldOp patterns
  // One is used by Shape Refinement, the other is a generic folder.
  patterns->add<FoldAddOpPattern,                //
                FoldBroadcastInDimSplatPattern,  //
                FoldConcatenateOpPattern,        //
                FoldMulOpPattern,                //
                FoldSqrtOpPattern,               //
                FoldSubtractOpPattern>(context, options);
}

class StablehloTargetIndependentOptimizationPass {
 private:
  StablehloTargetIndependentOptimizationPassOptions options;
};

class StablehloIndependentOptimizationPass {
 private:
  StablehloTargetIndependentOptimizationPassOptions options;
};

void populateStablehloShapeFolderPatterns(
    MLIRContext* context, RewritePatternSet* patterns,
    const StablehloAggressiveFolderPassOptions& options,
    PatternBenefit benefit) {
  patterns->add<EvalAddOpShapePattern>(context, options, benefit);
  patterns->add<EvalAndOpPattern>(context, options, benefit);
  patterns->add<EvalBroadcastInDimOpPattern>(context, options, benefit);
  patterns->add<EvalClampOpPattern>(context, options, benefit);
  patterns->add<EvalCompareOpPattern>(context, options, benefit);
  patterns->add<EvalConcatenateOpPattern>(context, options, benefit);
  patterns->add<EvalConvertOpPattern>(context, options, benefit);
  patterns->add<EvalDivOpPattern>(context, options, benefit);
  patterns->add<EvalGetDimensionSizeOpPattern>(context, options, benefit);
  patterns->add<EvalMaxOpPattern>(context, options, benefit);
  patterns->add<EvalMinOpPattern>(context, options, benefit);
  patterns->add<EvalMulOpPattern>(context, options, benefit);
  patterns->add<EvalOrOpPattern>(context, options, benefit);
  patterns->add<EvalRemOpPattern>(context, options, benefit);
  patterns->add<EvalReshapeOpPattern>(context, options, benefit);
  patterns->add<EvalSelectOpPattern>(context, options, benefit);
  patterns->add<EvalSignOpPattern>(context, options, benefit);
  patterns->add<EvalSliceOpPattern>(context, options, benefit);
  patterns->add<EvalSubtractOpPattern>(context, options, benefit);
}

void populateStablehloShapeFolderPatterns(MLIRContext* context,
                                          RewritePatternSet* patterns,
                                          PatternBenefit benefit) {
  populateStablehloShapeFolderPatterns(context, patterns, kDefaultOptions,
                                       benefit);
}

std::unique_ptr<::mlir::Pass> createStablehloAggressiveFolderPass(
    StablehloAggressiveFolderPassOptions options,
    GreedyRewriteConfig rewriteConfig) {
  return std::make_unique<StablehloAggressiveFolderPass>(options,
                                                         rewriteConfig);
}

}  // namespace mlir::stablehlo
