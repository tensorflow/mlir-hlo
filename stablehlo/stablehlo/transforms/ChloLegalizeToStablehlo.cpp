// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering CHLO ops to StableHLO and Shape dialect ops,
// taking care of CHLO's broadcasting semantics
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/BroadcastUtils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/ChloDecompositionUtils.h"
#include "stablehlo/transforms/PassUtils.h"
#include "stablehlo/transforms/Passes.h"

// This must precede all other headers, otherwise during Windows cross
// compilation, M_PI will not be defined.
#define _USE_MATH_DEFINES

#define DEBUG_TYPE "chlo-legalize-to-stablehlo"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_CHLOLEGALIZETOSTABLEHLOPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers.
//===----------------------------------------------------------------------===//

template <typename FromOpTy, typename ToOpTy>
struct HloNaryElementwiseAdaptor {
  static ToOpTy createOp(FromOpTy fromOp, Type resultType,
                         ValueRange broadcastedOperands, OpBuilder& builder) {
    return builder.create<ToOpTy>(fromOp.getLoc(), resultType,
                                  broadcastedOperands);
  }
};

static std::optional<mlir::stablehlo::ComparisonDirection>
toStableHloComparisonDirection(mlir::chlo::ComparisonDirection value) {
  switch (value) {
    case mlir::chlo::ComparisonDirection::EQ:
      return mlir::stablehlo::ComparisonDirection::EQ;
    case mlir::chlo::ComparisonDirection::NE:
      return mlir::stablehlo::ComparisonDirection::NE;
    case mlir::chlo::ComparisonDirection::GE:
      return mlir::stablehlo::ComparisonDirection::GE;
    case mlir::chlo::ComparisonDirection::GT:
      return mlir::stablehlo::ComparisonDirection::GT;
    case mlir::chlo::ComparisonDirection::LE:
      return mlir::stablehlo::ComparisonDirection::LE;
    case mlir::chlo::ComparisonDirection::LT:
      return mlir::stablehlo::ComparisonDirection::LT;
  }
  return {};
}

static std::optional<mlir::stablehlo::ComparisonType> toStableHloComparisonType(
    mlir::chlo::ComparisonType value) {
  switch (value) {
    case mlir::chlo::ComparisonType::NOTYPE:
      return mlir::stablehlo::ComparisonType::NOTYPE;
    case mlir::chlo::ComparisonType::FLOAT:
      return mlir::stablehlo::ComparisonType::FLOAT;
    case mlir::chlo::ComparisonType::TOTALORDER:
      return mlir::stablehlo::ComparisonType::TOTALORDER;
    case mlir::chlo::ComparisonType::SIGNED:
      return mlir::stablehlo::ComparisonType::SIGNED;
    case mlir::chlo::ComparisonType::UNSIGNED:
      return mlir::stablehlo::ComparisonType::UNSIGNED;
  }
  return {};
}

struct HloCompareAdaptor {
  static mlir::stablehlo::CompareOp createOp(
      mlir::chlo::BroadcastCompareOp fromOp, Type resultType,
      ValueRange broadcastedOperands, OpBuilder& builder) {
    auto chloDirection = fromOp.getComparisonDirection();
    auto hloDirection = toStableHloComparisonDirection(chloDirection);
    if (!hloDirection) return nullptr;
    auto chloType =
        fromOp.getCompareType().value_or(mlir::chlo::ComparisonType::NOTYPE);
    auto hloType = toStableHloComparisonType(chloType);
    if (!hloType) return nullptr;
    auto hloTypeAttr = fromOp.getCompareType()
                           ? mlir::stablehlo::ComparisonTypeAttr::get(
                                 builder.getContext(), *hloType)
                           : nullptr;
    return builder.create<mlir::stablehlo::CompareOp>(
        fromOp.getLoc(), resultType, broadcastedOperands[0],
        broadcastedOperands[1], *hloDirection, hloTypeAttr);
  }
};

// Populate a pattern for each Broadcasting Chlo op. This requires the pattern
// to take a ChloOpTy, NonBroadcastingOpTy, and an Adaptor as templated values.
template <template <typename, typename, typename> typename Pattern,
          typename... ConstructorArgs>
static void populateForBroadcastingBinaryOp(MLIRContext* context,
                                            RewritePatternSet* patterns,
                                            ConstructorArgs&&... args) {
#define POPULATE_BCAST(ChloOp, HloOp)                                          \
  patterns                                                                     \
      ->add<Pattern<ChloOp, HloOp, HloNaryElementwiseAdaptor<ChloOp, HloOp>>>( \
          context, args...);

  POPULATE_BCAST(mlir::chlo::BroadcastAddOp, mlir::stablehlo::AddOp);
  POPULATE_BCAST(mlir::chlo::BroadcastAndOp, mlir::stablehlo::AndOp);
  POPULATE_BCAST(mlir::chlo::BroadcastAtan2Op, mlir::stablehlo::Atan2Op);
  POPULATE_BCAST(mlir::chlo::BroadcastComplexOp, mlir::stablehlo::ComplexOp);
  POPULATE_BCAST(mlir::chlo::BroadcastDivOp, mlir::stablehlo::DivOp);
  POPULATE_BCAST(mlir::chlo::BroadcastMaxOp, mlir::stablehlo::MaxOp);
  POPULATE_BCAST(mlir::chlo::BroadcastMinOp, mlir::stablehlo::MinOp);
  POPULATE_BCAST(mlir::chlo::BroadcastMulOp, mlir::stablehlo::MulOp);
  POPULATE_BCAST(mlir::chlo::BroadcastNextAfterOp, mlir::chlo::NextAfterOp);
  POPULATE_BCAST(mlir::chlo::BroadcastOrOp, mlir::stablehlo::OrOp);
  POPULATE_BCAST(mlir::chlo::BroadcastPolygammaOp, mlir::chlo::PolygammaOp);
  POPULATE_BCAST(mlir::chlo::BroadcastPowOp, mlir::stablehlo::PowOp);
  POPULATE_BCAST(mlir::chlo::BroadcastRemOp, mlir::stablehlo::RemOp);
  POPULATE_BCAST(mlir::chlo::BroadcastShiftLeftOp,
                 mlir::stablehlo::ShiftLeftOp);
  POPULATE_BCAST(mlir::chlo::BroadcastShiftRightArithmeticOp,
                 mlir::stablehlo::ShiftRightArithmeticOp);
  POPULATE_BCAST(mlir::chlo::BroadcastShiftRightLogicalOp,
                 mlir::stablehlo::ShiftRightLogicalOp);
  POPULATE_BCAST(mlir::chlo::BroadcastSubOp, mlir::stablehlo::SubtractOp);
  POPULATE_BCAST(mlir::chlo::BroadcastXorOp, mlir::stablehlo::XorOp);
  POPULATE_BCAST(mlir::chlo::BroadcastZetaOp, mlir::chlo::ZetaOp);

#undef POPULATE_BCAST

  // Broadcasting ops requiring special construction.
  patterns->add<Pattern<mlir::chlo::BroadcastCompareOp,
                        mlir::stablehlo::CompareOp, HloCompareAdaptor>>(
      context, args...);
}

static Value getConstantLikeMaxFiniteValue(OpBuilder& b, Location loc,
                                           Value val) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(
      b, loc, llvm::APFloat::getLargest(ty.getFloatSemantics()), val);
}

static Value getConstantLikeInfValue(OpBuilder& b, Location loc, Value val,
                                     bool negative) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(
      b, loc, llvm::APFloat::getInf(ty.getFloatSemantics(), negative), val);
}

static Value getConstantLikeSmallestNormalizedValue(OpBuilder& b, Location loc,
                                                    Value val) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(
      b, loc, llvm::APFloat::getSmallestNormalized(ty.getFloatSemantics()),
      val);
}

// Broadcast using numpy-style broadcasting semantics.
// This is only valid if the CHLO op has static shaped operands, and no
// explicitly specified broadcast_dimensions.
//
// Asserts that input is ranked tensor type.
Value numpyBroadcastIfNeeded(Value op, RankedTensorType opResultType,
                             PatternRewriter& rewriter) {
  RankedTensorType inputType = cast<RankedTensorType>(op.getType());
  RankedTensorType broadcastedResultType =
      opResultType.clone(inputType.getElementType());

  // No broadcasting needed if input type matches broadcasted result type.
  if (inputType == broadcastedResultType) return op;

  // broadcast dims are the last dims for numpy style broadcasting.
  int64_t inputRank = inputType.getRank();
  int64_t resultRank = opResultType.getRank();
  auto broadcastDimensions =
      llvm::to_vector(llvm::seq<int64_t>(resultRank - inputRank, resultRank));
  return stablehlo::BroadcastInDimOp::create(rewriter, op.getLoc(),
                                             broadcastedResultType, op,
                                             broadcastDimensions)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Broadcasting Patterns.
//===----------------------------------------------------------------------===//

// Converts binary ops that statically are determined to not broadcast directly
// to the corresponding stablehlo non-broadcasting op.
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertTrivialNonBroadcastBinaryOp final
    : OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ChloOpTy op, typename ChloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    auto lhsType = dyn_cast<RankedTensorType>(adaptor.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(adaptor.getRhs().getType());
    if (!lhsType || !rhsType || lhsType.getShape() != rhsType.getShape() ||
        !lhsType.hasStaticShape() || !rhsType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op,
          "expected LHS and RHS to be ranked tensors with matching shapes that "
          "are all static");

    rewriter.replaceOp(
        op, ValueRange{Adaptor::createOp(op, op.getType(),
                                         adaptor.getOperands(), rewriter)});
    return success();
  }
};

// Converts binary ops that statically determined to use default numpy
// broadcasting to simple StableHLO broadcasting ops without shape dialect.
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertTrivialNumpyBroadcastBinaryOp final
    : OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ChloOpTy op, typename ChloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    auto lhsType = dyn_cast<RankedTensorType>(adaptor.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(adaptor.getRhs().getType());
    if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op,
          "expected LHS and RHS to be ranked tensor types with static "
          "shape");

    // Rely on CHLO type inference to figure out the proper broadcasted shape.
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "expected result to be a ranked tensor type with static shape");

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto broadcastDimensions = adaptor.getBroadcastDimensions();
    if (broadcastDimensions &&
        !hlo::isLegalNumpyRankedBroadcast(lhs, rhs, *broadcastDimensions))
      return rewriter.notifyMatchFailure(
          op,
          "expected implicit broadcast_dimensions or numpy-style broadcasting");

    LLVM_DEBUG(llvm::dbgs()
               << "CHLO Decomposing " << op->getName() << " with broadcast "
               << lhsType << " x " << rhsType << " -> " << resultType << "\n");

    // If operands are static directly create stablehlo broadcasting ops.
    // Use numpy-style broadcasting with using StableHLO broadcast ops,
    // when user didn't specify broadcast_dimensions.
    auto lhsBroadcast =
        numpyBroadcastIfNeeded(adaptor.getLhs(), resultType, rewriter);
    auto rhsBroadcast =
        numpyBroadcastIfNeeded(adaptor.getRhs(), resultType, rewriter);
    auto result = Adaptor::createOp(op, resultType,
                                    {lhsBroadcast, rhsBroadcast}, rewriter);
    rewriter.replaceOp(op, {result.getResult()});
    return success();
  }
};

// Converts a binary op with ranked broadcasting operands to explicitly
// broadcast and invoke the corresponding stablehlo non-broadcasting op.
// Note that dynamic broadcasting supported by this pattern is only valid for
// "numpy" broadcasting semantics as defined here:
//   https://docs.scipy.org/doc/numpy/reference/ufuncs.html
// Specifically, this includes the following cases:
//   - Same rank broadcast (operands have the same static rank).
//   - Different-rank broadcast, either without a broadcast_dims attribute or
//     with the broadcast_dims attribute set to map to a prefix padding.
//   - Legal combinations of degenerate (1-dim) implicit broadcasting.
// The restriction on broadcast_dims derives from the definition of the
// `shape.broadcast` op, which only supports prefix-padding.
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertRankedDynamicBroadcastBinaryOp final
    : OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ChloOpTy op, typename ChloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Only support ranked operands.
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!lhsType || !rhsType || !resultType) return failure();

    // Check for "numpy"-style rank broadcast.
    auto broadcastDimensions = op.getBroadcastDimensions();
    if (broadcastDimensions && !mlir::hlo::isLegalNumpyRankedBroadcast(
                                   lhs, rhs, *broadcastDimensions)) {
      // Note: It is unclear whether the general specification of explicit
      // broadcast_dimensions on binary ops is a feature we want to carry
      // forward. While it can technically be implemented for ranked-dynamic,
      // it is incompatible with unranked inputs. If this warning is emitted
      // in real programs, it is an indication that the feature should be
      // implemented versus just falling back on the more standard definition
      // of numpy-like prefix-padding.
      op.emitWarning() << "unsupported non prefix-padded dynamic rank "
                       << "broadcast_dimensions = " << *broadcastDimensions;
      return failure();
    }

    // Compute result shape.
    Location loc = op.getLoc();

    // Insert a constraint on the shapes being broadcastable and insert all
    // future code into an assuming block reliant on the constraint.
    Value lhsShape = rewriter.create<shape::ShapeOfOp>(loc, lhs);
    Value rhsShape = rewriter.create<shape::ShapeOfOp>(loc, rhs);
    auto broadcastableCstr =
        rewriter.create<shape::CstrBroadcastableOp>(loc, lhsShape, rhsShape);
    auto assumingOp = rewriter.create<shape::AssumingOp>(
        loc, ArrayRef<Type>{resultType}, broadcastableCstr.getResult());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&assumingOp.getDoRegion());

    int64_t resultRank = std::max(lhsType.getRank(), rhsType.getRank());
    Value resultExtents =
        hlo::computeBinaryElementwiseBroadcastingResultExtents(loc, lhs, rhs,
                                                               rewriter);

    // Note that we unconditionally emit DynamicBroadcastInDim ops and let
    // downstream canonicalizations fold them away if possible. This is
    // because, in the dynamic case, there are many corner cases regarding
    // when it is safe to omit, and some of them require analysis to prove
    // properly.
    auto lhsBroadcastDimensions = llvm::to_vector(
        llvm::seq<int64_t>(resultRank - lhsType.getRank(), resultRank));
    Value broadcastedLhs =
        rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
            loc,
            RankedTensorType::get(resultType.getShape(),
                                  lhsType.getElementType()),
            lhs, resultExtents,
            rewriter.getDenseI64ArrayAttr(lhsBroadcastDimensions));
    auto rhsBroadcastDimensions = llvm::to_vector(
        llvm::seq<int64_t>(resultRank - rhsType.getRank(), resultRank));
    Value broadcastedRhs =
        rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
            loc,
            RankedTensorType::get(resultType.getShape(),
                                  rhsType.getElementType()),
            rhs, resultExtents,
            rewriter.getDenseI64ArrayAttr(rhsBroadcastDimensions));

    // And generate the final non-broadcasted binary op.
    Value finalResult = Adaptor::createOp(
        op, resultType, {broadcastedLhs, broadcastedRhs}, rewriter);
    rewriter.create<shape::AssumingYieldOp>(loc, finalResult);
    rewriter.replaceOp(op, {assumingOp.getResult(0)});
    return success();
  }
};

struct ConvertConstantLikeOp final
    : OpConversionPattern<mlir::chlo::ConstantLikeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::ConstantLikeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto resultTy = cast<ShapedType>(op.getType());

    // Unranked uses are not supported.
    if (!resultTy.hasRank()) return failure();

    // Lower to HLO constant if statically shaped.
    if (resultTy.hasStaticShape()) {
      auto complexAttr = dyn_cast<mlir::complex::NumberAttr>(op.getValue());
      auto attr = DenseElementsAttr::get(
          resultTy, complexAttr ? complexAttr : op.getValue());
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, attr);
      return success();
    }

    // Lower to broadcasted constant.
    Location loc = op.getLoc();
    Value constant =
        rewriter.create<mlir::stablehlo::ConstantOp>(loc, op.getValue());
    Value shape = rewriter.create<shape::ShapeOfOp>(loc, adaptor.getOperand());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::DynamicBroadcastInDimOp>(
        op, resultTy, constant, shape, rewriter.getDenseI64ArrayAttr({}));
    return success();
  }
};

struct ConvertSelectOp final
    : OpConversionPattern<mlir::chlo::BroadcastSelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::BroadcastSelectOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Only support ranked operands.
    Value pred = adaptor.getPred();
    Value onTrue = adaptor.getOnTrue();
    Value onFalse = adaptor.getOnFalse();
    auto predType = dyn_cast<RankedTensorType>(pred.getType());
    auto onTrueType = dyn_cast<RankedTensorType>(onTrue.getType());
    auto onFalseType = dyn_cast<RankedTensorType>(onFalse.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!predType || !onTrueType || !onFalseType || !resultType) {
      return failure();
    }

    Location loc = op.getLoc();
    Value predShape = rewriter.createOrFold<shape::ShapeOfOp>(loc, pred);
    Value onTrueShape = rewriter.createOrFold<shape::ShapeOfOp>(loc, onTrue);
    Value onFalseShape = rewriter.createOrFold<shape::ShapeOfOp>(loc, onFalse);
    int64_t resultRank = std::max(
        {predType.getRank(), onTrueType.getRank(), onFalseType.getRank()});

    Value broadcastableCstr = rewriter.createOrFold<shape::CstrBroadcastableOp>(
        loc, ValueRange{predShape, onTrueShape, onFalseShape});
    auto assumingOp = rewriter.create<shape::AssumingOp>(
        loc, ArrayRef<Type>{resultType}, broadcastableCstr);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&assumingOp.getDoRegion());

    Value resultExtents = rewriter.createOrFold<shape::BroadcastOp>(
        loc, shape::getExtentTensorType(op.getContext()),
        ValueRange{predShape, onTrueShape, onFalseShape},
        /*error=*/nullptr);
    auto shapeType =
        RankedTensorType::get({resultRank}, rewriter.getIndexType());
    resultExtents =
        rewriter.createOrFold<tensor::CastOp>(loc, shapeType, resultExtents);

    Value broadcastedPred = pred;
    // Pred has an implicit broadcast for scalars, so use that when convenient.
    if (predType.getRank() > 0) {
      auto predBroadcastDimensions = llvm::to_vector(
          llvm::seq<int64_t>(resultRank - predType.getRank(), resultRank));
      broadcastedPred =
          rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
              loc,
              RankedTensorType::get(resultType.getShape(),
                                    predType.getElementType()),
              pred, resultExtents,
              rewriter.getDenseI64ArrayAttr(predBroadcastDimensions));
    }
    auto onTrueBroadcastDimensions = llvm::to_vector(
        llvm::seq<int64_t>(resultRank - onTrueType.getRank(), resultRank));
    Value broadcastedOnTrue =
        rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
            loc,
            RankedTensorType::get(resultType.getShape(),
                                  onTrueType.getElementType()),
            onTrue, resultExtents,
            rewriter.getDenseI64ArrayAttr(onTrueBroadcastDimensions));
    auto onFalseBroadcastDimensions = llvm::to_vector(
        llvm::seq<int64_t>(resultRank - onFalseType.getRank(), resultRank));
    Value broadcastedOnFalse =
        rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
            loc,
            RankedTensorType::get(resultType.getShape(),
                                  onFalseType.getElementType()),
            onFalse, resultExtents,
            rewriter.getDenseI64ArrayAttr(onFalseBroadcastDimensions));

    // And generate the final non-broadcasted ternary op.
    Value finalResult = rewriter.create<mlir::stablehlo::SelectOp>(
        loc, resultType, broadcastedPred, broadcastedOnTrue,
        broadcastedOnFalse);
    rewriter.create<shape::AssumingYieldOp>(loc, finalResult);
    rewriter.replaceOp(op, {assumingOp.getResult(0)});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Decomposition Patterns.
//===----------------------------------------------------------------------===//

struct ConvertConstantOp final : OpConversionPattern<mlir::chlo::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, op.getValue());
    return success();
  }
};

template <typename FTy>
static Value materializeChebyshevPolynomialApproximation(
    OpBuilder& rewriter, Location loc, Value x, ArrayRef<FTy> coefficients) {
  Value b0 = getConstantLike(rewriter, loc, 0.0, x);
  Value b1 = getConstantLike(rewriter, loc, 0.0, x);
  Value b2 = getConstantLike(rewriter, loc, 0.0, x);
  for (FTy c : coefficients) {
    b2 = b1;
    b1 = b0;
    b0 = rewriter.create<mlir::stablehlo::MulOp>(loc, x.getType(), x, b1);
    b0 = rewriter.create<mlir::stablehlo::SubtractOp>(loc, x.getType(), b0, b2);
    b0 = rewriter.create<mlir::stablehlo::AddOp>(
        loc, x.getType(), b0, getConstantLike(rewriter, loc, c, x));
  }
  Value result =
      rewriter.create<mlir::stablehlo::SubtractOp>(loc, x.getType(), b0, b2);
  result = rewriter.create<mlir::stablehlo::MulOp>(
      loc, x.getType(), result, getConstantLike(rewriter, loc, 0.5, x));
  return result;
}

template <typename FTy>
static Value materializeBesselI1eApproximation(OpBuilder& rewriter,
                                               Location loc, Value x,
                                               ArrayRef<FTy> kI1eCoeffsA,
                                               ArrayRef<FTy> kI1eCoeffsB) {
  Value z = rewriter.create<mlir::stablehlo::AbsOp>(loc, x);
  Value half = getConstantLike(rewriter, loc, 0.5, x);
  Value two = getConstantLike(rewriter, loc, 2.0, x);
  Value thirtyTwo = getConstantLike(rewriter, loc, 32.0, x);
  Value eight = getConstantLike(rewriter, loc, 8.0, x);

  Value tmp = rewriter.create<mlir::stablehlo::MulOp>(loc, half, z);
  tmp = rewriter.create<mlir::stablehlo::SubtractOp>(loc, tmp, two);

  Value xLe8 = materializeChebyshevPolynomialApproximation(rewriter, loc, tmp,
                                                           kI1eCoeffsA);
  xLe8 = rewriter.create<mlir::stablehlo::MulOp>(loc, z, xLe8);

  tmp = rewriter.create<mlir::stablehlo::DivOp>(loc, thirtyTwo, z);
  tmp = rewriter.create<mlir::stablehlo::SubtractOp>(loc, tmp, two);
  Value xGt8 = materializeChebyshevPolynomialApproximation(rewriter, loc, tmp,
                                                           kI1eCoeffsB);
  xGt8 = rewriter.create<mlir::stablehlo::DivOp>(
      loc, xGt8, rewriter.create<mlir::stablehlo::SqrtOp>(loc, z));

  Value isLe8 = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, z, eight, mlir::stablehlo::ComparisonDirection::LE);

  Value select =
      rewriter.create<mlir::stablehlo::SelectOp>(loc, isLe8, xLe8, xGt8);
  return rewriter.create<mlir::stablehlo::MulOp>(
      loc, rewriter.create<mlir::stablehlo::SignOp>(loc, x), select);
}

Value materializeBesselI1eApproximationF32(OpBuilder& rewriter, Location loc,
                                           ValueRange args) {
  Value x = args.front();
  assert(cast<ShapedType>(x.getType()).getElementType().isF32() &&
         "expect f32 element type");
  const float kI1eCoeffsA[] = {
      9.38153738649577178388E-9f, -4.44505912879632808065E-8f,
      2.00329475355213526229E-7f, -8.56872026469545474066E-7f,
      3.47025130813767847674E-6f, -1.32731636560394358279E-5f,
      4.78156510755005422638E-5f, -1.61760815825896745588E-4f,
      5.12285956168575772895E-4f, -1.51357245063125314899E-3f,
      4.15642294431288815669E-3f, -1.05640848946261981558E-2f,
      2.47264490306265168283E-2f, -5.29459812080949914269E-2f,
      1.02643658689847095384E-1f, -1.76416518357834055153E-1f,
      2.52587186443633654823E-1f};

  const float kI1eCoeffsB[] = {
      -3.83538038596423702205E-9f, -2.63146884688951950684E-8f,
      -2.51223623787020892529E-7f, -3.88256480887769039346E-6f,
      -1.10588938762623716291E-4f, -9.76109749136146840777E-3f,
      7.78576235018280120474E-1f};

  return materializeBesselI1eApproximation<float>(rewriter, loc, x, kI1eCoeffsA,
                                                  kI1eCoeffsB);
}

static Value materializeBesselI1eApproximationF64(OpBuilder& rewriter,
                                                  Location loc,
                                                  ValueRange args) {
  Value x = args.front();
  assert(cast<ShapedType>(x.getType()).getElementType().isF64() &&
         "expect f64 element type");

  const double kI1eCoeffsA[] = {
      2.77791411276104639959E-18, -2.11142121435816608115E-17,
      1.55363195773620046921E-16, -1.10559694773538630805E-15,
      7.60068429473540693410E-15, -5.04218550472791168711E-14,
      3.22379336594557470981E-13, -1.98397439776494371520E-12,
      1.17361862988909016308E-11, -6.66348972350202774223E-11,
      3.62559028155211703701E-10, -1.88724975172282928790E-9,
      9.38153738649577178388E-9,  -4.44505912879632808065E-8,
      2.00329475355213526229E-7,  -8.56872026469545474066E-7,
      3.47025130813767847674E-6,  -1.32731636560394358279E-5,
      4.78156510755005422638E-5,  -1.61760815825896745588E-4,
      5.12285956168575772895E-4,  -1.51357245063125314899E-3,
      4.15642294431288815669E-3,  -1.05640848946261981558E-2,
      2.47264490306265168283E-2,  -5.29459812080949914269E-2,
      1.02643658689847095384E-1,  -1.76416518357834055153E-1,
      2.52587186443633654823E-1};

  const double kI1eCoeffsB[] = {
      7.51729631084210481353E-18,  4.41434832307170791151E-18,
      -4.65030536848935832153E-17, -3.20952592199342395980E-17,
      2.96262899764595013876E-16,  3.30820231092092828324E-16,
      -1.88035477551078244854E-15, -3.81440307243700780478E-15,
      1.04202769841288027642E-14,  4.27244001671195135429E-14,
      -2.10154184277266431302E-14, -4.08355111109219731823E-13,
      -7.19855177624590851209E-13, 2.03562854414708950722E-12,
      1.41258074366137813316E-11,  3.25260358301548823856E-11,
      -1.89749581235054123450E-11, -5.58974346219658380687E-10,
      -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
      -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
      -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
      7.78576235018280120474E-1};

  return materializeBesselI1eApproximation<double>(rewriter, loc, x,
                                                   kI1eCoeffsA, kI1eCoeffsB);
}

static Value materializeWithUpcast(ConversionPatternRewriter& rewriter,
                                   Location loc, ValueRange args,
                                   FloatType minPrecisionTy,
                                   Value callback(OpBuilder&, Location,
                                                  ValueRange)) {
  Type originalTy = getElementTypeOrSelf(args.front().getType());
  auto floatOriginalTy = dyn_cast<FloatType>(originalTy);
  bool needsUpcast =
      floatOriginalTy && floatOriginalTy.getWidth() < minPrecisionTy.getWidth();

  // Upcast arguments if necessary.
  llvm::SmallVector<Value, 2> castedArgs;
  if (needsUpcast) {
    for (Value a : args) {
      castedArgs.push_back(
          rewriter.create<mlir::stablehlo::ConvertOp>(loc, a, minPrecisionTy));
    }
    args = castedArgs;
  }

  Value result = callback(rewriter, loc, args);

  // Cast back if necessary.
  if (needsUpcast) {
    result =
        rewriter.create<mlir::stablehlo::ConvertOp>(loc, result, originalTy);
  }

  return result;
}

struct ConvertBesselI1eOp final : OpConversionPattern<mlir::chlo::BesselI1eOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::BesselI1eOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value x = adaptor.getOperand();
    Type ty = cast<ShapedType>(x.getType()).getElementType();

    // For now, we support only f64, f32, f16 and bf16.
    // See https://www.tensorflow.org/api_docs/python/tf/math/bessel_i1e
    if (!ty.isF64() && !ty.isF32() && !ty.isF16() && !ty.isBF16()) {
      return failure();
    }

    if (ty.isF64()) {
      rewriter.replaceOp(
          op, materializeBesselI1eApproximationF64(rewriter, loc, x));
      return success();
    }

    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, loc, adaptor.getOperands(),
                                  rewriter.getF32Type(),
                                  &materializeBesselI1eApproximationF32));
    return success();
  }
};

template <typename FTy>
static Value materializePolynomialApproximation(OpBuilder& rewriter,
                                                Location loc, Value x,
                                                ArrayRef<FTy> coefficients) {
  if (coefficients.empty()) return getConstantLike(rewriter, loc, 0.0, x);

  Value poly = getConstantLike(rewriter, loc, coefficients[0], x);
  for (size_t i = 1, e = coefficients.size(); i < e; ++i) {
    poly = rewriter.create<mlir::stablehlo::MulOp>(loc, x.getType(), poly, x);
    poly = rewriter.create<mlir::stablehlo::AddOp>(
        loc, x.getType(), poly,
        getConstantLike(rewriter, loc, coefficients[i], x));
  }
  return poly;
}

// Precondition is |x| >= 1. Use erf approximation, otherwise.
//
// We rely on multiple polynomial approximations for x >= 1. We pass |x| as an
// argument and derive the final approximation for all |x| >= 1.
// This implementation is based on Cephes.
static Value materializeErfcApproximationF64ForMagnituteGeOne(
    ConversionPatternRewriter& rewriter, Location loc, ValueRange args) {
  Value x = args.front();
  assert(cast<ShapedType>(x.getType()).getElementType().isF64() &&
         "expect f64 element type");
  const double kMaxlog = 7.09782712893383996843E2;
  const double kErfcPCoefficients[] = {
      2.46196981473530512524E-10, 5.64189564831068821977E-1,
      7.46321056442269912687E0,   4.86371970985681366614E1,
      1.96520832956077098242E2,   5.26445194995477358631E2,
      9.34528527171957607540E2,   1.02755188689515710272E3,
      5.57535335369399327526E2};
  const double kErfcQCoefficients[] = {
      1.00000000000000000000E0, 1.32281951154744992508E1,
      8.67072140885989742329E1, 3.54937778887819891062E2,
      9.75708501743205489753E2, 1.82390916687909736289E3,
      2.24633760818710981792E3, 1.65666309194161350182E3,
      5.57535340817727675546E2};
  const double kErfcRCoefficients[] = {
      5.64189583547755073984E-1, 1.27536670759978104416E0,
      5.01905042251180477414E0,  6.16021097993053585195E0,
      7.40974269950448939160E0,  2.97886665372100240670E0};
  const double kErfcSCoefficients[] = {
      1.00000000000000000000E0, 2.26052863220117276590E0,
      9.39603524938001434673E0, 1.20489539808096656605E1,
      1.70814450747565897222E1, 9.60896809063285878198E0,
      3.36907645100081516050E0};

  // Let z = -x^2.
  Value xSq = rewriter.create<mlir::stablehlo::MulOp>(loc, x, x);
  Value z = rewriter.create<mlir::stablehlo::NegOp>(loc, xSq);

  // Materialize polynomial approximation for x in [1, 8) as
  //   erfc(x) = exp(z) P(|x|) / Q(|x|).
  Value expZ = rewriter.create<mlir::stablehlo::ExpOp>(loc, z);
  Value absX = rewriter.create<mlir::stablehlo::AbsOp>(loc, x);
  Value polP = materializePolynomialApproximation(
      rewriter, loc, absX, llvm::ArrayRef(kErfcPCoefficients));
  Value expZMulPolyP = rewriter.create<mlir::stablehlo::MulOp>(loc, expZ, polP);
  Value polQ = materializePolynomialApproximation(
      rewriter, loc, absX, llvm::ArrayRef(kErfcQCoefficients));
  Value erfcApprox18 =
      rewriter.create<mlir::stablehlo::DivOp>(loc, expZMulPolyP, polQ);

  // Materialize polynomial approximation for x in >= 8 as
  //   erfc(x) exp(z) R(|x|) / S(|x|).
  Value polR = materializePolynomialApproximation(
      rewriter, loc, absX, llvm::ArrayRef(kErfcRCoefficients));
  Value expZMulPolyR = rewriter.create<mlir::stablehlo::MulOp>(loc, expZ, polR);
  Value polS = materializePolynomialApproximation(
      rewriter, loc, absX, llvm::ArrayRef(kErfcSCoefficients));
  Value erfcApprox8Inf =
      rewriter.create<mlir::stablehlo::DivOp>(loc, expZMulPolyR, polS);

  // Combine polynomial approximations for x >= 1.
  Value eight = getConstantLike(rewriter, loc, 8.0, x);
  Value absXLt8 = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, absX, eight, mlir::stablehlo::ComparisonDirection::LT);
  Value erfcApprox = rewriter.create<mlir::stablehlo::SelectOp>(
      loc, absXLt8, erfcApprox18, erfcApprox8Inf);

  // Clamp to prevent overflow and materialize approximation for large x as
  //   erfc(x) = 0.
  Value zLtNegMaxlog = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, z, getConstantLike(rewriter, loc, -kMaxlog, x),
      mlir::stablehlo::ComparisonDirection::LT);
  Value zero = getConstantLike(rewriter, loc, 0.0, x);
  Value erfcApproxClamped = rewriter.create<mlir::stablehlo::SelectOp>(
      loc, zLtNegMaxlog, zero, erfcApprox);

  // Derive approximation for x <= -1 as
  //   erfc(x) = 2 - erfc(-x).
  // Reuse previously materialized approximations all of which take |x| as their
  // argument.
  Value xLtZero = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, x, zero, mlir::stablehlo::ComparisonDirection::LT);
  Value two = getConstantLike(rewriter, loc, 2.0, x);
  Value twoSubErfcApproxClamped =
      rewriter.create<mlir::stablehlo::SubtractOp>(loc, two, erfcApproxClamped);
  return rewriter.create<mlir::stablehlo::SelectOp>(
      loc, xLtZero, twoSubErfcApproxClamped, erfcApproxClamped);
}

// Precondition is |x| <= 1. Use erfc approximation, otherwise.
// This implementation is based on Cephes.
static Value materializeErfApproximationF64ForMagnituteLeOne(
    ConversionPatternRewriter& rewriter, Location loc, ValueRange args) {
  Value x = args.front();
  assert(cast<ShapedType>(x.getType()).getElementType().isF64() &&
         "expect f64 element type");
  const double kErfTCoefficients[] = {
      9.60497373987051638749E0, 9.00260197203842689217E1,
      2.23200534594684319226E3, 7.00332514112805075473E3,
      5.55923013010394962768E4};
  const double kErfUCoefficients[] = {
      1.00000000000000000000E0, 3.35617141647503099647E1,
      5.21357949780152679795E2, 4.59432382970980127987E3,
      2.26290000613890934246E4, 4.92673942608635921086E4};

  // Materialize polynomial approximation for |x| <= 1 as
  //   erf(x) = x T(x^2) / U(x^2).
  Value xSq = rewriter.create<mlir::stablehlo::MulOp>(loc, x, x);
  Value polyT = materializePolynomialApproximation(
      rewriter, loc, xSq, llvm::ArrayRef(kErfTCoefficients));
  Value xMulPolyT = rewriter.create<mlir::stablehlo::MulOp>(loc, x, polyT);
  Value polyU = materializePolynomialApproximation(
      rewriter, loc, xSq, llvm::ArrayRef(kErfUCoefficients));
  return rewriter.create<mlir::stablehlo::DivOp>(loc, xMulPolyT, polyU);
}

// This implementation is based on Cephes.
static Value materializeErfApproximationF64(ConversionPatternRewriter& rewriter,
                                            Location loc, ValueRange args) {
  Value x = args.front();
  assert(cast<ShapedType>(x.getType()).getElementType().isF64() &&
         "expect f64 element type");

  // Rely on erf approximation for |x| < 1
  //   erf(x) = erf_approx(x)
  Value erfApprox =
      materializeErfApproximationF64ForMagnituteLeOne(rewriter, loc, x);

  // Rely on erfc approximation for |x| >= 1 and materialize erf as
  //   erf(x) = 1 - erfc_approx(x)
  Value one = getConstantLike(rewriter, loc, 1.0, x);
  Value erfcApprox =
      materializeErfcApproximationF64ForMagnituteGeOne(rewriter, loc, x);
  Value erfcBasedApprox =
      rewriter.create<mlir::stablehlo::SubtractOp>(loc, one, erfcApprox);

  // Materialize approximation selection based on argument.
  Value absX = rewriter.create<mlir::stablehlo::AbsOp>(loc, x);
  Value absXLtOne = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, absX, one, mlir::stablehlo::ComparisonDirection::LT);
  return rewriter.create<mlir::stablehlo::SelectOp>(loc, absXLtOne, erfApprox,
                                                    erfcBasedApprox);
}

static Value materializeErfcApproximationF64(
    ConversionPatternRewriter& rewriter, Location loc, ValueRange args) {
  Value x = args.front();
  assert(cast<ShapedType>(x.getType()).getElementType().isF64() &&
         "expect f64 element type");

  // Rely on erfc approximation for |x| >= 1
  //   erfc(x) = erfc_approx(x)
  Value erfcApprox =
      materializeErfcApproximationF64ForMagnituteGeOne(rewriter, loc, x);

  // Rely on erf approximation for |x| < 1 and materialize erfc as
  //   erfc(x) = 1 - erf_approx(x)
  Value one = getConstantLike(rewriter, loc, 1.0, x);
  Value erfApprox =
      materializeErfApproximationF64ForMagnituteLeOne(rewriter, loc, x);
  Value erfBasedApprox =
      rewriter.create<mlir::stablehlo::SubtractOp>(loc, one, erfApprox);

  // Materialize approximation selection based on argument.
  Value absX = rewriter.create<mlir::stablehlo::AbsOp>(loc, x);
  Value absXLtOne = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, absX, one, mlir::stablehlo::ComparisonDirection::LT);
  return rewriter.create<mlir::stablehlo::SelectOp>(loc, absXLtOne,
                                                    erfBasedApprox, erfcApprox);
}

// Precondition is |x| >= 1. Use erf approximation, otherwise.
//
// We rely on multiple polynomial approximations for x >= 1. We pass |x| as an
// argument and derive the final approximation for all |x| >= 1.
// This implementation is based on Cephes.
static Value materializeErfcApproximationF32ForMagnitudeGeOne(
    OpBuilder& rewriter, Location loc, ValueRange args) {
  Value x = args.front();
  assert(cast<ShapedType>(x.getType()).getElementType().isF32() &&
         "expect f32 element type");
  const double kMaxlog = 88.72283905206835;
  const float kErfcPCoefficients[] = {
      +2.326819970068386E-2f, -1.387039388740657E-1f, +3.687424674597105E-1f,
      -5.824733027278666E-1f, +6.210004621745983E-1f, -4.944515323274145E-1f,
      +3.404879937665872E-1f, -2.741127028184656E-1f, +5.638259427386472E-1f,
  };
  const float kErfcRCoefficients[] = {
      -1.047766399936249E+1f, +1.297719955372516E+1f, -7.495518717768503E+0f,
      +2.921019019210786E+0f, -1.015265279202700E+0f, +4.218463358204948E-1f,
      -2.820767439740514E-1f, +5.641895067754075E-1f,
  };

  // Let z = -x^2.
  Value xSq = rewriter.create<mlir::stablehlo::MulOp>(loc, x, x);
  Value z = rewriter.create<mlir::stablehlo::NegOp>(loc, xSq);

  // Materialize polynomial approximation for x >= 1 as
  //   erfc(x) = exp(z) 1/x P(1/x^2)   if x in [1, 2)
  //   erfc(x) = exp(z) 1/x R(1/x^2)   if x >= 2
  Value absX = rewriter.create<mlir::stablehlo::AbsOp>(loc, x);
  Value one = getConstantLike(rewriter, loc, 1.0, x);
  Value reciprocalXSq = rewriter.create<mlir::stablehlo::DivOp>(loc, one, xSq);
  Value expZ = rewriter.create<mlir::stablehlo::ExpOp>(loc, z);
  Value oneDivAbsX = rewriter.create<mlir::stablehlo::DivOp>(loc, one, absX);
  Value expZMulOneDivAbsX =
      rewriter.create<mlir::stablehlo::MulOp>(loc, expZ, oneDivAbsX);
  Value two = getConstantLike(rewriter, loc, 2.0, x);
  Value absXLtTwo = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, absX, two, mlir::stablehlo::ComparisonDirection::LT);
  Value polP = materializePolynomialApproximation(
      rewriter, loc, reciprocalXSq, llvm::ArrayRef(kErfcPCoefficients));
  Value polR = materializePolynomialApproximation(
      rewriter, loc, reciprocalXSq, llvm::ArrayRef(kErfcRCoefficients));
  Value poly =
      rewriter.create<mlir::stablehlo::SelectOp>(loc, absXLtTwo, polP, polR);
  Value erfcApprox =
      rewriter.create<mlir::stablehlo::MulOp>(loc, expZMulOneDivAbsX, poly);

  // Clamp to prevent overflow and materialize approximation for large x as
  //   erfc(x) = 0.
  Value zLtNeqMaxlog = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, z, getConstantLike(rewriter, loc, -kMaxlog, x),
      mlir::stablehlo::ComparisonDirection::LT);
  Value zero = getConstantLike(rewriter, loc, 0.0, x);
  Value erfcApproxClamped = rewriter.create<mlir::stablehlo::SelectOp>(
      loc, zLtNeqMaxlog, zero, erfcApprox);

  // Derive approximation for x <= -1 as
  //   erfc(x) = 2 - erfc(-x).
  // Reuse previously materialized approximations all of which take |x| as their
  // argument.
  Value xLtZero = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, x, zero, mlir::stablehlo::ComparisonDirection::LT);
  Value twoSubErfcApprox =
      rewriter.create<mlir::stablehlo::SubtractOp>(loc, two, erfcApproxClamped);
  return rewriter.create<mlir::stablehlo::SelectOp>(
      loc, xLtZero, twoSubErfcApprox, erfcApproxClamped);
}

// Precondition is |x| <= 1. Use erfc approximation, otherwise.
// This implementation is based on Cephes.
static Value materializeErfApproximationF32ForMagnitudeLeOne(
    OpBuilder& rewriter, Location loc, ValueRange args) {
  Value x = args.front();
  assert(cast<ShapedType>(x.getType()).getElementType().isF32() &&
         "expect f32 element type");
  const float kErfTCoefficients[] = {
      +7.853861353153693E-5f, -8.010193625184903E-4f, +5.188327685732524E-3f,
      -2.685381193529856E-2f, +1.128358514861418E-1f, -3.761262582423300E-1f,
      +1.128379165726710E+0f,
  };

  // Materialize polynomial approximation for |x| <= 1 as
  //   erf(x) = x T(x^2).
  Value xSq = rewriter.create<mlir::stablehlo::MulOp>(loc, x, x);
  Value polyT = materializePolynomialApproximation(
      rewriter, loc, xSq, llvm::ArrayRef(kErfTCoefficients));
  return rewriter.create<mlir::stablehlo::MulOp>(loc, x, polyT);
}

// This is the same approximation as used in Eigen.
static Value materializeErfApproximationF32(OpBuilder& rewriter, Location loc,
                                            ValueRange args) {
  Value x = args.front();
  assert(cast<ShapedType>(x.getType()).getElementType().isF32() &&
         "expect f32 element type");
  const float kAlpha[] = {
      -2.72614225801306e-10f, 2.77068142495902e-08f,  -2.10102402082508e-06f,
      -5.69250639462346e-05f, -7.34990630326855e-04f, -2.95459980854025e-03f,
      -1.60960333262415e-02f,
  };
  const float kBeta[] = {
      -1.45660718464996e-05f, -2.13374055278905e-04f, -1.68282697438203e-03f,
      -7.37332916720468e-03f, -1.42647390514189e-02f,
  };

  // Clamp argument between -4 and 4.
  Value lb = getConstantLike(rewriter, loc, -4.0, x);
  Value ub = getConstantLike(rewriter, loc, 4.0, x);
  x = rewriter.create<mlir::stablehlo::ClampOp>(loc, x.getType(), lb, x, ub);
  Value xSq = rewriter.create<mlir::stablehlo::MulOp>(loc, x, x);

  // Materialize polynomial approximation for x in [-4, 4] as
  //   erf(x) = x * Alpha(x^2) / Beta(x^2).
  Value alphaPoly = materializePolynomialApproximation(rewriter, loc, xSq,
                                                       llvm::ArrayRef(kAlpha));
  Value betaPoly = materializePolynomialApproximation(rewriter, loc, xSq,
                                                      llvm::ArrayRef(kBeta));
  Value xMulAlphaPoly =
      rewriter.create<mlir::stablehlo::MulOp>(loc, x, alphaPoly);
  Value erf =
      rewriter.create<mlir::stablehlo::DivOp>(loc, xMulAlphaPoly, betaPoly);
  Value lbErf = getConstantLike(rewriter, loc, -1.0, x);
  Value ubErf = getConstantLike(rewriter, loc, 1.0, x);
  return rewriter.create<mlir::stablehlo::ClampOp>(loc, erf.getType(), lbErf,
                                                   erf, ubErf);
}

static Value materializeErfcApproximationF32(OpBuilder& rewriter, Location loc,
                                             ValueRange args) {
  Value x = args.front();
  assert(cast<ShapedType>(x.getType()).getElementType().isF32() &&
         "expect f32 element type");

  // Rely on erfc approximation for |x| >= 1
  //   erfc(x) = erfc_approx(x)
  Value erfcApprox =
      materializeErfcApproximationF32ForMagnitudeGeOne(rewriter, loc, x);

  // Rely on erf approximation for |x| < 1 and materialize erfc as
  //   erfc(x) = 1 - erf_approx(x)
  Value one = getConstantLike(rewriter, loc, 1.0, x);
  Value erfApprox =
      materializeErfApproximationF32ForMagnitudeLeOne(rewriter, loc, x);
  Value erfBasedApprox =
      rewriter.create<mlir::stablehlo::SubtractOp>(loc, one, erfApprox);

  // Materialize approximation selection based on argument.
  Value absX = rewriter.create<mlir::stablehlo::AbsOp>(loc, x);
  Value absXLtOne = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, absX, one, mlir::stablehlo::ComparisonDirection::LT);
  return rewriter.create<mlir::stablehlo::SelectOp>(loc, absXLtOne,
                                                    erfBasedApprox, erfcApprox);
}

struct ConvertErfOp final : OpConversionPattern<mlir::chlo::ErfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::ErfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value x = adaptor.getOperand();
    Type ty = cast<ShapedType>(x.getType()).getElementType();

    // For now, we support only f64, f32, f16 and bf16.
    if (!ty.isF64() && !ty.isF32() && !ty.isF16() && !ty.isBF16()) {
      return failure();
    }

    if (ty.isF64()) {
      rewriter.replaceOp(op, materializeErfApproximationF64(rewriter, loc, x));
      return success();
    }

    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, loc, adaptor.getOperands(),
                                  rewriter.getF32Type(),
                                  &materializeErfApproximationF32));
    return success();
  }
};

struct ConvertErfcOp final : OpConversionPattern<mlir::chlo::ErfcOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::ErfcOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value x = adaptor.getOperand();
    Type ty = cast<ShapedType>(x.getType()).getElementType();

    // For now, we support only f64, f32, f16 and bf16.
    if (!ty.isF64() && !ty.isF32() && !ty.isF16() && !ty.isBF16()) {
      return failure();
    }

    if (ty.isF64()) {
      rewriter.replaceOp(op, materializeErfcApproximationF64(rewriter, loc, x));
      return success();
    }

    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, loc, adaptor.getOperands(),
                                  rewriter.getF32Type(),
                                  &materializeErfcApproximationF32));
    return success();
  }
};

static Value erfInv32(OpBuilder& b, Location loc, ValueRange args) {
  constexpr int kDegree = 9;
  constexpr std::array<float, 9> wLessThan5Constants = {
      2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f,
      -4.39150654e-06f, 0.00021858087f,  -0.00125372503f,
      -0.00417768164f,  0.246640727f,    1.50140941f};
  constexpr std::array<float, 9> wGreaterThan5Constants = {
      -0.000200214257f, 0.000100950558f, 0.00134934322f,
      -0.00367342844f,  0.00573950773f,  -0.0076224613f,
      0.00943887047f,   1.00167406f,     2.83297682f};

  Value x = args[0];
  // Compute logarithm of (1+arg) using log1p(arg) which is more precise than
  // log(1+arg) when arg is close to zero. For more details, see
  // https://en.cppreference.com/w/cpp/numeric/math/log1p
  Value minusXSquared = b.create<mlir::stablehlo::MulOp>(
      loc, x, b.create<mlir::stablehlo::NegOp>(loc, x));
  Value w = b.create<mlir::stablehlo::NegOp>(
      loc, b.create<mlir::stablehlo::Log1pOp>(loc, minusXSquared));

  Value lt = b.create<mlir::stablehlo::CompareOp>(
      loc, w, getConstantLike(b, loc, 5.0, x),
      mlir::stablehlo::ComparisonDirection::LT);
  auto coefficient = [&](int i) {
    return b.create<mlir::stablehlo::SelectOp>(
        loc, lt, getConstantLike(b, loc, wLessThan5Constants[i], x),
        getConstantLike(b, loc, wGreaterThan5Constants[i], x));
  };
  w = b.create<mlir::stablehlo::SelectOp>(
      loc, lt,
      b.create<mlir::stablehlo::SubtractOp>(loc, w,
                                            getConstantLike(b, loc, 2.5, x)),
      b.create<mlir::stablehlo::SubtractOp>(
          loc, b.create<mlir::stablehlo::SqrtOp>(loc, w),
          getConstantLike(b, loc, 3.0, x)));
  Value p = coefficient(0);
  for (int i = 1; i < kDegree; ++i) {
    p = b.create<mlir::stablehlo::AddOp>(
        loc, coefficient(i), b.create<mlir::stablehlo::MulOp>(loc, p, w));
  }

  // Result modulo edge cases.
  Value result = b.create<mlir::stablehlo::MulOp>(loc, p, x);

  // Handle edge cases, namely erfinv(+/-1) = +/-inf.  (The above computation is
  // indeterminate, and can give nan or -/+inf.)
  return b.create<mlir::stablehlo::SelectOp>(
      loc,
      b.create<mlir::stablehlo::CompareOp>(
          loc, b.create<mlir::stablehlo::AbsOp>(loc, x),
          getConstantLike(b, loc, 1, x),
          mlir::stablehlo::ComparisonDirection::EQ),
      b.create<mlir::stablehlo::MulOp>(
          loc, x, getConstantLikeInfValue(b, loc, x, false)),
      result);
}

static Value erfInv64(ConversionPatternRewriter& b, Location loc,
                      ValueRange args) {
  constexpr std::array<double, 23> wLessThan625Constants = {
      -3.6444120640178196996e-21, -1.685059138182016589e-19,
      1.2858480715256400167e-18,  1.115787767802518096e-17,
      -1.333171662854620906e-16,  2.0972767875968561637e-17,
      6.6376381343583238325e-15,  -4.0545662729752068639e-14,
      -8.1519341976054721522e-14, 2.6335093153082322977e-12,
      -1.2975133253453532498e-11, -5.4154120542946279317e-11,
      1.051212273321532285e-09,   -4.1126339803469836976e-09,
      -2.9070369957882005086e-08, 4.2347877827932403518e-07,
      -1.3654692000834678645e-06, -1.3882523362786468719e-05,
      0.0001867342080340571352,   -0.00074070253416626697512,
      -0.0060336708714301490533,  0.24015818242558961693,
      1.6536545626831027356};
  constexpr std::array<double, 19> wLessThan16Constants = {
      2.2137376921775787049e-09,  9.0756561938885390979e-08,
      -2.7517406297064545428e-07, 1.8239629214389227755e-08,
      1.5027403968909827627e-06,  -4.013867526981545969e-06,
      2.9234449089955446044e-06,  1.2475304481671778723e-05,
      -4.7318229009055733981e-05, 6.8284851459573175448e-05,
      2.4031110387097893999e-05,  -0.0003550375203628474796,
      0.00095328937973738049703,  -0.0016882755560235047313,
      0.0024914420961078508066,   -0.0037512085075692412107,
      0.005370914553590063617,    1.0052589676941592334,
      3.0838856104922207635,
  };
  constexpr std::array<double, 17> wGreaterThan16Constants = {
      -2.7109920616438573243e-11, -2.5556418169965252055e-10,
      1.5076572693500548083e-09,  -3.7894654401267369937e-09,
      7.6157012080783393804e-09,  -1.4960026627149240478e-08,
      2.9147953450901080826e-08,  -6.7711997758452339498e-08,
      2.2900482228026654717e-07,  -9.9298272942317002539e-07,
      4.5260625972231537039e-06,  -1.9681778105531670567e-05,
      7.5995277030017761139e-05,  -0.00021503011930044477347,
      -0.00013871931833623122026, 1.0103004648645343977,
      4.8499064014085844221,
  };

  Value x = args[0];
  // Compute logarithm of (1+arg) using log1p(arg) which is more precise than
  // log(1+arg) when arg is close to zero. For more details, see
  // https://en.cppreference.com/w/cpp/numeric/math/log1p
  Value minusXSquared = b.create<mlir::stablehlo::MulOp>(
      loc, x, b.create<mlir::stablehlo::NegOp>(loc, x));
  Value w = b.create<mlir::stablehlo::NegOp>(
      loc, b.create<mlir::stablehlo::Log1pOp>(loc, minusXSquared));

  Value lt625 = b.create<mlir::stablehlo::CompareOp>(
      loc, w, getConstantLike(b, loc, 6.25, x),
      mlir::stablehlo::ComparisonDirection::LT);
  Value lt16 = b.create<mlir::stablehlo::CompareOp>(
      loc, w, getConstantLike(b, loc, 16, x),
      mlir::stablehlo::ComparisonDirection::LT);

  auto coefficient = [&](int i) {
    Value c = getConstantLike(b, loc, wLessThan625Constants[i], x);
    if (i < 19) {
      c = b.create<mlir::stablehlo::SelectOp>(
          loc, lt625, c, getConstantLike(b, loc, wLessThan16Constants[i], x));
    }
    if (i < 17) {
      c = b.create<mlir::stablehlo::SelectOp>(
          loc, lt16, c, getConstantLike(b, loc, wGreaterThan16Constants[i], x));
    }
    return c;
  };

  Value sqrtW = b.create<mlir::stablehlo::SqrtOp>(loc, w);
  Value wMinus3125 = b.create<mlir::stablehlo::SubtractOp>(
      loc, w, getConstantLike(b, loc, 3.125, x));
  Value select2 = b.create<mlir::stablehlo::SelectOp>(
      loc, lt16, getConstantLike(b, loc, 3.25, w),
      getConstantLike(b, loc, 5.0, w));
  Value select2Result =
      b.create<mlir::stablehlo::SubtractOp>(loc, sqrtW, select2);
  w = b.create<mlir::stablehlo::SelectOp>(loc, lt625, wMinus3125,
                                          select2Result);

  Value p = coefficient(0);
  for (int i = 1; i < 17; ++i) {
    p = b.create<mlir::stablehlo::AddOp>(
        loc, coefficient(i), b.create<mlir::stablehlo::MulOp>(loc, p, w));
  }
  for (int i = 17; i < 19; ++i) {
    p = b.create<mlir::stablehlo::SelectOp>(
        loc, lt16,
        b.create<mlir::stablehlo::AddOp>(
            loc, coefficient(i), b.create<mlir::stablehlo::MulOp>(loc, p, w)),
        p);
  }
  for (int i = 19; i < 23; ++i) {
    p = b.create<mlir::stablehlo::SelectOp>(
        loc, lt625,
        b.create<mlir::stablehlo::AddOp>(
            loc, coefficient(i), b.create<mlir::stablehlo::MulOp>(loc, p, w)),
        p);
  }

  // Result modulo edge cases.
  Value result = b.create<mlir::stablehlo::MulOp>(loc, p, x);

  // Handle edge cases, namely erfinv(+/-1) = +/-inf.  (The above computation is
  // indeterminate, and can give nan or -/+inf.)
  return b.create<mlir::stablehlo::SelectOp>(
      loc,
      b.create<mlir::stablehlo::CompareOp>(
          loc, b.create<mlir::stablehlo::AbsOp>(loc, x),
          getConstantLike(b, loc, 1, x),
          mlir::stablehlo::ComparisonDirection::EQ),
      b.create<mlir::stablehlo::MulOp>(
          loc, x, getConstantLikeInfValue(b, loc, x, false)),
      result);
}

struct ConvertErfInvOp final : OpConversionPattern<mlir::chlo::ErfInvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::ErfInvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    if (op.getType().getElementType().isF64()) {
      rewriter.replaceOp(op, erfInv64(rewriter, loc, adaptor.getOperands()));
      return success();
    }
    FloatType minPrecisionTy = rewriter.getF32Type();
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, loc, adaptor.getOperands(),
                                  minPrecisionTy, &erfInv32));
    return success();
  }
};

// Coefficients for the Lanczos approximation of the gamma function. The
// coefficients are uniquely determined by the choice of g and n (kLanczosGamma
// and kLanczosCoefficients.size() + 1). The coefficients below correspond to
// [7, 9]. [5, 7], [7, 9], [9, 10], and [607/128.0, 15] were evaluated and
// [7, 9] seemed to be the least sensitive to the quality of the log function.
// In particular, [5, 7] is the only choice where -1.5e-5 <= lgamma(2) <= 1.5e-5
// for a particularly inaccurate log function.
constexpr double kLanczosGamma = 7;  // aka g
constexpr double kBaseLanczosCoeff = 0.99999999999980993227684700473478;
constexpr std::array<double, 8> kLanczosCoefficients = {
    676.520368121885098567009190444019, -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,   -176.61502916214059906584551354,
    12.507343278686904814458936853,     -0.13857109526572011689554707,
    9.984369578019570859563e-6,         1.50563273514931155834e-7};

}  // namespace

// Compute the Lgamma function using Lanczos' approximation from "A Precision
// Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
// series B. Vol. 1:
//   lgamma(z + 1) = (log(2) + log(pi)) / 2
//                     + (z + 1/2) * log(t(z))
//                     - t(z) + log(a(z))
//   with   t(z) = z + kLanczosGamma + 1/2
//          a(z) = kBaseLanczosCoeff
//                   + sum(k = 1, n, kLanczosCoefficients[i] / (z + k))
Value materializeLgamma(OpBuilder& rewriter, Location loc, ValueRange args) {
  // If the input is less than 0.5 use Euler's reflection formula.
  //   gamma(x) = pi / (sin(pi * x) * gamma(1 - x))
  // Let z be
  //   z = -x      if x < 1/2
  //   z = x - 1   otheriwse
  Value x = args.front();
  Value half = getConstantLike(rewriter, loc, 0.5, x);
  Value needToReflect = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, x, half, mlir::stablehlo::ComparisonDirection::LT);
  Value negX = rewriter.create<mlir::stablehlo::NegOp>(loc, x);
  Value one = getConstantLike(rewriter, loc, 1, x);
  Value xSubOne = rewriter.create<mlir::stablehlo::SubtractOp>(loc, x, one);
  Value z = rewriter.create<mlir::stablehlo::SelectOp>(loc, needToReflect, negX,
                                                       xSubOne);

  // Materialize
  //   a(z) = kBaseLanczosCoeff
  //            + sum(k = 1, n, kLanczosCoefficients[i] / (z + k))
  Value a = getConstantLike(rewriter, loc, kBaseLanczosCoeff, x);
  for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    Value coeff = getConstantLike(rewriter, loc, kLanczosCoefficients[i], x);
    Value oneBasedIndex = getConstantLike(rewriter, loc, i + 1, x);
    Value quotient = rewriter.create<mlir::stablehlo::DivOp>(
        loc, coeff,
        rewriter.create<mlir::stablehlo::AddOp>(loc, z, oneBasedIndex));
    a = rewriter.create<mlir::stablehlo::AddOp>(loc, a, quotient);
  }

  // To improve accuracy on platforms with less-precise log implementations,
  // compute log(kLanczosGamma + 1/2) at compile time and use log1p on the
  // device.
  // Materialize as
  //   log(t) = log(kLanczosGamma + 1/2 + z)
  //          = log(kLanczosGamma + 1/2) + log1p(z / (kLanczosGamma + 1/2)).
  Value lanczosPlusHalf =
      getConstantLike(rewriter, loc, kLanczosGamma + 0.5, x);
  Value t = rewriter.create<mlir::stablehlo::AddOp>(loc, lanczosPlusHalf, z);
  Value logTerm =
      getConstantLike(rewriter, loc, std::log(kLanczosGamma + 0.5), x);
  Value log1pTerm = rewriter.create<mlir::stablehlo::Log1pOp>(
      loc, rewriter.create<mlir::stablehlo::DivOp>(loc, z, lanczosPlusHalf));
  Value logT = rewriter.create<mlir::stablehlo::AddOp>(loc, logTerm, log1pTerm);

  // Note that t(z) may be large and we need to be careful not to overflow to
  // infinity in the relevant term
  //   r = (z + 1/2) * log(t(z)) - t(z).
  // Therefore, we compute this as
  //   r = (z + 1/2 - t(z) / log(t(z))) * log(t(z)).
  Value tDivLogT = rewriter.create<mlir::stablehlo::DivOp>(loc, t, logT);
  Value sum = rewriter.create<mlir::stablehlo::SubtractOp>(
      loc, rewriter.create<mlir::stablehlo::AddOp>(loc, z, half), tDivLogT);
  Value r = rewriter.create<mlir::stablehlo::MulOp>(loc, sum, logT);

  // Compute the final result (modulo reflection) as
  //   lgamma(z + 1) = (log(2) + log(pi)) / 2 + r + log(a(z)).
  Value logA = rewriter.create<mlir::stablehlo::LogOp>(loc, a);
  Value lgamma = rewriter.create<mlir::stablehlo::AddOp>(
      loc,
      rewriter.create<mlir::stablehlo::AddOp>(
          loc,
          getConstantLike(rewriter, loc, (std::log(2) + std::log(M_PI)) / 2, x),
          r),
      logA);

  // Compute the reflected value for x < 0.5 as
  //   lgamma(x) = log(pi) - lgamma(1-x) - log(abs(sin(pi * x))).
  //
  // The abs is needed because lgamma is the log of the absolute value of the
  // gamma function.
  //
  // We have to be careful when computing the final term above. gamma(x) goes
  // to +/-inf at every integer x < 0, and this is controlled by the sin(pi * x)
  // term. The slope is large, so precision is particularly important.
  //
  // Because abs(sin(pi * x)) has period of 1 we can equivalently use
  // abs(sin(pi * frac(x))) where frac(x) is the fractional part of x. This is
  // more numerically accurate: It doesn't overflow to inf like pi * x would and
  // if x is an integer it evaluates to exactly 0 which is important because we
  // then take the log of this value, and log(0) is inf.
  //
  // We don't have a frac(x) primitive in HLO and computing it is tricky, but
  // because abs(sin(pi * x)) = abs(sin(pi * abs(x))), it's good enough for our
  // purposes to use abs(frac(x)) = abs(x) - floor(abs(x)).
  //
  // Furthermore, pi * abs(frac(x)) loses precision when abs(frac(x)) is close
  // to 1. To remedy this, we can use the fact that sin(pi * x) in the domain
  // [0, 1] is symmetric across the line Y=0.5.
  //

  // Convert values of abs_frac > 0.5 to (1 - abs_frac) to improve precision of
  // pi * abs_frac for values of abs_frac close to 1.
  Value abs = rewriter.create<mlir::stablehlo::AbsOp>(loc, x);
  Value absFrac = rewriter.create<mlir::stablehlo::SubtractOp>(
      loc, abs, rewriter.create<mlir::stablehlo::FloorOp>(loc, abs));
  Value reduceAbsFrac = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, half, absFrac, mlir::stablehlo::ComparisonDirection::LT);
  absFrac = rewriter.create<mlir::stablehlo::SelectOp>(
      loc, reduceAbsFrac,
      rewriter.create<mlir::stablehlo::SubtractOp>(loc, one, absFrac), absFrac);

  // Materialize reflection.
  Value reflectionDenom = rewriter.create<mlir::stablehlo::LogOp>(
      loc,
      rewriter.create<mlir::stablehlo::SineOp>(
          loc, rewriter.create<mlir::stablehlo::MulOp>(
                   loc, getConstantLike(rewriter, loc, M_PI, x), absFrac)));
  Value lgammaReflection = rewriter.create<mlir::stablehlo::SubtractOp>(
      loc,
      rewriter.create<mlir::stablehlo::SubtractOp>(
          loc, getConstantLike(rewriter, loc, std::log(M_PI), x),
          reflectionDenom),
      lgamma);

  // Avoid computing -inf - inf, which is nan. If reflection_denom is +/-inf,
  // then it "wins" and the result is +/-inf.
  Value finiteReflectionDenom =
      rewriter.create<mlir::stablehlo::IsFiniteOp>(loc, reflectionDenom);
  Value negReflectionDenom =
      rewriter.create<mlir::stablehlo::NegOp>(loc, reflectionDenom);
  lgammaReflection = rewriter.create<mlir::stablehlo::SelectOp>(
      loc, finiteReflectionDenom, lgammaReflection, negReflectionDenom);

  // Select whether or not to rely on the reflection.
  lgamma = rewriter.create<mlir::stablehlo::SelectOp>(loc, needToReflect,
                                                      lgammaReflection, lgamma);

  // Materialize +/-inf behavior as
  //   lgamma(+/-inf) = +inf.
  Value xIsInf = rewriter.create<chlo::IsInfOp>(loc, x);
  return rewriter.create<mlir::stablehlo::SelectOp>(
      loc, xIsInf,
      getConstantLikeInfValue(rewriter, loc, x, /*negative=*/false), lgamma);
}

namespace {

// Express `cosh` as
//   cosh(x) = (e^x + e^-x) / 2
//           = e^(x + log(1/2)) + e^(-x + log(1/2))
//
// The second formulation avoids overflowing when e^x = inf but (e^x)/2 is not.
//
// This incorrectly overflows to inf for two f32 input values, namely
// +/-89.4159851, due to rounding error when computing x +/- log(1/2).  The
// correct answer of 3.40281961e+38 (0x7f7fffec) is very close to max-float, so
// we deem this acceptable.
static Value materializeCoshApproximation(OpBuilder& rewriter, Location loc,
                                          ValueRange operands) {
  mlir::chlo::CoshOp::Adaptor transformed(operands);
  Value x = transformed.getOperand();

  Value logOneHalf = rewriter.create<mlir::stablehlo::LogOp>(
      loc, getConstantLike(rewriter, loc, 0.5, x));
  Value expAdd = rewriter.create<mlir::stablehlo::ExpOp>(
      loc, rewriter.create<mlir::stablehlo::AddOp>(loc, x, logOneHalf));
  Value expSub = rewriter.create<mlir::stablehlo::ExpOp>(
      loc, rewriter.create<mlir::stablehlo::SubtractOp>(loc, logOneHalf, x));
  return rewriter.create<mlir::stablehlo::AddOp>(loc, expAdd, expSub);
}

struct ConvertCoshOp final : OpConversionPattern<mlir::chlo::CoshOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::CoshOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, op.getLoc(), adaptor.getOperands(),
                                  rewriter.getF32Type(),
                                  &materializeCoshApproximation));
    return success();
  }
};

}  // namespace

// Compute the Digamma function using Lanczos' approximation from "A Precision
// Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
// series B. Vol. 1:
//   digamma(z + 1) = log(t(z)) + a'(z) / a(z) - kLanczosGamma / t(z)
//   with   t(z) = z + kLanczosGamma + 1/2
//          a(z) = kBaseLanczosCoeff
//                   + sum(k = 1, n, kLanczosCoefficients[i] / (z + k))
//          a'(z) = - sum(k = 1, n, kLanczosCoefficients[i] / (z + k) / (z + k))
Value materializeDigamma(OpBuilder& rewriter, Location loc, ValueRange args) {
  // If the input is less than 0.5 use Euler's reflection formula.
  //   digamma(x) = digamma(1 - x) - pi * cot(pi * x)
  // Let z be
  //   z = -x      if x < 1/2
  //   z = x - 1   otheriwse
  Value x = args.front();
  Value half = getConstantLike(rewriter, loc, 0.5, x);
  Value needToReflect = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, x, half, mlir::stablehlo::ComparisonDirection::LT);
  Value negX = rewriter.create<mlir::stablehlo::NegOp>(loc, x);
  Value one = getConstantLike(rewriter, loc, 1, x);
  Value xSubOne = rewriter.create<mlir::stablehlo::SubtractOp>(loc, x, one);
  Value z = rewriter.create<mlir::stablehlo::SelectOp>(loc, needToReflect, negX,
                                                       xSubOne);

  // Materialize
  //   a(z) = kBaseLanczosCoeff
  //            + sum(k = 1, n, kLanczosCoefficients[i] / (z + k))
  //   a'(z) = - sum(k = 1, n, kLanczosCoefficients[i] / (z + k) / (z + k))
  Value zero = getConstantLike(rewriter, loc, 0.0, x);
  Value a = getConstantLike(rewriter, loc, kBaseLanczosCoeff, x);
  Value aPrime = zero;
  for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    Value coeff = getConstantLike(rewriter, loc, kLanczosCoefficients[i], x);
    Value oneBasedIndex = getConstantLike(rewriter, loc, i + 1, x);
    Value zTerm =
        rewriter.create<mlir::stablehlo::AddOp>(loc, z, oneBasedIndex);
    aPrime = rewriter.create<mlir::stablehlo::SubtractOp>(
        loc, aPrime,
        rewriter.create<mlir::stablehlo::DivOp>(
            loc, coeff,
            rewriter.create<mlir::stablehlo::MulOp>(loc, zTerm, zTerm)));
    a = rewriter.create<mlir::stablehlo::AddOp>(
        loc, a, rewriter.create<mlir::stablehlo::DivOp>(loc, coeff, zTerm));
  }

  // To improve accuracy on platforms with less-precise log implementations,
  // compute log(kLanczosGamma + 1/2) at compile time and use log1p on the
  // device.
  // Materialize as
  //   log(t) = log(kLanczosGamma + 1/2 + z)
  //          = log(kLanczosGamma + 1/2) + log1p(z / (kLanczosGamma + 1/2)).
  Value lanczosPlusHalf =
      getConstantLike(rewriter, loc, kLanczosGamma + 0.5, x);
  Value t = rewriter.create<mlir::stablehlo::AddOp>(loc, lanczosPlusHalf, z);
  Value logTerm =
      getConstantLike(rewriter, loc, std::log(kLanczosGamma + 0.5), x);
  Value log1pTerm = rewriter.create<mlir::stablehlo::Log1pOp>(
      loc, rewriter.create<mlir::stablehlo::DivOp>(loc, z, lanczosPlusHalf));
  Value logT = rewriter.create<mlir::stablehlo::AddOp>(loc, logTerm, log1pTerm);

  // Materialize the final result (modulo reflection) as
  //   digamma(z + 1) = log(t(z)) + a'(z) / a(z) - kLanczosGamma / t(z).
  Value aPrimeDivA = rewriter.create<mlir::stablehlo::DivOp>(loc, aPrime, a);
  Value lanczosGammaDivT = rewriter.create<mlir::stablehlo::DivOp>(
      loc, getConstantLike(rewriter, loc, kLanczosGamma, x), t);
  Value digamma = rewriter.create<mlir::stablehlo::SubtractOp>(
      loc, rewriter.create<mlir::stablehlo::AddOp>(loc, logT, aPrimeDivA),
      lanczosGammaDivT);

  // We need to be careful how we compute cot(pi * input) below: For
  // near-integral arguments, pi * input can lose precision.
  //
  // Input is already known to be less than 0.5 (otherwise we don't have to
  // reflect). We shift values smaller than -0.5 into the range [-0.5, 0.5] to
  // increase precision of pi * x and the resulting cotangent.
  Value reducedX = rewriter.create<mlir::stablehlo::AddOp>(
      loc, x,
      rewriter.create<mlir::stablehlo::AbsOp>(
          loc, rewriter.create<mlir::stablehlo::FloorOp>(
                   loc, rewriter.create<mlir::stablehlo::AddOp>(
                            loc, x, getConstantLike(rewriter, loc, 0.5, x)))));

  // Materialize reflection for inputs less than 0.5 as
  //   digamma(x) = digamma(1 - x) - pi * cot(pi * x)
  //              = digamma(1 - x) - pi * cos(pi * x) / sin(pi * x)
  Value pi = getConstantLike(rewriter, loc, M_PI, x);
  Value piMulReducedX =
      rewriter.create<mlir::stablehlo::MulOp>(loc, pi, reducedX);
  Value cos = rewriter.create<mlir::stablehlo::CosineOp>(loc, piMulReducedX);
  Value sin = rewriter.create<mlir::stablehlo::SineOp>(loc, piMulReducedX);
  Value reflection = rewriter.create<mlir::stablehlo::SubtractOp>(
      loc, digamma,
      rewriter.create<mlir::stablehlo::DivOp>(
          loc, rewriter.create<mlir::stablehlo::MulOp>(loc, pi, cos), sin));

  // Select whether or not to rely on the reflection.
  digamma = rewriter.create<mlir::stablehlo::SelectOp>(loc, needToReflect,
                                                       reflection, digamma);

  // Digamma has poles at negative integers and zero; return nan for those.
  Value isLeZero = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, x, zero, mlir::stablehlo::ComparisonDirection::LE);
  Value isInt = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, x, rewriter.create<mlir::stablehlo::FloorOp>(loc, x),
      mlir::stablehlo::ComparisonDirection::EQ);
  Value isPole = rewriter.create<mlir::stablehlo::AndOp>(loc, isLeZero, isInt);
  return rewriter.create<mlir::stablehlo::SelectOp>(
      loc, isPole,
      getConstantLike(rewriter, loc, std::numeric_limits<double>::quiet_NaN(),
                      x),
      digamma);
}

namespace {

static Value getConstantLikeSmallestFiniteValue(OpBuilder& b, Location loc,
                                                Value val) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(
      b, loc, llvm::APFloat::getSmallest(ty.getFloatSemantics()), val);
}

static Value materializeZeta(OpBuilder& rewriter, Location loc,
                             ValueRange args) {
  // Implementation ported from:
  // https://github.com/openxla/xla/blob/7a067a7b88d2ffb15b1dc5e3c06f701a15f0391d/xla/client/lib/math.cc#L1912-L1917
  // Reference: Johansson, Fredrik.
  // "Rigorous high-precision computation of the Hurwitz zeta function and its
  // derivatives." Numerical Algorithms 69.2 (2015): 253-270.
  // https://arxiv.org/abs/1309.2877 - formula (5)
  // Notation is more or less kept as a reference to the whitepaper.
  assert(args.size() == 2);
  Value x = args[0];
  Value q = args[1];

  static constexpr auto kTerms = 12;
  static constexpr auto kIters = 9;
  static constexpr auto kTwoTermsMinusOne = 2 * kTerms - 1;
  static constexpr auto kZetaCoeffs = std::array<double, kTerms>{
      -7.1661652561756670113e18,
      1.8152105401943546773e17,
      -4.5979787224074726105e15,
      1.1646782814350067249e14,
      -2.950130727918164224e12,
      7.47242496e10,
      -1.8924375803183791606e9,
      47900160.0,
      -1209600.0,
      30240.0,
      -720.0,
      12.0,
  };

  // For speed we'll always use 9 iterations for the initial series estimate,
  // and a 12 term expansion for the Euler-Maclaurin formula.
  Value zero = getConstantLike(rewriter, loc, 0.0, q);
  Value one = getConstantLike(rewriter, loc, 1.0, q);
  Value acc = q;
  Value qNegPower = zero;
  Value negX = rewriter.create<NegOp>(loc, x);
  Value powerSum = rewriter.create<PowOp>(loc, q, negX);
  for (int i = 0; i < kIters; ++i) {
    acc = rewriter.create<AddOp>(loc, acc, one);
    qNegPower = rewriter.create<PowOp>(loc, acc, negX);
    powerSum = rewriter.create<AddOp>(loc, powerSum, qNegPower);
  }
  acc = rewriter.create<AddOp>(loc, acc, one);
  qNegPower = rewriter.create<PowOp>(loc, acc, negX);
  Value oneLikeX = getConstantLike(rewriter, loc, 1.0, x);
  Value correctionEulerMaclaurin =
      rewriter.create<DivOp>(loc, rewriter.create<MulOp>(loc, qNegPower, acc),
                             rewriter.create<SubtractOp>(loc, x, oneLikeX));

  // Manual reciprocal of the square root as RsqrtOp produces different results
  Value rsqrtAcc =
      rewriter.create<DivOp>(loc, one, rewriter.create<MulOp>(loc, acc, acc));

  // Use Horner's rule for this.
  // Note this differs from Cephes which does a 'naive' polynomial evaluation.
  // Using Horner's rule allows to avoid some NaN's and Infs from happening,
  // resulting in more numerically stable code.
  Value hornerSum = zero;
  Value hornerProduct = one;

  for (int i = 0; i < kTerms - 1; ++i) {
    Value factorLhs = rewriter.create<AddOp>(
        loc, x,
        getConstantLike(rewriter, loc, kTwoTermsMinusOne - 1 - 2 * i, x));
    Value factorRhs = rewriter.create<AddOp>(
        loc, x,
        getConstantLike(rewriter, loc, kTwoTermsMinusOne - 2 - 2 * i, x));
    hornerProduct = rewriter.create<MulOp>(loc, factorLhs, factorRhs);
    hornerSum = rewriter.create<MulOp>(
        loc, hornerProduct,
        rewriter.create<MulOp>(
            loc, rsqrtAcc,
            rewriter.create<AddOp>(
                loc, hornerSum,
                getConstantLike(rewriter, loc, 1. / kZetaCoeffs[i], acc))));
  }
  Value zeroPointFiveLikeQNegPower =
      getConstantLike(rewriter, loc, .5, qNegPower);
  Value xDivAcc = rewriter.create<DivOp>(loc, x, acc);
  Value bernoulliTailTerm = rewriter.create<MulOp>(
      loc, qNegPower,
      rewriter.create<AddOp>(
          loc, zeroPointFiveLikeQNegPower,
          rewriter.create<MulOp>(
              loc, xDivAcc,
              rewriter.create<AddOp>(
                  loc,
                  getConstantLike(rewriter, loc, 1. / kZetaCoeffs[kTerms - 1],
                                  acc),
                  hornerSum))));
  Value accurateResult = rewriter.create<AddOp>(
      loc, rewriter.create<AddOp>(loc, powerSum, correctionEulerMaclaurin),
      bernoulliTailTerm);

  // Use the initial zeta sum without the correction term coming
  // from Euler-Maclaurin if it is accurate enough.
  Value absQNegPower = rewriter.create<AbsOp>(loc, qNegPower);
  Value absPowerSum = rewriter.create<AbsOp>(loc, powerSum);
  Value output = rewriter.create<SelectOp>(
      loc,
      rewriter.create<CompareOp>(
          loc, absQNegPower,
          rewriter.create<MulOp>(
              loc, absPowerSum,
              getConstantLikeSmallestFiniteValue(rewriter, loc, acc)),
          ComparisonDirection::LT),
      powerSum, accurateResult);

  // Function is not defined for x < 1.
  Value nan = getConstantLike(rewriter, loc,
                              std::numeric_limits<double>::quiet_NaN(), x);
  output = rewriter.create<SelectOp>(
      loc,
      rewriter.create<CompareOp>(loc, x, oneLikeX, ComparisonDirection::LT),
      nan, output);

  // For q <= 0, x must be an integer.
  Value qLeZero =
      rewriter.create<CompareOp>(loc, q, zero, ComparisonDirection::LE);
  Value xNotInt = rewriter.create<CompareOp>(
      loc, x, rewriter.create<FloorOp>(loc, x), ComparisonDirection::NE);
  Value xDomainError = rewriter.create<AndOp>(loc, qLeZero, xNotInt);
  output = rewriter.create<SelectOp>(loc, xDomainError, nan, output);

  // For all integer q <= 0, zeta has a pole. The limit is only defined as
  // +inf if x is and even integer.
  Value inf = getConstantLike(rewriter, loc,
                              std::numeric_limits<double>::infinity(), x);
  Value qIsInt = rewriter.create<CompareOp>(
      loc, q, rewriter.create<FloorOp>(loc, q), ComparisonDirection::EQ);
  Value atPole = rewriter.create<AndOp>(loc, qLeZero, qIsInt);
  Value two = getConstantLike(rewriter, loc, 2.0, x);
  Value xIsInt = rewriter.create<CompareOp>(
      loc, x, rewriter.create<FloorOp>(loc, x), ComparisonDirection::EQ);
  Value xIsEven = rewriter.create<CompareOp>(
      loc, rewriter.create<RemOp>(loc, x, two), zero, ComparisonDirection::EQ);
  Value xIsEvenInt = rewriter.create<AndOp>(loc, xIsInt, xIsEven);
  output = rewriter.create<SelectOp>(
      loc, atPole, rewriter.create<SelectOp>(loc, xIsEvenInt, inf, nan),
      output);

  // For x = 1, this is the harmonic series and diverges.
  output = rewriter.create<SelectOp>(
      loc, rewriter.create<CompareOp>(loc, x, one, ComparisonDirection::EQ),
      inf, output);

  return output;
}

}  // namespace

Value materializePolygamma(OpBuilder& rewriter, Location loc, ValueRange args) {
  mlir::chlo::PolygammaOp::Adaptor transformed(args);
  Value n = transformed.getN();
  Value x = transformed.getX();

  // Handle integer n > 0.
  Value one = getConstantLike(rewriter, loc, 1.0, x);
  Value two = getConstantLike(rewriter, loc, 2.0, x);
  Value sign = rewriter.create<mlir::stablehlo::SubtractOp>(
      loc,
      rewriter.create<mlir::stablehlo::MulOp>(
          loc, two, rewriter.create<mlir::stablehlo::RemOp>(loc, n, two)),
      one);
  Value nPlusOne = rewriter.create<mlir::stablehlo::AddOp>(loc, n, one);
  Value expLgammaNp1 = rewriter.create<mlir::stablehlo::ExpOp>(
      loc, rewriter.create<chlo::LgammaOp>(loc, nPlusOne));
  Value zeta = rewriter.create<chlo::ZetaOp>(loc, nPlusOne, x);
  Value result = rewriter.create<mlir::stablehlo::MulOp>(
      loc, rewriter.create<mlir::stablehlo::MulOp>(loc, sign, expLgammaNp1),
      zeta);

  // Handle n = 0.
  Value zero = getConstantLike(rewriter, loc, 0.0, x);
  Value nEqZero = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, n, zero, mlir::stablehlo::ComparisonDirection::EQ);
  result = rewriter.create<mlir::stablehlo::SelectOp>(
      loc, nEqZero, rewriter.create<chlo::DigammaOp>(loc, x), result);

  // Check that n is a natural number. Return nan, otherwise.
  Value nonInt = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, n, rewriter.create<mlir::stablehlo::FloorOp>(loc, n),
      mlir::stablehlo::ComparisonDirection::NE);
  Value negative = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, n, zero, mlir::stablehlo::ComparisonDirection::LT);
  Value nonNatural =
      rewriter.create<mlir::stablehlo::OrOp>(loc, nonInt, negative);
  return rewriter.create<mlir::stablehlo::SelectOp>(
      loc, nonNatural,
      getConstantLike(rewriter, loc, std::numeric_limits<double>::quiet_NaN(),
                      x),
      result);
}

namespace {

struct ConvertLgammaOp final : OpConversionPattern<mlir::chlo::LgammaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::LgammaOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FloatType minPrecisionTy = rewriter.getF32Type();
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, op.getLoc(), adaptor.getOperands(),
                                  minPrecisionTy, &materializeLgamma));
    return success();
  }
};

struct ConvertDigammaOp final : OpConversionPattern<mlir::chlo::DigammaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::DigammaOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    FloatType minPrecisionTy = rewriter.getF32Type();
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, op.getLoc(), adaptor.getOperands(),
                                  minPrecisionTy, &materializeDigamma));
    return success();
  }
};

static Value materializeNextAfter(ConversionPatternRewriter& rewriter,
                                  Location loc, ValueRange operands) {
  mlir::chlo::NextAfterOp::Adaptor transformed(operands);
  Value x = transformed.getX();
  Value y = transformed.getY();
  auto resultTy = cast<ShapedType>(x.getType());
  auto bitwidth = resultTy.getElementType().getIntOrFloatBitWidth();
  mlir::ImplicitLocOpBuilder b(loc, rewriter);
  Type intTy = resultTy.clone(b.getIntegerType(bitwidth));
  auto xAsInt = b.create<mlir::stablehlo::BitcastConvertOp>(intTy, x);
  auto yAsInt = b.create<mlir::stablehlo::BitcastConvertOp>(intTy, y);

  // The result is NaN if either "x" or "y" are NaN.
  auto xIsNan = b.create<mlir::stablehlo::CompareOp>(
      x, x, mlir::stablehlo::ComparisonDirection::NE);
  auto yIsNan = b.create<mlir::stablehlo::CompareOp>(
      y, y, mlir::stablehlo::ComparisonDirection::NE);
  auto nanInput = b.create<mlir::stablehlo::OrOp>(xIsNan, yIsNan);
  auto resultForNan = getConstantLike(
      rewriter, loc, std::numeric_limits<double>::quiet_NaN(), x);
  auto resultForNanAsInt =
      b.create<mlir::stablehlo::BitcastConvertOp>(intTy, resultForNan);

  // The sign bit is the MSB.
  const int64_t signBit = int64_t{1} << (bitwidth - 1);
  // Discard the sign bit to make the result non-negative.
  Value signMask = getConstantLike(rewriter, loc, signBit, xAsInt);
  Value negatedSignMask = getConstantLike(rewriter, loc, ~signBit, xAsInt);
  auto xAbs = b.create<mlir::stablehlo::AndOp>(xAsInt, negatedSignMask);
  auto yAbs = b.create<mlir::stablehlo::AndOp>(yAsInt, negatedSignMask);

  // When both "x" and "y" are equal, the result is "y".
  auto xAndYAreEqual = b.create<mlir::stablehlo::CompareOp>(
      x, y, mlir::stablehlo::ComparisonDirection::EQ);
  auto resultForEqual = yAsInt;

  // When both "x" and "y" are 0, the result is "y". This is a separate case
  // from above because "x" and "y" might have a different sign.
  Value zero = getConstantLike(rewriter, loc, 0, xAsInt);
  auto xIsZero = b.create<mlir::stablehlo::CompareOp>(
      xAbs, zero, mlir::stablehlo::ComparisonDirection::EQ);
  auto yIsZero = b.create<mlir::stablehlo::CompareOp>(
      yAbs, zero, mlir::stablehlo::ComparisonDirection::EQ);
  auto resultForBothZero = yAsInt;

  auto xSign = b.create<mlir::stablehlo::AndOp>(xAsInt, signMask);
  auto ySign = b.create<mlir::stablehlo::AndOp>(yAsInt, signMask);

  // If from == 0 && to != 0, we need to return the smallest subnormal number
  // signed like "to".
  Value one = getConstantLike(rewriter, loc, 1, xAsInt);
  auto resultForXZeroYNonZero = b.create<mlir::stablehlo::OrOp>(ySign, one);

  // If the sign of "x" and "y" disagree:
  // - we need to make the magnitude of "from" smaller so that it is closer to
  //   zero.
  //
  // Otherwise the signs agree:
  // - "x" with a magnitude larger than "y" means we need to make the magnitude
  //   smaller.
  // - "x" with a magnitude smaller than "y" means we need to make the magnitude
  //   larger.
  auto signsDisagree = b.create<mlir::stablehlo::CompareOp>(
      xSign, ySign, mlir::stablehlo::ComparisonDirection::NE);
  auto xMagnitudeLargerThanY = b.create<mlir::stablehlo::CompareOp>(
      xAbs, yAbs, mlir::stablehlo::ComparisonDirection::GT);
  auto resultHasSmallerMagnitude =
      b.create<mlir::stablehlo::OrOp>(xMagnitudeLargerThanY, signsDisagree);
  auto minusOne = getConstantLike(rewriter, loc, -1, xAsInt);
  auto magnitudeAdjustment = b.create<mlir::stablehlo::SelectOp>(
      resultHasSmallerMagnitude, minusOne, one);
  Value result = b.create<mlir::stablehlo::AddOp>(xAsInt, magnitudeAdjustment);
  // Handle from == +-0.
  result = b.create<mlir::stablehlo::SelectOp>(
      xIsZero,
      b.create<mlir::stablehlo::SelectOp>(yIsZero, resultForBothZero,
                                          resultForXZeroYNonZero),
      result);
  // Handle from == to.
  result = b.create<mlir::stablehlo::SelectOp>(xAndYAreEqual, resultForEqual,
                                               result);
  // Handle isnan(x) || isnan(y).
  result =
      b.create<mlir::stablehlo::SelectOp>(nanInput, resultForNanAsInt, result);

  // Cast back to the original type.
  return b.create<mlir::stablehlo::BitcastConvertOp>(resultTy, result);
}

struct ConvertNextAfterOp final : OpConversionPattern<mlir::chlo::NextAfterOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::NextAfterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOp(
        op, materializeNextAfter(rewriter, op.getLoc(), adaptor.getOperands()));
    return success();
  }
};

struct ConvertPolygammaOp final : OpConversionPattern<mlir::chlo::PolygammaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::PolygammaOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    FloatType minPrecisionTy = rewriter.getF32Type();
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, loc, adaptor.getOperands(),
                                  minPrecisionTy, materializePolygamma));
    return success();
  }
};

// Sinh(x) = (e^x - e^-x) / 2
//         = e^(x + log(1/2)) - e^(-x + log(1/2)).
//
// The second formulation avoids overflowing when e^x = inf but (e^x)/2 is not
// inf.
//
// This incorrectly overflows to +/-inf for two f32 input values, namely
// +/-89.4159851, due to rounding error when computing x +/- log(1/2).  The
// correct answer of 3.40281961e+38 (0x7f7fffec) is very close to max-float, so
// we deem this acceptable.
static Value materializeSinhApproximationForLargeX(OpBuilder& rewriter,
                                                   Location loc,
                                                   ValueRange operands) {
  mlir::chlo::SinhOp::Adaptor transformed(operands);
  Value x = transformed.getOperand();

  Value logOneHalf = rewriter.create<mlir::stablehlo::LogOp>(
      loc, getConstantLike(rewriter, loc, 0.5, x));
  Value expAdd = rewriter.create<mlir::stablehlo::ExpOp>(
      loc, rewriter.create<mlir::stablehlo::AddOp>(loc, x, logOneHalf));
  Value expSub = rewriter.create<mlir::stablehlo::ExpOp>(
      loc, rewriter.create<mlir::stablehlo::SubtractOp>(loc, logOneHalf, x));
  return rewriter.create<mlir::stablehlo::SubtractOp>(loc, expAdd, expSub);
}

// Express `sinh` as
//   sinh(x) = (e^x - e^-x) / 2                     if |x| < 1
//           = e^(x + log(1/2)) - e^(-x + log(1/2)) otherwise.
static Value materializeSinhApproximation(OpBuilder& rewriter, Location loc,
                                          ValueRange operands) {
  Value largeSinhResult =
      materializeSinhApproximationForLargeX(rewriter, loc, operands);

  mlir::chlo::SinhOp::Adaptor transformed(operands);
  Value x = transformed.getOperand();

  // For smaller x, we get unwanted cancellations of e^x - e^-x, resulting in
  // 0.
  // Rewrite this to avoid that. We use expm1(x) because that preserves the
  // first order term of the taylor series of e^x.
  // (e^(x) - e^(-x)) / 2. =
  // (e^(x) - 1 + 1 - e^(-x)) / 2.
  // (expm1(x) + (e^(x) - 1) / e^x) / 2.
  // (expm1(x) + expm1(x) / (expm1(x) + 1)) / 2.
  Value expm1 = rewriter.create<mlir::stablehlo::Expm1Op>(loc, x);
  Value one = getConstantLike(rewriter, loc, 1.0, x);
  Value oneHalf = getConstantLike(rewriter, loc, 0.5, x);
  Value expm1PlusOne = rewriter.create<mlir::stablehlo::AddOp>(loc, expm1, one);
  Value ratio =
      rewriter.create<mlir::stablehlo::DivOp>(loc, expm1, expm1PlusOne);
  Value sum = rewriter.create<mlir::stablehlo::AddOp>(loc, expm1, ratio);
  Value smallSinhResult =
      rewriter.create<mlir::stablehlo::MulOp>(loc, oneHalf, sum);

  Value absX = rewriter.create<mlir::stablehlo::AbsOp>(loc, x);
  Value absXLtOne = rewriter.create<mlir::stablehlo::CompareOp>(
      loc, absX, one, mlir::stablehlo::ComparisonDirection::LT);
  return rewriter.create<mlir::stablehlo::SelectOp>(
      loc, absXLtOne, smallSinhResult, largeSinhResult);
}

namespace {

ArrayAttr convertPrecisionConfig(mlir::ArrayAttr precisionConfig,
                                 ConversionPatternRewriter& rewriter) {
  std::vector<Attribute> precisions;
  for (Attribute precision : precisionConfig.getValue()) {
    switch (dyn_cast<mlir::chlo::PrecisionAttr>(precision).getValue()) {
      case mlir::chlo::Precision::HIGHEST:
        precisions.push_back(rewriter.getAttr<mlir::stablehlo::PrecisionAttr>(
            mlir::stablehlo::Precision::HIGHEST));
        break;
      case mlir::chlo::Precision::HIGH:
        precisions.push_back(rewriter.getAttr<mlir::stablehlo::PrecisionAttr>(
            mlir::stablehlo::Precision::HIGH));
        break;
      default:
        precisions.push_back(rewriter.getAttr<mlir::stablehlo::PrecisionAttr>(
            mlir::stablehlo::Precision::DEFAULT));
        break;
    }
  }
  return ArrayAttr::get(rewriter.getContext(), precisions);
}

// Mode 1, where the ragged dimension is an lhs non-contracting dim (m).
//   lhs : [b, m, k]
//   rhs : [g, b, k, n]
//   group_sizes : [g]
//   result : [b, m, n]
// This pass basically does g iterations of [b, m, k] x [b, k, n] dot_general
// operations, apply partial mask of size group_sizes[i] and then add them
// together. This is a slow implementation that's simple enough to understand
// with the hope that there's already an efficient hardware kernel.
// Note:
// In this implementation, the IR size increases by a factor of g. If this
// becomes a problem, we can try adding stablehlo.while to reduce the IR size.
LogicalResult handleRaggedDotMode1(mlir::chlo::RaggedDotOp op,
                                   ConversionPatternRewriter& rewriter) {
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  chlo::RaggedDotDimensionNumbersAttr raggedDotDimensionNumbers =
      op.getRaggedDotDimensionNumbers();
  ArrayRef<int64_t> lhsBatchingDimensions =
      raggedDotDimensionNumbers.getLhsBatchingDimensions();
  ArrayRef<int64_t> lhsContractingDimensions =
      raggedDotDimensionNumbers.getLhsContractingDimensions();
  int64_t rhsGroupDimension =
      raggedDotDimensionNumbers.getRhsGroupDimensions()[0];

  auto groupSizes = op.getGroupSizes();
  auto precisionConfig = op.getPrecisionConfig();
  if (precisionConfig.has_value()) {
    precisionConfig = convertPrecisionConfig(precisionConfig.value(), rewriter);
  }
  RankedTensorType lhsTy = cast<RankedTensorType>(lhs.getType());
  RankedTensorType rhsTy = cast<RankedTensorType>(rhs.getType());
  int64_t lhsRank = lhsTy.getRank();
  int64_t rhsRank = rhsTy.getRank();
  auto outDType = op.getResult().getType().getElementType();

  int64_t m = lhsTy.getShape()[lhsTy.getRank() - 2];
  int64_t k = lhsTy.getShape()[lhsTy.getRank() - 1];
  int64_t g = rhsTy.getShape()[0];
  int64_t n = rhsTy.getShape()[rhsTy.getRank() - 1];

  std::vector<int64_t> outDims = {m, n};
  std::vector<int64_t> iotaShape = {m, 1};
  auto iotaDim = 0;
  std::vector<int64_t> rhsBatchingDims = {};
  std::vector<int64_t> rhsContractingDims = {0};
  std::vector<int64_t> rhsReshapedSliceShape = {k, n};

  // If LHS has batching dimension, then decompose ragged dot based on shape
  // [b, m, k], otherwise assume shape with no batch [m, k].
  if (lhsRank == 3) {
    int64_t b = lhsTy.getShape()[0];
    outDims = {b, m, n};
    iotaShape = {1, m, 1};
    iotaDim = 1;
    rhsBatchingDims = {0};
    rhsContractingDims = {1};
    rhsReshapedSliceShape = {b, k, n};
  }

  // result_iota = iota of shape [m, 1] or [1, m, 1]
  Value resultIota = rewriter.create<mlir::stablehlo::IotaOp>(
      op.getLoc(), RankedTensorType::get(iotaShape, rewriter.getI64Type()),
      /*dimension=*/iotaDim);
  Value start = rewriter.create<mlir::stablehlo::ConstantOp>(
      op.getLoc(),
      rewriter.getZeroAttr(RankedTensorType::get({1}, rewriter.getI64Type())));

  std::vector<int64_t> broadcastDimensions(lhsRank);
  std::iota(broadcastDimensions.begin(), broadcastDimensions.end(), 0);

  Value out = rewriter.create<mlir::stablehlo::ConstantOp>(
      op.getLoc(),
      rewriter.getZeroAttr(RankedTensorType::get(outDims, outDType)));

  Value outZeros = rewriter.create<mlir::stablehlo::ConstantOp>(
      op.getLoc(),
      rewriter.getZeroAttr(RankedTensorType::get(outDims, outDType)));
  for (auto i = 0; i < g; ++i) {
    // groupSize = group_sizes[i]
    Value groupSize = rewriter.create<mlir::stablehlo::SliceOp>(
        op.getLoc(), RankedTensorType::get({1}, rewriter.getI64Type()),
        groupSizes,
        /*startIndices=*/rewriter.getDenseI64ArrayAttr({i}),
        /*limitIndices=*/rewriter.getDenseI64ArrayAttr({i + 1}),
        /*strides=*/rewriter.getDenseI64ArrayAttr({1}));

    Value startBroadcasted = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
        op.getLoc(), resultIota.getType(), start,
        /*broadcast_dimensions=*/
        rewriter.getDenseI64ArrayAttr(0));

    // start <= result_iota
    Value startLEResultIota = rewriter.create<mlir::stablehlo::CompareOp>(
        op.getLoc(), startBroadcasted, resultIota, ComparisonDirection::LE);

    // result_iota < (start + size)
    Value resultIotaLTStartPlusGroupSize =
        rewriter.create<mlir::stablehlo::CompareOp>(
            op.getLoc(), resultIota,
            rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
                op.getLoc(), resultIota.getType(),
                rewriter.create<mlir::stablehlo::AddOp>(op.getLoc(), start,
                                                        groupSize),
                /*broadcast_dimensions=*/rewriter.getDenseI64ArrayAttr(0)),
            ComparisonDirection::LT);

    // (start <= result_iota) & (result_iota < (start + size))
    Value logicalAnd = rewriter.create<mlir::stablehlo::AndOp>(
        op.getLoc(), startLEResultIota, resultIotaLTStartPlusGroupSize);
    Value logicalAndBroadcasted =
        rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
            op.getLoc(),
            RankedTensorType::get(op.getResult().getType().getShape(),
                                  rewriter.getI1Type()),
            logicalAnd,
            /*broadcast_dimensions=*/
            rewriter.getDenseI64ArrayAttr(broadcastDimensions));

    // rhs_rehaped_slice = rhs[i, :, :, :]
    std::vector<int64_t> rhs_start_indices(rhsTy.getRank(), 0);
    rhs_start_indices[rhsGroupDimension] = i;
    std::vector<int64_t> rhs_limit_indices = rhsTy.getShape();
    rhs_limit_indices[rhsGroupDimension] = i + 1;
    Value rhsSliced = rewriter.create<mlir::stablehlo::SliceOp>(
        op.getLoc(), rhs,
        /*startIndices=*/rewriter.getDenseI64ArrayAttr(rhs_start_indices),
        /*limitIndices=*/rewriter.getDenseI64ArrayAttr(rhs_limit_indices),
        /*strides=*/
        rewriter.getDenseI64ArrayAttr(std::vector<int64_t>(rhsRank, 1)));
    Value rhsReshapedSlice = rewriter.create<mlir::stablehlo::ReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(rhsReshapedSliceShape, rhsTy.getElementType()),
        rhsSliced);

    // Einsum of (b)mk,(b)kn->(b)mn
    Value dotGeneral = rewriter.create<mlir::stablehlo::DotGeneralOp>(
        op.getLoc(), TypeRange{out.getType()},
        ValueRange{lhs, rhsReshapedSlice},
        ArrayRef<mlir::NamedAttribute>{
            rewriter.getNamedAttr(
                "dot_dimension_numbers",
                rewriter.getAttr<mlir::stablehlo::DotDimensionNumbersAttr>(
                    /*lhs_batching_dimensions=*/lhsBatchingDimensions,
                    /*rhs_batching_dimensions=*/rhsBatchingDims,
                    /*lhs_contracting_dimensions=*/lhsContractingDimensions,
                    /*rhs_contracting_dimensions=*/rhsContractingDims)),
            rewriter.getNamedAttr("precision_config",
                                  precisionConfig.value())});

    // Place the i'th dot_general to the corresponding position in the result.
    Value select = rewriter.create<mlir::stablehlo::SelectOp>(
        op.getLoc(), logicalAndBroadcasted, dotGeneral, outZeros);
    out = rewriter.create<mlir::stablehlo::AddOp>(op.getLoc(), out, select);
    start =
        rewriter.create<mlir::stablehlo::AddOp>(op.getLoc(), start, groupSize);
  }
  rewriter.replaceOp(op, {out});
  return success();
}

// Mode 2, where the ragged dimension is an lhs/rhs contracting dim (k).
//   lhs : [b, m, k]
//   rhs : [b, k, n]
//   group_sizes : [g]
//   result : [g, b, m, n]
LogicalResult handleRaggedDotMode2(mlir::chlo::RaggedDotOp op,
                                   ConversionPatternRewriter& rewriter) {
  return failure();
}

// Mode 3, where the ragged dimension is an lhs/rhs batch dim (b).
//   lhs : [b, m, k]
//   rhs : [b, k, n]
//   group_sizes : [g]
//   result : [b, m, n]
LogicalResult handleRaggedDotMode3(mlir::chlo::RaggedDotOp op,
                                   ConversionPatternRewriter& rewriter) {
  return failure();
}

}  // namespace

struct ConvertRaggedDotOp final : OpConversionPattern<mlir::chlo::RaggedDotOp> {
  using OpConversionPattern::OpConversionPattern;

  // RaggedDot has three general modes, based on the kind of the ragged
  // dimension.
  LogicalResult matchAndRewrite(
      mlir::chlo::RaggedDotOp op, OpAdaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (op.getLhs().getType().getRank() < op.getRhs().getType().getRank()) {
      return handleRaggedDotMode1(op, rewriter);
    } else if (op.getLhs().getType().getRank() <
               op.getResult().getType().getRank()) {
      return handleRaggedDotMode2(op, rewriter);
    } else {
      return handleRaggedDotMode3(op, rewriter);
    }
  }
};

struct ConvertSinhOp final : OpConversionPattern<mlir::chlo::SinhOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::SinhOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value x = adaptor.getOperand();
    if (isa<ComplexType>(cast<ShapedType>(x.getType()).getElementType())) {
      rewriter.replaceOp(op, materializeSinhApproximationForLargeX(
                                 rewriter, op.getLoc(), adaptor.getOperands()));
      return success();
    }
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, op.getLoc(), adaptor.getOperands(),
                                  rewriter.getF32Type(),
                                  &materializeSinhApproximation));
    return success();
  }
};

// Converts chlo.top_k to HLO iota, sort, and slice ops.
//
// chlo.top_k sorts along last dimension of the input tensor and then returns
// the top K components' values and indices. This is translated into a few
// ops in HLO: first generating an integer sequence for the indices,
// then sort both the original input tensor and the indices together, and
// at last slice out the top K components.
//
// For example, for the following IR:
//
// %0:2 = "chlo.top_k"(%input, k=8): tensor<16x16xf32> ->
//                                   (tensor<16x8xf32>, tensor<16x8xi32>)
//
// We will get:
//
// %1 = "hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<16x16xi32>
// %2 = "hlo.sort"(%input, %1) ({
// ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>,
//      %arg3: tensor<i32>, %arg4: tensor<i32>):
//   %7 = "hlo.compare"(%arg1, %arg2) {comparison_direction = "GT"}: ...
//   "hlo.return"(%7) : (tensor<i1>) -> ()
// }) {dimension = 1 : i64, is_stable = true} : ...
// %3 = "hlo.get_tuple_element"(%2) {index = 0 : i32} : ...
// %4 = "hlo.get_tuple_element"(%2) {index = 1 : i32} : ...
// %5 = "hlo.slice"(%3) {limit_indices = dense<[16, 8]> : tensor<2xi64>,
//                           start_indices dense<0> : tensor<2xi64>,
//                           strides = dense<1> : tensor<2xi64>} :
//                              (tensor<16x16xf32>) -> tensor<16x8xf32>
// %6 = "hlo.slice"(%4) ...
//
struct ConvertTopKOp final : OpConversionPattern<mlir::chlo::TopKOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::TopKOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {
    auto operandType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    if (!operandType) return failure();
    int64_t operandRank = operandType.getRank();
    int64_t lastDimIndex = operandRank - 1;
    int64_t lastDimSize = operandType.getDimSize(lastDimIndex);
    int64_t lastDimResultSize =
        mlir::hlo::isDynamicDimSize(lastDimSize)
            ? static_cast<int64_t>(op.getK())
            : std::min(static_cast<int64_t>(op.getK()), lastDimSize);
    int64_t isDynamic = !operandType.hasStaticShape();
    auto i32Type = rewriter.getIntegerType(32);
    Value opShapeValue, resultShapeValue;
    if (isDynamic) {
      SmallVector<Value> sizesI32x1;
      for (auto i = 0; i < operandType.getRank(); ++i) {
        auto sizeI32 = rewriter.create<mlir::stablehlo::GetDimensionSizeOp>(
            op.getLoc(), op.getOperand(), i);
        auto sizeI32x1 = rewriter.create<mlir::stablehlo::ReshapeOp>(
            op.getLoc(), RankedTensorType::get({1}, i32Type), sizeI32);
        sizesI32x1.push_back(sizeI32x1);
      }
      opShapeValue = rewriter.create<mlir::stablehlo::ConcatenateOp>(
          op.getLoc(), sizesI32x1,
          /*dimension=*/0);
      auto lastDimI32 = rewriter.create<mlir::stablehlo::ConstantOp>(
          op.getLoc(),
          rewriter.getI32IntegerAttr(static_cast<int32_t>(lastDimResultSize)));
      auto lastDimI32x1 = rewriter.create<mlir::stablehlo::ReshapeOp>(
          op.getLoc(), RankedTensorType::get({1}, i32Type), lastDimI32);
      sizesI32x1.back() = lastDimI32x1;
      resultShapeValue = rewriter.create<mlir::stablehlo::ConcatenateOp>(
          op.getLoc(), sizesI32x1,
          /*dimension=*/0);
    }

    // Create an Iota op for indices.
    Type iotaType = RankedTensorType::get(operandType.getShape(), i32Type);
    Value iotaOp;
    if (isDynamic) {
      iotaOp = rewriter.create<mlir::stablehlo::DynamicIotaOp>(
          op.getLoc(), iotaType, opShapeValue,
          rewriter.getI64IntegerAttr(lastDimIndex));
    } else {
      iotaOp = rewriter.create<mlir::stablehlo::IotaOp>(
          op.getLoc(), iotaType, rewriter.getI64IntegerAttr(lastDimIndex));
    }

    // Create the sort op. It takes two inputs, one for the original input, the
    // other for the indices. Use TOTALORDER comparison type instead of the
    // default comparison if the element type is of type float.
    Type elementType = operandType.getElementType();
    mlir::stablehlo::SortOp sortOp =
        createSortOp(&rewriter, op.getLoc(), {op.getOperand(), iotaOp},
                     {elementType, i32Type}, lastDimIndex,
                     /*isStable=*/true,
                     /*direction=*/mlir::stablehlo::ComparisonDirection::GT);

    // Get the sorted input and index tuple element.
    Value tupleFirstElement = sortOp.getResult(0);
    Value tupleSecondElement = sortOp.getResult(1);

    SmallVector<int64_t> beginIndices(operandRank, 0);
    auto endIndices = llvm::to_vector(operandType.getShape());
    endIndices.back() = lastDimResultSize;
    SmallVector<int64_t> strides(operandRank, 1);

    // Get the slice for the top K elements.
    auto indicesTy = RankedTensorType::get(operandRank, rewriter.getI64Type());
    Value values, indices;
    if (isDynamic) {
      Value startIndices = rewriter.create<mlir::stablehlo::ConstantOp>(
          op.getLoc(), DenseIntElementsAttr::get(indicesTy, beginIndices));
      Value lastIndices = rewriter.create<mlir::stablehlo::ConvertOp>(
          op.getLoc(), resultShapeValue, rewriter.getI64Type());
      Value stridesOp = rewriter.create<mlir::stablehlo::ConstantOp>(
          op.getLoc(), DenseIntElementsAttr::get(indicesTy, strides));

      SmallVector<int64_t> resultShape =
          llvm::to_vector(operandType.getShape());
      resultShape.back() = lastDimResultSize;
      RankedTensorType resultType = RankedTensorType::get(
          resultShape, elementType, operandType.getEncoding());
      RankedTensorType indexResultType =
          RankedTensorType::get(resultShape, i32Type);

      values = rewriter.create<mlir::stablehlo::RealDynamicSliceOp>(
          op.getLoc(), resultType, tupleFirstElement, startIndices, lastIndices,
          stridesOp);
      indices = rewriter.create<mlir::stablehlo::RealDynamicSliceOp>(
          op.getLoc(), indexResultType, tupleSecondElement, startIndices,
          lastIndices, stridesOp);
    } else {
      values = rewriter.create<mlir::stablehlo::SliceOp>(
          op.getLoc(), tupleFirstElement,
          rewriter.getDenseI64ArrayAttr(beginIndices),
          rewriter.getDenseI64ArrayAttr(endIndices),
          rewriter.getDenseI64ArrayAttr(strides));
      indices = rewriter.create<mlir::stablehlo::SliceOp>(
          op.getLoc(), tupleSecondElement,
          rewriter.getDenseI64ArrayAttr(beginIndices),
          rewriter.getDenseI64ArrayAttr(endIndices),
          rewriter.getDenseI64ArrayAttr(strides));
    }

    rewriter.replaceOp(op, {values, indices});
    return success();
  }
};

struct ConvertZetaOp final : OpConversionPattern<mlir::chlo::ZetaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::chlo::ZetaOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    FloatType minPrecisionTy = rewriter.getF32Type();
    rewriter.replaceOp(
        op, materializeWithUpcast(rewriter, loc, adaptor.getOperands(),
                                  minPrecisionTy, &materializeZeta));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition.
//===----------------------------------------------------------------------===//

struct ChloLegalizeToStablehloPass final
    : impl::ChloLegalizeToStablehloPassBase<ChloLegalizeToStablehloPass> {
  LogicalResult initialize(MLIRContext* context) override {
    target = std::make_shared<ConversionTarget>(*context);
    target->addIllegalDialect<chlo::ChloDialect>();
    target->addLegalDialect<mlir::stablehlo::StablehloDialect,
                            mlir::arith::ArithDialect, mlir::func::FuncDialect,
                            mlir::shape::ShapeDialect,
                            mlir::tensor::TensorDialect>();

    RewritePatternSet patterns_(context);
    populateChloToStablehloPatterns(context, &patterns_);
    patterns = std::move(patterns_);

    return success();
  }

  void runOnOperation() override {
    if (failed(applyPartialConversion(getOperation(), *target, patterns))) {
      return signalPassFailure();
    }
  }

 private:
  std::shared_ptr<ConversionTarget> target;
  FrozenRewritePatternSet patterns;
};

#include "stablehlo/transforms/ChloDecompositionPatterns.h.inc"
}  // namespace

namespace {
static void populateChloBroadcastingPatterns(MLIRContext* context,
                                             RewritePatternSet* patterns) {
  // Instantiate conversion templates for conforming binary elementwise ops
  // that do not have different dtypes between operands and results and do
  // not have special attributes that need to be preserved.
  populateForBroadcastingBinaryOp<ConvertTrivialNonBroadcastBinaryOp>(
      context, patterns, 10);
  populateForBroadcastingBinaryOp<ConvertTrivialNumpyBroadcastBinaryOp>(
      context, patterns, 10);
  populateForBroadcastingBinaryOp<ConvertRankedDynamicBroadcastBinaryOp>(
      context, patterns, 5);
  patterns->add<ConvertConstantLikeOp, ConvertSelectOp>(context);
}

static void populateChloDecompositionPatterns(MLIRContext* context,
                                              RewritePatternSet* patterns) {
  populateWithGenerated(*patterns);
  patterns
      ->add<ConvertConstantOp, ConvertBesselI1eOp, ConvertCoshOp,
            ConvertDigammaOp, ConvertErfOp, ConvertErfcOp, ConvertErfInvOp,
            ConvertLgammaOp, ConvertNextAfterOp, ConvertPolygammaOp,
            ConvertRaggedDotOp, ConvertSinhOp, ConvertTopKOp, ConvertZetaOp>(
          context);
}
}  // namespace

void populateChloToStablehloPatterns(MLIRContext* context,
                                     RewritePatternSet* patterns) {
  populateChloBroadcastingPatterns(context, patterns);
  populateChloDecompositionPatterns(context, patterns);
}

}  // namespace stablehlo
}  // namespace mlir
