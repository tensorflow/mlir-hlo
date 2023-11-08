/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.
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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/experimental/dialect/StablehloOps.h"
#include "stablehlo/experimental/transforms/Passes.h"

namespace mlir {
namespace stablehlo {
namespace experimental {

#define GEN_PASS_DEF_STABLEHLOCANONICALIZEDYNAMISMPASS
#include "stablehlo/experimental/transforms/Passes.h.inc"

namespace {

struct CanonicalizeCustomCallOpPattern : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> refinements;
    if (failed(hlo::getShapeRefinements(op.getLoc(), op, refinements)))
      return rewriter.notifyMatchFailure(op, "expected valid refinements");
    auto indicesAttr =
        op->getAttr("indices_of_shape_operands").cast<DenseIntElementsAttr>();
    DenseSet<int64_t> indices(indicesAttr.value_begin<int64_t>(),
                              indicesAttr.value_end<int64_t>());

    // Discard the indices_of_shape_operands attribute.
    // We rely on the verification logic implemented in getShapeRefinements to
    // make sure that its value is consistent with the result types.
    // In the future, when we upgrade indices_of_shape_operands from an
    // experiment to a full-fledged StableHLO feature, this logic will be moved
    // to a proper verifier.
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : op->getAttrs()) {
      if (attr.getName() == "indices_of_shape_operands") continue;
      if (attr.getName() == "operand_layouts") {
        // Drop the operand_layouts that correspond to indices_of_shape_operands
        ArrayAttr operandLayouts = op.getOperandLayoutsAttr();
        SmallVector<Attribute> newOperandLayouts;
        for (unsigned i = 0; i < operandLayouts.size(); ++i) {
          if (indices.contains(i)) continue;
          newOperandLayouts.push_back(operandLayouts[i]);
        }
        attr = NamedAttribute(attr.getName(),
                              rewriter.getArrayAttr(newOperandLayouts));
      }
      newAttrs.push_back(attr);
    }

    // Discard the operands that correspond to indices_of_shape_operands.
    // We rely on the verification logic implemented in getShapeRefinements to
    // make sure that: 1) these operands are static, 2) the values of these
    // operands are consistent with the result types.
    SmallVector<Value> newOperands;
    auto resultIndex = 0;
    for (auto& operand : op->getOpOperands()) {
      if (indices.contains(operand.getOperandNumber())) {
        auto resultType =
            op->getResult(resultIndex).getType().dyn_cast<ShapedType>();
        if (!resultType || !resultType.hasStaticShape())
          return rewriter.notifyMatchFailure(op,
                                             "expected static result types");
        ++resultIndex;
        continue;
      }
      newOperands.push_back(operand.get());
    }
    rewriter.replaceOpWithNewOp<CustomCallOp>(op, op.getResultTypes(),
                                              newOperands, newAttrs);
    return success();
  }
};

struct CanonicalizeDynamicBroadcastInDimOpPattern
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    // This pattern discards the output_dimensions operand as well as the
    // known_expanding_dimensions and known_nonexpanding_dimensions attributes.
    // We rely on the verifier to make sure that their values are consistent
    // with the result type.
    if (!op.getOperand().getType().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static operand type");
    if (!succeeded(hlo::matchInts(op.getOutputDimensions())))
      return rewriter.notifyMatchFailure(op,
                                         "expected static output_dimensions");
    if (!op.getType().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static result type");
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
        op, op.getType(), op.getOperand(), op.getBroadcastDimensions());
    return success();
  }
};

struct CanonicalizeDynamicConvOpPattern
    : public OpRewritePattern<DynamicConvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicConvOp op,
                                PatternRewriter& rewriter) const override {
    // ConvolutionOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.
    SmallVector<int64_t> padding;
    if (!succeeded(hlo::matchInts(op.getDPadding(), padding)))
      return rewriter.notifyMatchFailure(op, "expected static padding");
    auto paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(padding.size()) / 2, 2},
                              rewriter.getI64Type()),
        padding);
    rewriter.replaceOpWithNewOp<ConvolutionOp>(
        op, op.getType(), op.getLhs(), op.getRhs(), op.getWindowStridesAttr(),
        paddingAttr, op.getLhsDilationAttr(), op.getRhsDilationAttr(),
        op.getWindowReversalAttr(), op.getDimensionNumbers(),
        op.getFeatureGroupCount(), op.getBatchGroupCount(),
        op.getPrecisionConfigAttr());
    return success();
  }
};

struct CanonicalizeDynamicGatherOpPattern
    : public OpRewritePattern<DynamicGatherOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicGatherOp op,
                                PatternRewriter& rewriter) const override {
    // GatherOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.
    SmallVector<int64_t> sliceSizes;
    if (!succeeded(hlo::matchInts(op.getSliceSizes(), sliceSizes)))
      return rewriter.notifyMatchFailure(op, "expected static slice_sizes");
    rewriter.replaceOpWithNewOp<GatherOp>(
        op, op.getType(), op.getOperand(), op.getStartIndices(),
        op.getDimensionNumbersAttr(), rewriter.getI64TensorAttr(sliceSizes),
        op.getIndicesAreSortedAttr());
    return success();
  }
};

struct CanonicalizeDynamicIotaOpPattern
    : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicIotaOp op,
                                PatternRewriter& rewriter) const override {
    // This pattern discards the output_shape operand. We rely on the verifier
    // to make sure that its value is consistent with result type.
    SmallVector<int64_t> outputShape;
    if (!succeeded(hlo::matchInts(op.getOutputShape(), outputShape)))
      return rewriter.notifyMatchFailure(op, "expected static output_shape");
    if (!op.getType().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static result type");
    rewriter.replaceOpWithNewOp<IotaOp>(op, op.getType(),
                                        op.getIotaDimension());
    return success();
  }
};

struct CanonicalizeDynamicPadOpPattern : public OpRewritePattern<DynamicPadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicPadOp op,
                                PatternRewriter& rewriter) const override {
    // PadOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.
    SmallVector<int64_t> edgePaddingLow, edgePaddingHigh, interiorPadding;
    if (!succeeded(hlo::matchInts(op.getEdgePaddingLow(), edgePaddingLow)))
      return rewriter.notifyMatchFailure(op, "expected static low");
    if (!succeeded(hlo::matchInts(op.getEdgePaddingHigh(), edgePaddingHigh)))
      return rewriter.notifyMatchFailure(op, "expected static high");
    if (!succeeded(hlo::matchInts(op.getInteriorPadding(), interiorPadding)))
      return rewriter.notifyMatchFailure(op, "expected static interior");
    rewriter.replaceOpWithNewOp<PadOp>(
        op, op.getType(), op.getOperand(), op.getPaddingValue(),
        rewriter.getI64TensorAttr(edgePaddingLow),
        rewriter.getI64TensorAttr(edgePaddingHigh),
        rewriter.getI64TensorAttr(interiorPadding));
    return success();
  }
};

struct CanonicalizeDynamicReduceWindowOpPattern
    : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicReduceWindowOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicReduceWindowOpAdaptor op = *maybeOp;

    // ReduceWindowOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.
    SmallVector<int64_t> windowDimensions, windowStrides, baseDilations,
        windowDilations, padding;
    if (failed(hlo::matchInts(op.getWindowDimensions(), windowDimensions)))
      return rewriter.notifyMatchFailure(op,
                                         "expected static window_dimensions");
    if (failed(hlo::matchInts(op.getWindowStrides(), windowStrides)))
      return rewriter.notifyMatchFailure(op, "expected static window_strides");
    if (failed(hlo::matchInts(op.getBaseDilations(), baseDilations)))
      return rewriter.notifyMatchFailure(op, "expected static base_dilations");
    if (failed(hlo::matchInts(op.getWindowDilations(), windowDilations)))
      return rewriter.notifyMatchFailure(op,
                                         "expected static window_dilations");
    if (failed(hlo::matchInts(op.getPadding(), padding)))
      return rewriter.notifyMatchFailure(op, "expected static padding");
    auto newOp = rewriter.create<ReduceWindowOp>(
        op->getLoc(), op->getResultTypes(), op.getInputs(), op.getInitValues(),
        rewriter.getI64TensorAttr(windowDimensions),
        rewriter.getI64TensorAttr(windowStrides),
        rewriter.getI64TensorAttr(baseDilations),
        rewriter.getI64TensorAttr(windowDilations),
        hlo::getPaddingAttr(&rewriter, padding));

    // Inline the called computation into newOp.
    // This is somewhat annoying because we also have to rewrite the original
    // func::ReturnOp into stablehlo::ReturnOp.
    rewriter.cloneRegionBefore(op.getBody(), newOp.getBody(),
                               newOp.getBody().end());
    auto funcReturnOp =
        cast<func::ReturnOp>(newOp.getBody().front().getTerminator());
    rewriter.setInsertionPointToEnd(&newOp.getBody().front());
    rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(
        funcReturnOp, funcReturnOp.getOperands());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct CanonicalizeDynamicReshapeOpPattern
    : public OpRewritePattern<DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    // This pattern ignores and discards the output_shape operand. We rely on
    // the verifier to make sure that its value is consistent with result type.
    if (!succeeded(hlo::matchInts(op.getOutputShape())))
      return rewriter.notifyMatchFailure(op, "expected static output_shape");
    if (!op.getType().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static result type");
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getOperand());
    return success();
  }
};

struct CanonicalizeDynamicRngBitGeneratorOpPattern
    : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicRngBitGeneratorOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicRngBitGeneratorOpAdaptor op = *maybeOp;

    // This pattern ignores and discards the output_shape operand. We rely on
    // the verifier to make sure that its value is consistent with result type.
    if (!succeeded(hlo::matchInts(op.getOutputShape())))
      return rewriter.notifyMatchFailure(op, "expected static output_shape");
    if (!op.getOutput().getType().cast<ShapedType>().hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static output type");
    rewriter.replaceOpWithNewOp<RngBitGeneratorOp>(
        op, op->getResultTypes(), op.getRngAlgorithm(), op.getInitialState());
    return success();
  }
};

struct CanonicalizeDynamicTopKOpPattern
    : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicTopKOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicTopKOpAdaptor op = *maybeOp;

    SmallVector<int64_t> k;
    if (failed(hlo::matchInts(op.getK(), k)))
      return rewriter.notifyMatchFailure(impl, "expected constant k");

    // We rely on many of the properties checked by verification.
    auto valuesType = op.getValues().getType().cast<ShapedType>();
    auto valuesLastDimSize = valuesType.getShape()[valuesType.getRank() - 1];
    if (hlo::isDynamicDimSize(valuesLastDimSize) ||
        valuesLastDimSize != k[0])
      return rewriter.notifyMatchFailure(
          op,
          "expected value of k to match the values last dimension size of "
          "static values type (result #0)");

    rewriter.replaceOpWithNewOp<chlo::TopKOp>(
        op, op->getResultTypes(), op.getOperand(), k[0]);
    return success();
  }
};

struct CanonicalizeRealDynamicSliceOpToDynamicSliceOpPattern
    : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RealDynamicSliceOp op,
                                PatternRewriter& rewriter) const override {
    // DynamicSliceOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.

    // This rewrite only works for unit strides because DynamicSliceOp
    // doesn't support strides (i.e. it implicitly has unit strides).
    SmallVector<int64_t> strides;
    if (!succeeded(hlo::matchInts(op.getStrides(), strides)))
      return rewriter.notifyMatchFailure(op, "expected static strides");
    if (!llvm::all_of(strides, [&](int64_t stride) { return stride == 1; }))
      return rewriter.notifyMatchFailure(op, "expected unit strides");

    // Check that slice sizes are fully static (DynamicSliceOp style).
    // To detect that, we check whether `limit_indices` is defined as
    // `start_indices + constant` or `constant + start_indices`.
    DenseIntElementsAttr sliceSizesAttr;
    auto m_startIndices = matchers::m_Val(op.getStartIndices());
    if (!matchPattern(
            op.getLimitIndices(),
            m_Op<AddOp>(m_startIndices, m_Constant(&sliceSizesAttr))) &&
        !matchPattern(op.getLimitIndices(),
                      m_Op<AddOp>(m_Constant(&sliceSizesAttr), m_startIndices)))
      return rewriter.notifyMatchFailure(
          op, "expected limit indices equal to start indices plus constant");

    // RealDynamicSliceOp can take tensors of integer or index element types.
    // DynamicSliceOp::slice_sizes only supports i64 element type.
    // Adapt accordingly in order to be compatible with DynamicSliceOp.
    SmallVector<int64_t> sliceSizes;
    for (auto element : sliceSizesAttr.getValues<APInt>()) {
      sliceSizes.push_back(element.getSExtValue());
    }

    // RealDynamicSliceOp::start_indices is a 1-dimensional tensor.
    // DynamicSliceOp::start_indices is a vararg of 0-dimensional tensors.
    // Adapt accordingly in order to be compatible with DynamicSliceOp.
    SmallVector<Value> startIndices;
    for (auto i = 0; i < static_cast<int64_t>(sliceSizes.size()); ++i) {
      auto startIndexElementType =
          op.getStartIndices().getType().getElementType();
      auto startIndex1DType = RankedTensorType::get({1}, startIndexElementType);
      auto startIndex1D = rewriter.create<SliceOp>(
          op.getLoc(), startIndex1DType, op.getStartIndices(),
          rewriter.getI64TensorAttr(i), rewriter.getI64TensorAttr(i + 1),
          rewriter.getI64TensorAttr(1));
      auto startIndex0DType = RankedTensorType::get({}, startIndexElementType);
      auto startIndex0D = rewriter.create<ReshapeOp>(
          op.getLoc(), startIndex0DType, startIndex1D);
      startIndices.push_back(startIndex0D);
    }

    rewriter.replaceOpWithNewOp<DynamicSliceOp>(
        op, op.getType(), op.getOperand(), startIndices,
        rewriter.getI64TensorAttr(sliceSizes));
    return success();
  }
};

struct CanonicalizeRealDynamicSliceOpToSliceOpPattern
    : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RealDynamicSliceOp op,
                                PatternRewriter& rewriter) const override {
    // SliceOp supports dynamic shapes for operands and results, so we
    // don't check for that here unlike in some other patterns in this pass.
    SmallVector<int64_t> startIndices, limitIndices, strides;
    if (!succeeded(hlo::matchInts(op.getStartIndices(), startIndices)))
      return rewriter.notifyMatchFailure(op, "expected static start");
    if (!succeeded(hlo::matchInts(op.getLimitIndices(), limitIndices)))
      return rewriter.notifyMatchFailure(op, "expected static limit");
    if (!succeeded(hlo::matchInts(op.getStrides(), strides)))
      return rewriter.notifyMatchFailure(op, "expected static strides");
    rewriter.replaceOpWithNewOp<SliceOp>(
        op, op.getType(), op.getOperand(),
        rewriter.getI64TensorAttr(startIndices),
        rewriter.getI64TensorAttr(limitIndices),
        rewriter.getI64TensorAttr(strides));
    return success();
  }
};

struct StablehloCanonicalizeDynamismPass
    : public impl::StablehloCanonicalizeDynamismPassBase<
          StablehloCanonicalizeDynamismPass> {
  using StablehloCanonicalizeDynamismPassBase::
      StablehloCanonicalizeDynamismPassBase;

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = true;
    config.maxIterations = 2;
    config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    config.strictMode = GreedyRewriteStrictness::AnyOp;

    RewritePatternSet patterns(&getContext());
    patterns.add<CanonicalizeCustomCallOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicBroadcastInDimOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicConvOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicGatherOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicIotaOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicPadOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicReduceWindowOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicReshapeOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicRngBitGeneratorOpPattern>(&getContext());
    patterns.add<CanonicalizeDynamicTopKOpPattern>(&getContext());
    patterns.add<CanonicalizeRealDynamicSliceOpToDynamicSliceOpPattern>(
        &getContext());
    patterns.add<CanonicalizeRealDynamicSliceOpToSliceOpPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace experimental
}  // namespace stablehlo
}  // namespace mlir
