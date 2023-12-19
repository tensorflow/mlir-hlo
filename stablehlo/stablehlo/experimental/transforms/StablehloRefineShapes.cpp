/* Copyright 2022 The StableHLO Authors.
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

#include "stablehlo/transforms/StablehloRefineShapes.h"

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/experimental/dialect/StablehloOps.h"
#include "stablehlo/experimental/transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {
namespace experimental {

#define GEN_PASS_DEF_STABLEHLOREFINESHAPESPASS
#include "stablehlo/experimental/transforms/Passes.h.inc"

namespace {

struct RefineDynamicReduceWindowOpPattern
    : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicReduceWindowOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicReduceWindowOpAdaptor op = *maybeOp;

    // At the moment, we only support refining return types using fully static
    // shape values which serves the current use cases well.
    // Support for partially static shape values is left for future work.
    SmallVector<int64_t> windowDimensions, windowStrides, baseDilations,
        windowDilations, padding;
    if (failed(hlo::matchInts(op.getWindowDimensions(), windowDimensions)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant window_dimensions");
    if (failed(hlo::matchInts(op.getWindowStrides(), windowStrides)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant window_strides");
    if (failed(hlo::matchInts(op.getBaseDilations(), baseDilations)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant base_dilations");
    if (failed(hlo::matchInts(op.getWindowDilations(), windowDilations)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant window_dilations");
    if (failed(hlo::matchInts(op.getPadding(), padding)))
      return rewriter.notifyMatchFailure(op, "expected constant padding");

    SmallVector<ShapedTypeComponents> inferredReturnTypes;
    if (failed(hlo::inferReduceWindowOp(
            /*location=*/{}, op.getInputs(), op.getInitValues(),
            rewriter.getI64TensorAttr(windowDimensions),
            rewriter.getI64TensorAttr(windowStrides),
            rewriter.getI64TensorAttr(baseDilations),
            rewriter.getI64TensorAttr(windowDilations),
            hlo::getPaddingAttr(&rewriter, padding), inferredReturnTypes)))
      return rewriter.notifyMatchFailure(op, "inferReduceWindowOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnTypes);
  }
};

struct RefineDynamicRngBitGeneratorOpPattern
    : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicRngBitGeneratorOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicRngBitGeneratorOpAdaptor op = *maybeOp;

    // At the moment, we only support refining return types using fully static
    // shape values which serves the current use cases well.
    // Support for partially static shape values is left for future work.
    auto initialStateType = op.getInitialState().getType().cast<ShapedType>();
    SmallVector<int64_t> outputShape;
    if (failed(hlo::matchInts(op.getOutputShape(), outputShape)))
      return rewriter.notifyMatchFailure(op, "expected constant output_shape");

    // We only need to refine the shape of `output` (the second result).
    // The shape of `output_state` (the first result) is determined by the shape
    // of `initial_state`, so we ignore it and provide an empty refinement.
    return refineReturnTypes(rewriter, op, {{initialStateType}, {outputShape}});
  }
};

struct RefineDynamicTopKOpPattern : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getDynamicTopKOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    DynamicTopKOpAdaptor op = *maybeOp;

    auto operandType = op.getOperand().getType().cast<ShapedType>();
    SmallVector<int64_t> outputShape(operandType.getShape());
    SmallVector<int64_t> k;
    if (failed(hlo::matchInts(op.getK(), k)))
      return rewriter.notifyMatchFailure(op, "expected constant k");

    outputShape[operandType.getRank() - 1] = k[0];
    return refineReturnTypes(rewriter, op, {{outputShape}, {outputShape}});
  }
};

struct RefineTopKOpPattern : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getTopKOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    TopKOpAdaptor op = *maybeOp;

    auto operandType = op.getOperand().getType().cast<ShapedType>();
    SmallVector<int64_t> outputShape(operandType.getShape());
    outputShape.back() = op.getK();
    return refineReturnTypes(rewriter, op, {{outputShape}, {outputShape}});
  }
};

struct RefineTanOpPattern : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp impl,
                                PatternRewriter& rewriter) const override {
    auto maybeOp = getTanOp(impl);
    if (!maybeOp || failed(maybeOp->verify())) return failure();
    TanOpAdaptor op = *maybeOp;
    return refineReturnShape(rewriter, op,
                             op.getOperand().getType().getShape());
  }
};

struct StablehloRefineShapesPass
    : public impl::StablehloRefineShapesPassBase<StablehloRefineShapesPass> {
  using StablehloRefineShapesPassBase::StablehloRefineShapesPassBase;

  void runOnOperation() override {
    auto func = getStablehloRefineShapesTarget(getOperation());
    if (!func) return signalPassFailure();

    // The algorithm behind this pass consists of a single traversal of the
    // function. This is sufficient because we only support one function per
    // program at the moment.
    // TODO(#1048): Find out why .maxIterations = 1 no longer works.
    // There have been recent refactors to applyPatternsAndFoldGreedily
    // upstream, and that might be the reason.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = true;
    config.maxIterations = 2;
    config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    config.strictMode = GreedyRewriteStrictness::AnyOp;

    RewritePatternSet patterns(&getContext());
    populateStablehloRefineShapesPatterns(&patterns, &getContext());
    patterns.add<RefineDynamicReduceWindowOpPattern>(&getContext());
    patterns.add<RefineDynamicRngBitGeneratorOpPattern>(&getContext());
    patterns.add<RefineDynamicTopKOpPattern>(&getContext());
    patterns.add<RefineTanOpPattern>(&getContext());
    patterns.add<RefineTopKOpPattern>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace experimental
}  // namespace stablehlo
}  // namespace mlir
