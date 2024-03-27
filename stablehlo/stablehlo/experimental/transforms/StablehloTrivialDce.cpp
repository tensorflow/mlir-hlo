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

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/experimental/transforms/Passes.h"

namespace mlir {
namespace stablehlo {
namespace experimental {

#define GEN_PASS_DEF_STABLEHLOTRIVIALDCEPASS
#include "stablehlo/experimental/transforms/Passes.h.inc"

namespace {

struct StablehloTrivialDcePass
    : public impl::StablehloTrivialDcePassBase<StablehloTrivialDcePass> {
  using StablehloTrivialDcePassBase::StablehloTrivialDcePassBase;

  void runOnOperation() override {
    GreedyRewriteConfig config;

    // Hardcode defaults for stability.
    config.enableRegionSimplification = true;
    config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    config.strictMode = GreedyRewriteStrictness::AnyOp;

    // Run a single bottom up pass.
    config.useTopDownTraversal = false;
    config.maxIterations = 1;

    // Running a greedy rewrite will cause trivially dead values to be removed.
    // Doing it without patterns ensures that no other changes are made to the
    // IR. Doing it bottom-up ensures that values that are transitively dead are
    // also removed. Although 1 pass should be enough,
    // applyPatternsAndFoldGreedily will want to run at least 1 more iteration
    // to confirm convergence, but we don't need to check for convergence, so we
    // ignore the return value.
    (void)applyPatternsAndFoldGreedily(getOperation(), RewritePatternSet(&getContext()), config);
  }
};

}  // namespace
}  // namespace experimental
}  // namespace stablehlo
}  // namespace mlir
