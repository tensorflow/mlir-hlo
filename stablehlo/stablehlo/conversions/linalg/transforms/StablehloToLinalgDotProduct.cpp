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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/LegalizeToLinalgUtils.h"
#include "stablehlo/conversions/linalg/transforms/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo {
namespace {

template <typename LinalgOpTy, typename StablehloOpTy>
bool opMatchesLinalgTarget(StablehloOpTy op) {
  ArrayRef<int64_t> lhsShape = op.getLhs().getType().getShape();
  ArrayRef<int64_t> rhsShape = op.getRhs().getType().getShape();
  auto areCompatible = [](int64_t a, int64_t b) {
    return a == ShapedType::kDynamic || b == ShapedType::kDynamic || a == b;
  };
  if (lhsShape.size() == 1 && rhsShape.size() == 1 &&
      areCompatible(lhsShape[0], rhsShape[0])) {
    return std::is_same<LinalgOpTy, linalg::DotOp>::value;
  }
  if (lhsShape.size() == 2 && rhsShape.size() == 1 &&
      areCompatible(lhsShape[1], rhsShape[0])) {
    return std::is_same<LinalgOpTy, linalg::MatvecOp>::value;
  }
  if (lhsShape.size() == 1 && rhsShape.size() == 2 &&
      areCompatible(lhsShape[0], rhsShape[0])) {
    return std::is_same<LinalgOpTy, linalg::VecmatOp>::value;
  }
  if (lhsShape.size() == 2 && rhsShape.size() == 2 &&
      areCompatible(lhsShape[1], rhsShape[0])) {
    return std::is_same<LinalgOpTy, linalg::MatmulOp>::value;
  }
  return false;
}

template <typename LinalgOp>
SmallVector<Value, 2> getDotOpEmptyTensorDynSizes(OpBuilder& b, Location loc,
                                                  Value lhs, Value rhs) {
  SmallVector<Value, 2> dynShape;

  auto lhsType = cast<ShapedType>(lhs.getType());
  auto rhsType = cast<ShapedType>(rhs.getType());

  auto lhsIsMatrix = std::is_same<LinalgOp, linalg::MatvecOp>::value;
  auto rhsIsMatrix = std::is_same<LinalgOp, linalg::VecmatOp>::value;
  if (std::is_same<LinalgOp, linalg::MatmulOp>::value) {
    lhsIsMatrix = rhsIsMatrix = true;
  }

  if (lhsIsMatrix && lhsType.isDynamicDim(0))
    dynShape.push_back(tensor::DimOp::create(b, loc, lhs, 0));
  if (rhsIsMatrix && rhsType.isDynamicDim(1))
    dynShape.push_back(tensor::DimOp::create(b, loc, rhs, 1));
  return dynShape;
}

template <typename OpTy, typename LinalgOpTy>
LogicalResult lowerDotOp(ConversionPatternRewriter& rewriter,
                         const TypeConverter* typeConverter, OpTy op,
                         typename OpTy::Adaptor adaptor) {
  if (!opMatchesLinalgTarget<LinalgOpTy>(op)) return failure();

  auto loc = op.getLoc();

  // Convert unsigned to signed. This works because signed and unsigned
  // integer matmul is the same operation in two's complement.
  auto outputType = cast<ShapedType>(typeConverter->convertType(op.getType()));

  SmallVector<Value, 2> dynShape = getDotOpEmptyTensorDynSizes<LinalgOpTy>(
      rewriter, loc, adaptor.getLhs(), adaptor.getRhs());

  Value emptyTensor =
      !sparse_tensor::getSparseTensorEncoding(outputType)
          ? getEmptyTensor(rewriter, loc, outputType, dynShape)
          : getEmptySparseTensor(rewriter, loc, outputType, dynShape);
  Value zeroTensor = fillTensorWithZeros(rewriter, loc, emptyTensor);

  rewriter.replaceOpWithNewOp<LinalgOpTy>(
      op, TypeRange{outputType}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
      ValueRange{zeroTensor}, linalg::getPrunedAttributeList(op));
  return success();
}

template <typename LinalgOpTy>
struct DotOpConversion final : OpConversionPattern<mlir::stablehlo::DotOp> {
  using OpConversionPattern<mlir::stablehlo::DotOp>::OpConversionPattern;
  using OpAdaptor = mlir::stablehlo::DotOp::Adaptor;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::DotOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    return lowerDotOp<DotOp, LinalgOpTy>(rewriter, getTypeConverter(), op,
                                         adaptor);
  }
};

struct DotGeneralBatchMatMulOpConversion final
    : OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::DotGeneralOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (op.getType().getRank() != 3)
      return rewriter.notifyMatchFailure(op, "expected a batch matmul");

    if (op.getAlgorithm().has_value())
      return rewriter.notifyMatchFailure(
          op, "dot algorithms not yet supported in linalg conversion");

    mlir::stablehlo::DotDimensionNumbersAttr dimNumbers =
        op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims =
        dimNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dimNumbers.getRhsContractingDimensions();
    if (lhsBatchingDims.size() != 1 || lhsBatchingDims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected lhs batching dimensions exactly {0}");
    }
    if (rhsBatchingDims.size() != 1 || rhsBatchingDims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected rhs batching dimensions exactly {0}");
    }
    if (lhsContractingDims.size() != 1 || lhsContractingDims[0] != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected lhs contracting dimensions exactly {2}");
    }
    if (rhsContractingDims.size() != 1 || rhsContractingDims[0] != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected rhs contracting dimensions exactly {1}");
    }

    Location loc = op.getLoc();
    // Convert unsigned to signed. This works because signed and unsigned
    // integer matmul is the same operation in two's complement.
    auto outputType =
        cast<ShapedType>(typeConverter->convertType(op.getType()));
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, outputType, op, adaptor.getOperands());
    Value zeroTensor = fillTensorWithZeros(rewriter, loc, emptyTensor);
    Operation* linalgOp = linalg::BatchMatmulOp::create(
        rewriter, loc, /*resultTensorTypes=*/TypeRange{outputType},
        /*inputs=*/ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        /*outputBuffers=*/ValueRange{zeroTensor},
        linalg::getPrunedAttributeList(op));

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct DotGeneralOpConversion final
    : OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mlir::stablehlo::DotGeneralOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (op.getAlgorithm().has_value())
      return rewriter.notifyMatchFailure(
          op, "dot algorithms not yet supported in linalg conversion");

    if (op.isSimpleDot()) {
      if (succeeded(lowerDotOp<DotGeneralOp, linalg::MatmulOp>(
              rewriter, getTypeConverter(), op, adaptor)))
        return success();
      if (succeeded(lowerDotOp<DotGeneralOp, linalg::MatvecOp>(
              rewriter, getTypeConverter(), op, adaptor)))
        return success();
      if (succeeded(lowerDotOp<DotGeneralOp, linalg::VecmatOp>(
              rewriter, getTypeConverter(), op, adaptor)))
        return success();
      if (succeeded(lowerDotOp<DotGeneralOp, linalg::DotOp>(
              rewriter, getTypeConverter(), op, adaptor)))
        return success();
      std::string str;
      llvm::raw_string_ostream os(str);
      os << "supposedly simple DotGeneralOp could not be converted: ";
      op.print(os);
      llvm::report_fatal_error(str.c_str());
    }

    // Get various dimension iterator information
    mlir::stablehlo::DotDimensionNumbersAttr dimNumbers =
        op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims =
        dimNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dimNumbers.getRhsContractingDimensions();

    // Get shape information and initialize output
    assert(lhsContractingDims.size() == rhsContractingDims.size() &&
           "number of contracting dims must be equal");
    size_t numContracting = lhsContractingDims.size();
    // Convert unsigned to signed. This works because signed and unsigned
    // integer matmul is the same operation in two's complement.
    auto outputType =
        cast<ShapedType>(typeConverter->convertType(op.getType()));
    size_t targetRank = outputType.getRank();
    size_t totalLoopCount = numContracting + targetRank;

    int64_t lhsRank =
        llvm::cast<ShapedType>(adaptor.getLhs().getType()).getRank();
    size_t lhsExtraDims =
        lhsRank - lhsBatchingDims.size() - lhsContractingDims.size();
    int64_t rhsRank =
        llvm::cast<ShapedType>(adaptor.getRhs().getType()).getRank();

    Location loc = op.getLoc();
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, outputType, op, adaptor.getOperands());
    Value zeroTensor = fillTensorWithZeros(rewriter, loc, emptyTensor);
    SmallVector<AffineMap, 3> indexingMaps;

    auto getMap = [&](int64_t rank, ArrayRef<int64_t> batchingDims,
                      ArrayRef<int64_t> contractingDims, size_t extraDims) {
      llvm::SmallVector<AffineExpr> indices(rank);
      for (const auto& i : llvm::enumerate(batchingDims)) {
        indices[i.value()] = rewriter.getAffineDimExpr(i.index());
      }
      for (const auto& i : llvm::enumerate(contractingDims)) {
        indices[i.value()] = rewriter.getAffineDimExpr(i.index() + targetRank);
      }
      for (int i = 0; i < rank; ++i) {
        if (!indices[i]) {
          indices[i] = rewriter.getAffineDimExpr(extraDims++);
        }
      }
      indexingMaps.push_back(AffineMap::get(/*dimCount=*/totalLoopCount,
                                            /*symbolCount=*/0, indices,
                                            op->getContext()));
    };
    getMap(lhsRank, lhsBatchingDims, lhsContractingDims,
           lhsBatchingDims.size());
    getMap(rhsRank, rhsBatchingDims, rhsContractingDims,
           rhsBatchingDims.size() + lhsExtraDims);

    {
      SmallVector<AffineExpr> dimExprs;
      dimExprs.reserve(targetRank);
      for (unsigned i = 0; i < targetRank; ++i)
        dimExprs.push_back(rewriter.getAffineDimExpr(i));
      indexingMaps.push_back(AffineMap::get(/*dimCount=*/totalLoopCount,
                                            /*symbolCount=*/0, dimExprs,
                                            op.getContext()));
    }

    Operation* linalgOp = linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/TypeRange{outputType},
        /*inputs=*/ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        /*outputBuffers=*/ValueRange{zeroTensor}, indexingMaps,
        getParallelAndReductionIterators(
            /*nLoops=*/totalLoopCount,
            /*nReduction=*/numContracting),
        [](OpBuilder& b, Location loc, ValueRange) {
          ImplicitLocOpBuilder builder(loc, b);
          linalg::MatmulOp::regionBuilder(builder, *b.getInsertionBlock(), {},
                                          /*emitError=*/{});
        },
        linalg::getPrunedAttributeList(op));

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

}  // namespace

namespace detail {
void populateStablehloDotProdToLinalgConversionPatterns(
    MLIRContext* context, TypeConverter& typeConverter,
    RewritePatternSet* patterns) {
  // Ensure specialized patterns are higher priority than their generic
  // versions.
  patterns->add<
      DotOpConversion<linalg::MatmulOp>, DotOpConversion<linalg::MatvecOp>,
      DotOpConversion<linalg::VecmatOp>, DotOpConversion<linalg::DotOp>,
      DotGeneralBatchMatMulOpConversion>(typeConverter, context,
                                         PatternBenefit(2));
  patterns->add<DotGeneralOpConversion>(typeConverter, context,
                                        PatternBenefit(1));
}
}  // namespace detail
}  // namespace mlir::stablehlo
