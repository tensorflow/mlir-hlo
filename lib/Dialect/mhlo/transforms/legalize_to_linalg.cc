/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering HLO/LHLO dialect to Linalg dialect.

#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace {

template <typename OpTy>
SmallVector<NamedAttribute> PruneAttributeList(OpTy op) {
  auto op_attributes = op.getAttributeNames();
  llvm::StringSet<> elided_attrs;
  elided_attrs.insert(op_attributes.begin(), op_attributes.end());
  SmallVector<NamedAttribute> preserved_attrs;
  for (auto attr : op->getAttrs()) {
    if (elided_attrs.count(attr.getName())) continue;
    preserved_attrs.push_back(attr);
  }
  return preserved_attrs;
}

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
SmallVector<StringRef, 3> GetParallelAndReductionIterators(
    unsigned nLoops, unsigned nReduction) {
  SmallVector<StringRef, 3> res(nLoops - nReduction,
                                getParallelIteratorTypeName());
  res.append(nReduction, getReductionIteratorTypeName());
  return res;
}

SmallVector<StringRef, 3> GetNParallelLoopsAttrs(unsigned nParallelLoops) {
  return GetParallelAndReductionIterators(nParallelLoops, 0);
}

Value GetResultValue(Operation* op) { return op->getResult(0); }

ShapedType GetHloOpResultType(Operation* op) {
  return GetResultValue(op).getType().cast<ShapedType>();
}

bool VerifyHloOpBufferOrTensorSemantics(Operation* op) {
  auto verify_type = [&](Value val) -> bool {
    return val.getType().isa<RankedTensorType>();
  };
  if (!llvm::all_of(op->getOperands(), verify_type)) return false;
  return llvm::all_of(op->getResults(), verify_type);
}

Value GetInitTensor(OpBuilder& b, Location loc, ShapedType type,
                    ArrayRef<Value> dyn_sizes) {
  return b.create<linalg::InitTensorOp>(loc, dyn_sizes, type.getShape(),
                                        type.getElementType());
}

Value GetInitTensorFor(OpBuilder& b, Location loc, ShapedType result_type,
                       Operation* op, ValueRange operands) {
  SmallVector<Value> dyn_sizes;
  if (result_type.hasRank() && !result_type.hasStaticShape()) {
    // Ask the op for its output shape.
    auto shape_source = cast<InferShapedTypeOpInterface>(op);
    SmallVector<Value, 1> reified_shapes;
    (void)shape_source.reifyReturnTypeShapes(b, operands, reified_shapes);
    assert(reified_shapes.size() == 1 && "Expected one reified result");

    for (auto& en : llvm::enumerate(result_type.getShape())) {
      if (en.value() != ShapedType::kDynamicSize) continue;
      dyn_sizes.push_back(b.create<tensor::ExtractOp>(
          loc, reified_shapes[0],
          ValueRange{b.create<arith::ConstantIndexOp>(loc, en.index())}));
    }
  }
  return GetInitTensor(b, loc, result_type, dyn_sizes);
}

SmallVector<int64_t, 4> Extract1DVector(DenseIntElementsAttr elements) {
  SmallVector<int64_t, 4> ret;
  for (const APInt& element : elements) {
    ret.push_back(element.getLimitedValue());
  }
  return ret;
}

/// Returns the constant value associated with the init value if the defining
/// operation is a constant.
Attribute GetInitValueAsConst(Value init) {
  DenseElementsAttr attr;
  if (!matchPattern(init, m_Constant(&attr))) return {};
  auto type = attr.getType().dyn_cast<ShapedType>();
  if (!type || type.getRank() != 0) return {};
  return attr.getValues<Attribute>()[0];
}

/// Returns a permutation AffineMap that puts all reduction dimensions to the
/// last. The order of parallel loops and reduction loops are all sorted. E.g.,
/// if `rank` is 4 and `reductionDims` is {1, 3}, then
/// "(d0, d1, d2, d3) -> (d0, d2, d1, d3)" is used. The inverse permutation of
/// the AffineMap is returned.
AffineMap GetTransposeMapForReduction(MLIRContext* context, int rank,
                                      ArrayRef<int64_t> reduction_dims) {
  llvm::SmallSetVector<int, 4> s;
  for (auto dim : reduction_dims) s.insert(dim);

  SmallVector<unsigned, 4> permutation;
  for (int i = 0; i < rank; ++i)
    if (!s.count(i)) permutation.push_back(i);
  for (auto dim : reduction_dims) permutation.push_back(dim);

  auto map = AffineMap::getPermutationMap(permutation, context);
  return inversePermutation(map);
}

/// Returns true if the given `attr` is a splat of the given `value`.
bool isSplatValue(DenseIntElementsAttr attr, uint64_t value) {
  return attr.isSplat() && attr.getSplatValue<uint64_t>() == value;
}

/// Returns true if the given `dimensionNumbers` from a mhlo.convolution op
/// follows a canonical form:
///
/// * Input dimensions have order: (batch_count, spatial_dims,
///   input_channel_count).
/// * Filter dimensions have order: (spatial_dims, input_channel_count,
///   output_channel_count).
/// * Output dimensions have order: (batch_count, spatial_dims,
///   output_channel_count).
static bool HasCanonicalDimensionNumbers(
    mhlo::ConvDimensionNumbersAttr dimension_numbers) {
  const int input_spatial_rank =
      llvm::size(dimension_numbers.getInputSpatialDimensions());
  // The dimensions for input should follow the order of
  // batch_count, spatial_dims..., input_feature_count.
  if (dimension_numbers.getInputBatchDimension() != 0 ||
      dimension_numbers.getInputFeatureDimension() !=
          (input_spatial_rank + 1)) {
    return false;
  }

  const int kernel_spatial_rank =
      llvm::size(dimension_numbers.getKernelSpatialDimensions());
  // The dimensions for filter should follow the order of
  // spatial_dims..., input_feature_count, num_output_feature_count.
  if (dimension_numbers.getKernelInputFeatureDimension() !=
          kernel_spatial_rank ||
      dimension_numbers.getKernelOutputFeatureDimension() !=
          (kernel_spatial_rank + 1)) {
    return false;
  }

  const int output_spatial_rank =
      llvm::size(dimension_numbers.getOutputSpatialDimensions());
  // The dimensions for output should follow the order of
  // batch_count, spatial_dims.., output_feature_count.
  if (dimension_numbers.getOutputBatchDimension() != 0 ||
      dimension_numbers.getOutputFeatureDimension() !=
          (output_spatial_rank + 1)) {
    return false;
  }

  if (input_spatial_rank != output_spatial_rank ||
      input_spatial_rank != kernel_spatial_rank) {
    return false;
  }

  auto input_spatial_dim =
      dimension_numbers.getInputSpatialDimensions().begin();
  auto kernel_spatial_dim =
      dimension_numbers.getKernelSpatialDimensions().begin();
  auto output_spatial_dim =
      dimension_numbers.getOutputSpatialDimensions().begin();
  // Check spatial dims are ordered correctly.
  for (int i = 0; i < input_spatial_rank; ++i) {
    const int dim = i + 1;
    if ((*input_spatial_dim++) != dim || (*output_spatial_dim++) != dim ||
        (*kernel_spatial_dim++) != i) {
      return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// mhlo.RngUniformOp conversion patterns.
//===----------------------------------------------------------------------===//

// Pass to lower from rng_uniform to stateless uniform pseudo RNG with LCG
// algorithm
struct RngUniformConversion : public OpConversionPattern<mhlo::RngUniformOp> {
  using OpConversionPattern<mhlo::RngUniformOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::RngUniformOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // TODO(raikonenfnu): Handle other element types as well.
    auto min_ty = adaptor.getOperands()[0].getType().dyn_cast<ShapedType>();
    auto max_ty = adaptor.getOperands()[0].getType().dyn_cast<ShapedType>();
    if (!min_ty.getElementType().dyn_cast<FloatType>() ||
        !max_ty.getElementType().dyn_cast<FloatType>()) {
      return rewriter.notifyMatchFailure(
          op, "expected min/max for rng op to be FloatType");
    }
    auto target_ty = this->typeConverter->convertType(op.getResult().getType())
                         .cast<ShapedType>();
    if (!target_ty) {
      return rewriter.notifyMatchFailure(
          op, "expected target shape of rng op to be ShapedType");
    }
    auto loc = op.getLoc();
    Value init_tensor =
        GetInitTensorFor(rewriter, loc, target_ty, op, adaptor.getOperands());
    // Creates index map using target matrix's rank.
    auto target_rank = target_ty.getRank();
    SmallVector<AffineMap, 3> indexing_maps(
        2, AffineMap::get(target_rank, /*symbolCount=*/0,
                          SmallVector<AffineExpr>({}), rewriter.getContext()));
    indexing_maps.push_back(rewriter.getMultiDimIdentityMap(target_rank));
    const int kInitialSeed = 0;
    // Generic region with LCG Algorithm that make use of element index from:
    // https://reviews.llvm.org/D101364
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensors=*/target_ty,
        /*inputs=*/
        ValueRange{adaptor.getOperands()[0], adaptor.getOperands()[1]},
        /*outputs=*/init_tensor, indexing_maps,
        GetParallelAndReductionIterators(/*nLoops=*/target_rank,
                                         /*nReduction=*/0),
        [&](OpBuilder& b, Location loc, ValueRange args) {
          llvm::SmallVector<Value> update_vec = {b.create<arith::ConstantOp>(
              loc, b.getI32IntegerAttr(kInitialSeed))};
          Value multiplier =
              b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(1103515245));
          Value incrementStep =
              b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(12345));
          // For output matrix with rank N:
          // temp1 = (cast(I32, index(D.0)) + seed) * mult + incr
          // ...
          // tempN = (cast(I32, index(D.(N))) + tempN_1) * mult + incr
          for (int i = 0; i < target_rank; i++) {
            Value update = update_vec.back();
            Value ind = b.create<linalg::IndexOp>(loc, i);
            Value cast_ind =
                b.create<arith::IndexCastOp>(loc, b.getI32Type(), ind);
            Value add_res = b.create<arith::AddIOp>(loc, cast_ind, update);
            Value mult_res = b.create<arith::MulIOp>(loc, add_res, multiplier);
            Value inc_res =
                b.create<arith::AddIOp>(loc, mult_res, incrementStep);
            update_vec.push_back(inc_res);
          }
          // Scaling = (max - min) * const(F64, 2.3283064E-10)
          // which is derived from rand(min,max) = rand()/(RAND_MAX/(max-min)).
          Value epsilon = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(args[0].getType(), 2.3283064E-10));
          Value range = b.create<arith::SubFOp>(loc, args[1], args[0]);
          Value scale = b.create<arith::MulFOp>(loc, range, epsilon);
          // Res = cast(T, cast(F64, tempN) * scaling + min)
          Value update_cast = b.create<arith::UIToFPOp>(
              loc, target_ty.getElementType(), update_vec.back());
          Value scale_update = b.create<arith::MulFOp>(loc, update_cast, scale);
          Value res = b.create<arith::AddFOp>(loc, scale_update, args[0]);
          b.create<linalg::YieldOp>(loc, res);
        },
        PruneAttributeList(op));
    rewriter.replaceOp(op, linalg_op.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// mhlo.Einsum conversion patterns.
//===----------------------------------------------------------------------===//

// Looks through a set of dimension that has been marked as reduction axes,
// if it is found within the set, then we set it as "reduction", otherwise
// we can label it as "parallel".
SmallVector<StringRef, 3> GetEinsumLoopsAttrs(
    const llvm::SmallSetVector<StringRef, 4>& input_ind,
    const llvm::SmallSetVector<StringRef, 4>& reduction_dims) {
  SmallVector<StringRef, 3> res;
  for (StringRef dim : input_ind) {
    if (!reduction_dims.contains(dim)) {
      res.push_back(getParallelIteratorTypeName());
    } else {
      res.push_back(getReductionIteratorTypeName());
    }
  }
  return res;
}

SmallVector<Value, 2> ExtractDynamicEinsumSizes(
    OpBuilder& b, Location loc, Value lhs, Value rhs,
    const SmallVector<std::string>& lhs_loop_vec,
    const SmallVector<std::string>& rhs_loop_vec,
    const SmallVector<std::string>& output_loop_vec) {
  SmallVector<Value, 2> dyn_sizes;
  for (const std::string& dim_ind : output_loop_vec) {
    Value dim_size;
    auto dim_ind_it =
        std::find(lhs_loop_vec.begin(), lhs_loop_vec.end(), dim_ind);
    if (dim_ind_it != lhs_loop_vec.end()) {
      // Query from lhs vars.
      auto dim_ind_pos = dim_ind_it - lhs_loop_vec.begin();
      auto lhs_shape = lhs.getType().dyn_cast<RankedTensorType>().getShape();
      if (lhs_shape[dim_ind_pos] != ShapedType::kDynamicSize) continue;
      dim_size = b.create<tensor::DimOp>(loc, lhs, dim_ind_pos);
    } else {
      // query from rhs vars.
      dim_ind_it = std::find(rhs_loop_vec.begin(), rhs_loop_vec.end(), dim_ind);
      auto dim_ind_pos = dim_ind_it - rhs_loop_vec.begin();
      auto rhs_shape = rhs.getType().dyn_cast<RankedTensorType>().getShape();
      if (rhs_shape[dim_ind_pos] != ShapedType::kDynamicSize) continue;
      dim_size = b.create<tensor::DimOp>(loc, rhs, dim_ind_pos);
    }
    dyn_sizes.push_back(dim_size);
  }
  return dyn_sizes;
}

// Adds indices/axes that are missing from output set.
llvm::SmallSetVector<StringRef, 4> FindSummationAxes(
    const llvm::SmallSetVector<StringRef, 4>& input_set,
    const llvm::SmallSetVector<StringRef, 4>& output_set) {
  llvm::SmallSetVector<StringRef, 4> summation_axes;
  for (StringRef ind : input_set) {
    if (!output_set.contains(ind)) summation_axes.insert(ind);
  }
  return summation_axes;
}

// Given a 1:1 map from std::string -> affine dimension expression
// we can get the affine expression of dimensions that an
// operand will access based on the input_str of einsum_config.
// For example:
// let string_dim_umap = {'a' : d0, 'b' : d1, 'c' : d2}
// for einsum_config "abc,cb->acb"
// first_input_operand will get umap[{"a","b","c"}] -> (d0, d1, d2).
// second_input_operand will get umap[{"c","b"}] -> (d2, d1).
// ouput_operand will get umap[{"a","c","b"}] -> (d0, d2, d1).
SmallVector<AffineExpr> GetExprFromConfig(
    const SmallVector<std::string>& loop_dims,
    const DenseMap<StringRef, AffineExpr>& str_affine_dim_umap) {
  SmallVector<AffineExpr> exprs;
  for (const auto& dim : loop_dims) {
    exprs.push_back(str_affine_dim_umap.lookup(dim));
  }
  return exprs;
}

// Convert mhlo.einsum op into linalg.generic.
// Algorithm in general 3 steps:

// Step1) Dissect entire einsum_config to different operands
// e.g f("abc,cd->abd") = {lhs:["abc"], rhs:["cd"], out:["abd"]}.

// Step2) Split up the string into vector of the elements
// e.g {lhs:["abc"], rhs:["cd"], out:["abd"]} = {lhs:["a","b","c"],
// rhs:["c","d"], out:["a","b","d"]}.

// Step3) Convert the vector into data access
// patern represented by affineMaps with affineDimensions e.g
// {lhs:["a","b","c"], rhs:["c","d"], out:["a","b","d"]} = {lhs:[d0,d1,d2],
// rhs:[d2,d3], out:[d0,d1,d3]}.
class EinsumToLinalgConverter : public OpConversionPattern<mhlo::EinsumOp> {
 public:
  using OpConversionPattern<mhlo::EinsumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::EinsumOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto get_rank = [](Value v) {
      return v.getType().cast<ShapedType>().getRank();
    };
    auto einsum_config = op.einsum_config();

    // With the assumption of binary input operand and single output
    // get the inputs and output operands' indices.
    // einsum_config = "lhs_loop,rhs_loop->out_loop"
    std::size_t pos_arrow = einsum_config.find(kArrow);
    std::size_t pos_comma = einsum_config.find(kComma);

    StringRef lhs_loop = einsum_config.substr(0, pos_comma);
    StringRef rhs_loop = einsum_config.substr(
        pos_comma + kComma.size(), pos_arrow - (pos_comma + kComma.size()));
    StringRef out_loop = einsum_config.substr(pos_arrow + kArrow.size());

    // Check for Invalid Configs.
    // 1.Check that there is only maximum 2 inputs
    // 2.Check that there is only maximum 1 output
    // 3.Check that there is 1 kArrow
    if (rhs_loop.find(kComma) != std::string::npos ||
        out_loop.find(kComma) != std::string::npos ||
        out_loop.find(kArrow) != std::string::npos) {
      return rewriter.notifyMatchFailure(op, "Invalid einsum config!");
    }

    // Find result type, if on tensors.
    auto result_ty = this->typeConverter->convertType(GetHloOpResultType(op))
                         .dyn_cast<RankedTensorType>();

    // Check result type compatibility.
    if (!result_ty || !(result_ty.getElementType().isSignlessIntOrFloat())) {
      return rewriter.notifyMatchFailure(op, "Invalid result type");
    }

    // Convert the representation to vector<string>.
    SmallVector<std::string> lhs_ein =
        GetEinsumConfigAsVector(lhs_loop, get_rank(adaptor.lhs()));
    SmallVector<std::string> rhs_ein =
        GetEinsumConfigAsVector(rhs_loop, get_rank(adaptor.rhs()));
    SmallVector<std::string> out_ein =
        GetEinsumConfigAsVector(out_loop, result_ty.getRank());

    if (!CheckBatchHasEqualRank(lhs_ein.size(), lhs_loop, rhs_ein.size(),
                                rhs_loop, out_ein.size(), out_loop)) {
      return rewriter.notifyMatchFailure(
          op, "Invalid elipsis('...') within einsum config!");
    }

    // Find all unique indices in the input and output.
    llvm::SmallSetVector<StringRef, 4> input_ind;
    llvm::SmallSetVector<StringRef, 4> output_ind;

    input_ind.insert(lhs_ein.begin(), lhs_ein.end());
    input_ind.insert(rhs_ein.begin(), rhs_ein.end());
    output_ind.insert(out_ein.begin(), out_ein.end());

    llvm::SmallSetVector<StringRef, 4> reduction_axe =
        FindSummationAxes(input_ind, output_ind);

    // Find input/output values and types.
    auto loc = op.getLoc();

    // Prepare init tensor for linalg.generic op.
    auto dyn_sizes = ExtractDynamicEinsumSizes(
        rewriter, loc, adaptor.lhs(), adaptor.rhs(), lhs_ein, rhs_ein, out_ein);
    Value output = GetInitTensor(rewriter, loc, result_ty, dyn_sizes);
    if (!reduction_axe.empty()) {
      auto zero_attr = rewriter.getZeroAttr(result_ty.getElementType());
      Value zero = rewriter.create<arith::ConstantOp>(loc, zero_attr);
      output = rewriter.create<linalg::FillOp>(loc, zero, output).getResult(0);
    }

    // Create indexing maps.
    // Create a 1:1 map from f:strDimension -> affineDimension.
    int64_t nloops = input_ind.size();
    DenseMap<StringRef, AffineExpr> str_affine_dim_umap;
    for (auto it : llvm::enumerate(input_ind)) {
      str_affine_dim_umap[it.value()] = rewriter.getAffineDimExpr(it.index());
    }

    // From einsum_config of each operand in vector<string>, generate
    // the equivalent vector<AffineExpr>.
    SmallVector<AffineMap, 4> maps;
    for (const SmallVector<std::string>& loop_operand :
         {lhs_ein, rhs_ein, out_ein}) {
      auto exprs = GetExprFromConfig(loop_operand, str_affine_dim_umap);
      maps.push_back(AffineMap::get(nloops, 0, exprs, rewriter.getContext()));
    }

    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, result_ty ? result_ty : TypeRange{}, adaptor.getOperands(), output,
        maps, GetEinsumLoopsAttrs(input_ind, reduction_axe),
        [&](OpBuilder& b, Location nested_loc, ValueRange args) {
          Value result_val =
              b.create<mlir::arith::MulFOp>(nested_loc, args[0], args[1]);
          if (!reduction_axe.empty()) {
            result_val =
                b.create<mlir::arith::AddFOp>(nested_loc, args[2], result_val);
          }
          b.create<linalg::YieldOp>(nested_loc, result_val);
        },
        PruneAttributeList(op));
    rewriter.replaceOp(op, linalg_op.getResults());
    return success();
  }

 private:
  static constexpr StringRef kArrow = "->";
  static constexpr StringRef kComma = ",";
  static constexpr StringRef kEllipsis = "...";

  static bool CheckBatchHasEqualRank(size_t lhs_rank, StringRef lhs_loop,
                                     size_t rhs_rank, StringRef rhs_loop,
                                     size_t out_rank, StringRef out_loop);
  static SmallVector<std::string> GetEinsumConfigAsVector(StringRef loop,
                                                          size_t operand_rank);
};

// Definition of util const member variables.
constexpr StringRef EinsumToLinalgConverter::kArrow;
constexpr StringRef EinsumToLinalgConverter::kComma;
constexpr StringRef EinsumToLinalgConverter::kEllipsis;

// Convert the representation from string/vector<char> to vector<string>.
// i.e ("abc") -> {"a", "b", "c"}. For cases with ellipsis with batch rank 3:
// get loop_dim = f("ab...cde") = {"a","b","0","1","2","c","d","e"}
SmallVector<std::string> EinsumToLinalgConverter::GetEinsumConfigAsVector(
    StringRef loop, size_t operand_rank) {
  SmallVector<std::string> loop_dim;
  size_t pre_elip = loop.find(kEllipsis);
  bool has_elip = pre_elip != std::string::npos;
  if (!has_elip) pre_elip = loop.size();
  // Add the dimension until the end or up to ellipsis if it exist.
  for (int pre_elip_ind = 0; pre_elip_ind < pre_elip; pre_elip_ind++) {
    loop_dim.push_back(loop.substr(pre_elip_ind, 1).str());
  }
  if (!has_elip) return loop_dim;
  // Case where Ellipsis presence:
  size_t non_batch_rank = loop.size() - kEllipsis.size();
  size_t batch_rank = operand_rank - non_batch_rank;
  // Add the batch dimension ("0",...,"N") where N is rank of batch into the
  // loop.
  for (int batch_ind = 0; batch_ind < batch_rank; batch_ind++) {
    loop_dim.push_back(std::to_string(batch_ind));
  }
  // Add the dimension after ellipsis into the loop.
  int post_elip = pre_elip + kEllipsis.size();
  for (int post_elip_ind = post_elip; post_elip_ind < loop.size();
       post_elip_ind++) {
    loop_dim.push_back(loop.substr(post_elip_ind, 1).str());
  }
  return loop_dim;
}

// Returns true if all operand's batch has same rank.
bool EinsumToLinalgConverter::CheckBatchHasEqualRank(
    size_t lhs_rank, StringRef lhs_loop, size_t rhs_rank, StringRef rhs_loop,
    size_t out_rank, StringRef out_loop) {
  SmallVector<int, 3> batch_rank_vec;
  if (lhs_rank != lhs_loop.size()) {
    size_t lhs_batch_rank = lhs_rank - (lhs_loop.size() - kEllipsis.size());
    batch_rank_vec.push_back(lhs_batch_rank);
  }
  if (rhs_rank != rhs_loop.size()) {
    size_t rhs_batch_rank = rhs_rank - (rhs_loop.size() - kEllipsis.size());
    batch_rank_vec.push_back(rhs_batch_rank);
  }
  if (out_rank != out_loop.size()) {
    size_t out_batch_rank = out_rank - (out_loop.size() - kEllipsis.size());
    batch_rank_vec.push_back(out_batch_rank);
  }
  bool batch_has_equal_rank = true;

  // Condition is valid if only 1 operand or less have batches.
  if (batch_rank_vec.size() < 2) return batch_has_equal_rank;
  if (!std::equal(batch_rank_vec.begin() + 1, batch_rank_vec.end(),
                  batch_rank_vec.begin()) &&
      batch_rank_vec.size() > 1)
    batch_has_equal_rank = false;
  return batch_has_equal_rank;
}

template <typename OpTy>
class PointwiseToLinalgConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // Find maximum rank / number of loops.
    auto get_rank = [](Value v) {
      return v.getType().cast<ShapedType>().getRank();
    };
    auto is_scalar = [&](Value v) { return get_rank(v) == 0; };
    auto it = llvm::find_if_not(adaptor.getOperands(), is_scalar);
    Value max_rank_arg =
        it != adaptor.getOperands().end() ? *it : adaptor.getOperands().front();
    int64_t nloops = get_rank(max_rank_arg);

    // Apply only if all operands are scalar or have the same rank. Some ops,
    // like `mhlo.select`, support implicit broadcasting of scalars.
    if (!llvm::all_of(adaptor.getOperands(), [&](Value v) {
          int64_t r = get_rank(v);
          return r == 0 || r == nloops;
        })) {
      return rewriter.notifyMatchFailure(
          op, "Operands must be os same rank or scalar.");
    }

    // Find result type, if on tensors.
    Optional<ShapedType> result_ty;
    result_ty = this->typeConverter->convertType(op->getResultTypes().front())
                    .template dyn_cast<ShapedType>();

    // Check result type compatibility.
    if (!result_ty || !result_ty->hasRank() || result_ty->getRank() != nloops ||
        !(result_ty->getElementType().isSignlessIntOrFloat() ||
          result_ty->getElementType().isa<ComplexType>())) {
      return rewriter.notifyMatchFailure(
          op, "mismatched operand/result types or iterator count");
    }

    // Find input/output values and types.
    auto loc = op.getLoc();
    ValueRange inputs = adaptor.getOperands();
    Value output =
        GetInitTensorFor(rewriter, loc, *result_ty, op, adaptor.getOperands());

    // Create indexing maps.
    AffineMap scalar_map = AffineMap::get(nloops, 0, rewriter.getContext());
    AffineMap id_map = rewriter.getMultiDimIdentityMap(nloops);
    SmallVector<AffineMap, 4> maps;
    for (Value v : inputs) maps.push_back(is_scalar(v) ? scalar_map : id_map);
    maps.push_back(id_map);

    // Build `linalg.generic` op.
    bool failed = false;
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, result_ty ? *result_ty : TypeRange{}, inputs, output, maps,
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
          Type inner_result_ty = getElementTypeOrSelf(output);
          Value inner_result = mhlo::MhloOpToStdScalarOp::map<OpTy>(
              op, inner_result_ty,
              llvm::to_vector<2>(args.take_front(inputs.size())), &rewriter);
          if (inner_result == nullptr) {
            failed = true;
          } else {
            nested_builder.create<linalg::YieldOp>(loc, inner_result);
          }
        },
        PruneAttributeList(op));
    if (failed) return failure();
    rewriter.replaceOp(op, linalg_op->getResults());
    return success();
  }
};

template <typename MhloOp>
class ScalarPointwiseToStandardConverter : public OpConversionPattern<MhloOp> {
 public:
  using OpConversionPattern<MhloOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MhloOp mhlo_op, typename MhloOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = mhlo_op.getLoc();
    auto arg_type =
        mhlo_op.getOperand(0).getType().template dyn_cast<ShapedType>();
    if (!arg_type || !arg_type.getElementType().isSignlessIntOrFloat() ||
        (arg_type.getRank() != 0)) {
      return failure();
    }

    // Create two loads from the input.
    auto lhs = rewriter.create<memref::LoadOp>(loc, mhlo_op.lhs());
    auto rhs = rewriter.create<memref::LoadOp>(loc, mhlo_op.rhs());
    Value op_result = mhlo::MhloOpToStdScalarOp::map<MhloOp>(
        mhlo_op, arg_type.getElementType(), llvm::ArrayRef<Value>{lhs, rhs},
        &rewriter);
    rewriter.create<memref::StoreOp>(loc, op_result, mhlo_op.out());
    rewriter.eraseOp(mhlo_op);
    return success();
  }
};

/// Base class for lowering HLO operations that have one operand and one result,
/// and are semantically equivalent to a copy of the input to the output (like
/// transpose, some reshape, etc.). The derived classes need to provide a method
/// `getIndexingMaps` that returns AffineMaps for the index maps of the input
/// and the output.
template <typename Derived, typename OpTy>
class DataMovementOpConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (!VerifyHloOpBufferOrTensorSemantics(op)) return failure();
    auto result_type = GetHloOpResultType(op);
    result_type = this->typeConverter->convertType(result_type)
                      .template cast<ShapedType>();

    SmallVector<AffineMap, 2> indexing_maps =
        Derived::getIndexingMaps(op, &rewriter);
    if (indexing_maps.empty()) return failure();

    auto nloops = result_type.getRank();
    auto loc = op.getLoc();
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/result_type,
        /*inputs=*/adaptor.getOperands().front(),
        /*outputBuffers=*/

        ValueRange{GetInitTensorFor(rewriter, loc, result_type, op,
                                    adaptor.getOperands())},
        indexing_maps, GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
          nested_builder.create<linalg::YieldOp>(loc, *args.begin());
        },
        PruneAttributeList(op));
    rewriter.replaceOp(op, linalg_op.getOperation()->getResults());
    return success();
  }
};

/// Pattern to convert BroadcastOp to Linalg ops.
template <typename OpTy>
class BroadcastConverter
    : public DataMovementOpConverter<BroadcastConverter<OpTy>, OpTy> {
 public:
  using DataMovementOpConverter<BroadcastConverter,
                                OpTy>::DataMovementOpConverter;

  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy broadcast_op,
                                                   Builder* b) {
    ShapedType input_type =
        broadcast_op.operand().getType().template cast<ShapedType>();
    unsigned input_rank = input_type.getRank();
    unsigned nloops = GetHloOpResultType(broadcast_op).getRank();

    // BroadcastOp prepends the dimensions in the `broadcast_sizes` attribute to
    // the input's dimensions.
    unsigned num_prepended_dims = llvm::size(broadcast_op.broadcast_sizes());
    SmallVector<AffineExpr, 4> input_dim_exprs;
    input_dim_exprs.reserve(input_rank);
    for (unsigned i = 0; i < input_rank; ++i) {
      input_dim_exprs.push_back(b->getAffineDimExpr(num_prepended_dims + i));
    }

    AffineMap input_map;
    MLIRContext* context = b->getContext();
    if (input_dim_exprs.empty()) {
      // The input is a scalar, i.e. this is a scalar broadcast op.
      input_map = AffineMap::get(nloops, /*symbolCount=*/0, context);
    } else {
      input_map =
          AffineMap::get(nloops, /*symbolCount=*/0, input_dim_exprs, context);
    }
    return {input_map, b->getMultiDimIdentityMap(nloops)};
  }
};

class HloBroadcastInDimConverter
    : public DataMovementOpConverter<HloBroadcastInDimConverter,
                                     mhlo::BroadcastInDimOp> {
 public:
  using DataMovementOpConverter<
      HloBroadcastInDimConverter,
      mhlo::BroadcastInDimOp>::DataMovementOpConverter;

  static SmallVector<AffineMap, 2> getIndexingMaps(
      mhlo::BroadcastInDimOp broadcast_op, Builder* b) {
    auto result_type = GetHloOpResultType(broadcast_op);
    auto operand_type =
        broadcast_op.operand().getType().template cast<ShapedType>();
    unsigned nloops = result_type.getRank();

    // The input is a scalar, i.e. this is a scalar broadcast op.
    if (operand_type.getRank() == 0) {
      return {AffineMap::get(nloops, /*symbolCount=*/0, b->getContext()),
              b->getMultiDimIdentityMap(nloops)};
    }

    auto operand_shape = operand_type.getShape();
    SmallVector<AffineExpr, 4> dim_exprs;
    dim_exprs.reserve(nloops);

    if (broadcast_op.broadcast_dimensions()) {
      for (const auto& broadcastDim :
           enumerate(broadcast_op.broadcast_dimensions().getValues<APInt>())) {
        int size = broadcastDim.value().getSExtValue();
        bool expansion_needed = operand_shape[broadcastDim.index()] == 1 &&
                                result_type.getShape()[size] != 1;
        dim_exprs.push_back(expansion_needed ? b->getAffineConstantExpr(0)
                                             : b->getAffineDimExpr(size));
      }
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, dim_exprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

class HloDynamicBroadcastInDimConverter
    : public OpConversionPattern<mhlo::DynamicBroadcastInDimOp> {
 public:
  using OpConversionPattern<mhlo::DynamicBroadcastInDimOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicBroadcastInDimOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // If the input has a static shape we know exactly when the broadcast must
    // expand (the dimension is 1, which also trivially expands to 1) or will
    // never expand (the dimension is not 1). This means we can lower the
    // broadcast just as we would lower a fully static broadcast and go directly
    // to linalg.generic. This also covers the important case of broadcasting a
    // scalar.

    // Ideally the pattern (`mhlo.constant` -> `mhlo.dynamic_broadcast_in_dim`)
    // should be converted to an Tensor-dialect op similar to TF ConstantLikeOp.

    Value operand = adaptor.operand();
    auto operand_type = operand.getType().dyn_cast<RankedTensorType>();
    if (!operand_type || !operand_type.hasStaticShape()) return failure();

    auto result_type =
        typeConverter->convertType(op.getType()).dyn_cast<RankedTensorType>();
    if (!result_type) return failure();

    Location loc = op.getLoc();
    int64_t nloops = result_type.getRank();
    auto operand_shape = operand_type.getShape();
    SmallVector<AffineExpr, 4> dim_exprs;
    dim_exprs.reserve(nloops);

    if (op.broadcast_dimensions()) {
      for (const auto& broadcast_dim :
           enumerate(op.broadcast_dimensions().getValues<APInt>())) {
        int64_t size = broadcast_dim.value().getSExtValue();
        bool expansion_needed = operand_shape[broadcast_dim.index()] == 1;
        dim_exprs.push_back(expansion_needed ? rewriter.getAffineConstantExpr(0)
                                             : rewriter.getAffineDimExpr(size));
      }
    }

    Value init =
        GetInitTensorFor(rewriter, loc, result_type, op, adaptor.getOperands());
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, TypeRange{init.getType()}, ValueRange{operand},
        /*outputBuffers=*/ValueRange{init},
        llvm::makeArrayRef(
            {AffineMap::get(/*dimCount=*/nloops, /*symbolCount=*/0, dim_exprs,
                            rewriter.getContext()),
             rewriter.getMultiDimIdentityMap(nloops)}),
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
          nested_builder.create<linalg::YieldOp>(loc, *args.begin());
        },
        PruneAttributeList(op));
    return success();
  }
};

template <typename OpTy>
class TransposeConverter
    : public DataMovementOpConverter<TransposeConverter<OpTy>, OpTy> {
 public:
  using DataMovementOpConverter<TransposeConverter<OpTy>,
                                OpTy>::DataMovementOpConverter;
  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy op, Builder* b) {
    auto result_type = GetHloOpResultType(op).template cast<ShapedType>();
    auto nloops = result_type.getRank();
    SmallVector<AffineExpr, 2> input_exprs;
    input_exprs.resize(result_type.getRank());
    for (auto permutation : llvm::enumerate(op.permutation())) {
      input_exprs[permutation.value().getZExtValue()] =
          b->getAffineDimExpr(permutation.index());
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, input_exprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

// Converts reshape ops that can be proven to be either a collapse of dimensions
// or expansion of dimensions of the operand.
class ReshapeOpConverter : public OpConversionPattern<mhlo::ReshapeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ReshapeOp reshape_op, mhlo::ReshapeOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (!VerifyHloOpBufferOrTensorSemantics(reshape_op)) return failure();
    auto operand = adaptor.operand();
    auto operand_type = operand.getType().cast<ShapedType>();
    auto elem_type = operand_type.getElementType();
    auto result_type = reshape_op.getType().cast<ShapedType>();

    if (!result_type.hasStaticShape()) return failure();

    result_type = typeConverter->convertType(result_type).cast<ShapedType>();

    if (result_type.getRank() == 0 && !operand_type.hasStaticShape()) {
      // This means all dimensions of the operand need to be 1. We add a cast to
      // cast the dynamic dimensions to 1.
      auto static_type = RankedTensorType::get(
          llvm::SmallVector<int64_t>(operand_type.getRank(), 1), elem_type);
      operand = rewriter.create<tensor::CastOp>(reshape_op.getLoc(),
                                                static_type, operand);
      operand_type = static_type;
    }

    // Compute the reassociation maps for the linalg operation. This will
    // succeed if the reshape can be done with a single expand_shape or
    // collapse_shape.
    if (Optional<SmallVector<ReassociationIndices>> reassociation_map =
            getReassociationIndicesForReshape(operand_type, result_type)) {
      if (result_type.getRank() < operand_type.getRank()) {
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
            reshape_op, result_type, operand, *reassociation_map);
      } else {
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            reshape_op, result_type, operand, *reassociation_map);
      }
      return success();
    }

    Value collapsed_op = operand;
    Location loc = reshape_op.getLoc();
    auto get_identity_exprs = [&rewriter](int64_t n) {
      SmallVector<AffineExpr, 4> exprs;
      for (int i = 0; i < n; ++i) exprs.push_back(rewriter.getAffineDimExpr(i));
      return exprs;
    };
    // Otherwise, we need to first reduce all source dimensions into one and
    // then expand to the destination dimensions. If there is only a single
    // source dimension, the reduce step can be skipped. TensorCollapseShape
    // expects a different rank of operand and result.
    if (operand_type.getRank() != 1) {
      SmallVector<ReassociationExprs, 4> collapsing_map = {
          // Use operand_type here because we need to collapse all operands
          // dimensions.
          get_identity_exprs(operand_type.getRank())};

      collapsed_op = rewriter.create<tensor::CollapseShapeOp>(loc, operand,
                                                              collapsing_map);
    }
    // Cast to a known static type if the input has dynamic dimensions.
    int64_t total_elems = result_type.getNumElements();
    auto collapsed_type = RankedTensorType::get({total_elems}, elem_type);
    collapsed_op =
        rewriter.create<tensor::CastOp>(loc, collapsed_type, collapsed_op);
    if (result_type.getRank() == 1) {
      rewriter.replaceOp(reshape_op, collapsed_op);
    } else {
      SmallVector<ReassociationExprs, 4> expanding_map = {
          // Use result_type here because we need to expand to all result
          // dimensions.
          get_identity_exprs(result_type.getRank())};
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
          reshape_op, result_type, collapsed_op, expanding_map);
    }
    return success();
  }
};

template <typename OpTy>
class IotaConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy iota_op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    ShapedType result_shaped_type = GetHloOpResultType(iota_op);
    if (!result_shaped_type) return failure();
    result_shaped_type = this->typeConverter->convertType(result_shaped_type)
                             .template dyn_cast<ShapedType>();

    auto result_element_type = result_shaped_type.getElementType();
    if (!result_element_type.isSignlessIntOrFloat()) return failure();

    // Construct the indexing maps needed for linalg.generic ops.
    unsigned nloops = result_shaped_type.getRank();

    Location loc = iota_op.getLoc();
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/
        ArrayRef<Type>{result_shaped_type},
        /*inputs=*/ValueRange{},
        /*outputBuffers=*/

        ValueRange{GetInitTensorFor(rewriter, loc, result_shaped_type, iota_op,
                                    adaptor.getOperands())},
        llvm::makeArrayRef(rewriter.getMultiDimIdentityMap(nloops)),
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location nested_loc, ValueRange args) {
          Value index_op = nested_builder.create<linalg::IndexOp>(
              nested_loc, iota_op.iota_dimension());
          Value cast_op = nested_builder.create<arith::IndexCastOp>(
              nested_loc, index_op,
              nested_builder.getIntegerType(
                  result_element_type.getIntOrFloatBitWidth()));
          if (result_element_type.template isa<FloatType>()) {
            cast_op = nested_builder.create<arith::SIToFPOp>(
                nested_loc, cast_op, result_element_type);
          }
          nested_builder.create<linalg::YieldOp>(nested_loc, cast_op);
        },
        PruneAttributeList(iota_op));
    rewriter.replaceOp(iota_op, linalg_op.result_tensors());
    return success();
  }
};

/// Converts mhlo.concatenate operation to a linalg.generic op.
struct ConcatenateConverter : public OpConversionPattern<mhlo::ConcatenateOp> {
  using OpConversionPattern<mhlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConcatenateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Shortcut the one-operand case, simplifies code below.
    if (adaptor.getOperands().size() == 1) {
      rewriter.replaceOp(op, adaptor.getOperands()[0]);
      return success();
    }

    auto result_type =
        this->typeConverter->convertType(op.getResult().getType())
            .dyn_cast<RankedTensorType>();
    if (!result_type) return failure();

    uint64_t dim = op.dimension();
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Allocate the output tensor with init_tensor.
    Value result =
        GetInitTensorFor(rewriter, loc, result_type, op, adaptor.getOperands());

    // Generate a generic op to gather the elements of the concatenate. This is
    // awkward standalone but allows fusion with other generic ops.
    int64_t nloops = result_type.getRank();
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op,
        /*resultTensorTypes=*/result_type,
        /*inputs=*/ValueRange{}, /*outputBuffers=*/result,
        llvm::makeArrayRef(rewriter.getMultiDimIdentityMap(nloops)),
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nested_builder, Location loc, ValueRange) {
          OpBuilder b = nested_builder;
          Value concat_dim_size = zero;
          Value result;

          SmallVector<Value, 4> extract_indices;
          extract_indices.reserve(nloops);
          for (int64_t i = 0; i < nloops; i++) {
            extract_indices.push_back(b.create<linalg::IndexOp>(loc, i));
          }

          Value index_op = b.create<linalg::IndexOp>(loc, dim);
          for (auto it : llvm::enumerate(adaptor.getOperands())) {
            Value arg = it.value();
            Value new_concat_dim_size;
            scf::IfOp if_op;
            if (it.index() != (adaptor.getOperands().size() - 1)) {
              // Calculate how far along we have iterated along the concatenate
              // dimension. That way we can tell which input to select.
              new_concat_dim_size = b.create<arith::AddIOp>(
                  loc, concat_dim_size, b.create<tensor::DimOp>(loc, arg, dim));
              Value cmp = b.create<arith::CmpIOp>(
                  loc, rewriter.getI1Type(), arith::CmpIPredicate::ult,
                  index_op, new_concat_dim_size);
              if_op = b.create<scf::IfOp>(loc, result_type.getElementType(),
                                          cmp, true);
              if (result) {
                b.create<scf::YieldOp>(loc, if_op->getResults()[0]);
              } else {
                result = if_op->getResults()[0];
              }

              b = if_op.getThenBodyBuilder(b.getListener());
            }

            // Now adjust the index for the concatenated dimension to fit into
            // the selected tensor and do an extract at that position.
            extract_indices[dim] =
                b.create<arith::SubIOp>(loc, index_op, concat_dim_size);
            Value extract =
                b.create<tensor::ExtractOp>(loc, arg, extract_indices);
            b.create<scf::YieldOp>(loc, extract);

            if (if_op) {
              b = if_op.getElseBodyBuilder(b.getListener());
              concat_dim_size = new_concat_dim_size;
            }
          }
          nested_builder.create<linalg::YieldOp>(loc, result);
        },
        PruneAttributeList(op));
    return success();
  }
};

class ConstConverterTensor : public OpConversionPattern<mhlo::ConstOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConstOp const_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto value_attr = const_op.value().cast<DenseElementsAttr>();
    auto type =
        typeConverter->convertType(const_op.getType()).cast<ShapedType>();
    if (type != const_op.getType()) {
      // Signedness conversion.
      value_attr = value_attr.mapValues(type.getElementType(),
                                        [](const APInt& i) { return i; });
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(const_op, type, value_attr);
    return success();
  }
};

// TODO(b/156787842): Support the lowering for dynamic shapes.
template <typename OpTy>
class ReverseConverter
    : public DataMovementOpConverter<ReverseConverter<OpTy>, OpTy> {
 public:
  using DataMovementOpConverter<ReverseConverter<OpTy>,
                                OpTy>::DataMovementOpConverter;
  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy op, Builder* b) {
    auto result_type = GetHloOpResultType(op).template cast<ShapedType>();
    auto nloops = result_type.getRank();
    SmallVector<AffineExpr, 2> input_exprs;
    input_exprs.reserve(nloops);
    for (int i = 0; i < nloops; ++i)
      input_exprs.push_back(b->getAffineDimExpr(i));
    for (auto dim : op.dimensions()) {
      int i = dim.getZExtValue();
      if (result_type.isDynamicDim(i)) return {};
      int n = result_type.getShape()[i];
      input_exprs[i] = b->getAffineConstantExpr(n - 1) - input_exprs[i];
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, input_exprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

template <typename OpTy>
class SliceConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy slice_op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto arg_type =
        adaptor.getOperands()[0].getType().template dyn_cast<ShapedType>();
    if (!arg_type || !arg_type.hasRank()) {
      return rewriter.notifyMatchFailure(slice_op, "expects known-rank args");
    }

    SmallVector<OpFoldResult, 3> offsets, sizes, strides;
    for (int i = 0, e = arg_type.getRank(); i < e; ++i) {
      auto start = slice_op.start_indices().template getValues<int64_t>()[i];
      auto limit = slice_op.limit_indices().template getValues<int64_t>()[i];
      auto stride = slice_op.strides().template getValues<int64_t>()[i];
      offsets.push_back(rewriter.getI64IntegerAttr(start));
      // Say that there are k elements in total, we have condition:
      //   start + (k - 1) * strides <= limit - 1
      // ->
      //   k <= (limit - 1 - start) / strides + 1
      sizes.push_back(
          rewriter.getI64IntegerAttr((limit - 1 - start) / stride + 1));
      strides.push_back(rewriter.getI64IntegerAttr(stride));
    }
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        slice_op, adaptor.getOperands()[0], offsets, sizes, strides);
    return success();
  }
};

class DynamicSliceConverter : public OpConversionPattern<mhlo::DynamicSliceOp> {
 public:
  using OpConversionPattern<mhlo::DynamicSliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicSliceOp dynamic_slice_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = dynamic_slice_op.getLoc();
    auto arg_type = adaptor.operand().getType().dyn_cast<ShapedType>();
    if (!arg_type || !arg_type.hasRank()) {
      return rewriter.notifyMatchFailure(dynamic_slice_op,
                                         "require known-rank args");
    }

    auto index_type = rewriter.getIndexType();
    SmallVector<OpFoldResult, 3> start_indices, sizes;
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.start_indices()[0]
                                      .getType()
                                      .cast<RankedTensorType>()
                                      .getElementType()));
    for (auto en : llvm::enumerate(
             llvm::zip(adaptor.start_indices(),
                       dynamic_slice_op.slice_sizes().getValues<int64_t>()))) {
      int64_t size = std::get<1>(en.value());
      sizes.push_back(rewriter.getI64IntegerAttr(size));

      // By mhlo.DynamicSlice definition:
      //   `start_indices[i] = clamp(start_indices[i],
      //       0, operand.dimension_size[i] - size_indices[i])`
      Value start_index =
          rewriter.create<tensor::ExtractOp>(loc, std::get<0>(en.value()));
      Value ub = rewriter.createOrFold<tensor::DimOp>(loc, adaptor.operand(),
                                                      en.index());
      // ClampOp lowering does not support index type, so cast it into integer
      // type.
      ub = rewriter.createOrFold<arith::IndexCastOp>(loc, start_index.getType(),
                                                     ub);
      ub = rewriter.createOrFold<arith::SubIOp>(
          loc, ub,
          rewriter.create<arith::ConstantOp>(
              loc, rewriter.getIntegerAttr(start_index.getType(), size)));
      start_index = mhlo::MhloOpToStdScalarOp::map<mhlo::ClampOp>(
          loc, start_index.getType(),
          ArrayRef<Type>{start_index.getType(), start_index.getType(),
                         start_index.getType()},
          ArrayRef<Value>{zero, start_index, ub}, &rewriter);
      start_indices.push_back(
          rewriter.create<arith::IndexCastOp>(loc, index_type, start_index)
              .getResult());
    }

    int64_t rank = arg_type.getRank();
    SmallVector<OpFoldResult, 3> strides(rank, rewriter.getI64IntegerAttr(1));

    auto result_type =
        this->typeConverter->convertType(dynamic_slice_op.getType())
            .cast<RankedTensorType>();

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        dynamic_slice_op, result_type, adaptor.operand(), start_indices, sizes,
        strides);
    return success();
  }
};

/// Lowers mhlo.RealDynamicSliceOp to tensor.extract_slice and other
/// arith/tensor dialect ops.
class RealDynamicSliceConverter
    : public OpConversionPattern<mhlo::RealDynamicSliceOp> {
 public:
  using OpConversionPattern<mhlo::RealDynamicSliceOp>::OpConversionPattern;

  /// Computes size of a slice as :-
  /// size = (limit - 1 - start)/(stride + 1)
  static Value computeSize(Location loc, Value start, Value limit, Value stride,
                           ConversionPatternRewriter& rewriter) {
    Value one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    limit = rewriter.create<arith::SubIOp>(loc, limit, one);
    limit = rewriter.create<arith::SubIOp>(loc, limit, start);
    stride = rewriter.create<arith::AddIOp>(loc, stride, one);
    // TODO: Use FloorDivUIOp once it is added in Arithmetic dialect since the
    //       numerator is guaranteed to be positive.
    return rewriter.create<arith::FloorDivSIOp>(loc, limit, stride);
  }

  LogicalResult matchAndRewrite(
      mhlo::RealDynamicSliceOp real_dynamic_slice_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Location loc = real_dynamic_slice_op.getLoc();
    auto arg_type = adaptor.operand().getType().dyn_cast<ShapedType>();
    if (!arg_type || !arg_type.hasRank()) {
      return rewriter.notifyMatchFailure(real_dynamic_slice_op,
                                         "require known-rank args");
    }
    auto result_type =
        this->typeConverter->convertType(real_dynamic_slice_op.getType())
            .cast<RankedTensorType>();
    Type index_type = rewriter.getIndexType();
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    Type i64_type = rewriter.getI64Type();
    SmallVector<OpFoldResult, 4> offsets, sizes, strides;
    SmallVector<Type, 3> clamp_type(3, i64_type);
    for (auto i : llvm::seq<unsigned>(0, arg_type.getRank())) {
      Value dim =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
      Value start =
          rewriter.create<tensor::ExtractOp>(loc, adaptor.start_indices(), dim);
      Value limit =
          rewriter.create<tensor::ExtractOp>(loc, adaptor.limit_indices(), dim);
      Value stride =
          rewriter.create<tensor::ExtractOp>(loc, adaptor.strides(), dim);

      // Compute i-th dimension size of the result : size[i].
      // If the i-th dimension of the result type is known, we go ahead with it
      // else we compute it using limit, start and stride values.
      int64_t result_dim_size = result_type.getDimSize(i);
      Value size;
      if (result_dim_size != -1)
        size = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(result_dim_size));
      else
        size = computeSize(loc, start, limit, stride, rewriter);

      // Fetch i-th dimension size of the operand and calculate upper bound as
      // :-
      //    ub = operand_dim[i] - size[i]
      Value operand_dim_size =
          rewriter.createOrFold<tensor::DimOp>(loc, adaptor.operand(), dim);
      Value upper_bound =
          rewriter.createOrFold<arith::SubIOp>(loc, operand_dim_size, size);

      // We clamp the start_index to keep it bounded as :-
      // start index : 0 <= start_index[i] <= ub
      // ClampOp lowering does not support index type, so we cast it into
      // integer type.
      start = rewriter.create<arith::IndexCastOp>(loc, i64_type, start);
      upper_bound =
          rewriter.create<arith::IndexCastOp>(loc, i64_type, upper_bound);
      start = mhlo::MhloOpToStdScalarOp::map<mhlo::ClampOp>(
          loc, i64_type, clamp_type, ValueRange{zero, start, upper_bound},
          &rewriter);

      offsets.push_back(
          rewriter.create<arith::IndexCastOp>(loc, index_type, start)
              .getResult());
      if (result_dim_size != -1)
        sizes.push_back(rewriter.getI64IntegerAttr(result_dim_size));
      else
        sizes.push_back(size);
      strides.push_back(stride);
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        real_dynamic_slice_op, result_type, adaptor.operand(), offsets, sizes,
        strides);
    return success();
  }
};

class DynamicUpdateSliceConverter
    : public OpConversionPattern<mhlo::DynamicUpdateSliceOp> {
 public:
  using OpConversionPattern<mhlo::DynamicUpdateSliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto operand_type =
        adaptor.operand().getType().dyn_cast<RankedTensorType>();
    if (!operand_type || !operand_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "require static ranked type for operand");
    }

    auto update_type = adaptor.update().getType().dyn_cast<RankedTensorType>();
    if (!update_type || !update_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "require static ranked type for operand");
    }

    // We do not have to clamp sizes because the semantic of `update`
    // guarantees that it is always in the bounds. See
    // https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice
    SmallVector<OpFoldResult, 3> sizes;
    for (auto size : update_type.getShape()) {
      sizes.push_back(rewriter.getIndexAttr(size));
    }

    auto index_type = rewriter.getIndexType();
    SmallVector<OpFoldResult, 3> start_indices;
    Type start_index_type = adaptor.start_indices()[0]
                                .getType()
                                .cast<RankedTensorType>()
                                .getElementType();
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(start_index_type));
    for (auto en : llvm::enumerate(adaptor.start_indices())) {
      // By mhlo.DynamicUpdateSlice definition:
      //   `start_indices[i] = clamp(start_indices[i],
      //       0, operand.dimension_size[i] - update.dimension_size[i])`
      Value start_index = rewriter.create<tensor::ExtractOp>(loc, en.value());
      Value ub = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(start_index_type,
                                       operand_type.getDimSize(en.index()) -
                                           update_type.getDimSize(en.index())));
      start_index = mhlo::MhloOpToStdScalarOp::map<mhlo::ClampOp>(
          loc, start_index_type,
          ArrayRef<Type>{start_index_type, start_index_type, start_index_type},
          ArrayRef<Value>{zero, start_index, ub}, &rewriter);
      start_indices.push_back(
          rewriter.create<arith::IndexCastOp>(loc, index_type, start_index)
              .getResult());
    }

    int64_t rank = operand_type.getRank();
    SmallVector<OpFoldResult, 3> strides(rank, rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, adaptor.update(), adaptor.operand(), start_indices, sizes, strides);
    return success();
  }
};

enum class DotOperationType {
  kVectorDot = 0,
  kMatrixVector = 1,
  kMatrixMatrix = 2,
  kUnsupported = 3
};

DotOperationType GetDotOperationType(mhlo::DotOp dot_op) {
  ArrayRef<int64_t> lhs_shape =
      dot_op.lhs().getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhs_shape =
      dot_op.rhs().getType().cast<ShapedType>().getShape();
  auto shape_matches = [](int64_t a, int64_t b) {
    return a == ShapedType::kDynamicSize || b == ShapedType::kDynamicSize ||
           a == b;
  };
  if (lhs_shape.size() == 1 && rhs_shape.size() == 1 &&
      shape_matches(lhs_shape[0], rhs_shape[0])) {
    return DotOperationType::kVectorDot;
  }
  if (lhs_shape.size() == 2 && rhs_shape.size() == 1 &&
      shape_matches(lhs_shape[1], rhs_shape[0])) {
    return DotOperationType::kMatrixVector;
  }
  if (rhs_shape.size() == 2 && rhs_shape.size() == 2 &&
      shape_matches(lhs_shape[1], rhs_shape[0])) {
    return DotOperationType::kMatrixMatrix;
  }
  return DotOperationType::kUnsupported;
}

SmallVector<Value, 2> GetDotOpInitTensorDynSizes(OpBuilder& b, Location loc,
                                                 Value lhs, Value rhs,
                                                 DotOperationType type) {
  SmallVector<Value, 2> dyn_shape;
  switch (type) {
    case DotOperationType::kMatrixMatrix: {
      if (lhs.getType().cast<ShapedType>().isDynamicDim(0))
        dyn_shape.push_back(b.create<tensor::DimOp>(loc, lhs, 0));
      if (rhs.getType().cast<ShapedType>().isDynamicDim(1))
        dyn_shape.push_back(b.create<tensor::DimOp>(loc, rhs, 1));
      break;
    }
    case DotOperationType::kMatrixVector: {
      if (lhs.getType().cast<ShapedType>().isDynamicDim(0))
        dyn_shape.push_back(b.create<tensor::DimOp>(loc, lhs, 0));
      break;
    }
    case DotOperationType::kVectorDot:
    case DotOperationType::kUnsupported:
    default: {
      break;
    }
  }
  return dyn_shape;
}

template <DotOperationType op_type, typename LinalgOp>
class DotOpOnTensorsConversion : public OpConversionPattern<mhlo::DotOp> {
 public:
  using OpConversionPattern<mhlo::DotOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::DotOp op, mhlo::DotOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (!VerifyHloOpBufferOrTensorSemantics(op)) {
      return failure();
    }
    if (GetDotOperationType(op) != op_type) return failure();

    Location loc = op.getLoc();
    auto output_type = op.getType().cast<ShapedType>();
    auto output_el_type = output_type.getElementType();
    auto zero_attr = rewriter.getZeroAttr(output_el_type);
    Value zero = rewriter.create<arith::ConstantOp>(loc, zero_attr);
    SmallVector<Value, 2> dyn_shape = GetDotOpInitTensorDynSizes(
        rewriter, loc, adaptor.lhs(), adaptor.rhs(), op_type);
    auto init_tensor = GetInitTensor(rewriter, loc, output_type, dyn_shape);
    Value zero_tensor =
        rewriter.create<linalg::FillOp>(loc, zero, init_tensor).getResult(0);
    rewriter.replaceOpWithNewOp<LinalgOp>(
        op, TypeRange{op.getType()}, ValueRange{adaptor.lhs(), adaptor.rhs()},
        ValueRange{zero_tensor}, PruneAttributeList(op));
    return success();
  }
};

SmallVector<Value, 8> GetDotGeneralOpInitTensorDynSizes(
    OpBuilder& b, Location loc, Value lhs, Value rhs, ShapedType result_type) {
  SmallVector<Value, 8> dyn_shape;
  if (result_type.isDynamicDim(0))
    dyn_shape.push_back(b.create<tensor::DimOp>(loc, lhs, 0));
  if (result_type.isDynamicDim(1))
    dyn_shape.push_back(b.create<tensor::DimOp>(loc, lhs, 1));
  if (result_type.isDynamicDim(2))
    dyn_shape.push_back(b.create<tensor::DimOp>(loc, rhs, 2));
  return dyn_shape;
}

class DotGeneralOpOnTensorsConversion
    : public OpConversionPattern<mhlo::DotGeneralOp> {
 public:
  using OpConversionPattern<mhlo::DotGeneralOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::DotGeneralOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (!VerifyHloOpBufferOrTensorSemantics(op)) {
      return failure();
    }

    mhlo::DotDimensionNumbersAttr dim_numbers = op.dot_dimension_numbers();
    auto lhs_batching_dims = dim_numbers.getLhsBatchingDimensions();
    auto rhs_batching_dims = dim_numbers.getRhsBatchingDimensions();
    auto lhs_contracting_dims = dim_numbers.getLhsContractingDimensions();
    auto rhs_contracting_dims = dim_numbers.getRhsContractingDimensions();
    if (lhs_batching_dims.size() != 1 || lhs_batching_dims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected lhs batching dimensions exactly {0}");
    }
    if (rhs_batching_dims.size() != 1 || rhs_batching_dims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected rhs batching dimensions exactly {0}");
    }
    if (lhs_contracting_dims.size() != 1 || lhs_contracting_dims[0] != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected lhs contracting dimensions exactly {2}");
    }
    if (rhs_contracting_dims.size() != 1 || rhs_contracting_dims[0] != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected rhs contracting dimensions exactly {1}");
    }

    Location loc = op.getLoc();
    auto output_type = op.getType().cast<ShapedType>();
    auto output_el_type = output_type.getElementType();
    SmallVector<Value, 8> dyn_shape = GetDotGeneralOpInitTensorDynSizes(
        rewriter, loc, adaptor.lhs(), adaptor.rhs(), output_type);
    auto zero_attr = rewriter.getZeroAttr(output_el_type);
    Value zero = rewriter.create<arith::ConstantOp>(loc, zero_attr);
    auto init_tensor = GetInitTensor(rewriter, loc, output_type, dyn_shape);
    Value zero_tensor =
        rewriter.create<linalg::FillOp>(loc, zero, init_tensor).getResult(0);
    Operation* linalg_op = rewriter.create<linalg::BatchMatmulOp>(
        loc, /*resultTensorTypes=*/TypeRange{op.getType()},
        /*inputs=*/ValueRange{adaptor.lhs(), adaptor.rhs()},
        /*outputBuffers=*/ValueRange{zero_tensor}, PruneAttributeList(op));

    rewriter.replaceOp(op, linalg_op->getResults());
    return success();
  }
};

bool IsInBodyOfLinalgOps(Operation* op) {
  auto parent_op = op->getParentRegion()->getParentOp();
  return parent_op->getDialect() ==
         parent_op->getContext()->getLoadedDialect<linalg::LinalgDialect>();
}

template <typename OpTy>
struct ReduceRegionXLAOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (!IsInBodyOfLinalgOps(op)) {
      return failure();
    }
    if (!op.getResult().getType().template isa<TensorType>()) return failure();
    if (llvm::all_of(adaptor.getOperands(), [](Value arg) {
          return arg.getType().template isa<TensorType>();
        })) {
      return failure();
    }
    Value result = mhlo::MhloOpToStdScalarOp::map<OpTy>(
        op, getElementTypeOrSelf(op.getType()), adaptor.getOperands(),
        &rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

SmallVector<Value, 8> GetReduceOpInitTensorDynSizes(
    OpBuilder& b, Location loc, Value arg, ShapedType result_type,
    ArrayRef<int64_t> reduction_dims) {
  llvm::SmallSetVector<int, 4> s;
  for (auto dim : reduction_dims) s.insert(dim);

  SmallVector<unsigned, 4> parallel_dims;
  SmallVector<Value, 8> dyn_shape;
  int rank = arg.getType().cast<RankedTensorType>().getRank();
  for (int i = 0, j = 0; i < rank; ++i) {
    if (s.count(i)) continue;
    if (!result_type.isDynamicDim(j++)) continue;
    dyn_shape.push_back(b.create<tensor::DimOp>(loc, arg, i));
  }

  return dyn_shape;
}

class ReduceRegionReturnOpConversion
    : public OpConversionPattern<mhlo::ReturnOp> {
 public:
  using OpConversionPattern<mhlo::ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (!IsInBodyOfLinalgOps(op)) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

class ReduceOnTensorsConversion : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern<mhlo::ReduceOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Location loc = op.getLoc();

    int num_inputs = static_cast<int>(adaptor.inputs().size());
    auto src_type = adaptor.inputs()[0].getType().cast<ShapedType>();
    int src_rank = src_type.getRank();
    if (!src_rank) {
      return rewriter.notifyMatchFailure(op, "expects known-rank args");
    }

    SmallVector<int64_t, 4> reduction_dims = Extract1DVector(op.dimensions());

    SmallVector<Value> inputs, outputs;
    SmallVector<AffineMap, 3> indexing_maps;
    for (int i = 0; i < num_inputs; ++i) {
      Value src = adaptor.inputs()[i];
      if (src.getType() != src_type) return failure();

      // Check if init_value is constant. If so, inline the value into the
      // region.
      Value init_value = adaptor.init_values()[i];
      Attribute init_const_val = GetInitValueAsConst(init_value);
      if (init_const_val) {
        init_value = rewriter.create<arith::ConstantOp>(
            init_value.getDefiningOp()->getLoc(), init_const_val);
      } else {
        init_value = rewriter.create<tensor::ExtractOp>(loc, init_value);
      }

      inputs.push_back(src);
      auto result_type = op.getResult(i).getType().cast<ShapedType>();
      SmallVector<Value, 8> dyn_shape = GetReduceOpInitTensorDynSizes(
          rewriter, loc, src, result_type, reduction_dims);
      auto init_tensor = GetInitTensor(rewriter, loc, result_type, dyn_shape);
      Value filled_tensor =
          rewriter.create<linalg::FillOp>(loc, init_value, init_tensor)
              .result();
      outputs.push_back(filled_tensor);
    }

    // Prepare indexing maps for linalg generic op. The elements are for src
    // and dst. Transpose `src` to make the reduction loops be the innermost,
    // because it's easier to fully utilize processors.
    indexing_maps.append(
        num_inputs, GetTransposeMapForReduction(rewriter.getContext(), src_rank,
                                                reduction_dims));

    // The indexing map of `dst` should drop the reduction loops. Since the
    // reduction loops now are all in the innermost, drops
    // `reduction_dims.size()` dimensions. We don't need an inverse
    // permutation here because they are the same.
    SmallVector<AffineExpr, 4> exprs;
    for (int i = 0, e = src_rank - reduction_dims.size(); i < e; ++i)
      exprs.push_back(rewriter.getAffineDimExpr(i));
    indexing_maps.append(num_inputs,
                         AffineMap::get(src_rank, /*symbolCount=*/0, exprs,
                                        rewriter.getContext()));

    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/op.getResultTypes(), inputs,
        /*outputBuffers=*/ValueRange{outputs}, indexing_maps,
        GetParallelAndReductionIterators(src_rank, reduction_dims.size()),
        /*bodyBuild=*/nullptr, PruneAttributeList(op));

    // Convert the signature of the body. The reduce op region apply function
    // has a signature (lhs, rhs) -> output, all of the same tensor type t.
    // This is converted to a function with the same signature but with
    // element types. E.g., "(tensor<f32>, tensor<f32>) -> tensor<f32>" will
    // be converted to "(f32, f32, f32)".
    Region& region = linalg_op.region();
    rewriter.inlineRegionBefore(op.body(), region, region.end());
    TypeConverter::SignatureConversion signature_converter(num_inputs * 2);
    for (int i = 0; i < num_inputs * 2; ++i)
      signature_converter.addInputs(i, src_type.getElementType());
    rewriter.applySignatureConversion(&region, signature_converter);
    rewriter.replaceOp(op, linalg_op.getResults());
    return success();
  }
};

/// Converts mhlo.pad operation to linalg.pad_tensor op.
struct PadOpOnTensorsConversion : public OpConversionPattern<mhlo::PadOp> {
  using OpConversionPattern<mhlo::PadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::PadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (llvm::any_of(
            op.interior_padding().getValues<APInt>(),
            [](const APInt& int_val) { return int_val.getZExtValue() != 0; })) {
      return rewriter.notifyMatchFailure(op, "expected no interior padding");
    }

    auto loc = op.getLoc();
    Value padding_val =
        rewriter.createOrFold<tensor::ExtractOp>(loc, adaptor.padding_value());

    const auto& edge_padding_low = op.edge_padding_low();
    const auto& edge_padding_high = op.edge_padding_high();
    SmallVector<OpFoldResult, 4> low, high;
    for (auto it : llvm::zip(edge_padding_low, edge_padding_high)) {
      low.push_back(rewriter.createOrFold<arith::ConstantIndexOp>(
          loc, std::get<0>(it).getZExtValue()));
      high.push_back(rewriter.createOrFold<arith::ConstantIndexOp>(
          loc, std::get<1>(it).getZExtValue()));
    }
    Type result_type = op.getResult().getType();
    auto pad_tensor_op = linalg::PadTensorOp::createPadScalarOp(
        result_type, adaptor.operand(), padding_val, low, high,
        /*nofold=*/false, loc, rewriter);
    rewriter.replaceOp(op, pad_tensor_op.getResult());
    return success();
  }
};

/// Converts mhlo.conv operation to linalg named op. This only covers normal
/// convolution cases. The op must have canonical dimension numbers. Depthwise
/// convolution and pointwise convolution are not handled in the conversion.
struct NormalConvOpOnTensorsConversion
    : public OpConversionPattern<mhlo::ConvOp> {
  using OpConversionPattern<mhlo::ConvOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!HasCanonicalDimensionNumbers(op.dimension_numbers())) return failure();
    if (op.feature_group_count() != 1u) return failure();

    Location loc = op.getLoc();
    Value input = adaptor.lhs();
    Value filter = adaptor.rhs();
    auto result_type = op.getResult().getType().cast<ShapedType>();
    int64_t rank = result_type.getRank();

    // Check if padding is zero.
    DenseIntElementsAttr padding = op.paddingAttr();
    if (padding && !isSplatValue(*op.padding(), 0)) {
      return rewriter.notifyMatchFailure(op, "expected no padding");
    }

    // The output shape is N spatial_dims F.
    SmallVector<Value, 8> dyn_sizes;
    if (result_type.isDynamicDim(0)) {
      dyn_sizes.push_back(rewriter.create<tensor::DimOp>(loc, input, 0));
    }
    for (int64_t i = 1, e = rank - 1; i < e; ++i) {
      if (result_type.isDynamicDim(i)) {
        return rewriter.notifyMatchFailure(
            op, "expected output spatial dims to be static shapes");
      }
    }
    if (result_type.isDynamicDim(rank - 1)) {
      dyn_sizes.push_back(
          rewriter.create<tensor::DimOp>(loc, filter, rank - 1));
    }
    Value init_tensor = rewriter.create<linalg::InitTensorOp>(
        loc, dyn_sizes, result_type.getShape(), result_type.getElementType());
    auto zero_attr = rewriter.getZeroAttr(result_type.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, zero_attr);
    Value zero_tensor =
        rewriter.create<linalg::FillOp>(loc, zero, init_tensor).getResult(0);
    linalg::LinalgOp res;
    Attribute strides = op.window_stridesAttr();
    // TODO(ataei): Only support dilated kernel right now. We need to consider
    // input dilation for deconvolution cases.
    Attribute dilations = op.rhs_dilationAttr();
    switch (rank) {
      case 3: {
        res = rewriter.create<linalg::Conv1DNwcWcfOp>(
            loc, result_type, ValueRange{input, filter},
            ValueRange{zero_tensor}, strides, dilations,
            PruneAttributeList(op));
        break;
      }
      case 4: {
        res = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
            loc, result_type, ValueRange{input, filter},
            ValueRange{zero_tensor}, strides, dilations,
            PruneAttributeList(op));
        break;
      }
      case 5: {
        res = rewriter.create<linalg::Conv3DNdhwcDhwcfOp>(
            loc, result_type, ValueRange{input, filter},
            ValueRange{zero_tensor}, strides, dilations,
            PruneAttributeList(op));
        break;
      }
      default:
        return rewriter.notifyMatchFailure(op, "expected 1/2/3D conv op");
    }
    rewriter.replaceOp(op, res.getOperation()->getResults());
    return success();
  }
};

/// Converts mhlo.convolution operation to
/// linalg.depthwise_conv_2d_input_nhwc_filter_hwcf op or
/// depthwise_conv_2d_input_nhwc_filter_hwc op.
struct DepthwiseConvOpOnTensorsConversion
    : public OpConversionPattern<mhlo::ConvOp> {
  using OpConversionPattern<mhlo::ConvOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConvOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (op.batch_group_count() != 1) return failure();

    if (op.padding() && !isSplatValue(*op.padding(), 0)) {
      return rewriter.notifyMatchFailure(op,
                                         "non-zero padding unsupported yet");
    }

    if ((op.lhs_dilation() && !isSplatValue(*op.lhs_dilation(), 1))) {
      return rewriter.notifyMatchFailure(
          op, "non-one lhs- dialation unsupported yet");
    }

    if (const mhlo::ConvDimensionNumbersAttr& dimension_numbers =
            op.dimension_numbers()) {
      // Make sure that this is 2-D convolution.
      const auto spatial_rank =
          llvm::size(dimension_numbers.getInputSpatialDimensions());
      if (spatial_rank != 2) {
        return rewriter.notifyMatchFailure(op,
                                           "only support 2-D cases for now");
      }

      // Make sure that this is depthwise convolution.
      int64_t input_feature_dim = dimension_numbers.getInputFeatureDimension();
      int64_t input_feature_count =
          op.lhs().getType().cast<ShapedType>().getDimSize(input_feature_dim);
      if (op.feature_group_count() != input_feature_count) {
        return rewriter.notifyMatchFailure(op, "not depth-wise convolution");
      }

      // Make sure that this convolution has a canonical form.
      if (!HasCanonicalDimensionNumbers(dimension_numbers)) {
        return rewriter.notifyMatchFailure(op, "does not have canonical form");
      }
    }

    DenseIntElementsAttr window_strides;
    if (op.window_strides()) {
      window_strides = op.window_strides().getValue();
    } else {
      window_strides = rewriter.getI64VectorAttr({1, 1});
    }

    DenseIntElementsAttr rhs_dilation;
    if (op.rhs_dilation()) {
      rhs_dilation = op.rhs_dilation().getValue();
    } else {
      rhs_dilation = rewriter.getI64VectorAttr({1, 1});
    }

    Location loc = op.getLoc();
    Value input = adaptor.lhs();
    Value filter = adaptor.rhs();
    auto result_type = op.getResult().getType().cast<RankedTensorType>();
    if (!result_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected output has static shapes");
    }

    auto filter_dims =
        llvm::to_vector<4>(op.rhs().getType().cast<ShapedType>().getShape());

    auto get_indices_vector = [](int start, int end) {
      return llvm::to_vector<2>(llvm::seq<int64_t>(start, end));
    };

    if (filter_dims[2] * filter_dims[3] != op.feature_group_count()) {
      // For cases where channel multiplier != 1
      auto output_dims = result_type.getShape();
      auto channel_multiplier = filter_dims[3];
      SmallVector<int64_t> reshaped_output_dims;
      reshaped_output_dims.assign(output_dims.begin(), output_dims.end());
      reshaped_output_dims.push_back(channel_multiplier);
      reshaped_output_dims[3] /= channel_multiplier;

      Value init_tensor = rewriter.create<linalg::InitTensorOp>(
          loc, reshaped_output_dims, result_type.getElementType());
      auto zero_attr = rewriter.getZeroAttr(result_type.getElementType());
      Value zero = rewriter.create<arith::ConstantOp>(loc, zero_attr);
      Value zero_tensor =
          rewriter.create<linalg::FillOp>(loc, zero, init_tensor).getResult(0);

      auto reshaped_output_type = RankedTensorType::get(
          reshaped_output_dims, result_type.getElementType());
      auto conv = rewriter.create<linalg::DepthwiseConv2DNhwcHwcmOp>(
          op.getLoc(), reshaped_output_type, ValueRange{input, filter},
          ValueRange{zero_tensor}, window_strides, rhs_dilation,
          PruneAttributeList(op));

      // Create a Linalg reshape op that converts the output from 5 dimensions
      // into 4 dimensions (by collapsing the last two dimensions). This is
      // needed because linalg.depthwise_conv_2d_input_nhwc_filter_hwcf returns
      // 5 dimensions for the output.
      SmallVector<ReassociationIndices, 4> collapsed_dim_list = {
          get_indices_vector(0, 1), get_indices_vector(1, 2),
          get_indices_vector(2, 3), get_indices_vector(3, 5)};
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, result_type, conv.getResult(0), collapsed_dim_list);
    } else {
      // For cases where channel multiplier == 1
      Value init_tensor = rewriter.create<linalg::InitTensorOp>(
          loc, result_type.getShape(), result_type.getElementType());
      auto zero_attr = rewriter.getZeroAttr(result_type.getElementType());
      Value zero = rewriter.create<arith::ConstantOp>(loc, zero_attr);
      Value zero_tensor =
          rewriter.create<linalg::FillOp>(loc, zero, init_tensor).getResult(0);

      // Create a Linalg reshape op that converts the filter from 4 dimensions
      // into 3 dimensions (by droping the unit dimension). This is needed
      // because linalg.depthwise_conv_2d_input_nhwc_filter_hwc expects 3
      // dimensions for the filter.

      filter_dims[2] = static_cast<int64_t>(op.feature_group_count());
      filter_dims.pop_back();

      RankedTensorType filter_shape =
          RankedTensorType::get(filter_dims, op.getType().getElementType());

      SmallVector<ReassociationIndices, 4> collapsed_dim_list = {
          get_indices_vector(0, 1), get_indices_vector(1, 2),
          get_indices_vector(2, 4)};

      Value reshaped_filter = rewriter.create<tensor::CollapseShapeOp>(
          loc, filter_shape, filter, collapsed_dim_list);

      rewriter.replaceOpWithNewOp<linalg::DepthwiseConv2DNhwcHwcOp>(
          op, result_type, ValueRange{input, reshaped_filter},
          ValueRange{zero_tensor}, window_strides, rhs_dilation,
          PruneAttributeList(op));
    }

    return success();
  }
};

struct ReduceWindowOpOnTensorsConversion
    : public OpConversionPattern<mhlo::ReduceWindowOp> {
  using OpConversionPattern<mhlo::ReduceWindowOp>::OpConversionPattern;

  /// mhlo.reduce_window is mapped to a linalg.pooling operation. The type of
  /// the pooling is determined based on the body of the reduce window
  /// operation. This class enumerates the different variants.
  enum class PoolingType {
    kInvalid,
    k2DMin,
    k3DMin,
    k2DMax,
    k3DMax,
    k2DAdd,
    k3DAdd,
  };

  static PoolingType getPoolingType(mhlo::ReduceWindowOp reduce_op,
                                    int result_index) {
    auto rank =
        reduce_op.getResultTypes()[result_index].cast<ShapedType>().getRank();
    if (Operation* op = reduce_op.getReductionOp(result_index)) {
      if (isa<mhlo::MinOp>(*op) && rank == 4) return PoolingType::k2DMin;
      if (isa<mhlo::MinOp>(*op) && rank == 5) return PoolingType::k3DMin;
      if (isa<mhlo::MaxOp>(*op) && rank == 4) return PoolingType::k2DMax;
      if (isa<mhlo::MaxOp>(*op) && rank == 5) return PoolingType::k3DMax;
      if (isa<mhlo::AddOp>(*op) && rank == 4) return PoolingType::k2DAdd;
      if (isa<mhlo::AddOp>(*op) && rank == 5) return PoolingType::k3DAdd;
    }
    return PoolingType::kInvalid;
  }

  LogicalResult matchAndRewrite(
      mhlo::ReduceWindowOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    int rank = op.getResultTypes()[0].cast<ShapedType>().getRank();
    if (rank != 4 && rank != 5) {
      return rewriter.notifyMatchFailure(
          op, "expected NHWC/NDHWC pooling-based op");
    }

    if (op.padding() && !isSplatValue(*op.padding(), 0)) {
      return rewriter.notifyMatchFailure(op, "require paddings are all zero");
    }

    int last_dim = rank - 1;
    SmallVector<int64_t, 2> fake_window_shapes;
    for (int i = 1; i < last_dim; ++i) {
      fake_window_shapes.push_back(
          op.window_dimensions().getValues<int64_t>()[i]);
    }

    if (op.window_strides() &&
        (op.window_strides().getValue().getValues<int64_t>()[0] != 1 ||
         op.window_strides().getValue().getValues<int64_t>()[last_dim] != 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected window_strides to be [1,x,y,(z),1]");
    }
    if (op.window_dimensions() &&
        (op.window_dimensions().getValues<int64_t>()[0] != 1 ||
         op.window_dimensions().getValues<int64_t>()[last_dim] != 1)) {
      return rewriter.notifyMatchFailure(
          op, "expected window_dimensions to be [1,x,y,(z),1]");
    }

    Attribute strides;
    SmallVector<int64_t> vec;
    if (op.window_stridesAttr()) {
      for (int i = 1; i < last_dim; ++i) {
        vec.push_back(op.window_strides().getValue().getValues<int64_t>()[i]);
      }
    } else {
      vec.assign(rank - 2, 1);
    }
    strides = rewriter.getI64VectorAttr(vec);

    Attribute dilations;
    vec.clear();
    if (op.window_dilations()) {
      for (int i = 1; i < last_dim; ++i) {
        vec.push_back(op.window_dilations().getValue().getValues<int64_t>()[i]);
      }
    } else {
      vec.assign(rank - 2, 1);
    }
    dilations = rewriter.getI64VectorAttr(vec);

    SmallVector<Value> pooling_ops;

    ValueRange inputs = adaptor.inputs();
    ValueRange init_values = adaptor.init_values();
    for (auto it : llvm::zip(op.getResults(), inputs, init_values)) {
      OpResult result = std::get<0>(it);
      Value input = std::get<1>(it);
      Value init_value = std::get<2>(it);
      auto result_type = result.getType().cast<ShapedType>();
      if (!input.getType().cast<ShapedType>().getElementType().isF32()) {
        return rewriter.notifyMatchFailure(op,
                                           "expected element type to be f32");
      }

      // Create a fake window dimension.
      auto fake_window_dims = rewriter.create<linalg::InitTensorOp>(
          loc, fake_window_shapes, result_type.getElementType());

      SmallVector<Value> result_dynamic_dims;
      for (auto& en : llvm::enumerate(result_type.getShape())) {
        if (en.value() != ShapedType::kDynamicSize) continue;
        Value dim_size = rewriter.create<tensor::DimOp>(loc, input, en.index());
        if (en.index() == 0 || en.index() == rank - 1) {
          // batch dims and channel dims can be derived from input dims
          // directly.
          result_dynamic_dims.push_back(dim_size);
        } else {
          auto i = en.index() - 1;
          auto stride =
              strides.cast<DenseIntElementsAttr>().getValues<int64_t>()[i];
          auto dilation =
              dilations.cast<DenseIntElementsAttr>().getValues<int64_t>()[i];
          // let j = i * stride
          // output[i] = reduce( input[j, j + window_size * dilation) )
          Value offset = rewriter.create<arith::ConstantIndexOp>(
              loc, fake_window_shapes[i] * dilation);
          dim_size = rewriter.create<arith::SubIOp>(loc, dim_size, offset);
          dim_size = rewriter.create<arith::DivUIOp>(
              loc, dim_size,
              rewriter.create<arith::ConstantIndexOp>(loc, stride));
          dim_size = rewriter.create<arith::AddIOp>(
              loc, dim_size, rewriter.create<arith::ConstantIndexOp>(loc, 1));
          result_dynamic_dims.push_back(dim_size);
        }
      }
      Value init_tensor = rewriter.create<linalg::InitTensorOp>(
          loc, result_dynamic_dims, result_type.getShape(),
          result_type.getElementType());

      init_value = rewriter.create<tensor::ExtractOp>(loc, init_value);
      Value filled_init_tensor =
          rewriter.create<linalg::FillOp>(loc, init_value, init_tensor)
              .getResult(0);
      auto create_op = [&](auto* type_ptr) -> linalg::LinalgOp {
        return cast<linalg::LinalgOp>(
            rewriter
                .create<std::remove_pointer_t<decltype(type_ptr)>>(
                    loc, ArrayRef<Type>{result_type},
                    ValueRange{input, fake_window_dims.getResult()},
                    filled_init_tensor, strides, dilations,
                    PruneAttributeList(op))
                .getOperation());
      };
      linalg::LinalgOp pooling_op;
      PoolingType pooling_type = getPoolingType(op, result.getResultNumber());
      switch (pooling_type) {
        case PoolingType::k2DMin: {
          pooling_op =
              create_op(static_cast<linalg::PoolingNhwcMinOp*>(nullptr));
          break;
        }
        case PoolingType::k3DMin: {
          pooling_op =
              create_op(static_cast<linalg::PoolingNdhwcMinOp*>(nullptr));
          break;
        }
        case PoolingType::k2DMax: {
          pooling_op =
              create_op(static_cast<linalg::PoolingNhwcMaxOp*>(nullptr));
          break;
        }
        case PoolingType::k3DMax: {
          pooling_op =
              create_op(static_cast<linalg::PoolingNdhwcMaxOp*>(nullptr));
          break;
        }
        case PoolingType::k2DAdd: {
          pooling_op =
              create_op(static_cast<linalg::PoolingNhwcSumOp*>(nullptr));
          break;
        }
        case PoolingType::k3DAdd: {
          pooling_op =
              create_op(static_cast<linalg::PoolingNdhwcSumOp*>(nullptr));
          break;
        }
        case PoolingType::kInvalid:
          return rewriter.notifyMatchFailure(op, "unknown reduction operation");
      }
      pooling_ops.push_back(pooling_op->getResult(0));
    }
    rewriter.replaceOp(op, pooling_ops);
    return success();
  }
};

/// Converts xla-hlo.torch_index_select op to a linalg.generic op.
struct TorchIndexSelectOpOnTensorsConversion
    : public OpConversionPattern<mhlo::TorchIndexSelectOp> {
  using OpConversionPattern<mhlo::TorchIndexSelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::TorchIndexSelectOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    int axis = static_cast<int>(op.dim());
    int batch = static_cast<int>(op.batch_dims());
    auto index_shaped_type = adaptor.index().getType().cast<ShapedType>();
    int num_indices = static_cast<int>(index_shaped_type.getRank());
    auto input_shaped_type = adaptor.input().getType().cast<ShapedType>();
    if (axis < 0) axis += static_cast<int>(input_shaped_type.getRank());
    if (batch < 0) batch += num_indices;

    Location loc = op.getLoc();
    auto result_type =
        this->typeConverter->convertType(op.getResult().getType())
            .cast<ShapedType>();
    int rank = static_cast<int>(result_type.getRank());

    SmallVector<AffineMap, 2> indexing_maps;
    SmallVector<AffineExpr, 4> exprs;
    for (int i = 0; i < batch; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    for (int i = 0, e = num_indices - batch; i < e; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(axis + i));
    }
    indexing_maps.emplace_back(
        AffineMap::get(rank, /*symbolCount=*/0, exprs, rewriter.getContext()));
    indexing_maps.emplace_back(rewriter.getMultiDimIdentityMap(rank));

    // The output shape is
    //   `params[:axis] + indices[batch_dims:] + params[axis + 1:]`
    SmallVector<Value, 4> dyn_sizes;
    for (int i = 0; i < rank; ++i) {
      if (!result_type.isDynamicDim(i)) continue;
      if (i < axis) {
        dyn_sizes.push_back(
            rewriter.create<tensor::DimOp>(loc, adaptor.input(), i));
      } else if (i < (axis + num_indices - batch)) {
        int idx = i - axis + batch;
        dyn_sizes.push_back(
            rewriter.create<tensor::DimOp>(loc, adaptor.index(), idx));
      } else {
        int idx = i - (axis + num_indices - batch) + axis + 1;
        dyn_sizes.push_back(
            rewriter.create<tensor::DimOp>(loc, adaptor.input(), idx));
      }
    }
    Value init_op = rewriter.create<linalg::InitTensorOp>(
        loc, dyn_sizes, result_type.getShape(), result_type.getElementType());
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensors=*/ArrayRef<Type>{result_type},
        /*inputs=*/adaptor.index(),
        /*outputs=*/init_op, indexing_maps, GetNParallelLoopsAttrs(rank),
        /*bodyBuild=*/nullptr, PruneAttributeList(op));

    SmallVector<Type, 4> body_arg_types;
    SmallVector<Value, 2> linalg_op_args = {adaptor.index()};
    // Add a block to the region.
    auto* region = &linalg_op.region();
    auto* block = rewriter.createBlock(region, region->end());
    for (auto block_args : linalg_op_args) {
      body_arg_types.push_back(
          block_args.getType().cast<ShapedType>().getElementType());
    }
    block->addArguments(body_arg_types);
    block->addArguments(result_type.getElementType());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(block);

    Value casted_value = rewriter.create<arith::IndexCastOp>(
        loc, block->getArgument(0), rewriter.getIndexType());

    SmallVector<Value, 4> indices;
    for (int i = 0; i < axis; ++i) {
      indices.push_back(rewriter.create<linalg::IndexOp>(loc, i));
    }
    indices.push_back(casted_value);
    for (int i = axis + num_indices - batch; i < rank; ++i) {
      indices.push_back(rewriter.create<linalg::IndexOp>(loc, i));
    }
    Value res =
        rewriter.create<tensor::ExtractOp>(loc, adaptor.input(), indices);
    rewriter.create<linalg::YieldOp>(loc, res);

    rewriter.replaceOp(op, linalg_op.getResults());
    return success();
  }
};

/// This lowering encompasses the full range of the Gather operation and
/// therefore is very general and just loops over the output and calculate the
/// corresponding input index. It follows the explanation at
/// https://www.tensorflow.org/xla/operation_semantics#gather. The compiler
/// should be able to optimize that a bit, but in order to get efficient
/// lowerings, special-cases of gather should be extracted in separate
/// lowerings, and ideally encapsulated as separate ops or canonicalization
/// patterns.
struct GatherOnTensorsConversion : public OpConversionPattern<mhlo::GatherOp> {
  using OpConversionPattern<mhlo::GatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::GatherOp gatherOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Location loc = gatherOp.getLoc();

    Value startIndices = adaptor.start_indices();
    Value operand = adaptor.operand();

    RankedTensorType resultType =
        gatherOp.getResult().getType().dyn_cast<RankedTensorType>();
    RankedTensorType startIndicesType =
        startIndices.getType().dyn_cast<RankedTensorType>();
    // We could actually deal with an unranked result by inferring the result
    // rank, but the current reifyReturnTypes doesn't support unranked either.
    if (!resultType || !startIndicesType)
      return rewriter.notifyMatchFailure(gatherOp,
                                         "unranked start indices or result");

    int resultRank = resultType.getRank();
    // slice_sizes has to have the same size as operand.rank, and doing it this
    // way permits an unranked operand.
    int operandRank = gatherOp.slice_sizes().getNumElements();

    int64_t indexVectorDim = gatherOp.dimension_numbers().getIndexVectorDim();

    ArrayRef<int64_t> offsetDims = gatherOp.dimension_numbers().getOffsetDims();
    ArrayRef<int64_t> collapsedSliceDims =
        gatherOp.dimension_numbers().getCollapsedSliceDims();
    ArrayRef<int64_t> startIndexMap =
        gatherOp.dimension_numbers().getStartIndexMap();

    auto extractAsIndex = [&](Value input, ArrayRef<Value> index) -> Value {
      return rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(),
          rewriter.create<tensor::ExtractOp>(loc, input, index));
    };

    // We'll need these later and creating them on demand we end up with
    // duplicates, which also makes lit tests really hard to write.
    SmallVector<Value> constants;
    for (unsigned i = 0; i < std::max(resultRank, operandRank); ++i)
      constants.push_back(
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(i)));

    // Create ops to calculate the dynamic dimensions of the return shape, which
    // are needed for the init tensor.
    SmallVector<Value> dynDimSizes;
    if (!resultType.hasStaticShape()) {
      SmallVector<Value> returnShapes;
      if (failed(gatherOp.reifyReturnTypeShapes(rewriter, adaptor.getOperands(),
                                                returnShapes)))
        return rewriter.notifyMatchFailure(gatherOp,
                                           "could not reify return shape");
      assert(returnShapes.size() == 1);
      Value returnShape = returnShapes[0];

      for (int i = 0; i < resultRank; ++i)
        if (resultType.isDynamicDim(i))
          dynDimSizes.push_back(extractAsIndex(returnShape, constants[i]));
    }

    Value initOp = rewriter.create<linalg::InitTensorOp>(
        loc, dynDimSizes, resultType.getShape(), resultType.getElementType());

    ValueRange ins;
    SmallVector<AffineMap, 1> indexingMaps(
        {rewriter.getMultiDimIdentityMap(resultRank)});
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/resultType,
        /*inputs=*/ins,
        /*outputs=*/initOp, indexingMaps, GetNParallelLoopsAttrs(resultRank),
        /*bodyBuild=*/nullptr, PruneAttributeList(gatherOp));

    // Now populate the linalg generic region
    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    block->addArguments(resultType.getElementType());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(block);

    // Dimensions in the result that aren't offset dimensions are called batch.
    SmallVector<int64_t> batchDims;
    for (int dim = 0; dim < resultRank; ++dim)
      if (!llvm::is_contained(offsetDims, dim)) batchDims.push_back(dim);

    // Same as with the constants. Creating these all up front is easier than
    // potentially getting duplicates later.
    SmallVector<Value> linalgIndices;
    for (unsigned i = 0; i < resultRank; ++i)
      linalgIndices.push_back(rewriter.create<linalg::IndexOp>(loc, i));

    // Now the complicated part. For a given output dimension we build up an
    // index into the input. It's composed of two parts: the index coming from
    // start_indices, and the offset from that index along the offset
    // dimensions. Everything includes dimension shuffling and remapping as well
    // because of the way gather is defined to allow for any-layout input by
    // adding more attributes.

    // The base gather index (`G` in the documentation) points to a place in
    // start_indices along the batch dimensions.
    SmallVector<Value> gatherIndex;
    for (auto dim : batchDims) gatherIndex.push_back(linalgIndices[dim]);

    SmallVector<Value> indexFromStartIndices;
    for (unsigned i = 0; i < startIndexMap.size(); ++i) {
      // The index along the index_vector dimension of start_indices varies.
      // Basically indexFromStartIndices indexes into a "row" along
      // index_vector_dim, where the row is selected by the current output
      // index.
      // But if index_vector_dim is equal to start_indices.rank, then
      // start_indices gets a trailing 1 dimension added. So the row we're
      // extracting always has length 1 and the index into it is always 0, so we
      // just use the gather index directly
      SmallVector<Value> gCombine(gatherIndex);
      if (indexVectorDim != startIndicesType.getRank()) {
        assert(indexVectorDim <= gCombine.size());
        gCombine.insert(gCombine.begin() + indexVectorDim, constants[i]);
      }

      indexFromStartIndices.push_back(extractAsIndex(startIndices, gCombine));
    }

    // But then start indices are shuffled by the start index map. To make a
    // full index into the operand, all missing indices are zeroes.
    SmallVector<Value> remappedIndexFromIndices(operandRank, constants[0]);
    for (auto& it : llvm::enumerate(startIndexMap))
      remappedIndexFromIndices[it.value()] = indexFromStartIndices[it.index()];

    // Now we construct the index based on the offset. First we need to remap
    // the offset dimensions by dropping the collapsed indices.
    SmallVector<unsigned> remappedOffsetDims;
    for (unsigned i = 0; i < operandRank; ++i)
      if (!llvm::is_contained(collapsedSliceDims, i))
        remappedOffsetDims.push_back(i);

    assert(remappedOffsetDims.size() == offsetDims.size());

    // For the (remapped) offset dimensions, the index is the current index in
    // the output. As before this is expanded to a full index into the operand
    // by using zeroe for the missing indices.
    SmallVector<Value> indexFromOffset(operandRank, constants[0]);
    for (unsigned k = 0; k < offsetDims.size(); ++k)
      indexFromOffset[remappedOffsetDims[k]] = linalgIndices[offsetDims[k]];

    // Now we add together our two indices to get the final index into the
    // operand.
    SmallVector<Value> combinedIndex;
    for (unsigned i = 0; i < operandRank; ++i)
      combinedIndex.push_back(rewriter.create<arith::AddIOp>(
          loc, rewriter.getIndexType(), remappedIndexFromIndices[i],
          indexFromOffset[i]));

    Value element =
        rewriter.create<tensor::ExtractOp>(loc, operand, combinedIndex);
    rewriter.create<linalg::YieldOp>(loc, element);

    rewriter.replaceOp(gatherOp, linalgOp.getResults());

    return success();
  }
};

struct ScatterUpdateOnTensorsConversion
    : public OpConversionPattern<mhlo::ScatterOp> {
  using OpConversionPattern<mhlo::ScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // Check if it is a tensor_scatter_nd_update-like op.
    auto& body_ops = op.getRegion().front().getOperations();
    if (body_ops.size() != 1) return failure();
    auto ret_arg = body_ops.front().getOperand(0).dyn_cast<BlockArgument>();
    if (!ret_arg || ret_arg.getArgNumber() != 1) return failure();

    auto operand_ty = adaptor.operand().getType().dyn_cast<RankedTensorType>();
    auto indices_ty =
        adaptor.scatter_indices().getType().dyn_cast<RankedTensorType>();
    if (!operand_ty || !indices_ty) return failure();

    // Linalg operations put all the computation to the innermost loop. Since we
    // also iterate over scatter_indices() with some loops, we can only check
    // one scatter index in one iteration. If there are multiple indices (ie,
    // the index depth is greater than 1), we don't have a way to keep the
    // comparison state. E.g., if the index_depth is 2, like indices = [[0, 1]],
    // we should use the update value only if (i == 0 and j == 1). However, we
    // can not get both indices in one iteration unless we pack them together.
    auto index_vector_dim = op.scatter_dimension_numbers().getIndexVectorDim();
    if (indices_ty.getDimSize(index_vector_dim) != 1)
      return rewriter.notifyMatchFailure(op, "require index depth to be 1");
    if (index_vector_dim != indices_ty.getRank() - 1) {
      return rewriter.notifyMatchFailure(
          op, "require index_vector_dim to be the last dim");
    }

    // One of indices dims is index depth vector.
    int64_t nloops = operand_ty.getRank() + indices_ty.getRank() - 1;
    SmallVector<AffineMap, 3> indexing_maps;
    {
      SmallVector<AffineExpr> exprs;
      for (int64_t i = 0, e = operand_ty.getRank(); i < e; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(i));
      indexing_maps.push_back(AffineMap::get(nloops, /*symbolCount=*/0, exprs,
                                             rewriter.getContext()));
    }
    {
      SmallVector<AffineExpr> exprs;
      for (int64_t i = operand_ty.getRank(); i < nloops; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(i));
      // The index depth is 1.
      exprs.push_back(rewriter.getAffineConstantExpr(0));
      indexing_maps.push_back(AffineMap::get(nloops, /*symbolCount=*/0, exprs,
                                             rewriter.getContext()));

      exprs.pop_back();
      auto update_window_dims =
          op.scatter_dimension_numbers().getUpdateWindowDims();
      for (auto d : update_window_dims)
        exprs.push_back(rewriter.getAffineDimExpr(d));
      indexing_maps.push_back(AffineMap::get(nloops, /*symbolCount=*/0, exprs,
                                             rewriter.getContext()));
    }
    indexing_maps.push_back(indexing_maps.front());

    auto result_ty = this->typeConverter->convertType(op.getResult().getType())
                         .cast<ShapedType>();
    auto scatter_dims_to_operand_dims =
        op.scatter_dimension_numbers().getScatterDimsToOperandDims();
    assert(scatter_dims_to_operand_dims.size() == 1);
    // Do not need init_tensor because we'd like to initialize the output as
    // operand.
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        op.getLoc(), /*resultTensors=*/ArrayRef<Type>{result_ty},
        /*inputs=*/
        ValueRange{adaptor.operand(), adaptor.scatter_indices(),
                   adaptor.updates()},
        /*outputs=*/adaptor.operand(), indexing_maps,
        GetNParallelLoopsAttrs(nloops),
        [&](OpBuilder& b, Location loc, ValueRange args) {
          Value cmp_idx =
              b.create<linalg::IndexOp>(loc, scatter_dims_to_operand_dims[0]);
          Value idx =
              b.create<arith::IndexCastOp>(loc, b.getIndexType(), args[1]);
          Value pred = b.create<arith::CmpIOp>(
              loc, b.getI1Type(), arith::CmpIPredicate::eq, cmp_idx, idx);
          // Use the output arg, so some update values won't be init value
          // again.
          Value res = b.create<SelectOp>(loc, args[2].getType(), pred, args[2],
                                         args[3]);
          b.create<linalg::YieldOp>(loc, res);
        },
        PruneAttributeList(op));
    rewriter.replaceOp(op, linalg_op.getResults());
    return success();
  }
};

struct ComputeReshapeShapeConversion
    : public OpConversionPattern<mhlo::ComputeReshapeShapeOp> {
  using OpConversionPattern<mhlo::ComputeReshapeShapeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ComputeReshapeShapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto ctx = op->getContext();
    Value neg_one = rewriter.create<arith::ConstantIndexOp>(loc, -1);
    auto index_type = rewriter.getIndexType();
    auto num_elements = adaptor.getOperands()[0];
    auto target_shape_type =
        adaptor.getOperands()[1].getType().cast<ShapedType>();
    auto extent_type =
        shape::getExtentTensorType(ctx, target_shape_type.getDimSize(0));

    // Calculate the computed actual extent for a possible dynamic extent.
    auto new_shape = target_shape_type.getElementType().isIndex()
                         ? adaptor.getOperands()[1]
                         : rewriter.create<arith::IndexCastOp>(
                               loc, extent_type, adaptor.getOperands()[1]);
    Value new_shape_rank =
        rewriter.create<shape::RankOp>(loc, index_type, new_shape);
    // The product begins with a -1 seed which will cancel out a -1 extent in
    // the input shape if there is one. If there is not, this computed result
    // will never be used, so it's okay to compute a negative number of
    // elements.
    auto accounted_num_els =
        rewriter.create<shape::ReduceOp>(loc, new_shape, neg_one);
    {
      PatternRewriter::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(accounted_num_els.getBody());
      Value lhs = accounted_num_els.getBody()->getArgument(1);
      Value rhs = accounted_num_els.getBody()->getArgument(2);
      rewriter.create<shape::YieldOp>(
          loc, rewriter.create<arith::MulIOp>(loc, lhs, rhs).getResult());
    }
    Value missing_dim_val = rewriter.create<arith::DivUIOp>(
        loc, num_elements, accounted_num_els->getResult(0));

    // Create the final target shape with a possible dynamic extent replace with
    // the calculated extent.
    SmallVector<Value> dynamic_extent;
    if (!target_shape_type.hasStaticShape())
      dynamic_extent.push_back(new_shape_rank);
    auto gen = rewriter.create<tensor::GenerateOp>(
        loc, target_shape_type, dynamic_extent,
        [&](OpBuilder& b, Location loc, ValueRange indices) {
          Value extent = b.create<shape::GetExtentOp>(loc, index_type,
                                                      new_shape, indices[0]);
          Value use_missing_dim_val = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, extent, neg_one);
          Value dim_val = b.create<SelectOp>(loc, use_missing_dim_val,
                                             missing_dim_val, extent);
          dim_val = target_shape_type.getElementType().isIndex()
                        ? dim_val
                        : b.create<arith::IndexCastOp>(
                              loc, target_shape_type.getElementType(), dim_val);
          b.create<tensor::YieldOp>(loc, dim_val);
        });
    rewriter.replaceOp(op, gen.result());

    return success();
  }
};

struct CstrReshapableConversion
    : public OpConversionPattern<mhlo::CstrReshapableOp> {
  using OpConversionPattern<mhlo::CstrReshapableOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::CstrReshapableOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto ctx = op->getContext();
    Value neg_one = rewriter.create<arith::ConstantIndexOp>(loc, -1);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto num_elements = adaptor.getOperands()[0];
    auto target_shape_type =
        adaptor.getOperands()[1].getType().cast<ShapedType>();
    auto extent_type =
        shape::getExtentTensorType(ctx, target_shape_type.getDimSize(0));

    // Calculate the computed actual extent for a possible dynamic extent.
    auto new_shape = target_shape_type.getElementType().isIndex()
                         ? adaptor.getOperands()[1]
                         : rewriter.create<arith::IndexCastOp>(
                               loc, extent_type, adaptor.getOperands()[1]);
    auto reduction = rewriter.create<shape::ReduceOp>(
        loc, new_shape, llvm::makeArrayRef({one, zero, zero}));
    {
      PatternRewriter::InsertionGuard g(rewriter);
      auto body = reduction.getBody();
      rewriter.setInsertionPointToEnd(body);
      Value extent = body->getArgument(1);
      Value is_dynamic = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, neg_one, extent);
      Value is_invalid = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, extent, neg_one);
      Value total_dynamic = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<SelectOp>(loc, is_dynamic, one, zero),
          body->getArgument(3));
      Value total_invalid = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<SelectOp>(loc, is_invalid, one, zero),
          body->getArgument(4));
      Value extent_or_one =
          rewriter.create<SelectOp>(loc, is_dynamic, one, extent);
      Value total_elements = rewriter.create<arith::MulIOp>(
          loc, extent_or_one, body->getArgument(2));
      rewriter.create<shape::YieldOp>(
          loc,
          llvm::makeArrayRef({total_elements, total_dynamic, total_invalid}));
    }
    // Avoid division by zero.
    Value is_zero_elements = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, reduction->getResult(0), zero);
    Value divisor = rewriter.create<SelectOp>(loc, is_zero_elements, one,
                                              reduction->getResult(0));
    Value is_divisible = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, zero,
        rewriter.create<arith::RemSIOp>(loc, num_elements, divisor));
    // Must have 0 or 1 dynamic dimensions.
    Value acceptably_dynamic = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ule, reduction->getResult(1), one);
    // Must have no invalid dimensions.
    Value no_invalid = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, reduction->getResult(2), zero);
    // If the old shape has size zero, the new shape must have size zero too.
    // This can be a zero factor or a -1.
    Value has_one_dynamic = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, reduction->getResult(1), one);
    Value equal_if_empty = rewriter.create<arith::OrIOp>(
        loc, has_one_dynamic,
        rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, is_zero_elements,
            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           num_elements, zero)));

    Value all_passing = rewriter.create<arith::AndIOp>(
        loc, is_divisible,
        rewriter.create<arith::AndIOp>(
            loc, acceptably_dynamic,
            rewriter.create<arith::AndIOp>(loc, no_invalid, equal_if_empty)));

    rewriter.replaceOpWithNewOp<shape::CstrRequireOp>(
        op, all_passing, "Required valid reshape shape input");

    return success();
  }
};

struct HloLegalizeToLinalgPass
    : public mhlo::HloLegalizeToLinalgPassBase<HloLegalizeToLinalgPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    complex::ComplexDialect, math::MathDialect,
                    memref::MemRefDialect, shape::ShapeDialect>();
  }

  void runOnFunction() override {
    MLIRContext& ctx = getContext();
    OwningRewritePatternList patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<arith::ArithmeticDialect, complex::ComplexDialect,
                           linalg::LinalgDialect, math::MathDialect,
                           StandardOpsDialect, tensor::TensorDialect,
                           scf::SCFDialect, shape::ShapeDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    mhlo::RemoveSignTypeConverter type_converter;
    auto func = getFunction();
    mhlo::populateHLOToLinalgConversionPattern(&ctx, type_converter, &patterns);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mhlo {

void populateHLOToLinalgConversionPattern(MLIRContext* context,
                                          TypeConverter& type_converter,
                                          OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<
      BroadcastConverter<mhlo::BroadcastOp>, ConcatenateConverter,
      ConstConverterTensor, HloDynamicBroadcastInDimConverter,
      HloBroadcastInDimConverter, IotaConverter<mhlo::IotaOp>,
      EinsumToLinalgConverter,
      IotaConverter<mhlo::DynamicIotaOp>,
      PointwiseToLinalgConverter<mhlo::AbsOp>,
      PointwiseToLinalgConverter<mhlo::AddOp>,
      PointwiseToLinalgConverter<mhlo::AndOp>,
      PointwiseToLinalgConverter<mhlo::Atan2Op>,
      PointwiseToLinalgConverter<mhlo::BitcastConvertOp>,
      PointwiseToLinalgConverter<mhlo::CbrtOp>,
      PointwiseToLinalgConverter<mhlo::CeilOp>,
      PointwiseToLinalgConverter<mhlo::ClampOp>,
      PointwiseToLinalgConverter<mhlo::CompareOp>,
      PointwiseToLinalgConverter<mhlo::ComplexOp>,
      PointwiseToLinalgConverter<mhlo::ConvertOp>,
      PointwiseToLinalgConverter<mhlo::CopyOp>,
      PointwiseToLinalgConverter<mhlo::CosOp>,
      PointwiseToLinalgConverter<mhlo::DivOp>,
      PointwiseToLinalgConverter<mhlo::ExpOp>,
      PointwiseToLinalgConverter<mhlo::Expm1Op>,
      PointwiseToLinalgConverter<mhlo::FloorOp>,
      PointwiseToLinalgConverter<mhlo::ImagOp>,
      PointwiseToLinalgConverter<mhlo::IsFiniteOp>,
      PointwiseToLinalgConverter<mhlo::LogOp>,
      PointwiseToLinalgConverter<mhlo::LogisticOp>,
      PointwiseToLinalgConverter<mhlo::Log1pOp>,
      PointwiseToLinalgConverter<mhlo::MaxOp>,
      PointwiseToLinalgConverter<mhlo::MinOp>,
      PointwiseToLinalgConverter<mhlo::MulOp>,
      PointwiseToLinalgConverter<mhlo::NegOp>,
      PointwiseToLinalgConverter<mhlo::NotOp>,
      PointwiseToLinalgConverter<mhlo::OrOp>,
      PointwiseToLinalgConverter<mhlo::PowOp>,
      PointwiseToLinalgConverter<mhlo::RealOp>,
      PointwiseToLinalgConverter<mhlo::RemOp>,
      PointwiseToLinalgConverter<mhlo::RsqrtOp>,
      PointwiseToLinalgConverter<mhlo::SelectOp>,
      PointwiseToLinalgConverter<mhlo::ShiftLeftOp>,
      PointwiseToLinalgConverter<mhlo::ShiftRightArithmeticOp>,
      PointwiseToLinalgConverter<mhlo::ShiftRightLogicalOp>,
      PointwiseToLinalgConverter<mhlo::SignOp>,
      PointwiseToLinalgConverter<mhlo::SinOp>,
      PointwiseToLinalgConverter<mhlo::SqrtOp>,
      PointwiseToLinalgConverter<mhlo::SubOp>,
      PointwiseToLinalgConverter<mhlo::TanhOp>,
      PointwiseToLinalgConverter<mhlo::XorOp>,
      RealDynamicSliceConverter,
      ReshapeOpConverter,
      ReverseConverter<mhlo::ReverseOp>,
      SliceConverter<mhlo::SliceOp>,
      ComputeReshapeShapeConversion,
      CstrReshapableConversion,
      DynamicSliceConverter,
      DynamicUpdateSliceConverter,
      TransposeConverter<mhlo::TransposeOp>,
      DotOpOnTensorsConversion<DotOperationType::kMatrixMatrix,
                               linalg::MatmulOp>,
      DotOpOnTensorsConversion<DotOperationType::kMatrixVector,
                               linalg::MatvecOp>,
      DotOpOnTensorsConversion<DotOperationType::kVectorDot, linalg::DotOp>,
      DotGeneralOpOnTensorsConversion,
      NormalConvOpOnTensorsConversion,
      DepthwiseConvOpOnTensorsConversion,
      ReduceOnTensorsConversion,
      ReduceWindowOpOnTensorsConversion,
      RngUniformConversion,
      ScatterUpdateOnTensorsConversion,
      GatherOnTensorsConversion,
      TorchIndexSelectOpOnTensorsConversion,
      PadOpOnTensorsConversion>(type_converter, context);
  // clang-format on
  patterns->insert<ReduceRegionXLAOpConversion<mhlo::AddOp>,
                   ReduceRegionXLAOpConversion<mhlo::AndOp>,
                   ReduceRegionXLAOpConversion<mhlo::CompareOp>,
                   ReduceRegionXLAOpConversion<mhlo::MaxOp>,
                   ReduceRegionXLAOpConversion<mhlo::MinOp>,
                   ReduceRegionXLAOpConversion<mhlo::MulOp>,
                   ReduceRegionXLAOpConversion<mhlo::OrOp>,
                   ReduceRegionXLAOpConversion<mhlo::SelectOp>,
                   ReduceRegionReturnOpConversion>(context,
                                                   PatternBenefit(1000));
}

std::unique_ptr<OperationPass<FuncOp>> createLegalizeHloToLinalgPass() {
  return std::make_unique<HloLegalizeToLinalgPass>();
}

std::unique_ptr<TypeConverter> createHloToLinalgSignedIntegerConverter() {
  return std::make_unique<RemoveSignTypeConverter>();
}

}  // namespace mhlo
}  // namespace mlir
