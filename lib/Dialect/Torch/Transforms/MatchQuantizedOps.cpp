//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
namespace mlir::torch::Torch {

#define GEN_PASS_DEF_MATCHQUANTIZEDCUSTOMOPS
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h.inc"

namespace {

Type getQuantizedType(MLIRContext *context, Type t) {
  if (t.isSignlessInteger(8) || t.isUnsignedInteger(8))
    return Torch::QUInt8Type::get(context);
  if (t.isInteger(8) || t.isSignedInteger(8))
    return Torch::QInt8Type::get(context);
  if (t.isInteger(16))
    return Torch::QInt16Type::get(context);
  if (t.isInteger(32))
    return Torch::QInt32Type::get(context);
  return {};
}

class MatchQuantizeOperator : public OpRewritePattern<OperatorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(OperatorOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getName() == "torch.quantized_decomposed.quantize_per_tensor") {
      auto resultTy = cast<ValueTensorType>(op.getType(0));
      auto qeTy = getQuantizedType(rewriter.getContext(), resultTy.getDtype());
      if (!qeTy)
        qeTy = resultTy.getDtype();

      auto qTy =
          rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qeTy);
      Value quant = AtenQuantizePerTensorOp::create(
          rewriter, op.getLoc(), qTy,
          /*self=*/op.getOperand(0), /*scale=*/op.getOperand(1),
          /*zero_point=*/op.getOperand(2), /*dtype=*/op.getOperand(5));

      if (qTy != resultTy) {
        quant = AtenIntReprOp::create(rewriter, op.getLoc(), resultTy, quant);
      }

      rewriter.replaceOpWithNewOp<AtenClampOp>(
          op, resultTy, quant, op.getOperand(3), op.getOperand(4));
      return success();
    }

    auto prepareDequantize = [&](Value quantMin, Value quantMax, Value &clamp,
                                 Type &qTy) {
      clamp =
          AtenClampOp::create(rewriter, op.getLoc(), op.getOperand(0).getType(),
                              op.getOperand(0), quantMin, quantMax);

      auto clampTy = cast<Torch::ValueTensorType>(clamp.getType());
      if (!clampTy.hasDtype())
        return rewriter.notifyMatchFailure(op,
                                           "dequantization has unknown dtype");

      Type dtype = clampTy.getDtype();
      Type qetype = getQuantizedType(op.getContext(), dtype);
      if (!qetype)
        return rewriter.notifyMatchFailure(op,
                                           "dequantization has unknown qtype");

      qTy = Torch::ValueTensorType::get(op.getContext(),
                                        clampTy.getOptionalSizes(), qetype);
      return success();
    };

    if (op.getName() == "torch.quantized_decomposed.dequantize_per_tensor") {
      Value clamp;
      Type qTy;
      if (failed(prepareDequantize(op.getOperand(3), op.getOperand(4), clamp,
                                   qTy)))
        return failure();

      auto quant = Aten_MakePerTensorQuantizedTensorOp::create(
          rewriter, op.getLoc(), qTy, clamp, op.getOperand(1),
          op.getOperand(2));
      rewriter.replaceOpWithNewOp<AtenDequantizeTensorOp>(
          op, op.getResultTypes(), quant);
      return success();
    }

    if (op.getName() == "torch.quantized_decomposed.dequantize_per_channel") {
      Value clamp;
      Type qTy;
      if (failed(prepareDequantize(op.getOperand(4), op.getOperand(5), clamp,
                                   qTy)))
        return failure();
      auto quant = Aten_MakePerChannelQuantizedTensorOp::create(
          rewriter, op.getLoc(), qTy, clamp, op.getOperand(1), op.getOperand(2),
          op.getOperand(3));
      rewriter.replaceOpWithNewOp<AtenDequantizeSelfOp>(op, op.getResultTypes(),
                                                        quant);
      return success();
    }

    // --- Dynamic (per-token, runtime-scale) symmetric quantization ---
    // torchao pt2e is_dynamic=True with a symmetric activation spec emits, per
    // activation: choose_qparams_symmetric.tensor (compute scale at runtime) ->
    // quantize_per_tensor.tensor -> dequantize_per_tensor.tensor (the ".tensor"
    // variants carry scale/zp as rank-1 tensors rather than constant attrs).
    // We legalize all three so a dynamically-quantized symmetric activation
    // folds + routes to the int8 mmt4d ukernel like the static path.
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    auto getScalarTensorTy = [&](Type elem) {
      return rewriter.getType<ValueTensorType>(ArrayRef<int64_t>{}, elem);
    };

    if (op.getName() ==
        "torch.quantized_decomposed.choose_qparams_symmetric.tensor") {
      // returns (scale: vtensor<[1],f64>, zero_point: vtensor<[1],si64>).
      // Symmetric (torchao): scale = max(|x|) / ((qmax - qmin) / 2), >= eps;
      // zero_point = 0.  (max(-min(min,0), max(max,0)) == amax(|x|) for any sign.)
      Value x = op.getOperand(0);
      int64_t qmin, qmax;
      if (!matchPattern(op.getOperand(1), m_TorchConstantInt(&qmin)) ||
          !matchPattern(op.getOperand(2), m_TorchConstantInt(&qmax)))
        return failure();
      Value eps = op.getOperand(3);
      auto scaleTy = cast<ValueTensorType>(op.getResult(0).getType());
      auto zpTy = cast<ValueTensorType>(op.getResult(1).getType());
      Type f32 = cast<ValueTensorType>(x.getType()).getDtype();
      auto scalarTy = getScalarTensorTy(f32);

      Value absx = AtenAbsOp::create(rewriter, loc, x.getType(), x);
      Value amax = AtenMaxOp::create(rewriter, loc, scalarTy, absx);
      double divisor = static_cast<double>(qmax - qmin) / 2.0;
      Value divV = ConstantFloatOp::create(
          rewriter, loc, rewriter.getF64FloatAttr(divisor));
      Value scaled = AtenDivScalarOp::create(rewriter, loc, scalarTy, amax, divV);
      Value clamped = AtenClampMinOp::create(rewriter, loc, scalarTy, scaled, eps);
      // rank-0 -> [1], then cast to the declared scale dtype (f64).
      Value zero = ConstantIntOp::create(rewriter, loc,
                                         rewriter.getI64IntegerAttr(0));
      auto oneF32Ty = rewriter.getType<ValueTensorType>(ArrayRef<int64_t>{1}, f32);
      Value scale1 =
          AtenUnsqueezeOp::create(rewriter, loc, oneF32Ty, clamped, zero);
      Value cstFalse = ConstantBoolOp::create(rewriter, loc, false);
      Value none = ConstantNoneOp::create(rewriter, loc);
      Value f64dtype = ConstantIntOp::create(rewriter, loc,
                                             rewriter.getI64IntegerAttr(7));
      Value scaleVal = AtenToDtypeOp::create(rewriter, loc, scaleTy, scale1,
                                             f64dtype, cstFalse, cstFalse, none);
      // zero_point = zeros([1], si64)
      Value one = ConstantIntOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(1));
      Value sizeList = Torch::PrimListConstructOp::create(
          rewriter, loc, rewriter.getType<Torch::ListType>(
                             rewriter.getType<Torch::IntType>()),
          ValueRange{one});
      Value i64dtype = ConstantIntOp::create(rewriter, loc,
                                             rewriter.getI64IntegerAttr(4));
      Value zpVal = AtenZerosOp::create(rewriter, loc, zpTy, sizeList, i64dtype,
                                        none, none, none);
      rewriter.replaceOp(op, {scaleVal, zpVal});
      return success();
    }

    // quantize/dequantize ".tensor" variants: scale/zp arrive as rank-1
    // tensors; extract the scalars (aten.item) and reuse the per-tensor path.
    if (op.getName() == "torch.quantized_decomposed.quantize_per_tensor.tensor") {
      Value scaleS = AtenItemOp::create(rewriter, loc,
                                        rewriter.getType<Torch::FloatType>(),
                                        op.getOperand(1));
      Value zpS = AtenItemOp::create(rewriter, loc,
                                     rewriter.getType<Torch::IntType>(),
                                     op.getOperand(2));
      auto resultTy = cast<ValueTensorType>(op.getType(0));
      auto qeTy = getQuantizedType(ctx, resultTy.getDtype());
      if (!qeTy)
        qeTy = resultTy.getDtype();
      auto qTy =
          rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qeTy);
      Value quant = AtenQuantizePerTensorOp::create(
          rewriter, loc, qTy, op.getOperand(0), scaleS, zpS, op.getOperand(5));
      if (qTy != resultTy)
        quant = AtenIntReprOp::create(rewriter, loc, resultTy, quant);
      rewriter.replaceOpWithNewOp<AtenClampOp>(op, resultTy, quant,
                                               op.getOperand(3),
                                               op.getOperand(4));
      return success();
    }

    if (op.getName() ==
        "torch.quantized_decomposed.dequantize_per_tensor.tensor") {
      Value scaleS = AtenItemOp::create(rewriter, loc,
                                        rewriter.getType<Torch::FloatType>(),
                                        op.getOperand(1));
      Value zpS = AtenItemOp::create(rewriter, loc,
                                     rewriter.getType<Torch::IntType>(),
                                     op.getOperand(2));
      Value clamp;
      Type qTy;
      if (failed(prepareDequantize(op.getOperand(3), op.getOperand(4), clamp,
                                   qTy)))
        return failure();
      auto quant = Aten_MakePerTensorQuantizedTensorOp::create(
          rewriter, loc, qTy, clamp, scaleS, zpS);
      rewriter.replaceOpWithNewOp<AtenDequantizeTensorOp>(
          op, op.getResultTypes(), quant);
      return success();
    }

    return failure();
  }
};

class MatchQuantizedCustomOpsPass
    : public impl::MatchQuantizedCustomOpsBase<MatchQuantizedCustomOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<MatchQuantizeOperator>(context);

    GreedyRewriteConfig config;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createMatchQuantizedCustomOpsPass() {
  return std::make_unique<MatchQuantizedCustomOpsPass>();
}

} // namespace mlir::torch::Torch
