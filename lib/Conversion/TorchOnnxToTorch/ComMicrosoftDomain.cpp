//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

void mlir::torch::onnx_c::populateComMicrosoftDomain(
    OnnxCustomOpConversionPattern &patterns) {
  patterns.onOp(
      "RotaryEmbedding", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        int64_t interleaved, isPackedBatching, numHeads, rotaryEmbeddingDim;
        float scale;
        Value input, positionIds, cosCache, sinCache;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(positionIds, 1) ||
            binder.tensorOperandAtIndex(cosCache, 2) ||
            binder.tensorOperandAtIndex(sinCache, 3) ||
            binder.s64IntegerAttr(interleaved, "interleaved", 0) ||
            binder.s64IntegerAttr(isPackedBatching, "is_packed_batching", 0) ||
            binder.s64IntegerAttr(numHeads, "num_heads", 0) ||
            binder.s64IntegerAttr(rotaryEmbeddingDim, "rotary_embedding_dim",
                                  0) ||
            binder.f32FloatAttr(scale, "scale", 1.0)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get required inputs");
        }

        Torch::ValueTensorType resultType;
        if (binder.tensorResultType(resultType)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "result type bind failure");
        }

        Value cstInterleaved = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(interleaved));
        Value cstIsPackedBatching = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(isPackedBatching));
        Value cstNumHeads = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(numHeads));
        Value cstRotaryEmbeddingDim = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(rotaryEmbeddingDim));
        Value cstScale = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(scale));

        rewriter.replaceOpWithNewOp<Torch::OnnxVariantRotaryEmbeddingOp>(
            binder.op, resultType, input, positionIds, cosCache, sinCache,
            cstInterleaved, cstIsPackedBatching, cstNumHeads,
            cstRotaryEmbeddingDim, cstScale);
        return success();
      });
  patterns.onOp(
      "SimplifiedLayerNormalization", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();

        // Bind required operands: input and scale (gamma)
        Value input, gamma;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(gamma, 1))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get required inputs");

        // Bind attributes
        float epsilon;
        int64_t axis;
        if (binder.f32FloatAttr(epsilon, "epsilon", 1e-5f))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get epsilon");
        if (binder.s64IntegerAttr(axis, "axis", -1))
          return rewriter.notifyMatchFailure(binder.op, "Failed to get axis");

        // Get result types (there can be 1 or more outputs)
        SmallVector<Type> resultTypes;
        if (binder.tensorResultTypes(resultTypes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "result types bind failure");

        // Get input type to determine shapes and dtype
        Torch::ValueTensorType inputType =
            cast<Torch::ValueTensorType>(input.getType());
        if (!inputType.hasDtype())
          return rewriter.notifyMatchFailure(binder.op,
                                             "input should have dtype");

        // Get tensor rank to normalize axis
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unranked input tensor");
        unsigned inputRank = *maybeRank;
        if (inputRank == 0)
          return rewriter.notifyMatchFailure(binder.op,
                                             "scalar input not supported");

        // Normalize negative axis
        if (axis < 0)
          axis += inputRank;

        // Create constants
        Value cstOne = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(1.0));
        Value cstEpsilon = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(epsilon));
        Value cstAxis = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(axis));
        Value cstTrue = Torch::ConstantBoolOp::create(rewriter, loc, true);
        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);

        // Step 1: Compute input^2
        Value inputSquared = Torch::AtenMulTensorOp::create(
            rewriter, loc, inputType, input, input);

        // Step 2: Compute mean(input^2, dim=axis, keepdim=true)
        // Create dim list with the axis
        Value dimList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            SmallVector<Value>{cstAxis});

        // Result type for mean (same shape as input but reduced on axis dim)
        SmallVector<int64_t> meanSizes(inputType.getSizes());
        if (!meanSizes.empty() &&
            static_cast<size_t>(axis) < meanSizes.size() &&
            meanSizes[axis] != Torch::kUnknownSize)
          meanSizes[axis] = 1; // keepdim=true, so axis dim becomes 1
        Torch::ValueTensorType meanType =
            cast<Torch::ValueTensorType>(inputType.getWithSizesAndDtype(
                meanSizes, inputType.getOptionalDtype()));

        Value meanSquared = Torch::AtenMeanDimOp::create(
            rewriter, loc, meanType, inputSquared, dimList, cstTrue, cstNone);

        // Step 3: Compute rms = sqrt(mean(input^2) + epsilon)
        Value meanPlusEpsilon = Torch::AtenAddScalarOp::create(
            rewriter, loc, meanType, meanSquared, cstEpsilon, cstOne);
        Value rms =
            Torch::AtenSqrtOp::create(rewriter, loc, meanType, meanPlusEpsilon);

        // Step 4: Compute output = (input / rms) * gamma
        // rms needs to be expanded/broadcasted to match input's shape
        Value rmsExpanded =
            Torch::AtenExpandAsOp::create(rewriter, loc, inputType, rms, input);

        Value normalized = Torch::AtenDivTensorOp::create(
            rewriter, loc, inputType, input, rmsExpanded);

        Value output = Torch::AtenMulTensorOp::create(rewriter, loc, inputType,
                                                      normalized, gamma);

        // Return outputs (may have optional inv_std_var output)
        if (resultTypes.size() == 1) {
          rewriter.replaceOp(binder.op, {output});
        } else if (resultTypes.size() == 2) {
          // Second output is inv_std_var (1/rms)
          Value invStdVar =
              Torch::AtenReciprocalOp::create(rewriter, loc, meanType, rms);
          rewriter.replaceOp(binder.op, {output, invStdVar});
        } else if (resultTypes.size() == 3) {
          // Third output is mean (not typically used but part of spec)
          Value invStdVar =
              Torch::AtenReciprocalOp::create(rewriter, loc, meanType, rms);
          rewriter.replaceOp(binder.op, {output, invStdVar, meanSquared});
        } else {
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected 1-3 result types");
        }

        return success();
      });
  patterns.onOp(
      "SkipSimplifiedLayerNormalization", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();

        // Bind required operands
        Value input, skip, gamma;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(skip, 1) ||
            binder.tensorOperandAtIndex(gamma, 2))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get required inputs");

        // Bind optional bias (index 3)
        Value bias;
        bool hasBias = !binder.tensorOperandAtIndex(bias, 3);

        // Bind epsilon attribute
        float epsilon;
        if (binder.f32FloatAttr(epsilon, "epsilon", 1e-5f))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get epsilon");

        // Get result types (there can be 1 or 2 outputs)
        SmallVector<Type> resultTypes;
        if (binder.tensorResultTypes(resultTypes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "result types bind failure");

        // Get input type to determine shapes and dtype
        Torch::ValueTensorType inputType =
            cast<Torch::ValueTensorType>(input.getType());
        if (!inputType.hasDtype())
          return rewriter.notifyMatchFailure(binder.op,
                                             "input should have dtype");

        // Get tensor rank to compute last dimension
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unranked input tensor");
        unsigned inputRank = *maybeRank;
        if (inputRank == 0)
          return rewriter.notifyMatchFailure(binder.op,
                                             "scalar input not supported");

        // Create constants
        Value cstOne = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(1.0));
        Value cstEpsilon = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(epsilon));
        Value cstLastDim = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(inputRank - 1));
        Value cstTrue = Torch::ConstantBoolOp::create(rewriter, loc, true);
        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);

        // Step 1: Compute s = input + skip + bias (if present)
        Value s = Torch::AtenAddTensorOp::create(rewriter, loc, inputType,
                                                 input, skip, cstOne);

        if (hasBias) {
          s = Torch::AtenAddTensorOp::create(rewriter, loc, inputType, s, bias,
                                             cstOne);
        }

        // Step 2: Compute s^2
        Value sSquared =
            Torch::AtenMulTensorOp::create(rewriter, loc, inputType, s, s);

        // Step 3: Compute mean(s^2, dim=-1, keepdim=true)
        // Create dim list with last dimension
        Value dimList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            SmallVector<Value>{cstLastDim});

        // Result type for mean (same shape as input but reduced on last dim)
        SmallVector<int64_t> meanSizes(inputType.getSizes());
        if (!meanSizes.empty() && meanSizes.back() != Torch::kUnknownSize)
          meanSizes.back() = 1; // keepdim=true, so last dim becomes 1
        Torch::ValueTensorType meanType =
            cast<Torch::ValueTensorType>(inputType.getWithSizesAndDtype(
                meanSizes, inputType.getOptionalDtype()));

        Value meanSquared = Torch::AtenMeanDimOp::create(
            rewriter, loc, meanType, sSquared, dimList, cstTrue, cstNone);

        // Step 4: Compute rms = sqrt(mean(s^2) + epsilon)
        Value meanPlusEpsilon = Torch::AtenAddScalarOp::create(
            rewriter, loc, meanType, meanSquared, cstEpsilon, cstOne);
        Value rms =
            Torch::AtenSqrtOp::create(rewriter, loc, meanType, meanPlusEpsilon);

        // Step 5: Compute output = (s / rms) * gamma
        // rms needs to be expanded/broadcasted to match s's shape
        Value rmsExpanded =
            Torch::AtenExpandAsOp::create(rewriter, loc, inputType, rms, s);

        Value normalized = Torch::AtenDivTensorOp::create(
            rewriter, loc, inputType, s, rmsExpanded);

        Value output = Torch::AtenMulTensorOp::create(rewriter, loc, inputType,
                                                      normalized, gamma);

        // Compute inv_std_var = 1/rms when needed
        Value invStdVar;
        if (resultTypes.size() >= 3) {
          invStdVar =
              Torch::AtenReciprocalOp::create(rewriter, loc, meanType, rms);
        }

        // Note: While the ONNX spec says output order is (output, mean,
        // inv_std_var, input_skip_bias_sum), in practice ORT and real models
        // expect the 2-output case to return (output, input_skip_bias_sum)
        // since that's what's needed for transformer residual connections.
        if (resultTypes.size() == 1) {
          rewriter.replaceOp(binder.op, {output});
        } else if (resultTypes.size() == 2) {
          // Return input_skip_bias_sum as 2nd output for backward compatibility
          rewriter.replaceOp(binder.op, {output, s});
        } else if (resultTypes.size() == 3) {
          rewriter.replaceOp(binder.op, {output, meanSquared, invStdVar});
        } else if (resultTypes.size() == 4) {
          rewriter.replaceOp(binder.op, {output, meanSquared, invStdVar, s});
        } else {
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected 1-4 result types");
        }

        return success();
      });
  patterns.onOp(
      "GroupQueryAttention", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        SmallVector<Value> operands;
        SmallVector<Type> resultTypes;
        int64_t doRotary, kvNumHeads, localWindowSize, numHeads,
            rotaryInterleaved, smoothSoftmax;
        float scale, softcap;
        if (binder.tensorOperandsList(operands))
          return rewriter.notifyMatchFailure(binder.op,
                                             "operands bind failure");

        if (binder.tensorResultTypes(resultTypes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "result types bind failure");

        if (resultTypes.size() != 3)
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected 3 result types");

        if (binder.s64IntegerAttr(doRotary, "do_rotary") ||
            binder.s64IntegerAttr(kvNumHeads, "kv_num_heads") ||
            binder.s64IntegerAttr(localWindowSize, "local_window_size", -1) ||
            binder.s64IntegerAttr(numHeads, "num_heads") ||
            binder.s64IntegerAttr(rotaryInterleaved, "rotary_interleaved") ||
            binder.f32FloatAttr(scale, "scale") ||
            binder.s64IntegerAttr(smoothSoftmax, "smooth_softmax") ||
            binder.f32FloatAttr(softcap, "softcap"))
          return rewriter.notifyMatchFailure(binder.op,
                                             "op attributes bind failure");

        // This lowering supports two input formats:
        // 1. Separate Q, K, V inputs (9 operands with rotary, 7 without):
        //    query, key, value, past_key, past_value, seqlens_k, total_seq_len,
        //    [cos_cache, sin_cache]
        // 2. Packed QKV input (7 operands with rotary, 5 without):
        //    packed_qkv, past_key, past_value, seqlens_k, total_seq_len,
        //    [cos_cache, sin_cache]
        bool isPackedQKV = false;
        if (doRotary) {
          if (operands.size() == 7) {
            isPackedQKV = true;
          } else if (operands.size() != 9) {
            return rewriter.notifyMatchFailure(
                binder.op,
                "Expected 7 operands (packed QKV) or 9 operands (separate Q, "
                "K, V) when do_rotary is enabled");
          }
        } else {
          if (operands.size() == 5) {
            isPackedQKV = true;
          } else if (operands.size() != 7) {
            return rewriter.notifyMatchFailure(
                binder.op,
                "Expected 5 operands (packed QKV) or 7 operands (separate Q, "
                "K, V) when do_rotary is disabled");
          }
        }

        if (kvNumHeads == 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "kv_num_heads is a required attribute and should be non-zero");

        if (localWindowSize != -1)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: local_window_size attribute is not supported, "
              "hence it should have default value equal to -1");

        if (numHeads == 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "num_heads is a required attribute and should be non-zero");

        if (smoothSoftmax > 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: smooth_softmax attribute is not supported, hence "
              "it should have a value <= 0 (disabled)");

        if (softcap != 0.0f)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: softcap attribute is not supported, "
                         "hence it should have default value equal to 0.0");

        Location loc = binder.getLoc();
        MLIRContext *context = binder.op->getContext();
        Value query, key, value, pastKey, pastValue, seqlensK,
            totalSequenceLength;
        Value cosCache, sinCache;

        if (isPackedQKV) {
          // Packed QKV mode: first operand contains Q, K, V concatenated
          Value packedQKV = operands[0];
          pastKey = operands[1];
          pastValue = operands[2];
          seqlensK = operands[3];
          totalSequenceLength = operands[4];
          if (doRotary) {
            cosCache = operands[5];
            sinCache = operands[6];
          }

          // Split packed QKV into separate Q, K, V tensors
          // packed_qkv shape: [batch, seq, q_hidden + k_hidden + v_hidden]
          // where q_hidden = num_heads * head_size
          //       k_hidden = kv_num_heads * head_size
          //       v_hidden = kv_num_heads * head_size
          Torch::ValueTensorType packedType =
              cast<Torch::ValueTensorType>(packedQKV.getType());
          if (!packedType.hasSizes() || packedType.getSizes().size() != 3)
            return rewriter.notifyMatchFailure(
                binder.op, "Expected packed QKV input to have 3 dimensions");

          SmallVector<int64_t> packedDims{packedType.getSizes()};
          int64_t batchSize = packedDims[0];        // may be dynamic
          int64_t sequenceLength = packedDims[1];   // may be dynamic
          int64_t packedHiddenSize = packedDims[2]; // must be static

          if (packedHiddenSize == Torch::kUnknownSize)
            return rewriter.notifyMatchFailure(
                binder.op,
                "Expected packed QKV hidden dimension (dim 2) to be static");

          // Calculate head_size from past_key shape: [batch, kv_num_heads,
          // past_seq, head_size]
          Torch::ValueTensorType pastKeyType =
              cast<Torch::ValueTensorType>(pastKey.getType());
          if (!(pastKeyType.hasSizes() && pastKeyType.getSizes().size() == 4))
            return rewriter.notifyMatchFailure(
                binder.op, "Expected past_key to have 4 dimensions");

          int64_t headSize = pastKeyType.getSizes()[3];
          if (headSize == Torch::kUnknownSize)
            return rewriter.notifyMatchFailure(
                binder.op, "Expected past_key head_size (dim 3) to be static");

          int64_t qHiddenSize = numHeads * headSize;
          int64_t kvHiddenSize = kvNumHeads * headSize;

          // Validate packed hidden size
          if (packedHiddenSize != qHiddenSize + 2 * kvHiddenSize)
            return rewriter.notifyMatchFailure(
                binder.op, "Packed QKV hidden size mismatch: expected " +
                               std::to_string(qHiddenSize + 2 * kvHiddenSize) +
                               " but got " + std::to_string(packedHiddenSize));

          Value cstOne = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(1));
          Value cstTwo = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(2));
          Value cstZero = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(0));
          Value cstQHidden = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(qHiddenSize));
          Value cstQPlusKVHidden = Torch::ConstantIntOp::create(
              rewriter, loc,
              rewriter.getI64IntegerAttr(qHiddenSize + kvHiddenSize));
          Value cstPackedHidden = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(packedHiddenSize));

          // Slice Q: packed_qkv[:, :, 0:q_hidden]
          // batch and seq dimensions may be dynamic
          SmallVector<int64_t> querySizes{batchSize, sequenceLength,
                                          qHiddenSize};
          Torch::ValueTensorType queryType = Torch::ValueTensorType::get(
              context, querySizes, packedType.getOptionalDtype());
          query = Torch::AtenSliceTensorOp::create(
              rewriter, loc, queryType, packedQKV,
              /*dim=*/cstTwo, /*start=*/cstZero, /*end=*/cstQHidden,
              /*step=*/cstOne);

          // Slice K: packed_qkv[:, :, q_hidden:q_hidden+kv_hidden]
          SmallVector<int64_t> kvSizes{batchSize, sequenceLength, kvHiddenSize};
          Torch::ValueTensorType keyType = Torch::ValueTensorType::get(
              context, kvSizes, packedType.getOptionalDtype());
          key = Torch::AtenSliceTensorOp::create(rewriter, loc, keyType,
                                                 packedQKV,
                                                 /*dim=*/cstTwo,
                                                 /*start=*/cstQHidden,
                                                 /*end=*/cstQPlusKVHidden,
                                                 /*step=*/cstOne);

          // Slice V: packed_qkv[:, :, q_hidden+kv_hidden:]
          Torch::ValueTensorType valueType = Torch::ValueTensorType::get(
              context, kvSizes, packedType.getOptionalDtype());
          value = Torch::AtenSliceTensorOp::create(
              rewriter, loc, valueType, packedQKV,
              /*dim=*/cstTwo, /*start=*/cstQPlusKVHidden,
              /*end=*/cstPackedHidden,
              /*step=*/cstOne);
        } else {
          // Separate Q, K, V mode
          query = operands[0];
          key = operands[1];
          value = operands[2];
          pastKey = operands[3];
          pastValue = operands[4];
          seqlensK = operands[5];
          totalSequenceLength = operands[6];
          if (doRotary) {
            cosCache = operands[7];
            sinCache = operands[8];
          }
        }

        Torch::ValueTensorType queryType =
            cast<Torch::ValueTensorType>(query.getType());
        if (!queryType.hasSizes() || queryType.getSizes().size() != 3)
          return rewriter.notifyMatchFailure(
              binder.op, "Expected `query` input to have 3 dimensions");

        SmallVector<int64_t> queryDims{queryType.getSizes()};
        int64_t batchSize = queryDims[0];      // may be dynamic
        int64_t sequenceLength = queryDims[1]; // may be dynamic
        int64_t hiddenSize = queryDims[2];     // must be static
        if (hiddenSize == Torch::kUnknownSize)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected `query` hidden dimension (dim 2) to be static");
        int64_t headSize = hiddenSize / numHeads;

        // For dynamic dimensions, use aten.size.int to get runtime values
        Type intType = rewriter.getType<Torch::IntType>();

        Value cstBatchSize, cstSequenceLength;
        if (batchSize == Torch::kUnknownSize) {
          Value cstZeroDim = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(0));
          cstBatchSize = Torch::AtenSizeIntOp::create(rewriter, loc, intType,
                                                      query, cstZeroDim);
        } else {
          cstBatchSize = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(batchSize));
        }
        if (sequenceLength == Torch::kUnknownSize) {
          Value cstOneDim = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));
          cstSequenceLength = Torch::AtenSizeIntOp::create(
              rewriter, loc, intType, query, cstOneDim);
        } else {
          cstSequenceLength = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(),
              rewriter.getI64IntegerAttr(sequenceLength));
        }

        Value cstHiddenSize = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(hiddenSize));
        Value cstHeadSize = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(headSize));
        Value cstNumHeads = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(numHeads));
        Value cstKVNumHeads = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(kvNumHeads));

        // Reshape Query, Key and Value as follows:
        // Query: (batch_size, sequence_length, hidden_size)
        //     -> (batch_size, num_heads, sequence_length, head_size)
        // Key: (batch_size, kv_sequence_length, kv_hidden_size)
        //   -> (batch_size, kv_num_heads, sequence_length, head_size)
        // Value: (batch_size, kv_sequence_length, kv_hidden_size)
        //     -> (batch_size, kv_num_heads, sequence_length, head_size)

        // Reshaping query.
        SmallVector<int64_t> queryReshapeSizesInt{batchSize, numHeads,
                                                  sequenceLength, headSize};
        Value queryReshapeSizesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(query.getContext())),
            llvm::SmallVector<Value>{cstBatchSize, cstNumHeads,
                                     cstSequenceLength, cstHeadSize});
        Value qInput = Torch::AtenReshapeOp::create(
            rewriter, loc,
            queryType.getWithSizesAndDtype(queryReshapeSizesInt,
                                           queryType.getOptionalDtype()),
            query, queryReshapeSizesList);

        // Reshaping key.
        SmallVector<int64_t> kvReshapeSizesInt{batchSize, kvNumHeads,
                                               sequenceLength, headSize};
        Value kvReshapeSizesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(query.getContext())),
            llvm::SmallVector<Value>{cstBatchSize, cstKVNumHeads,
                                     cstSequenceLength, cstHeadSize});
        Torch::ValueTensorType keyType =
            cast<Torch::ValueTensorType>(key.getType());
        Value kInput = Torch::AtenReshapeOp::create(
            rewriter, loc,
            keyType.getWithSizesAndDtype(kvReshapeSizesInt,
                                         keyType.getOptionalDtype()),
            key, kvReshapeSizesList);

        // Reshaping value.
        Torch::ValueTensorType valueType =
            cast<Torch::ValueTensorType>(value.getType());
        Value vInput = Torch::AtenReshapeOp::create(
            rewriter, loc,
            valueType.getWithSizesAndDtype(kvReshapeSizesInt,
                                           valueType.getOptionalDtype()),
            value, kvReshapeSizesList);

        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);

        Value qRotary = qInput, kRotary = kInput;
        if (doRotary) {
          // `totalSequenceLength` is a scalar tensor.
          Value scalarTotalSeqLens = Torch::AtenItemOp::create(
              rewriter, loc, rewriter.getType<Torch::IntType>(),
              totalSequenceLength);
          Value cstIntOne = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));
          Type boolTy = rewriter.getType<Torch::BoolType>();
          Value condA = Torch::AtenGtIntOp::create(
              rewriter, loc, boolTy, cstSequenceLength, cstIntOne);
          Value condB = Torch::AtenNeIntOp::create(
              rewriter, loc, boolTy, cstSequenceLength, scalarTotalSeqLens);
          //   if (sequence_length > 1 && sequence_length !=
          //   total_sequence_length)
          //         is_subsequent_prompt = false;  // Subsequent prompt
          Value isSubsequentPrompt = Torch::Aten__And__BoolOp::create(
              rewriter, loc, boolTy, condA, condB);

          // Generating position_ids for rotary_embedding as follows:
          //   pos_ids_a = torch.zeros((batch_size, seq_len), dtype=torch.int64)
          //
          //   total_seqlens = seqlens_k + 1
          //   past_seqlens = total_seqlens - sequence_length
          //   pos_ids = torch.arange(sequence_length,
          //             dtype=torch.int64).repeat(batch_size, 1)
          //   pos_ids = pos_ids + past_seqlens.view(-1, 1)
          //   cond = pos_ids < total_seqlens.view(-1, 1)
          //   one_tensor = torch.tensor(1, dtype=torch.int64)
          //   pos_ids_b = torch.where(cond, pos_ids, one_tensor)
          //
          //  if subsequent_prompt:
          //      pos_ids = pos_ids_b
          //  else:
          //      pos_ids = pos_ids_a
          SmallVector<int64_t> positionIdsSizeInt{batchSize, sequenceLength};
          Torch::ValueTensorType positionIdsType = Torch::ValueTensorType::get(
              context, positionIdsSizeInt,
              IntegerType::get(context, 64, IntegerType::Signed));
          Value cstInt64Dtype = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(),
              rewriter.getI64IntegerAttr(
                  (int)torch_upstream::ScalarType::Long));

          Value cstInterleaved = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(),
              rewriter.getI64IntegerAttr(rotaryInterleaved));
          Value cstIntZero = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(0));
          Value cstFloatOne = Torch::ConstantFloatOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(1.0));

          Value positionIdsA, positionIdsB;

          Value posIdsSizeList = Torch::PrimListConstructOp::create(
              rewriter, loc,
              rewriter.getType<Torch::ListType>(
                  rewriter.getType<Torch::IntType>()),
              SmallVector<Value>{cstBatchSize, cstSequenceLength});
          positionIdsA = Torch::AtenZerosOp::create(
              rewriter, loc, positionIdsType, /*size=*/posIdsSizeList,
              /*dtype=*/cstInt64Dtype,
              /*layout=*/cstNone, /*device=*/cstNone,
              /*pin_memory=*/cstNone);

          // Convert seqlens_k which is a tensor of type si32 to si64.
          Torch::ValueTensorType seqLensKType =
              cast<Torch::ValueTensorType>(seqlensK.getType());
          seqlensK = Torch::AtenToDtypeOp::create(
              rewriter, loc,
              seqLensKType.getWithSizesAndDtype(
                  std::nullopt,
                  rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true)),
              seqlensK, cstInt64Dtype, /*non_blocking=*/cstFalse,
              /*copy=*/cstFalse, /*memory_format=*/cstNone);
          Value totalSeqLens = Torch::AtenAddScalarOp::create(
              rewriter, loc, seqlensK.getType(), /*self=*/seqlensK,
              /*other=*/cstIntOne,
              /*alpha=*/cstIntOne);
          Value pastSeqLens = Torch::AtenSubScalarOp::create(
              rewriter, loc, totalSeqLens.getType(), /*self=*/totalSeqLens,
              /*other=*/cstSequenceLength, /*alpha=*/cstIntOne);
          Torch::ValueTensorType initPosIdsType = Torch::ValueTensorType::get(
              context, {sequenceLength},
              IntegerType::get(context, 64, IntegerType::Signed));
          Value initPosIds = Torch::AtenArangeOp::create(
              rewriter, loc, initPosIdsType, cstSequenceLength, cstInt64Dtype,
              /*layout=*/cstNone,
              /*device=*/cstNone, /*pin_memory=*/cstNone);
          Value repeatValuesList = Torch::PrimListConstructOp::create(
              rewriter, binder.getLoc(),
              Torch::ListType::get(Torch::IntType::get(context)),
              llvm::SmallVector<Value>{cstBatchSize, cstIntOne});
          positionIdsB = Torch::AtenRepeatOp::create(
              rewriter, loc, positionIdsType, initPosIds,
              /*repeats=*/repeatValuesList);

          Value cstIntMinusOne = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(-1));
          Value viewSizeList = Torch::PrimListConstructOp::create(
              rewriter, binder.getLoc(),
              Torch::ListType::get(Torch::IntType::get(context)),
              llvm::SmallVector<Value>{cstIntMinusOne, cstIntOne});

          Torch::ValueTensorType seqLensViewType = Torch::ValueTensorType::get(
              context, llvm::SmallVector<int64_t>{batchSize, 1},
              IntegerType::get(context, 64, IntegerType::Signed));
          pastSeqLens = Torch::AtenViewOp::create(
              rewriter, loc, seqLensViewType, pastSeqLens, viewSizeList);

          positionIdsB = Torch::AtenAddTensorOp::create(
              rewriter, loc, positionIdsType, positionIdsB, pastSeqLens,
              /*alpha=*/cstIntOne);

          totalSeqLens = Torch::AtenViewOp::create(
              rewriter, loc, seqLensViewType, totalSeqLens, viewSizeList);
          Value cond = Torch::AtenLtTensorOp::create(
              rewriter, loc,
              positionIdsType.getWithSizesAndDtype(positionIdsType.getSizes(),
                                                   rewriter.getI1Type()),
              positionIdsB, totalSeqLens);

          Value cstOneTensorDataList = Torch::PrimListConstructOp::create(
              rewriter, loc,
              rewriter.getType<Torch::ListType>(
                  rewriter.getType<Torch::IntType>()),
              SmallVector<Value>{cstIntOne});
          Value cstOneTensor = Torch::AtenTensorOp::create(
              rewriter, loc,
              Torch::ValueTensorType::get(
                  context, {},
                  IntegerType::get(context, 64, IntegerType::Signed)),
              cstOneTensorDataList, /*dtype=*/cstInt64Dtype,
              /*layout=*/cstNone, /*requires_grad=*/cstFalse);

          positionIdsB = Torch::AtenWhereSelfOp::create(
              rewriter, loc, positionIdsType, cond, positionIdsB, cstOneTensor);

          isSubsequentPrompt = Torch::AtenIntBoolOp::create(
              rewriter, loc, rewriter.getType<Torch::IntType>(),
              isSubsequentPrompt);
          isSubsequentPrompt = Torch::AtenFullOp::create(
              rewriter, loc,
              Torch::ValueTensorType::get(context, positionIdsSizeInt,
                                          rewriter.getI1Type()),
              /*size=*/posIdsSizeList, /*fill_value=*/isSubsequentPrompt,
              /*dtype=*/
              Torch::ConstantIntOp::create(
                  rewriter, binder.getLoc(),
                  rewriter.getI64IntegerAttr(
                      (int)torch_upstream::ScalarType::Bool)),
              /*layout=*/cstNone, /*device=*/cstNone, /*pin_memory=*/cstNone);
          Value positionIds = Torch::AtenWhereSelfOp::create(
              rewriter, loc, positionIdsType, isSubsequentPrompt, positionIdsB,
              positionIdsA);

          // Performing RotaryEmbedding over Query and Key.
          qRotary = Torch::OnnxVariantRotaryEmbeddingOp::create(
              rewriter, loc, qInput.getType(), qInput, positionIds, cosCache,
              sinCache, cstInterleaved, /*is_packed_batching=*/cstIntZero,
              /*num_heads=*/cstIntZero, /*rotary_embedding_dim=*/cstIntZero,
              /*scale=*/cstFloatOne);

          kRotary = Torch::OnnxVariantRotaryEmbeddingOp::create(
              rewriter, loc, kInput.getType(), kInput, positionIds, cosCache,
              sinCache, cstInterleaved, /*is_packed_batching=*/cstIntZero,
              /*num_heads=*/cstIntZero, /*rotary_embedding_dim=*/cstIntZero,
              /*scale=*/cstFloatOne);
        }

        // Compute present_key and present_value by concatenating past with
        // current. These are used both for attention AND as output for the next
        // iteration's KV cache.
        // present_key = torch.cat([past_key, key], dim=2)
        // present_value = torch.cat([past_value, value], dim=2)
        // Always concatenate, even if past_key/past_value are empty (sequence
        // length 0), as concatenating empty tensors still produces correct
        // results.
        Value cstConcatDim = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(2));

        Type kvListElemType = keyType.getWithSizesAndDtype(
            /*optionalSizes=*/std::nullopt,
            /*optionalDtype=*/nullptr);
        Type kvListType = Torch::ListType::get(kvListElemType);
        Value keyList = Torch::PrimListConstructOp::create(
            rewriter, loc, kvListType, SmallVector<Value>{pastKey, kRotary});
        Value presentKey = Torch::AtenCatOp::create(
            rewriter, loc, resultTypes[1], keyList, cstConcatDim);

        Value valueList = Torch::PrimListConstructOp::create(
            rewriter, loc, kvListType, SmallVector<Value>{pastValue, vInput});
        Value presentValue = Torch::AtenCatOp::create(
            rewriter, loc, resultTypes[2], valueList, cstConcatDim);

        // Generate attention mask combining:
        // 1. Valid length mask: mask out positions beyond seqlens_k + seq_q
        // 2. Causal mask: each query position can only attend to positions up
        //    to past_len + q (where past_len = seqlens_k, q is query position)
        //
        // Per ONNX GQA spec: "Only supports causal and local attention."
        // Mask shape: [batch, 1, seqLen, kvSeqLen] where masked positions are
        // -inf.
        Value attnMask = cstNone;

        // Get the KV sequence length from presentKey shape
        Torch::ValueTensorType presentKeyType =
            cast<Torch::ValueTensorType>(presentKey.getType());
        if (presentKeyType.hasSizes() &&
            presentKeyType.getSizes().size() == 4) {
          int64_t kvSeqLen = presentKeyType.getSizes()[2];

          // Only generate mask if KV sequence length is dynamic or > 0
          // For dynamic shapes or non-trivial sequences, we need to mask
          if (kvSeqLen == Torch::kUnknownSize || kvSeqLen > 0) {
            // seqlens_k is already converted to i64 dtype in rotary section,
            // but we need to handle non-rotary case too
            Value seqlensKInt64 = seqlensK;
            Torch::ValueTensorType seqLensKType =
                cast<Torch::ValueTensorType>(seqlensK.getType());
            if (seqLensKType.getOptionalDtype() &&
                seqLensKType.getOptionalDtype().isInteger(32)) {
              Value cstInt64Dtype = Torch::ConstantIntOp::create(
                  rewriter, binder.getLoc(),
                  rewriter.getI64IntegerAttr(
                      (int)torch_upstream::ScalarType::Long));
              seqlensKInt64 = Torch::AtenToDtypeOp::create(
                  rewriter, loc,
                  seqLensKType.getWithSizesAndDtype(
                      std::nullopt,
                      rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true)),
                  seqlensK, cstInt64Dtype, /*non_blocking=*/cstFalse,
                  /*copy=*/cstFalse, /*memory_format=*/cstNone);
            }

            Value cstIntOne = Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));

            // Get KV sequence dimension size
            Value cstDim2 = Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(2));
            Value kvSeqLenVal = Torch::AtenSizeIntOp::create(
                rewriter, loc, rewriter.getType<Torch::IntType>(), presentKey,
                cstDim2);

            // Create range tensors for causal mask computation
            Value cstInt64Dtype = Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(),
                rewriter.getI64IntegerAttr(
                    (int)torch_upstream::ScalarType::Long));

            // kRange: [0, 1, 2, ..., kvSeqLen-1] shape [kvSeqLen]
            Torch::ValueTensorType rangeType = Torch::ValueTensorType::get(
                context, {kvSeqLen},
                rewriter.getIntegerType(64, /*isSigned=*/true));
            Value kRange = Torch::AtenArangeOp::create(
                rewriter, loc, rangeType, kvSeqLenVal, cstInt64Dtype,
                /*layout=*/cstNone, /*device=*/cstNone, /*pin_memory=*/cstNone);

            // qRange: [0, 1, 2, ..., seqLen-1] shape [seqLen]
            Torch::ValueTensorType qRangeType = Torch::ValueTensorType::get(
                context, {sequenceLength},
                rewriter.getIntegerType(64, /*isSigned=*/true));
            Value qRange = Torch::AtenArangeOp::create(
                rewriter, loc, qRangeType, cstSequenceLength, cstInt64Dtype,
                /*layout=*/cstNone, /*device=*/cstNone, /*pin_memory=*/cstNone);

            // Reshape for broadcasting:
            // seqlensK: [batch] -> [batch, 1, 1] for broadcasting
            // qRange: [seqLen] -> [1, seqLen, 1]
            // kRange: [kvSeqLen] -> [1, 1, kvSeqLen]
            Value cstMinusOne = Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(-1));

            // seqlensK -> [batch, 1, 1]
            Value seqlensViewList = Torch::PrimListConstructOp::create(
                rewriter, loc,
                rewriter.getType<Torch::ListType>(
                    rewriter.getType<Torch::IntType>()),
                SmallVector<Value>{cstMinusOne, cstIntOne, cstIntOne});
            SmallVector<int64_t> seqlensViewSizes{batchSize, 1, 1};
            Torch::ValueTensorType seqlensViewType =
                Torch::ValueTensorType::get(
                    context, seqlensViewSizes,
                    rewriter.getIntegerType(64, /*isSigned=*/true));
            Value seqlensKView = Torch::AtenViewOp::create(
                rewriter, loc, seqlensViewType, seqlensKInt64, seqlensViewList);

            // qRange -> [1, seqLen, 1]
            Value qViewList = Torch::PrimListConstructOp::create(
                rewriter, loc,
                rewriter.getType<Torch::ListType>(
                    rewriter.getType<Torch::IntType>()),
                SmallVector<Value>{cstIntOne, cstMinusOne, cstIntOne});
            SmallVector<int64_t> qViewSizes{1, sequenceLength, 1};
            Torch::ValueTensorType qViewType = Torch::ValueTensorType::get(
                context, qViewSizes,
                rewriter.getIntegerType(64, /*isSigned=*/true));
            Value qRangeView = Torch::AtenViewOp::create(
                rewriter, loc, qViewType, qRange, qViewList);

            // kRange -> [1, 1, kvSeqLen]
            Value kViewList = Torch::PrimListConstructOp::create(
                rewriter, loc,
                rewriter.getType<Torch::ListType>(
                    rewriter.getType<Torch::IntType>()),
                SmallVector<Value>{cstIntOne, cstIntOne, cstMinusOne});
            SmallVector<int64_t> kViewSizes{1, 1, kvSeqLen};
            Torch::ValueTensorType kViewType = Torch::ValueTensorType::get(
                context, kViewSizes,
                rewriter.getIntegerType(64, /*isSigned=*/true));
            Value kRangeView = Torch::AtenViewOp::create(
                rewriter, loc, kViewType, kRange, kViewList);

            // Compute causal mask: k <= seqlens_k + q
            // Equivalently: k - seqlens_k <= q, or k <= seqlens_k + q
            // This allows query at position q to attend to KV positions 0..
            // (seqlens_k + q), which gives proper causal behavior considering
            // the past KV cache.
            //
            // seqlens_k + q: [batch, 1, 1] + [1, seqLen, 1] -> [batch, seqLen,
            // 1]
            SmallVector<int64_t> seqlensQSizes{batchSize, sequenceLength, 1};
            Torch::ValueTensorType seqlensQType = Torch::ValueTensorType::get(
                context, seqlensQSizes,
                rewriter.getIntegerType(64, /*isSigned=*/true));
            Value seqlensKPlusQ = Torch::AtenAddTensorOp::create(
                rewriter, loc, seqlensQType, seqlensKView, qRangeView,
                cstIntOne);

            // k <= seqlens_k + q: [1, 1, kvSeqLen] <= [batch, seqLen, 1]
            // -> [batch, seqLen, kvSeqLen]
            SmallVector<int64_t> maskBoolSizes{batchSize, sequenceLength,
                                               kvSeqLen};
            Torch::ValueTensorType maskBoolType = Torch::ValueTensorType::get(
                context, maskBoolSizes, rewriter.getI1Type());
            Value causalMask = Torch::AtenLeTensorOp::create(
                rewriter, loc, maskBoolType, kRangeView, seqlensKPlusQ);

            // Convert bool mask to float mask: True -> 0, False -> -inf
            Value cstZeroFloat = Torch::ConstantFloatOp::create(
                rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                rewriter.getF64FloatAttr(0.0));
            Value cstNegInf = Torch::ConstantFloatOp::create(
                rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                rewriter.getF64FloatAttr(
                    -std::numeric_limits<double>::infinity()));

            Value cstFloatDtype = Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(),
                rewriter.getI64IntegerAttr(
                    (int)torch_upstream::ScalarType::Float));
            Torch::ValueTensorType scalarTensorType =
                Torch::ValueTensorType::get(context, {}, rewriter.getF32Type());
            Value zeroTensor = Torch::AtenFullOp::create(
                rewriter, loc, scalarTensorType,
                /*size=*/
                Torch::PrimListConstructOp::create(
                    rewriter, loc,
                    rewriter.getType<Torch::ListType>(
                        rewriter.getType<Torch::IntType>()),
                    SmallVector<Value>{}),
                /*fill_value=*/cstZeroFloat, /*dtype=*/cstFloatDtype,
                /*layout=*/cstNone, /*device=*/cstNone, /*pin_memory=*/cstNone);
            Value negInfTensor = Torch::AtenFullOp::create(
                rewriter, loc, scalarTensorType,
                /*size=*/
                Torch::PrimListConstructOp::create(
                    rewriter, loc,
                    rewriter.getType<Torch::ListType>(
                        rewriter.getType<Torch::IntType>()),
                    SmallVector<Value>{}),
                /*fill_value=*/cstNegInf, /*dtype=*/cstFloatDtype,
                /*layout=*/cstNone, /*device=*/cstNone, /*pin_memory=*/cstNone);

            // mask_float = where(causalMask, 0, -inf)
            SmallVector<int64_t> maskFloatSizes{batchSize, sequenceLength,
                                                kvSeqLen};
            Torch::ValueTensorType maskFloatType = Torch::ValueTensorType::get(
                context, maskFloatSizes, rewriter.getF32Type());
            Value maskFloat = Torch::AtenWhereSelfOp::create(
                rewriter, loc, maskFloatType, causalMask, zeroTensor,
                negInfTensor);

            // Reshape to [batch, 1, seqLen, kvSeqLen] for SDPA
            Value maskReshapeSizeList = Torch::PrimListConstructOp::create(
                rewriter, loc,
                rewriter.getType<Torch::ListType>(
                    rewriter.getType<Torch::IntType>()),
                SmallVector<Value>{cstBatchSize, cstIntOne, cstSequenceLength,
                                   kvSeqLenVal});
            SmallVector<int64_t> attnMaskSizes{batchSize, 1, sequenceLength,
                                               kvSeqLen};
            Torch::ValueTensorType attnMaskType = Torch::ValueTensorType::get(
                context, attnMaskSizes, rewriter.getF32Type());
            attnMask = Torch::AtenReshapeOp::create(
                rewriter, loc, attnMaskType, maskFloat, maskReshapeSizeList);
          }
        }

        // Do attention with full KV cache (past + current) and mask.
        Value cstEnableGQA = Torch::ConstantBoolOp::create(rewriter, loc, true);
        Value cstFloatZero = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(0.0));
        Value cstScale = cstNone;
        if (scale != 0.0f)
          cstScale = Torch::ConstantFloatOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(scale));

        // Use presentKey/presentValue (full KV cache) for attention, not just
        // the current token's K/V. This is essential for proper KV caching.
        Value attention = Torch::AtenScaledDotProductAttentionOp::create(
            rewriter, loc, qRotary.getType(), qRotary, presentKey, presentValue,
            /*attn_mask=*/attnMask,
            /*dropout_p=*/cstFloatZero, /*is_causal=*/cstFalse, cstScale,
            cstEnableGQA);

        // Reshaping the attention result from:
        //    (batch_size, num_heads, sequence_length, head_size)
        // -> (batch_size, sequence_length, hidden_size)
        Value attentionResultSizesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(attention.getContext())),
            llvm::SmallVector<Value>{cstBatchSize, cstSequenceLength,
                                     cstHiddenSize});
        attention = Torch::AtenReshapeOp::create(
            rewriter, loc, resultTypes[0], attention, attentionResultSizesList);

        rewriter.replaceOp(binder.op, {attention, presentKey, presentValue});
        return success();
      });
  patterns.onOp(
      "MultiHeadAttention", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        MLIRContext *context = binder.op->getContext();

        // Get all operands as a list (some may be optional/none)
        SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands))
          return rewriter.notifyMatchFailure(binder.op,
                                             "operands bind failure");

        // Get result type
        Torch::ValueTensorType resultType;
        if (binder.tensorResultType(resultType))
          return rewriter.notifyMatchFailure(binder.op,
                                             "result type bind failure");

        // Get attributes
        int64_t numHeads;
        float scale;
        int64_t unidirectional;
        if (binder.s64IntegerAttr(numHeads, "num_heads"))
          return rewriter.notifyMatchFailure(binder.op,
                                             "num_heads is required");
        (void)binder.f32FloatAttr(scale, "scale", 0.0f); // 0 means use default
        (void)binder.s64IntegerAttr(unidirectional, "unidirectional", 0);

        if (numHeads == 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "num_heads is a required attribute and should be non-zero");

        if (operands.size() == 0)
          return rewriter.notifyMatchFailure(
              binder.op, "At least query input is required");

        // Extract Q, K, V from operands
        // query (index 0) is required
        Value query = operands[0];
        // key (index 1) is optional, defaults to query for self-attention
        Value key = operands.size() > 1 && operands[1] ? operands[1] : query;
        // value (index 2) is optional, defaults to key
        Value value = operands.size() > 2 && operands[2] ? operands[2] : key;

        // Index 3: bias (projection bias) - not supported yet
        if (operands.size() > 3 && operands[3] &&
            !isa<Torch::NoneType>(operands[3].getType()))
          return rewriter.notifyMatchFailure(
              binder.op, "projection bias (input 3) not yet supported");

        // Index 4: key_padding_mask - not supported yet
        if (operands.size() > 4 && operands[4] &&
            !isa<Torch::NoneType>(operands[4].getType()))
          return rewriter.notifyMatchFailure(
              binder.op, "key_padding_mask (input 4) not yet supported");

        // Index 5: attention_bias - pass through as attn_mask
        Value attnBias;
        if (operands.size() > 5 && operands[5] &&
            !isa<Torch::NoneType>(operands[5].getType()))
          attnBias = operands[5];

        // Get input shape info - only hidden_size needs to be static
        Torch::ValueTensorType queryType =
            cast<Torch::ValueTensorType>(query.getType());
        if (!queryType.hasSizes())
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expected query to have sizes");

        SmallVector<int64_t> queryDims{queryType.getSizes()};
        if (queryDims.size() < 3)
          return rewriter.notifyMatchFailure(
              binder.op, "Expected query to have at least 3 dimensions");

        // hidden_size must be static to compute head_size
        int64_t hiddenSize = queryDims[2];
        if (hiddenSize == Torch::kUnknownSize)
          return rewriter.notifyMatchFailure(
              binder.op, "hidden_size (last dimension) must be static");

        if (hiddenSize % numHeads != 0)
          return rewriter.notifyMatchFailure(
              binder.op, "hidden_size must be divisible by num_heads");

        int64_t headSize = hiddenSize / numHeads;

        // batch and sequence dimensions can be dynamic
        int64_t batchSize = queryDims[0];
        int64_t sequenceLength = queryDims[1];

        // Create constants for static values
        Value cstHeadSize = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(headSize));
        Value cstNumHeads = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(numHeads));
        Value cstHiddenSize = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(hiddenSize));
        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);

        // Get batch and sequence dimensions dynamically
        Value cstZero = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value cstOne = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value batchSizeVal = Torch::AtenSizeIntOp::create(
            rewriter, loc, rewriter.getType<Torch::IntType>(), query, cstZero);
        Value seqLenVal = Torch::AtenSizeIntOp::create(
            rewriter, loc, rewriter.getType<Torch::IntType>(), query, cstOne);

        // Reshape Q, K, V from (batch, seq, hidden) to (batch, num_heads, seq,
        // head_size)
        SmallVector<int64_t> reshapeSizesInt{batchSize, numHeads,
                                             sequenceLength, headSize};
        Value reshapeSizesList = Torch::PrimListConstructOp::create(
            rewriter, loc, Torch::ListType::get(Torch::IntType::get(context)),
            SmallVector<Value>{batchSizeVal, cstNumHeads, seqLenVal,
                               cstHeadSize});

        // Reshape query
        Value qInput = Torch::AtenReshapeOp::create(
            rewriter, loc,
            queryType.getWithSizesAndDtype(reshapeSizesInt,
                                           queryType.getOptionalDtype()),
            query, reshapeSizesList);

        // Reshape key
        Torch::ValueTensorType keyType =
            cast<Torch::ValueTensorType>(key.getType());
        Value kInput = Torch::AtenReshapeOp::create(
            rewriter, loc,
            keyType.getWithSizesAndDtype(reshapeSizesInt,
                                         keyType.getOptionalDtype()),
            key, reshapeSizesList);

        // Reshape value
        Torch::ValueTensorType valueType =
            cast<Torch::ValueTensorType>(value.getType());
        Value vInput = Torch::AtenReshapeOp::create(
            rewriter, loc,
            valueType.getWithSizesAndDtype(reshapeSizesInt,
                                           valueType.getOptionalDtype()),
            value, reshapeSizesList);

        // Create scale value - if scale is 0, use None for default behavior
        Value cstScale = cstNone;
        if (scale != 0.0f) {
          cstScale = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(scale));
        }

        // Create is_causal based on unidirectional attribute
        Value isCausal =
            Torch::ConstantBoolOp::create(rewriter, loc, unidirectional != 0);

        // Handle attention bias/mask if present
        Value attnMask = cstNone;
        if (attnBias) {
          // If attnBias is provided, use it as attn_mask
          // The shape should be (batch, num_heads, seq_len, seq_len) or
          // compatible
          attnMask = attnBias;
        }

        // Apply scaled dot product attention
        // enable_gqa should be false for standard MHA (unlike GQA)
        Value cstEnableGQA =
            Torch::ConstantBoolOp::create(rewriter, loc, false);
        Value cstFloatZero = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(0.0));

        Value attention = Torch::AtenScaledDotProductAttentionOp::create(
            rewriter, loc, qInput.getType(), qInput, kInput, vInput,
            /*attn_mask=*/attnMask,
            /*dropout_p=*/cstFloatZero, /*is_causal=*/isCausal, cstScale,
            cstEnableGQA);

        // Reshape output back from (batch, num_heads, seq, head_size)
        // to (batch, seq, hidden)
        Value attentionResultSizesList = Torch::PrimListConstructOp::create(
            rewriter, loc, Torch::ListType::get(Torch::IntType::get(context)),
            SmallVector<Value>{batchSizeVal, seqLenVal, cstHiddenSize});
        Value output = Torch::AtenReshapeOp::create(
            rewriter, loc, resultType, attention, attentionResultSizesList);

        rewriter.replaceOp(binder.op, {output});
        return success();
      });
  patterns.onOp(
      "QLinearAdd", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 8)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 8 input operands");

        Value a, aScale, aZp, b, bScale, bZp, cScale, cZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], aScale, aZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[4],
                /*zero_point=*/operands[5], bScale, bZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[6],
                /*zero_point=*/operands[7], cScale, cZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/aScale, /*zero_point=*/aZp,
                                          /*output=*/a)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `a` because of "
                         "missing sizes");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[3],
                                          /*scale=*/bScale, /*zero_point=*/bZp,
                                          /*output=*/b)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `b` because of "
                         "missing sizes");

        // Computing the result of "Add".
        auto cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value alpha = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(1.0));
        Value c = Torch::AtenAddTensorOp::create(rewriter, binder.getLoc(), cTy,
                                                 a, b, alpha);

        // Quantizing the result of "Add" operation.
        cTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(cTy.getDtype()))));
        c = Torch::AtenQuantizePerTensorOp::create(rewriter, binder.getLoc(),
                                                   cTy, c, cScale, cZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          c);
        return success();
      });
  patterns.onOp(
      "QLinearLeakyRelu", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        float alpha;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType) ||
            binder.f32FloatAttr(alpha, "alpha"))
          return failure();

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the LeakyRelu result.
        Value constAlpha = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr((double)alpha));
        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value y =
            Torch::AtenLeakyReluOp::create(rewriter, loc, yTy, x, constAlpha);

        // Quantizing the result of LeakyRelu op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        y = Torch::AtenQuantizePerTensorOp::create(rewriter, loc, yTy, y,
                                                   yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          y);
        return success();
      });
  patterns.onOp(
      "QLinearConcat", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        SmallVector<Value> operands;
        int64_t axis;
        if (binder.tensorOperandsList(operands) ||
            binder.s64IntegerAttr(axis, "axis") ||
            binder.tensorResultType(resultType))
          return failure();

        SmallVector<Value> inputs, inputScales, inputZeroPoints;
        for (unsigned i = 2; i < operands.size(); i = i + 3) {
          inputs.push_back(operands[i]);
          inputScales.push_back(operands[i + 1]);
          inputZeroPoints.push_back(operands[i + 2]);
        }

        unsigned numInputs = (operands.size() - 2) / 3;
        if (!(llvm::all_equal({inputs.size(), inputScales.size(),
                               inputZeroPoints.size()}) &&
              inputs.size() == numInputs))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible number of input operands, scales and/or "
                         "zero-points");

        // Preparing the dequantized inputs.
        SmallVector<Value> dequantizedInputs;
        for (unsigned i = 0; i < numInputs; i++) {
          Value scale, zeroPoint;
          if (failed(extractPerTensorQuantizationArguments(
                  rewriter, loc, /*scale=*/inputScales[i],
                  /*zero_point=*/inputZeroPoints[i], scale, zeroPoint)))
            return rewriter.notifyMatchFailure(
                binder.op, "Incompatible scale and zero-points argument for "
                           "per-tensor quantization");

          Value dequantizedInput;
          if (failed(createDequantizeTensor(rewriter, loc, inputs[i], scale,
                                            zeroPoint,
                                            /*output=*/dequantizedInput)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to dequantize the input tensor because of "
                           "missing sizes");

          dequantizedInputs.push_back(dequantizedInput);
        }

        // Concatenating the inputs.
        Type listElemType =
            cast<Torch::BaseTensorType>(dequantizedInputs[0].getType())
                .getWithSizesAndDtype(/*optionalSizes=*/std::nullopt,
                                      /*optionalDtype=*/nullptr);
        Type listType = Torch::ListType::get(listElemType);
        Value tensorList = Torch::PrimListConstructOp::create(
            rewriter, binder.op->getLoc(), listType, dequantizedInputs);
        Value cstAxis = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(axis));
        auto concatTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value concat = Torch::AtenCatOp::create(rewriter, loc, concatTy,
                                                tensorList, cstAxis);

        // Quantizing the result of concatenated inputs.
        Value yScale, yZp;
        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[0],
                /*zero_point=*/operands[1], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible scale and zero-points argument for "
                         "per-tensor quantization");
        Torch::ValueTensorType yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        Value result = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, concat, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          result);
        return success();
      });
  patterns.onOp(
      "QLinearGlobalAveragePool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        int64_t channelsLast;
        if (binder.tensorOperands(operands, 5) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(channelsLast, "channels_last"))
          return failure();

        // TODO: Add support for channels_last attribute.
        if (channelsLast)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: support not present for channels_last attribute");

        auto xTy = dyn_cast<Torch::ValueTensorType>(operands[0].getType());
        if (!xTy || !xTy.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input argument `x` to have sizes");
        ArrayRef<int64_t> inputShape = xTy.getSizes();

        if (!resultType || !resultType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected result type having sizes");
        }
        ArrayRef<int64_t> resultShape = resultType.getSizes();

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the AvgPool result.
        SmallVector<Value> cstKernel, cstPadding, cstStrides;
        Value cstZero = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value cstOne = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        unsigned inputRank = inputShape.size();
        for (unsigned i = 2; i < inputRank; i++) {
          if (inputShape[i] == Torch::kUnknownSize) {
            Value dim = Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(i));
            Value inputDimSize =
                Torch::AtenSizeIntOp::create(rewriter, loc, x, dim);
            cstKernel.push_back(inputDimSize);
          } else {
            int64_t kernelSize = inputShape[i] - resultShape[i] + 1;
            cstKernel.push_back(Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(kernelSize)));
          }
          cstPadding.push_back(cstZero);
          cstStrides.push_back(cstOne);
        }
        Value kernelSizeList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value stridesList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstStrides);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
        Value cstCeilMode = cstFalse;
        Value cstCountIncludePad = cstFalse;
        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);

        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultShape, rewriter.getF32Type());
        Value avgpool;
        if (inputRank == 3) {
          avgpool = Torch::AtenAvgPool1dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad);
        } else if (inputRank == 4) {
          avgpool = Torch::AtenAvgPool2dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
        } else if (inputRank == 5) {
          avgpool = Torch::AtenAvgPool3dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
        } else {
          return failure();
        }

        // Quantizing the result of AvgPool op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        avgpool = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, avgpool, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          avgpool);
        return success();
      });
  patterns.onOp(
      "QLinearSigmoid", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the Sigmoid result.
        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value y = Torch::AtenSigmoidOp::create(rewriter, loc, yTy, x);

        // Quantizing the result of Sigmoid op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        y = Torch::AtenQuantizePerTensorOp::create(rewriter, loc, yTy, y,
                                                   yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          y);
        return success();
      });
  patterns.onOp(
      "QLinearAveragePool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        int64_t channelsLast;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(channelsLast, "channels_last"))
          return failure();

        // TODO: Add support for channels_last attribute.
        if (channelsLast)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: support not present for channels_last attribute");

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Creating Onnx.AveragePool op.
        llvm::SmallVector<Value> newOperands = {x};
        llvm::SmallVector<NamedAttribute> newAttributes;
        newAttributes.push_back(rewriter.getNamedAttr(
            "name", rewriter.getStringAttr("onnx.AveragePool")));
        for (auto namedAttr : binder.op->getAttrDictionary()) {
          if (namedAttr.getName().getValue().compare("name") == 0)
            continue;
          newAttributes.push_back(namedAttr);
        }

        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value averagePool = Torch::OperatorOp::create(
                                rewriter, binder.getLoc(), yTy, newOperands,
                                newAttributes, binder.op->getRegions().size())
                                .getResult(0);

        // Quantizing the result of AveragePool op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        averagePool = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, averagePool, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          averagePool);
        return success();
      });
  patterns.onOp(
      "FusedMatMul", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value lhs, rhs;
        int64_t transA, transB, transBatchA, transBatchB;
        if (binder.tensorOperands(lhs, rhs) ||
            binder.s64IntegerAttr(transA, "transA", 0) ||
            binder.s64IntegerAttr(transB, "transB", 0) ||
            binder.s64IntegerAttr(transBatchA, "transBatchA", 0) ||
            binder.s64IntegerAttr(transBatchB, "transBatchB", 0) ||
            binder.tensorResultType(resultType))
          return failure();

        // Transposing the LHS argument.
        Value transposedLhs = lhs;
        if (transA) {
          // Determine the rank of lhs tensor.
          std::optional<unsigned> maybeRank = Torch::getTensorRank(lhs);
          if (!maybeRank)
            return rewriter.notifyMatchFailure(
                binder.op, "Unimplemented: unranked lhs tensor");
          unsigned lhsRank = *maybeRank;
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), lhs,
                  /*dimA=*/lhsRank - 2, /*dimB=*/lhsRank - 1, transposedLhs)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op for lhs");
        }

        // Transposing the RHS argument.
        Value transposedRhs = rhs;
        if (transB) {
          std::optional<unsigned> maybeRank = Torch::getTensorRank(rhs);
          if (!maybeRank)
            return rewriter.notifyMatchFailure(
                binder.op, "Unimplemented: unranked rhs tensor");
          unsigned rhsRank = *maybeRank;
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), rhs,
                  /*dimA=*/rhsRank - 2, /*dimB=*/rhsRank - 1, transposedRhs)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op for rhs");
        }

        // TODO: Add support for `transBatchA` and `transBatchB`
        // attribute.
        if (transBatchA || transBatchB)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: support not present for "
                         "transBatchA and transBatchB attribute");

        rewriter.replaceOpWithNewOp<Torch::AtenMatmulOp>(
            binder.op, resultType, transposedLhs, transposedRhs);
        return success();
      });
  patterns.onOp(
      "QLinearMul", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 8)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 8 input operands");

        Value a, b, aScale, aZp, bScale, bZp, cScale, cZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], aScale, aZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[4],
                /*zero_point=*/operands[5], bScale, bZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[6],
                /*zero_point=*/operands[7], cScale, cZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/aScale, /*zero_point=*/aZp,
                                          /*output=*/a)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `a` because of "
                         "missing sizes");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[3],
                                          /*scale=*/bScale, /*zero_point=*/bZp,
                                          /*output=*/b)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `b` because of "
                         "missing sizes");

        // Computing the Mul result.
        auto cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value c = Torch::AtenMulTensorOp::create(rewriter, binder.getLoc(), cTy,
                                                 a, b);

        // Quantizing the result of Mul operation.
        cTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(cTy.getDtype()))));
        c = Torch::AtenQuantizePerTensorOp::create(rewriter, binder.getLoc(),
                                                   cTy, c, cScale, cZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          c);
        return success();
      });
}
