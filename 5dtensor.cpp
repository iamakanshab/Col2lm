/*
  This implementation focuses specifically on lowering the ONNX Col2Im operation to its Torch equivalent, with explicit support for 5D tensors.
  Here are the key aspects of this implementation:

1. Input Handling:
   - Expects a 3D input tensor (C * D * H * W, N * oD * oH, oW) as per ONNX specification.
   - The output shape is provided as a separate tensor operand, which is expected to be 5D.

2. Attribute Validation:
   - Validates that blockShape, dilations, and strides have 3 elements each (for D, H, W).
   - Validates that pads has 6 elements (start and end padding for D, H, W).

3. Tensor Reshaping:
   - Reshapes the input from (C * D * H * W, N * oD * oH, oW) to (N, C * D * H * W, oD * oH, oW).
   - This step is necessary to match the expected input format of Torch's Col2Im operation.

4. Torch Operation Creation:
   - Creates a constant tensor for blockShape using Torch::ConstantOp.
   - Uses Torch::AtenCol2ImOp for the actual Col2Im operation, which supports 5D tensors.

5. Flexible Output Shape:
   - The output shape is set to (-1, -1, -1, -1, -1), allowing for dynamic shape inference in Torch.

6. ONNX to Torch Mapping:
   - Maps the ONNX Col2Im operation directly to its Torch equivalent, maintaining the semantics of the operation while adapting to Torch's specific requirements.

This implementation provides a clear lowering strategy from the ONNX Col2Im operation to its Torch equivalent, with explicit support for 5D tensors.
It handles the necessary reshaping and attribute conversions to ensure compatibility between the ONNX and Torch versions of the operation.
*/

LogicalResult col2imONNXToTorchLowering(OpBinder binder, ConversionPatternRewriter &rewriter) {
  // Tensor operands
  Value input, outputShape;
  
  // Attributes
  SmallVector<int64_t> blockShape, dilations, pads, strides;

  // Bind operands and attributes
  if (binder.tensorOperandAtIndex(input, 0) ||
      binder.tensorOperandAtIndex(outputShape, 1) ||
      binder.s64IntegerArrayAttr(blockShape, "blockShape") ||
      binder.s64IntegerArrayAttr(dilations, "dilations") ||
      binder.s64IntegerArrayAttr(pads, "pads") ||
      binder.s64IntegerArrayAttr(strides, "strides"))
    return failure();

  // Validate input tensor
  auto inputTy = cast<Torch::ValueTensorType>(input.getType());
  auto inputShape = inputTy.getSizes();
  if (inputShape.size() != 3) {
    return rewriter.notifyMatchFailure(binder.op, "Expected input to be a 3D tensor");
  }

  // Validate output shape
  auto outputShapeTy = cast<Torch::ValueTensorType>(outputShape.getType());
  auto outputShapeSizes = outputShapeTy.getSizes();
  if (outputShapeSizes[0] != 5) {
    return rewriter.notifyMatchFailure(binder.op, "Expected output shape to be 5D");
  }

  // Validate attribute sizes
  if (blockShape.size() != 3 || dilations.size() != 3 || strides.size() != 3 || pads.size() != 6) {
    return rewriter.notifyMatchFailure(binder.op, "Attribute sizes don't match 5D tensor requirements");
  }

  // Create Torch tensor for blockShape
  auto blockShapeTensor = rewriter.create<Torch::ConstantOp>(
    binder.op->getLoc(),
    Torch::ValueTensorType::get(rewriter.getContext(), {3}, rewriter.getI64Type()),
    rewriter.getDenseI64ArrayAttr(blockShape));

  // Reshape input from (C * D * H * W, N * oD * oH, oW) to (N, C * D * H * W, oD * oH, oW)
  auto reshapedInput = rewriter.create<Torch::AtenReshapeOp>(
    binder.op->getLoc(),
    Torch::ValueTensorType::get(rewriter.getContext(), {-1, inputShape[0], -1, inputShape[2]}, inputTy.getDtype()),
    input,
    rewriter.getDenseI64ArrayAttr({-1, inputShape[0], -1, inputShape[2]}));

  // Create the Torch Col2Im operation
  auto col2imOp = rewriter.create<Torch::AtenCol2ImOp>(
    binder.op->getLoc(),
    Torch::ValueTensorType::get(rewriter.getContext(), {-1, -1, -1, -1, -1}, inputTy.getDtype()),
    reshapedInput,
    outputShape,
    blockShapeTensor.getResult(),
    rewriter.getI64ArrayAttr(dilations),
    rewriter.getI64ArrayAttr(pads),
    rewriter.getI64ArrayAttr(strides));

  // Replace the original op with the new col2im op
  rewriter.replaceOp(binder.op, col2imOp.getResult());

  return success();
}

// Register the pattern
patterns.onOp("ONNXCol2ImOp", 18, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
  return col2imONNXToTorchLowering(binder, rewriter);
});
