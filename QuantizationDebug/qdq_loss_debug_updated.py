# --------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Utilities to run a given ONNX model, while saving input/output tensors of
eligible operator nodes.

A use case is to debug quantization induced accuracy drop. An AI engineer can
run the original float32 model and the quantized model with the same inputs,
then compare the corresponding activations between the two models to find
where the divergence is.

Example Usage:

```python
    class ExampleDataReader(CalibrationDataReader):
        def __init__(self):
            ...
        def get_next(self):
            ...

    input_data_reader = ExampleDataReader()

    augmented_model_path = str(Path(self._tmp_model_dir.name).joinpath("augmented_model.onnx"))
    modify_model_output_intermediate_tensors (path_to_onnx_model, augmented_model_path)

    tensor_dict = collect_activations(augmented_model_path, input_data_reader)
```

`tensor_dict` points to a dictionary where the keys are tensor names and each value
is a list of tensors, one from each model run.

This is an updated version of the onnxruntime tool that also supports newer int4 models

"""

import logging
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

import numpy
import onnx
from onnx import helper, numpy_helper

import onnxruntime

from onnxruntime.quantization.calibrate import CalibraterBase, CalibrationDataReader
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import (
    DEQUANT_OP_NAME,
    DEQUANT_OUTPUT_SUFFIX,
    QUANT_INPUT_SUFFIX,
    TENSOR_NAME_QUANT_SUFFIX,
    find_by_name,
    load_model_with_shape_infer,
)

_TENSOR_SAVE_POSTFIX = "_ReshapedSavedOutput"
_TENSOR_SAVE_POSTFIX_LEN = len(_TENSOR_SAVE_POSTFIX)

TENSOR_NAME_QUANT_SUFFIX_ALT = "_DQ_Q4"
TENSOR_NAME_QUANT_SUFFIX_ALT2 = "_DQ_Q8"
TENSOR_NAME_QUANT_SUFFIX_ALT3 = "_Q4"
TENSOR_NAME_QUANT_SUFFIX_ALT4 = "_Q8"
DEQUANT_OUTPUT_SUFFIX_ALT = "_DQ_Q4_output"
MATMUL_OUTPUT_POSTFIX = "_output_0"


def modify_model_output_intermediate_tensors(
    input_model_path: str | Path,
    output_model_path: str | Path,
    op_types_for_saving: Sequence[str] | None = None,
    save_as_external_data: bool = False,
) -> None:
    """Augment a given ONNX model to save node input/output tensors.

    Add all input/output tensors of operator nodes to model outputs
    so that their values can be retrieved for debugging purposes.

    Args:
        input_model: the path to load the model.
        op_types_for_saving: Operator types for which the
                input/output should be saved. By default, saving all the
                float32/float16 tensors.

    Returns:
        The augmented ONNX model
    """

    if op_types_for_saving is None:
        op_types_for_saving = []
    saver = CalibraterBase(input_model_path, op_types_to_calibrate=op_types_for_saving)
    model_to_augment = saver.model
    tensors, value_infos = saver.select_tensors_to_calibrate(model_to_augment)
    reshape_shape_name = "LinearReshape_" + str(time.time())
    reshape_shape = numpy_helper.from_array(numpy.array([-1], dtype=numpy.int64), reshape_shape_name)
    model_to_augment.graph.initializer.append(reshape_shape)

    for tensor_name in tensors:
        reshape_output = tensor_name + _TENSOR_SAVE_POSTFIX
        reshape_node = onnx.helper.make_node(
            "Reshape",
            inputs=[tensor_name, reshape_shape_name],
            outputs=[reshape_output],
            name=reshape_output,
        )
        model_to_augment.graph.node.append(reshape_node)
        reshape_output_value_info = helper.make_tensor_value_info(
            reshape_output, value_infos[tensor_name].type.tensor_type.elem_type, [-1]
        )
        model_to_augment.graph.output.append(reshape_output_value_info)

    onnx.save(
        model_to_augment,
        output_model_path,
        save_as_external_data=save_as_external_data,
    )


def collect_activations(
    augmented_model: str,
    input_reader: CalibrationDataReader,
    session_options=None,
    execution_providers: Sequence[str] | None = None,
) -> dict[str, list[numpy.ndarray]]:
    """Run augmented model and collect activations tensors.

    Args:
        augmented_model: Path to augmented model created by modify_model_output_intermediate_tensors ()
        input_reader: Logic for reading input for the model, augmented model have the same
            input with the original model.
        session_options: Optional OnnxRuntime session options for controlling model run.
            By default graph optimization is turned off
        execution_providers: Collection of execution providers for running the model.
            Only CPU EP is used by default.

    Returns:
        A dictionary where the key is tensor name and values are list of tensors from each batch
    """

    if session_options is None:
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    if execution_providers is None:
        execution_providers = ["CPUExecutionProvider"]

    inference_session = onnxruntime.InferenceSession(
        augmented_model,
        sess_options=session_options,
        providers=execution_providers,
    )

    intermediate_outputs = []
    for input_d in input_reader:
        intermediate_outputs.append(inference_session.run(None, input_d))
    if not intermediate_outputs:
        raise RuntimeError("No data is collected while running augmented model!")

    output_dict = {}
    output_info = inference_session.get_outputs()
    for batch in intermediate_outputs:
        for output, output_data in zip(output_info, batch, strict=False):
            if output.name.endswith(_TENSOR_SAVE_POSTFIX):
                output_name = output.name[:-_TENSOR_SAVE_POSTFIX_LEN]
                output_dict.setdefault(output_name, []).append(output_data)

    return output_dict


_POST_QDQ_POSTFIX1 = DEQUANT_OUTPUT_SUFFIX + "_1"


def _add_pre_post_qdq_pair(
    qdq_cmp: dict[str, dict[str, Sequence[numpy.ndarray]]],
    activation_name: str,
    pre_qdq_tensors: Sequence[numpy.ndarray] | None,
    post_qdq_tensors: Sequence[numpy.ndarray] | None,
) -> None:
    if post_qdq_tensors is not None and pre_qdq_tensors is not None:
        qdq_cmp[activation_name] = {}
        qdq_cmp[activation_name]["pre_qdq"] = pre_qdq_tensors
        qdq_cmp[activation_name]["post_qdq"] = post_qdq_tensors

def _add_post_qdq_float_pair(
    qdq_cmp: dict[str, dict[str, Sequence[numpy.ndarray]]],
    activation_name: str,
    post_qdq_tensors: Sequence[numpy.ndarray] | None,
    float_tensors: Sequence[numpy.ndarray] | None,
) -> None:
    if post_qdq_tensors is not None and float_tensors is not None:
        qdq_cmp[activation_name] = {}
        qdq_cmp[activation_name]["post_qdq"] = post_qdq_tensors
        qdq_cmp[activation_name]["float"] = float_tensors


def create_activation_matching(
    qdq_activations: dict[str, Sequence[numpy.ndarray]],
    float_activations: dict[str, Sequence[numpy.ndarray]] | None = None,
) -> dict[str, dict[str, Sequence[numpy.ndarray]]]:
    """Comparing activation values to help debugging accuracy loss due to quantization.

    This functions takes saved activations from the QDQ model and (optionally) the
    float point model, and provides a data structure for comparing:
        * from the qdq model, activation values before and after QDQ operation
        * across both models, activations from the orignal model vs the corresponding
          activations in the QDQ model

    Arg:
        qdq_activations: Output of `collect_activations`. This must be from a quantized
            model with QDQ format.
        float_activations: Output of `collect_activations`. This must be from the float
            point model.

    Returns:
        Dict for comparing pre and post quantized activation tensors. E.g.
        ```
        qdq_cmp = cmp_qdq_input_output(qdq_activations)
        print(qdq_cmp['activation1']['pre_qdq'][0])
        print(qdq_cmp['activation1'][`post_qdq'][0])


        qdq_cmp = cmp_qdq_input_output(qdq_activations, float_activations)
        print(qdq_cmp['activation1']['float'][0])
        print(qdq_cmp['activation1']['pre_qdq'][0])
        print(qdq_cmp['activation1'][`post_qdq'][0])
        ```
    """

    qdq_cmp: dict[str, dict[str, Sequence[numpy.ndarray]]] = {}
    for tensor_name, tensors in qdq_activations.items():
        if tensor_name.endswith(QUANT_INPUT_SUFFIX):
            pre_name = tensor_name[: -len(QUANT_INPUT_SUFFIX)]
            post_qdq_tensors = qdq_activations.get(pre_name)
            pre_qdq_tensors = tensors
            _add_pre_post_qdq_pair(qdq_cmp, pre_name, pre_qdq_tensors, post_qdq_tensors)

        elif tensor_name.endswith(DEQUANT_OUTPUT_SUFFIX):
            pre_name = tensor_name[: -len(DEQUANT_OUTPUT_SUFFIX)]
            pre_qdq_tensors = qdq_activations.get(pre_name)
            post_qdq_tensors = tensors
            _add_pre_post_qdq_pair(qdq_cmp, pre_name, pre_qdq_tensors, post_qdq_tensors)
        elif tensor_name.endswith(DEQUANT_OUTPUT_SUFFIX_ALT):
            pre_name = tensor_name[: -len(DEQUANT_OUTPUT_SUFFIX_ALT)]
            pre_qdq_tensors = qdq_activations.get(pre_name)
            post_qdq_tensors = tensors
            _add_pre_post_qdq_pair(qdq_cmp, pre_name, pre_qdq_tensors, post_qdq_tensors)

        elif tensor_name.endswith(_POST_QDQ_POSTFIX1):
            pre_name = tensor_name[: -len(_POST_QDQ_POSTFIX1)]
            pre_qdq_tensors = qdq_activations.get(pre_name)
            post_qdq_tensors = tensors
            _add_pre_post_qdq_pair(qdq_cmp, pre_name, pre_qdq_tensors, post_qdq_tensors)

    if not float_activations:
        return qdq_cmp

    for act_name, act_values in qdq_cmp.items():
        float_acts = float_activations.get(act_name)
        if float_acts is not None:
            act_values["float"] = float_acts

    return qdq_cmp


def create_activation_matching_updated(
    qdq_activations: dict[str, Sequence[numpy.ndarray]],
    float_activations: dict[str, Sequence[numpy.ndarray]],
) -> dict[str, dict[str, Sequence[numpy.ndarray]]]:
    """Comparing activation values to help debugging accuracy loss due to quantization.

    This functions takes saved activations from the QDQ model and the
    float point model, and provides a data structure for comparing across both models,
    activations from the orignal model vs the corresponding activations in the QDQ model

    Arg:
        qdq_activations: Output of `collect_activations`. This must be from a quantized
            model with QDQ format.
        float_activations: Output of `collect_activations`. This must be from the float
            point model.

    Returns:
        Dict for comparing quantized activation tensors with the corrisponding float activation tensors. E.g.
        ```
        qdq_cmp = cmp_qdq_input_output(qdq_activations, float_activations)
        print(qdq_cmp['activation1']['float'][0])
        print(qdq_cmp['activation1'][`post_qdq'][0])
        ```
    """

    qdq_cmp: dict[str, dict[str, Sequence[numpy.ndarray]]] = {}
    for tensor_name, tensors in qdq_activations.items():
        if tensor_name in float_activations:
            post_qdq_tensors = tensors
            float_tensors = float_activations.get(tensor_name)
            _add_post_qdq_float_pair(qdq_cmp, tensor_name, post_qdq_tensors, float_tensors)

    return qdq_cmp


def _run_dequantize_linear(
    weight_tensor: numpy.ndarray,
    weight_scale: numpy.ndarray,
    weight_zp: numpy.ndarray,
    axis: int,
    block_size: int = 0,
) -> Optional[numpy.ndarray]:
    """
    Supports ONNX DequantizeLinear granularities:
      - per-tensor: scale/zp are scalar
      - per-axis:   scale/zp are 1-D (len = D_axis)
      - blocked:    scale/zp have same rank as input; axis dim is ceil(D_axis / block_size)

    Notes:
      * For blocked quantization, ONNX uses the 'block_size' attribute and 'axis'. :contentReference[oaicite:1]{index=1}
      * This assumes weight_tensor already contains *unpacked* quantized integer values
        (i.e., not packed 4-bit nibbles).
    """
    if weight_scale.shape != weight_zp.shape:
        raise ValueError(f"scale and zero_point must have same shape, got {weight_scale.shape} vs {weight_zp.shape}")

    r = weight_tensor.ndim
    if r == 0:
        # scalar tensor
        return (weight_tensor.astype(numpy.float32) - weight_zp.astype(numpy.float32)) * weight_scale.astype(numpy.float32)

    # normalize axis
    if axis < 0:
        axis += r
    if not (0 <= axis < r):
        raise ValueError(f"axis out of range: axis={axis}, rank={r}")

    # ---- per-tensor ----
    if weight_zp.size == 1:
        # (x - zp) * scale with numpy broadcasting
        return (weight_tensor.astype(numpy.float32) - weight_zp.astype(numpy.float32)) * weight_scale.astype(numpy.float32)

    # ---- per-axis ----
    if weight_scale.ndim == 1 and block_size == 0:
        if weight_scale.shape[0] != weight_tensor.shape[axis]:
            raise ValueError(
                f"Per-axis expects scale len == weight.shape[axis]; "
                f"got len={weight_scale.shape[0]} vs D_axis={weight_tensor.shape[axis]}"
            )
        # reshape to broadcast on axis
        reshape = [1] * r
        reshape[axis] = weight_tensor.shape[axis]
        s = weight_scale.astype(numpy.float32).reshape(reshape)
        z = weight_zp.astype(numpy.float32).reshape(reshape)
        return (weight_tensor.astype(numpy.float32) - z) * s

    # ---- blocked quantization ----
    # ONNX blocked: scale/zp have same rank as input, but axis dimension is ceil(D_axis / B). :contentReference[oaicite:2]{index=2}
    if weight_scale.ndim == r:
        if block_size <= 0:
            raise ValueError(
                "Blocked quantization detected (scale.ndim == input.ndim) but block_size <= 0. "
                "Pass DequantizeLinear's 'block_size' attribute."
            )

        # sanity: all dims except axis must match
        for d in range(r):
            if d == axis:
                continue
            if weight_scale.shape[d] != weight_tensor.shape[d]:
                raise ValueError(
                    f"Blocked quantization expects scale.shape[d]==weight.shape[d] for d != axis; "
                    f"mismatch at d={d}: {weight_scale.shape[d]} vs {weight_tensor.shape[d]}"
                )

        d_axis = weight_tensor.shape[axis]
        expected = int(math.ceil(d_axis / block_size))
        if weight_scale.shape[axis] != expected:
            # Some exporters may choose a different compressed length; you *can* relax this,
            # but it's usually a bug/mismatch worth surfacing.
            raise ValueError(
                f"Blocked quantization axis dim mismatch: scale.shape[axis]={weight_scale.shape[axis]} "
                f"but expected ceil({d_axis}/{block_size})={expected}"
            )

        # expand scale/zp along axis: repeat each block scale block_size times, then trim to d_axis
        s_full = numpy.repeat(weight_scale.astype(numpy.float32), repeats=block_size, axis=axis)
        z_full = numpy.repeat(weight_zp.astype(numpy.float32), repeats=block_size, axis=axis)

        slicer = [slice(None)] * r
        slicer[axis] = slice(0, d_axis)
        s_full = s_full[tuple(slicer)]
        z_full = z_full[tuple(slicer)]

        return (weight_tensor.astype(numpy.float32) - z_full) * s_full

    # If you get here, scale has an unexpected rank/shape for ONNX DQ.
    raise ValueError(
        f"Unsupported DequantizeLinear scale rank: input rank={r}, scale rank={weight_scale.ndim}. "
        "Expected scalar, 1-D (per-axis), or same-rank (blocked)."
    )


def _run_dequantize_linear_matmulnbits(
    B_blob_u8: numpy.ndarray,          # uint8 flat (initializer)
    scales: numpy.ndarray,             # float16/float32 flat, len = N * n_blocks
    zero_points_u8: numpy.ndarray | None,  # uint8 flat; may be optional
    K: int,
    N: int,
    bits: int = 4,
    block_size: int = 128,
    out_dtype=numpy.float32,
    return_shape: str = "KxN",      # "KxN" for standard MatMul B, or "NxK" if you want stored layout
) -> numpy.ndarray:
    """
    Reconstructs dequantized weight from com.microsoft::MatMulNBits storage.

    ORT schema: B stored as uint8 with shape [N][n_blocks_per_col][blob_size]
      n_blocks_per_col = ceil(K / block_size)
      blob_size = block_size/8 * bits
    scales shape: [N * n_blocks_per_col]
    zero_points shape:
      - [(N * n_blocks_per_col + 1) / 2] if bits <= 4 (2 zp per byte)
      - [N * n_blocks_per_col]           if bits > 4 (1 zp per byte)
    """
    if bits not in (4, 8):
        raise NotImplementedError("This helper implements bits=4 and bits=8. Extend if you need 2/3/5/6/7.")

    if block_size <= 0 or (block_size & (block_size - 1)) != 0 or block_size < 16:
        raise ValueError("block_size must be a power of 2 and >= 16 for MatMulNBits.")

    n_blocks = (K + block_size - 1) // block_size
    blob_size = (block_size // 8) * bits  # per schema

    expected_B = N * n_blocks * blob_size
    if B_blob_u8.size != expected_B:
        raise ValueError(f"B blob has {B_blob_u8.size} bytes, expected {expected_B} (=N*n_blocks*blob_size).")

    if scales.size != N * n_blocks:
        raise ValueError(f"scales has {scales.size} elems, expected {N*n_blocks} (=N*n_blocks).")

    if zero_points_u8 is not None:
        if bits <= 4:
            expected_zp = (N * n_blocks + 1) // 2
        else:
            expected_zp = N * n_blocks
        if zero_points_u8.size != expected_zp:
            raise ValueError(f"zero_points has {zero_points_u8.size} elems, expected {expected_zp}.")

    # reshape B into [N, n_blocks, blob_size]
    B = B_blob_u8.reshape(N, n_blocks, blob_size)

    # We'll build weight in the logical (stored) orientation: [N, K]
    W_NK = numpy.empty((N, K), dtype=out_dtype)

    scales_f = scales.astype(out_dtype, copy=False)

    def get_zp(block_index: int) -> int:
        # block_index in [0, N*n_blocks)
        if zero_points_u8 is None:
            # If zero_points is omitted, we return the default zero point 2^(bits - 1), according to MatMulNBits documentation
            return pow(2, bits-1)
        if bits <= 4:
            b = int(zero_points_u8[block_index // 2])
            return (b & 0x0F) if (block_index % 2 == 0) else (b >> 4)
        else:
            return int(zero_points_u8[block_index])

    for n in range(N):
        for blk in range(n_blocks):
            block_index = n * n_blocks + blk
            scale = scales_f[block_index]
            zp = get_zp(block_index)

            k0 = blk * block_size
            k1 = min(k0 + block_size, K)
            valid = k1 - k0

            blob = B[n, blk]  # uint8[blob_size]

            if bits == 8:
                # blob_size = block_size, values are direct (uint8) for this block
                q = blob[:valid].astype(out_dtype)
            else:
                # bits == 4
                # blob_size = block_size/2, each byte packs 2 nibbles
                packed = blob  # length block_size/2
                # expand to block_size values
                q_full = numpy.empty(block_size, dtype=numpy.uint8)
                q_full[0::2] = packed & 0x0F
                q_full[1::2] = packed >> 4
                q = q_full[:valid].astype(out_dtype)

            W_NK[n, k0:k1] = (q - out_dtype(zp)) * scale

    if return_shape.upper() == "NXK":
        return W_NK
    if return_shape.upper() == "KXN":
        return W_NK.T
    raise ValueError("return_shape must be 'KxN' or 'NxK'.")


def create_weight_matching(float_model_path: str, qdq_model_path: str) -> dict[str, dict[str, numpy.ndarray]]:
    """Comparing weight values to help debugging accuracy loss due to quantization.

    This functions takes the float model and the qdq model, and provides a data structure for comparing
    their corresponding weights to locate quantization errors

    Arg:
        float_model_path: Path points to the float point model.
        qdq_model_path: Path points to the qdq model.

    Returns:
        Dict for comparing weight tensors. E.g.
        ```
        qdq_weight_cmp = create_weight_matching(float_model, qdq_model)
        print(qdq_weight_cmp['activation1']['float'])
        print(qdq_weight_cmp['activation1']['dequantized'])
        ```
    """
    float_onnx_model = ONNXModel(load_model_with_shape_infer(Path(float_model_path)))
    qdq_onnx_model = ONNXModel(load_model_with_shape_infer(Path(qdq_model_path)))

    matched_weights: dict[str, dict[str, numpy.ndarray]] = {}
    initializers = qdq_onnx_model.initializer()
    for node in qdq_onnx_model.nodes():
        if node.op_type != DEQUANT_OP_NAME:
            continue  # Only care about DQ node
        weight_name: str = node.input[0]
        weight_values = find_by_name(weight_name, initializers)
        if not weight_values:
            continue  # Only care about DQ node with const inputs
        if not (weight_name.endswith(TENSOR_NAME_QUANT_SUFFIX) or weight_name.endswith(TENSOR_NAME_QUANT_SUFFIX_ALT)):
            logging.error(f"Model Error in '{qdq_model_path}': Dequantized tensor name '{weight_name}' not recognized!")
            continue

        axis = -1
        block_size = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
            if(attr.name == "block_size"):
                block_size = attr.i

        weight_tensor = numpy_helper.to_array(weight_values)
        weight_scale = numpy_helper.to_array(find_by_name(node.input[1], initializers))
        if len(node.input) > 2:
            weight_zp = numpy_helper.to_array(find_by_name(node.input[2], initializers))
        else:
            weight_zp = numpy.zeros(weight_scale.shape, dtype=numpy.int32)

        # Perform dequantization:
        if weight_scale.size == weight_zp.size == 1:
            # Avoids the confusion between a scaler and a tensor of one element.
            weight_scale = weight_scale.reshape(())
            weight_zp = weight_zp.reshape(())
        if weight_scale.shape != weight_zp.shape:
            raise RuntimeError(
                f"scale and zero_point must have the same shape but {weight_scale.shape} != {weight_zp.shape}"
            )
        weight_quant = _run_dequantize_linear(weight_tensor, weight_scale, weight_zp, axis=axis, block_size=block_size)
        if(weight_name.endswith(TENSOR_NAME_QUANT_SUFFIX)):
            weight_name = weight_name[: -len(TENSOR_NAME_QUANT_SUFFIX)]
        elif(weight_name.endswith(TENSOR_NAME_QUANT_SUFFIX_ALT)):
            weight_name = weight_name[: -len(TENSOR_NAME_QUANT_SUFFIX_ALT)]
        if weight_quant is None:
            logging.error(f"Model Error in '{qdq_model_path}': '{weight_name}' per-channel quantization on 0 channel")
            continue

        float_values = find_by_name(weight_name, float_onnx_model.initializer())
        if not float_values:
            logging.error(f"Model Error in '{float_model_path}': weight tensor '{weight_name}' not found!")
            continue
        weight_float = numpy_helper.to_array(float_values)
        matched_weights[weight_name] = {"float": weight_float, "dequantized": weight_quant}

    return matched_weights



def create_weight_matching_qoperator(float_model_path: str, qoperator_model_path: str) -> dict[str, dict[str, numpy.ndarray]]:
    """Comparing weight values to help debugging accuracy loss due to quantization.

    This functions takes the float model and the qdq model, and provides a data structure for comparing
    their corresponding weights to locate quantization errors

    Arg:
        float_model_path: Path points to the float point model.
        qdq_model_path: Path points to the qdq model.

    Returns:
        Dict for comparing weight tensors. E.g.
        ```
        qdq_weight_cmp = create_weight_matching(float_model, qdq_model)
        print(qdq_weight_cmp['activation1']['float'])
        print(qdq_weight_cmp['activation1']['dequantized'])
        ```
    """
    float_onnx_model = ONNXModel(load_model_with_shape_infer(Path(float_model_path)))
    qdq_onnx_model = ONNXModel(load_model_with_shape_infer(Path(qoperator_model_path)))

    matched_weights: dict[str, dict[str, numpy.ndarray]] = {}
    initializers = qdq_onnx_model.initializer()
    for node in qdq_onnx_model.nodes():
        if node.op_type != "MatMul" and node.op_type != "MatMulNBits":
            continue  # Only care about MatMul node and variants
        weight_name: str = node.input[1]
        weight_values = find_by_name(weight_name, initializers)
        if not weight_values:
            continue  # Only care about MatMul node with const B matrix
        if not (weight_name.endswith((TENSOR_NAME_QUANT_SUFFIX, TENSOR_NAME_QUANT_SUFFIX_ALT, TENSOR_NAME_QUANT_SUFFIX_ALT2, TENSOR_NAME_QUANT_SUFFIX_ALT3, TENSOR_NAME_QUANT_SUFFIX_ALT4))):
            logging.error(f"Model Error in '{qoperator_model_path}': Dequantized tensor name '{weight_name}' not recognized!")
            continue

        axis = 0
        block_size = 0
        k = -1
        n = -1
        bits = 4
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
            if(attr.name == "block_size"):
                block_size = attr.i
            if(attr.name == "K"):
                k = attr.i
            if(attr.name == "N"):
                n = attr.i
            if(attr.name == "bits"):
                bits = attr.i

        weight_tensor = numpy_helper.to_array(weight_values)
        weight_scale = numpy_helper.to_array(find_by_name(node.input[2], initializers))
        weight_zp: None | numpy.ndarray = None
        if len(node.input) > 3:
            weight_zp = numpy_helper.to_array(find_by_name(node.input[3], initializers))

        # Perform dequantization:
        if weight_scale.size == 1:
            # Avoids the confusion between a scaler and a tensor of one element.
            weight_scale = weight_scale.reshape(())
        if (weight_zp is not None and weight_zp.size == 1):
            # Avoids the confusion between a scaler and a tensor of one element.
            weight_zp = weight_zp.reshape(())

        weight_quant = _run_dequantize_linear_matmulnbits(weight_tensor, weight_scale, weight_zp, K=k, N=n, bits=bits, block_size=block_size)
        if(weight_name.endswith(TENSOR_NAME_QUANT_SUFFIX)):
            weight_name = weight_name[: -len(TENSOR_NAME_QUANT_SUFFIX)]
        elif(weight_name.endswith(TENSOR_NAME_QUANT_SUFFIX_ALT)):
            weight_name = weight_name[: -len(TENSOR_NAME_QUANT_SUFFIX_ALT)]
        elif(weight_name.endswith(TENSOR_NAME_QUANT_SUFFIX_ALT2)):
            weight_name = weight_name[: -len(TENSOR_NAME_QUANT_SUFFIX_ALT2)]
        elif(weight_name.endswith(TENSOR_NAME_QUANT_SUFFIX_ALT3)):
            weight_name = weight_name[: -len(TENSOR_NAME_QUANT_SUFFIX_ALT3)]
        elif(weight_name.endswith(TENSOR_NAME_QUANT_SUFFIX_ALT4)):
            weight_name = weight_name[: -len(TENSOR_NAME_QUANT_SUFFIX_ALT4)]
        if weight_quant is None:
            logging.error(f"Model Error in '{qoperator_model_path}': '{weight_name}' per-channel quantization on 0 channel")
            continue

        float_values = find_by_name(weight_name, float_onnx_model.initializer())
        if not float_values:
            logging.error(f"Model Error in '{float_model_path}': weight tensor '{weight_name}' not found!")
            continue
        weight_float = numpy_helper.to_array(float_values)
        matched_weights[weight_name] = {"float": weight_float, "dequantized": weight_quant}

    return matched_weights


def compute_signal_to_quantization_noice_ratio(
    x: Sequence[numpy.ndarray] | numpy.ndarray, y: Sequence[numpy.ndarray] | numpy.ndarray
) -> float:
    if isinstance(x, numpy.ndarray):
        xlist = [x]
    else:
        xlist = x
    if isinstance(y, numpy.ndarray):
        ylist = [y]
    else:
        ylist = y
    if len(xlist) != len(ylist):
        raise RuntimeError("Unequal number of tensors to compare!")

    left = numpy.concatenate(xlist).flatten()
    right = numpy.concatenate(ylist).flatten()

    epsilon = numpy.finfo("float").eps
    tensor_norm = max(numpy.linalg.norm(left), epsilon)
    diff_norm = max(numpy.linalg.norm(left - right), epsilon)
    res = tensor_norm / diff_norm
    return 20 * math.log10(res)

def compute_signal_to_quantization_relative_norm_percentage(
    x: Sequence[numpy.ndarray] | numpy.ndarray, y: Sequence[numpy.ndarray] | numpy.ndarray
) -> float:
    if isinstance(x, numpy.ndarray):
        xlist = [x]
    else:
        xlist = x
    if isinstance(y, numpy.ndarray):
        ylist = [y]
    else:
        ylist = y
    if len(xlist) != len(ylist):
        raise RuntimeError("Unequal number of tensors to compare!")

    left = numpy.concatenate(xlist).flatten()
    right = numpy.concatenate(ylist).flatten()

    epsilon = numpy.finfo("float").eps
    tensor_norm = max(numpy.linalg.norm(left), epsilon)
    diff_norm = max(numpy.linalg.norm(left - right), epsilon)
    res = 100 * float(diff_norm / tensor_norm)
    return res


def compute_weight_error(
    weights_match: dict[str, dict[str, numpy.ndarray]],
    err_func: Callable[[numpy.ndarray, numpy.ndarray], float] = compute_signal_to_quantization_noice_ratio,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for weight_name, weight_match in weights_match.items():
        result[weight_name] = err_func(weight_match["float"], weight_match["dequantized"])
    return result


def compute_activation_error(
    activations_match: dict[str, dict[str, Sequence[numpy.ndarray]]],
    err_func: Callable[
        [Sequence[numpy.ndarray], Sequence[numpy.ndarray]], float
    ] = compute_signal_to_quantization_noice_ratio,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, match in activations_match.items():
        err_result: dict[str, float] = {}
        if("pre_qdq" in match):
            err_result["qdq_err"] = err_func(match["pre_qdq"], match["post_qdq"])
        float_activation = match["float"]
        if float_activation:
            err_result["xmodel_err"] = err_func(float_activation, match["post_qdq"])
        result[name] = err_result
    return result

def compute_activation_error_clean(
    activations_match: dict[str, dict[str, Sequence[numpy.ndarray]]],
    err_func: Callable[
        [Sequence[numpy.ndarray], Sequence[numpy.ndarray]], float
    ] = compute_signal_to_quantization_noice_ratio,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, match in activations_match.items():
        float_activation = match["float"]
        if float_activation:
            result[name] = err_func(float_activation, match["post_qdq"])
    return result

'''def compute_activation_error_avg(activation_error: dict[str, dict[str, float]]) -> dict[str, float]:
    activation_error_avg: dict[str, float] | None = None
    for key in activation_error:
        avg_error = 0
        for innerkey in activation_error[key]:
            avg_error = activation_error[key][innerkey]
        avg_error = avg_error / len(activation_error[key])
        activation_error_avg[key] = avg_error
    return activation_error_avg'''