from pathlib import Path
import random
from onnxruntime.quantization import quantize_dynamic, QuantFormat, QuantizationMode, QuantType, quant_pre_process
from onnxruntime.quantization import (
    matmul_nbits_quantizer,
    quant_utils,
    quantize
)

from DatasetData import getDatasetData
from MadladCalibrationDataReader import MadladDecoderCalibrationDataReader, MadladEncoderCalibrationDataReader


def quantize_madlad_model_GPTQ(
    fp32_model_path: str,
    int4_model_path: str,
    calibration_data_reader,
    percdamp=0.01,
    block_size=128,
    actorder=False,
    mse=False,
    perchannel=True,
):

    # Build GPTQ config (QOperator is required for GPTQ here).
    quant_config = matmul_nbits_quantizer.GPTQWeightOnlyQuantConfig(
        calibration_data_reader=calibration_data_reader,
        percdamp=percdamp,
        block_size=block_size,
        actorder=actorder,
        mse=mse,
        perchannel=perchannel,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),  # GPTQ is typically used for MatMul
    )

    # Create the INT4 MatMul quantizer and run it.
    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        fp32_model_path,
        nodes_to_exclude=None,
        nodes_to_include=None,
        algo_config=quant_config,
    )
    quantizer.process()

    # Save model; external data is useful for large models (>2GB).
    quantizer.model.save_model_to_file(Path(int4_model_path), use_external_data_format=False)


if __name__ == '__main__':
    rng = random.Random()
    f = open("hf_token.txt")
    dataset = getDatasetData(256, hf_token=f.read())
    targetLanguagesEncoder = []
    targetLanguagesDecoder = []
    for data in dataset:
        targetLanguagesEncoder.append(data["lang"])
        targetLanguagesDecoder.append(data["lang"])
    rng.shuffle(targetLanguagesEncoder)
    rng.shuffle(targetLanguagesDecoder)
    calibrationDataReaderEncoder = MadladEncoderCalibrationDataReader(dataset, targetLanguagesEncoder)
    calibrationDataReaderDecoder = MadladDecoderCalibrationDataReader(dataset, targetLanguagesDecoder)
    quantize_madlad_model_GPTQ(
        "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx",
        "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/GPTQ/madlad_decoder_4bit.onnx",
        calibrationDataReaderDecoder
    )
    quantize_madlad_model_GPTQ(
        "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx",
        "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/GPTQ/madlad_encoder_4bit.onnx",
        calibrationDataReaderEncoder
    )