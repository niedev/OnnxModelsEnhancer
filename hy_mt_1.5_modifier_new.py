import json
import os
from pathlib import Path
import optimum.exporters.onnx
import huggingface_hub
from huggingface_hub import constants, snapshot_download
from onnxruntime_genai.models.builder import create_model
from onnxruntime.quantization import (
    matmul_nbits_quantizer,
    quant_utils,
    quantize
)
from onnxruntime.quantization import quantize_dynamic, QuantFormat, QuantizationMode, QuantType, quant_pre_process
from transformers import HunYuanDenseV1ForCausalLM, Gemma3ForCausalLM
import hy_mt_optimum_exporter
import onnx_execution
import hy_mt_custom_exporter
from kernels import has_kernel


PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", constants.HF_HUB_CACHE)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb — the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."


def create_hy_final_model():
    #hy_mt_optimum_exporter.convert_hy_cache_optimum()
    quantize_hy(modelPath="onnx/HY-MT/Optimum_Cache_Optimized/model_optimized.onnx", 
                     outputFolder="onnx/HY-MT/Optimum_Cache_Optimized/Quantized/", bits=4)
    quantize_hy(modelPath="onnx/HY-MT/HuggingFace/model.onnx", 
                     outputFolder="onnx/HY-MT/HuggingFace/Quantized/", bits=4)


def quantize_hy(modelPath = "onnx/HY-MT/HuggingFace/model.onnx", outputFolder = "onnx/HY-MT/HuggingFace/Quantized/", bits=4):
    accuracy_level = 4
    quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=128, # 2's exponential and >= 16 (128)
        is_symmetric=False, # if true, quantize to Int4. otherwise, quantize to uint4.
        accuracy_level=4, # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35,
        quant_format = quant_utils.QuantFormat.QOperator,
        op_types_to_quantize={"MatMul", "Gather"},
        bits=bits)
    
    os.makedirs(outputFolder, exist_ok=True)
    
    #quantization of decoder
    model_fp32_path=modelPath
    model_quant_path=outputFolder + ("model.onnx" if bits == 4 else "model_int8.onnx")
    model_quant_final_path=outputFolder + "model_int8_final.onnx"

    if(not Path(model_quant_path).is_file()):
        _quantize_weight_only(model_fp32_path, model_quant_path, quant_config, None, accuracy_level, True)
        if(bits == 8):
            _quantize_dynamic_int8(model_quant_path, model_quant_final_path, op_types_to_quantize=["Gather", "MatMul"], save_external=True)


def _quantize_weight_only(model_fp32_path: str, model_int_path: str, quant_config, nodes_to_exclude=None, accuracy_level=None, save_external=False):
    model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        accuracy_level=accuracy_level,
        nodes_to_exclude=nodes_to_exclude, # specify a list of nodes to exclude from quantization
        algo_config=quant_config,)
    quant.process()
    quant.model.save_model_to_file(model_int_path, save_external)


def _quantize_dynamic_int8(model_fp32_path: str, model_int8_path: str, op_types_to_quantize=None, nodes_to_exclude=None, save_external=False):
    quantize_dynamic(Path(model_fp32_path), Path(model_int8_path),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=op_types_to_quantize,
                     use_external_data_format = save_external, nodes_to_exclude=nodes_to_exclude,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})



if __name__ == '__main__':
    #create_hy_final_model()

    onnx_execution.onnx_execution_hy_cache(text=en_text, src_lang="English", tgt_lang="Italian", decoder_path="onnx/HY-MT/Optimum_Cache_Optimized/Quantized/model_int8_final.onnx")
    #print("Model Optimized:")
    #onnx_execution.onnx_execution_hy_cache(text=en_text, src_lang="English", tgt_lang="Italian", decoder_path="onnx/HY-MT/Optimum_Cache_Optimized/Quantized/model_int8_final.onnx")
    #print("Model:")
    #onnx_execution.onnx_execution_hy_cache(text=en_text, src_lang="English", tgt_lang="Italian", decoder_path="onnx/HY-MT/Optimum_Cache_Optimized/Quantized/model.onnx")
    #print("Model Group Query Attention:")

    print("quantization-gptq available:", has_kernel("kernels-community/quantization-gptq"))

    #onnx_execution.execute_decoder_only_hf(text=en_text, src_lang="English", tgt_lang="Italian", quantized=False, log=True, model_name="tencent/HY-MT1.5-1.8B")
    #onnx_execution.execute_decoder_only_hf(text=en_text, src_lang="English", tgt_lang="Italian", quantized=True, log=True, model_name="tencent/HY-MT1.5-1.8B-GPTQ-Int4")

    '''onnx_execution.compare_models_quality_multi_language(
        decoder_path="onnx/HY-MT/Optimum_Cache_Optimized/model_optimized.onnx",
        decoder_quant_path="onnx/HY-MT/Optimum_Cache_Optimized/Quantized/model_int8_final.onnx",
        modelType = onnx_execution.ModelType.HYMT, logFile = True, logFileFolder = "onnx/HY-MT/Optimum_Cache_Optimized/Quantized/Quality/RTNInt8/", logFileName = "translate_hy_Int8"
    )

    onnx_execution.compare_models_quality_multi_language(
        decoder_path="onnx/HY-MT/Optimum_Cache_Optimized/model_optimized.onnx",
        decoder_quant_path="onnx/HY-MT/Optimum_Cache_Optimized/Quantized/model.onnx",
        modelType = onnx_execution.ModelType.HYMT, logFile = True, logFileFolder = "onnx/HY-MT/Optimum_Cache_Optimized/Quantized/Quality/RTN/", logFileName = "translate_hy_Int4"
    )

    onnx_execution.compare_models_quality_multi_language(
        decoder_path="tencent/HY-MT1.5-1.8B",
        decoder_quant_path="tencent/HY-MT1.5-1.8B-FP8",
        modelType = onnx_execution.ModelType.HYMT_hf, logFile = True, logFileFolder = "onnx/HY-MT/Optimum_Cache_Optimized/Quantized/Quality/HF_Fp8/", logFileName = "translate_hy_Fp8"
    )'''

