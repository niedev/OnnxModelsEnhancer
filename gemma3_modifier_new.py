

import json
import os
from pathlib import Path
import optimum.exporters.onnx
from transformers import Gemma3ForConditionalGeneration
import huggingface_hub
from huggingface_hub import constants, snapshot_download
from onnxruntime_genai.models.builder import create_model
from onnxruntime.quantization import (
    matmul_nbits_quantizer,
    quant_utils,
    quantize
)
from onnxruntime.quantization import quantize_dynamic, QuantFormat, QuantizationMode, QuantType, quant_pre_process

import onnx_execution


PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", constants.HF_HUB_CACHE)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb — the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."


def create_gemma3_final_model():
    convert_gemma3_cache_optimum(False)
    #convert_gemma3_cache_optimum(True)
    quantize_gemma3_4bit()



def convert_gemma3_cache_optimum(quantize = False):
    # metodo per esportare Gemma3 in formato Onnx con onnxruntime gen-ai e kv cache
    model_name = 'google/translategemma-4b-it'   #google/gemma-3-4b-it-qat-q4_0-unquantized
    save_directory = 'onnx/TranslateGemma/Huggingface'
    if(quantize):
        save_onnx_directory = "onnx/TranslateGemma/Onnx/Quantized"
    else:
        save_onnx_directory = "onnx/TranslateGemma/Onnx"

    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(save_onnx_directory, exist_ok=True)

    token = open("hf_token.txt").read()
    huggingface_hub.login(token=token)

    if len(os.listdir(save_directory)) == 0:  #if the directory is empty
        #save the hugging face files locally
        snapshot_download(
            repo_id=model_name,
            local_dir=save_directory,
            local_dir_use_symlinks=False,
        )
        
        """
        Force Gemma3ForCausalLM so builder uses the Gemma3ForCausalLM branch
        (text-only path) and we can keep token embedding inside the model
        => input_ids instead of inputs_embeds.
        """
        cfg_path = Path(save_directory + "/config.json")
        if not cfg_path.exists():
            raise FileNotFoundError(f"config.json not found in {save_directory}")
        cfg = json.loads(cfg_path.read_text())
        
        # 1) Force CausalLM export path
        cfg["architectures"] = ["Gemma3ForCausalLM"]

        # 2) Promote text_config keys to root (builder expects root attributes)
        text_cfg = cfg.get("text_config", {})
        for k, v in text_cfg.items():
            cfg.setdefault(k, v)

        # 3) Ensure max_position_embeddings exists
        # Some Gemma3 configs miss it; default for Gemma 3 is 131072.
        cfg.setdefault("max_position_embeddings", text_cfg.get("max_position_embeddings", 131072))

        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    
    if(not Path(save_onnx_directory+"/model.onnx").exists()):  #if the directory contains the model
        extra_options = {
            "exclude_embeds": False,
        }

        create_model(model_name=model_name, input_path=save_directory, output_dir=save_onnx_directory, precision=("int4" if quantize else "fp32"), execution_provider="cpu", cache_dir=TRANSFORMERS_CACHE, extra_options=extra_options)


def quantize_gemma3_4bit(outputFolder = "onnx/TranslateGemma/Onnx/Quantized/RTN32/"):
    accuracy_level = 4
    quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=32, # 2's exponential and >= 16 (128)
        is_symmetric=False, # if true, quantize to Int4. otherwise, quantize to uint4.
        accuracy_level=4, # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35,
        quant_format = quant_utils.QuantFormat.QOperator,
        op_types_to_quantize={"MatMul", "Gather"})
    
    #quantization of decoder
    model_fp32_path="onnx/TranslateGemma/Onnx/model.onnx"
    model_int4_path=outputFolder + "translate_gemma_decoder_4bit.onnx"
    model_int4_8_path=outputFolder + "gemma3_decoder_4-8bit.onnx"

    if(not Path(model_int4_path).is_file()):
        _quantize_weight_only(model_fp32_path, model_int4_path, quant_config, None, accuracy_level, True)
    #if(not Path(model_int4_8_path).is_file()):
        #_quantize_dynamic_int8(model_int4_path, model_int4_8_path, save_external=True)
        #set_model_matmul_accuracy_level(model_int4_path, model_int4_path, accuracy_level)


def _quantize_weight_only(model_fp32_path: str, model_int_path: str, quant_config, nodes_to_exclude=None, accuracy_level=None, save_external=False):
    model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        accuracy_level=accuracy_level,
        nodes_to_exclude=nodes_to_exclude, # specify a list of nodes to exclude from quantization
        algo_config=quant_config,)
    quant.process()
    quant.model.save_model_to_file(model_int_path, save_external)


def _quantize_dynamic_int8(model_fp32_path: str, model_int8_path: str, nodes_to_exclude=None, save_external=False):
    quantize_dynamic(Path(model_fp32_path), Path(model_int8_path),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = save_external, nodes_to_exclude=nodes_to_exclude,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})



if __name__ == '__main__':
   #create_gemma3_final_model()
   #onnx_execution.onnx_execution_gemma3_cache(text=en_text, src_lang="English", tgt_lang="Italian", decoder_path="onnx/Gemma3/Onnx/Quantized/gemma3_decoder_4bit.onnx")
   #onnx_execution.onnx_execution_translate_gemma_cache(text=en_text, src_lang="en", tgt_lang="it", decoder_path="onnx/TranslateGemma/Onnx/Quantized/RTN32/gemma3_decoder_4bit.onnx")

   '''onnx_execution.compare_models_quality_multi_language(
        decoder_path="onnx/TranslateGemma/Onnx/model.onnx",
        decoder_quant_path="onnx/TranslateGemma/Onnx/Quantized/RTN32/gemma3_decoder_4bit.onnx",
        modelType = onnx_execution.ModelType.TRANSLATEGEMMA, logFile = True, logFileFolder = "onnx/TranslateGemma/Onnx/Quantized/Quality/RTN32/", logFileName = "translate_gemma_quality_Int4"
    )'''
   
   onnx_execution.compare_models_quality(
        decoder_path="onnx/TranslateGemma/Onnx/model.onnx",
        decoder_quant_path="onnx/TranslateGemma/Onnx/Quantized/RTN32/gemma3_decoder_4bit.onnx",
        modelType = onnx_execution.ModelType.TRANSLATEGEMMA, logFile = True, logFileFolder = "onnx/TranslateGemma/Onnx/Quantized/Quality/RTN32/", logFileName = "translate_gemma_quality_Int4",
        data_dir="en-ja", src_lan="en", tgt_lan="ja"
    )
