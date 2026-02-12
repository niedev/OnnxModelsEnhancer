import logging
from pathlib import Path
from typing import Optional

from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model import OnnxModel
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
#from T5OnnxModelMod import T5OnnxModel
from onnxruntime.transformers.optimizer import optimize_model, MODEL_TYPES
from onnxsim import model_info
from optimum.onnxruntime.configuration import OptimizationConfig
from transformers.utils.sentencepiece_model_pb2 import ModelProto

import onnx
import optimum.exporters.onnx
import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTOptimizer
from transformers import T5ForConditionalGeneration, T5Tokenizer, WhisperForConditionalGeneration, \
    M2M100ForConditionalGeneration
from olive.passes.onnx import transformer_optimization

from Olive.olive.constants import Framework, ModelFileFormat
from Olive.olive.hardware.accelerator import DEFAULT_CPU_ACCELERATOR
from Olive.olive.model.handler.base import OliveModelHandler
from Olive.olive.model.handler.onnx import ONNXModelHandler
from T5OnnxModelMod import T5OnnxModel

logger = logging.getLogger(__name__)



def convert_madlad_cache():
    en_text_extended_128 = ("Pre-trained Transformer models have achieved state-of-the-art performance on natural language processing tasks and have been adopted as feature extractors for solving downstream tasks such as question answering, natural language inference, and sentiment analysis. The current state-of-the-art Transformer-based pre-trained models consist of dozens of layers and millions of parameters. While deeper and wider models yield better performance, they also need large GPU/TPU memory. For example, BERT-large (Devlin et al., 2019")
    model_name = 'jbochi/madlad400-3b-mt'

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # con torch script = True la conversione funziona
    model = T5ForConditionalGeneration.from_pretrained(model_name, use_cache=True)
    model.eval()
    inputs = tokenizer(en_text_extended_128, return_tensors="pt")

    # conversione del decoder di seamless in onnx (con dynamo export)
    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)  #per scegliere se usare le dynamic_shapes
    encoder_output_demo = torch.rand((1, len(inputs.input_ids[0]), 1024), dtype=torch.float32)
    kwargs = {"input_ids": inputs.input_ids, 'attention_mask':inputs.attention_mask, "encoder_attention_mask": inputs.attention_mask, "encoder_hidden_states": encoder_output_demo}
    onnx_model = torch.onnx.dynamo_export(model.decoder, export_options=export_options, **kwargs)
    onnx_model.save("onnx/Madlad/DynamoCache/Seamless_decoder.onnx")


def convert_madlad_cache_attention_optimum():
    # metodo per esportare Madlad in formato Onnx con optimum e kv cache
    model_name = 'jbochi/madlad400-3b-mt'
    save_directory = "onnx/Madlad/OptimumAttention"
    out_directory = "onnx/Madlad/OptimumAttention/Optimized"
    encoder_name = "encoder_model"
    decoder_name = "decoder_with_past_model"

    model = T5ForConditionalGeneration.from_pretrained(model_name, use_cache=True, attn_implementation="eager")
    model.eval()
    optimum.exporters.onnx.onnx_export_from_model(model, Path(save_directory), opset=17, optimize=None)

    #optimize_optimum(save_directory, encoder_name, out_directory)
    optimize_optimum(save_directory, decoder_name, out_directory)



def convert_whisper_cache_optimum():
    # metodo per esportare Madlad in formato Onnx con optimum e kv cache
    model_name = "openai/whisper-small"
    save_directory = "onnx/WhisperOptimum/Optimized"

    model = WhisperForConditionalGeneration.from_pretrained(model_name, use_cache=True, attn_implementation="eager")  #output_attentions=True #togliere attn_implementation="eager" per usare la classica attention_mask
    model.eval()
    optimum.exporters.onnx.onnx_export_from_model(model, Path(save_directory), opset=17, optimize="O3")


def convert_nllb_cache_attention_optimum():
    # metodo per esportare Madlad in formato Onnx con optimum e kv cache
    model_name = 'facebook/nllb-200-distilled-600M'
    save_directory = "onnx/NLLBOptimum"

    model = M2M100ForConditionalGeneration.from_pretrained(model_name, use_cache=True)   #attn_implementation="eager"
    model.eval()
    optimum.exporters.onnx.onnx_export_from_model(model, Path(save_directory), opset=17, optimize="O1")


def convert_madlad_script_encoder():
    en_text_extended_128 = ("Pre-trained Transformer models have achieved state-of-the-art performance on natural language processing tasks and have been adopted as feature extractors for solving downstream tasks such as question answering, natural language inference, and sentiment analysis. The current state-of-the-art Transformer-based pre-trained models consist of dozens of layers and millions of parameters. While deeper and wider models yield better performance, they also need large GPU/TPU memory. For example, BERT-large (Devlin et al., 2019")
    model_name = 'jbochi/madlad400-3b-mt'

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # con torch script = True la conversione funziona
    model = T5ForConditionalGeneration.from_pretrained(model_name, use_cache=False, torchscript=True)
    model.eval()
    inputs = tokenizer(en_text_extended_128, return_tensors="pt")

    # conversione dell'encoder in onnx (con script export)
    torch.onnx.export(model.encoder,  # model being run
                      (inputs.input_ids, inputs.attention_mask),  # model input (or a tuple for multiple inputs)
                      "onnx/Madlad/Script/Madlad_encoder.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      verbose=True,
                      opset_version=17,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input_ids']+['attention_mask'],  # the model's input names
                      output_names=['last_hidden_state'],  # the model's output names
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX)



def optimize_optimum(model_path_dir, model_path_name, model_path_out):
    '''# Create the optimizer
    optimizer = ORTOptimizer.from_pretrained(Path(model_path_dir), file_names=[model_path_name+".onnx"])
    # Define the optimization strategy by creating the appropriate configuration
    optimization_config = OptimizationConfig(optimization_level=2,
                                             enable_gelu_approximation=True,
                                             use_multi_head_attention=False)
    # Optimize the model
    optimizer.optimize(save_dir=model_path_out, optimization_config=optimization_config)'''

    #model = ONNXModelHandler(model_path_dir+"/"+model_path_name+".onnx")
    model = onnx.load_model(model_path_dir+"/"+model_path_name+".onnx")
    accelerator = DEFAULT_CPU_ACCELERATOR
    config = {"model_type": "t5", "num_heads": 16, "hidden_size": 2048, "optimization_options": {"use_multi_head_attention": True}}
    #optimizer = transformer_optimization.OrtTransformersOptimization(accelerator, config, True)
    #optimizer.run(model, data_root=model_path_dir+"/"+model_path_name+".onnx_data", output_model_path=model_path_out+"/"+model_path_name+"_optimized.onnx")
    optimization_options = FusionOptions("t5")
    optimization_options.use_multi_head_attention = True
    optimization_options.enable_layer_norm = True
    optimization_options.enable_embed_layer_norm = False
    optimization_options.enable_bias_skip_layer_norm = False
    optimization_options.enable_skip_layer_norm = True
    optimization_options.enable_rotary_embeddings = False
    #optimizer = optimize_model(model_path_dir+"/"+model_path_name+".onnx", model_type="t5", num_heads=16, hidden_size=2048, optimization_options=optimization_options, only_onnxruntime=False, opt_level=0, verbose=True)
    optimizer = optimize_by_fusion_mod(model, model_type="t5", num_heads=16, hidden_size=2048, optimization_options=optimization_options)
    # Topologically sort the graph at the end since previous optimizations may have broken it
    optimizer.topological_sort()
    model = onnx.load_model(model_path_dir+"/"+model_path_name+".onnx")
    model_optimized = optimizer.model
    model_info.print_simplifying_info(model, model_optimized)


def optimize_by_fusion_mod(
    model: ModelProto,
    model_type: str = "bert",
    num_heads: int = 0,
    hidden_size: int = 0,
    optimization_options: Optional[FusionOptions] = None,
):
    """Optimize Model by graph fusion logic.

    Note that ONNXRuntime graph optimizations (like constant folding) will not be applied. So it is better to enable
    constant folding during exporting ONNX model, or run optimize_by_onnxruntime on the model first like optimize_model.

    For BERT model, num_heads and hidden_size are optional. For other model types, you need to specify these parameters.

    Args:
        model (ModelProto): model object
        model_type (str, optional): model type - like bert, bert_tf, bert_keras or gpt2. Defaults to 'bert'.
        num_heads (int, optional): number of attention heads. Defaults to 0.
                                   0 allows detect the parameter from graph automatically.
        hidden_size (int, optional): hidden size. Defaults to 0.
                                     0 allows detect the parameter from graph automatically.
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions.
                                                        Defaults to None.

     Returns:
        object of an optimizer class.
    """
    if model_type not in ["bert", "swin", "unet", "vae", "clip"] and (num_heads == 0 or hidden_size == 0):
        logger.warning(f"Please specify parameters of num_heads and hidden_size for model_type {model_type}")

    if model_type not in MODEL_TYPES:
        logger.warning(f"Unsupported model type: {model_type} for graph fusion, directly return model.")
        return OnnxModel(model)

    (optimizer_class, producer, _) = MODEL_TYPES[model_type]

    if model.producer_name and producer != model.producer_name:
        logger.warning(
            f'Model producer not matched: Expected "{producer}", Got "{model.producer_name}".'
            "Please specify correct --model_type parameter."
        )

    if optimization_options is None:
        optimization_options = FusionOptions(model_type)

    optimizer = T5OnnxModel(model, num_heads, hidden_size)

    optimizer.optimize(optimization_options)

    optimizer.topological_sort()

    optimizer.model.producer_name = "onnxruntime.transformers"
    from onnxruntime import __version__ as onnxruntime_version

    optimizer.model.producer_version = onnxruntime_version

    return optimizer