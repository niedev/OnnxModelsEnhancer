import os
from pathlib import Path
import huggingface_hub
import onnx
from optimum.exporters.onnx.model_configs import TextDecoderOnnxConfig, NormalizedTextConfig, DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator
from packaging import version
from transformers import AutoConfig, HunYuanDenseV1ForCausalLM
import optimum.exporters.onnx
from onnxruntime.transformers.fusion_options import FusionOptions, AttentionOpType
from onnxruntime.transformers.optimizer import optimize_model
from onnxsim import simplify, model_info
from onnxruntime_genai.models.builders.gemma import Gemma2Model
from huggingface_hub import constants, snapshot_download
from onnxruntime_genai.models.builder import set_io_dtype, set_onnx_dtype

PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", constants.HF_HUB_CACHE)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)


class HYMTOnnxConfig(TextDecoderOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, GemmaDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator
    MIN_TRANSFORMERS_VERSION = version.parse("4.38.0")



def convert_hy_cache_optimum():
    # metodo per esportare Madlad in formato Onnx con optimum e kv cache
    model_name = 'tencent/HY-MT1.5-1.8B'
    save_directory = "onnx/HY-MT/Optimum_Cache_Optimized"

    os.makedirs(save_directory, exist_ok=True)

    if((not Path(save_directory + "/model.onnx").is_file())):
        model = HunYuanDenseV1ForCausalLM.from_pretrained(model_name, use_cache=True, attn_implementation="eager")   #device_map="cpu" for bf16
        model.eval()

        onnx_config = HYMTOnnxConfig(model.config, task="text-generation", use_past=True, use_past_in_inputs=True, float_dtype="fp32")  #float_dtype= "bf16" if the weights are bf16 (verify if otherwise it can have problems)

        optimum.exporters.onnx.onnx_export_from_model(model, Path(save_directory), opset=18, optimize=None, custom_onnx_configs={"model": onnx_config}, slim=True)

    if((not Path(save_directory + "/model_optimized.onnx").is_file())):
        '''from optimum.onnxruntime import AutoOptimizationConfig, ORTOptimizer
        optimizer = ORTOptimizer.from_pretrained(save_directory, file_names="model.onnx")
        optimization_config = AutoOptimizationConfig.with_optimization_level(optimization_level="O4")
        optimization_config.disable_shape_inference = True
        optimizer.optimize(save_dir=save_directory, optimization_config=optimization_config, file_suffix="_optimized")'''

        optimize_onnxruntime_plus(model_path_dir=save_directory, model_path_name="model", model_path_out=save_directory+"/model_optimized.onnx")





def optimize_onnxruntime_plus(model_path_dir, model_path_name, model_path_out):
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
    #accelerator = DEFAULT_CPU_ACCELERATOR
    optimization_options = FusionOptions("llama")
    optimization_options.set_attention_op_type(AttentionOpType)
    optimizer = optimize_model(model_path_dir+"/"+model_path_name+".onnx", model_type="t5", num_heads=16, hidden_size=2048, optimization_options=optimization_options, only_onnxruntime=True, opt_level=99, verbose=False)
    # Topologically sort the graph at the end since previous optimizations may have broken it
    optimizer.topological_sort()
    model = onnx.load_model(model_path_dir+"/"+model_path_name+".onnx")
    model_optimized = optimizer.model
    model_info.print_simplifying_info(model, model_optimized)
    onnx.save_model(model_optimized, model_path_out, save_as_external_data=True)
    onnx.shape_inference.infer_shapes_path(model_path_out, model_path_out)

if __name__ == '__main__':
    convert_hy_cache_optimum()