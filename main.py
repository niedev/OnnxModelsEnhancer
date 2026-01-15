import functools
import operator
import pathlib
from typing import List, Mapping, Optional, Any, Dict

import numpy
from onnx.helper import make_node
from onnxruntime.quantization import shape_inference, quantize_dynamic, QuantType
from onnxruntime.transformers.optimizer import optimize_model
"""from optimum.exporters.onnx.base import ConfigBehavior
from optimum.exporters.onnx.model_configs import BartDummyTextInputGenerator
from optimum.exporters.onnx.model_patcher import PatchingSpec
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig, \
    default_quantization_parameters, QuantizationConfig, ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION, \
    ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QOPS, ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QDQ"""
from optimum.utils import DummySeq2SeqDecoderTextInputGenerator, DummyDecoderTextInputGenerator, \
    DummySeq2SeqPastKeyValuesGenerator, NormalizedSeq2SeqConfig, DummyPastKeyValuesGenerator
from tokenizers import Encoding
#from optimum.exporters.onnx.model_configs import M2M100OnnxConfig

import compose_modified
#import madlad_modifier
import onnx
import optimum.utils
import torch
from pathlib import Path
from transformers.convert_graph_to_onnx import convert
from torchinfo import summary
# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, \
    pipeline, AutoProcessor, SeamlessM4Tv2Model, SeamlessM4TModel, SeamlessM4Tv2ForTextToText, \
    SeamlessM4Tv2ForSpeechToText, AutoConfig, T5ForConditionalGeneration, SeamlessM4TTokenizer, SeamlessM4TConfig, \
    NllbTokenizer, M2M100Model, BatchEncoding, M2M100Config, PretrainedConfig, T5Tokenizer, MarianMTModel, \
    MarianTokenizer, WhisperModel, WhisperForConditionalGeneration
from progress.bar import Bar
from pathlib import Path
from onnxsim import simplify, model_info
import onnx
#from onnxconverter_common import float16
import onnxruntime
#from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer
from onnxruntime.tools.convert_onnx_models_to_ort import convert_onnx_models_to_ort, OptimizationStyle

from prettytable import PrettyTable
from onnxruntime.tools.mobile_helpers import usability_checker
import logging

#import onnx_converter
#import onnx_execution
#import static_quantizer_madlad
from onnx import inliner, version_converter
from onnxruntime.tools import convert_onnx_models_to_ort
from onnxruntime.tools.update_onnx_opset import update_onnx_opset
#from optimum.exporters.onnx import TextSeq2SeqOnnxConfig, OnnxConfig, main_export, export_models, \
#    get_decoder_models_for_export, OnnxConfigWithPast
#from optimum.exporters.onnx import export
#from optimum.exporters.onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast
import os
import numpy as np
from transformers import AutoConfig, AutoProcessor



_folder = Path.cwd()
saved_models_path = _folder.joinpath('models/T2T')

Bar.check_tty = False

def testM2M100():
    en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb, the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."

    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to("cuda:0")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    # translate en to it
    tokenizer.src_lang = "en"
    encoded_hi = tokenizer(en_text, return_tensors="pt").to("cuda:0")
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("it"))
    res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(res)


def testNLLB():
    en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb, the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

    # translate en to it
    translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang="ita_Latn", max_length=400, device=0)
    res=translator(en_text)
    print(res)
    print("")
    count_parameters_nllb(model)

def testNLLBBig():
    en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb, the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")


    # translate en to it
    translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang="ita_Latn", max_length=400, device=0)
    res=translator(en_text)
    print(res)

def testMarianMT():
    en_text = "Originally, Google Translate was released as a statistical machine translation service.The input text had to be translated into English first before being translated into the selected language. Since SMT uses predictive algorithms to translate text, it had poor grammatical accuracy. Despite this, Google initially did not hire experts to resolve this limitation due to the ever-evolving nature of language."
    en_text2 = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb, the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-it")
    # translate en to it
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-it", max_length=128)
    res=translator(en_text2)
    count_parameters(model)
    print(res)

def testMarianMTBig():
    en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb, the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."

    # translate en to it
    #translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-it", max_length=400, device=0)
    #res=translator(en_text)

    model_name = "Helsinki-NLP/opus-mt-tc-big-en-it"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer(en_text, return_tensors="pt", padding=True))
    for t in translated:
        print(tokenizer.decode(t, skip_special_tokens=True))

    count_parameters(model)

def testSeamlessBigV2():
    en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb, the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."

    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    text_inputs = processor(text=en_text, src_lang="eng", return_tensors="pt")

    # translate en to it
    output_tokens = model.generate(**text_inputs, tgt_lang="ita", generate_speech=False)
    res = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    print(res)
    print(model)

def testSeamlessMedium():
    en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb, the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."

    processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
    model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
    text_inputs = processor(text=en_text, src_lang="eng", return_tensors="pt")

    # translate en to it
    output_tokens = model.generate(**text_inputs, tgt_lang="ita", generate_speech=False)
    res = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    print(res)
    print(model)

def countParamsOfWhisper():
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    count_parameters_whisper(model)

def countParamsOfMadlad():
    model_name = 'jbochi/madlad400-3b-mt'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    count_parameters_madlad(model)



def count_parameters(model):
    table = PrettyTable(["Model Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def count_parameters_nllb(model):
    table = PrettyTable(["Model Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    table = PrettyTable(["Encoder Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.model.encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    table = PrettyTable(["Decoder Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.model.decoder.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    table = PrettyTable(["LM Head Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.lm_head.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    '''for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)'''

    return total_params

def count_parameters_whisper(model):
    table = PrettyTable(["Model Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    table = PrettyTable(["Encoder Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.model.encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    table = PrettyTable(["Decoder Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.model.decoder.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    return total_params

def count_parameters_madlad(model):
    table = PrettyTable(["Model Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    table = PrettyTable(["Encoder Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    table = PrettyTable(["Decoder Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.decoder.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    table = PrettyTable(["LM Head Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.lm_head.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    return total_params




if __name__ == '__main__':
    name = "world"
    #print(torch.cuda.is_available())

    #madlad_modifier.simplify_and_quantize_madlad()

    #testSeamlessBigV2()
    en_text = "Also, unlike in 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb, the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."
    it_text = "Inoltre, a differenza del 2014, non ci sono quasi tante lacune. Non si può semplicemente acquistare una lampadina a 150 watt o una lampadina a tre vie, il divieto copre qualsiasi lampadina normale che genera meno di 45 lumeni al watt, che esclude praticamente sia la tecnologia a incandescenza che l'alogeno nella loro interezza."
    en_text_extended_512 = ("Pre-trained Transformer models have achieved state-of-the-art performance on natural language processing tasks and have been adopted as feature extractors for solving downstream tasks such as question answering, natural language inference, and sentiment analysis. The current state-of-the-art Transformer-based pre-trained models consist of dozens of layers and millions of parameters. While deeper and wider models yield better performance, they also need large GPU/TPU memory. For example, BERT-large (Devlin et al., 2019) is trained with 335 million parameters, and requires at least 24 GB of GPU memory to load. The larger size of these models limits their applicability in time- and memory-constrained environments.\
                       Several methods have been proposed to reduce the size of pre-trained models. Notable approaches include pruning parts of the network after training (Michel et al., 2019a, Voita et al., 2019b, McCarley, 2019), reduction through weight factorization and sharing (Lan et al., 2019), compression via knowledge-distillation (Sanh et al., 2019) and quantization (Zafrir et al., 2019, Shen et al., 2019). Our work falls under the class of pruning methods.\
                       The central argument governing pruning methods is that deep neural models are over-parameterized and that not all parameters are strictly needed, especially at the inference time. For example, previous research has shown that most of the attention heads can be removed (Michel et al., 2019b, Voita et al., 2019b) or reallocated (Peng et al., 2020) without significantly impacting performance. Gordon et al. (2019) pruned the least important weights in the network. We build our work based on similar observations, but we are interested in (i) whether it is necessary to use all layers of a pre-trained model for downstream tasks, and if not, (ii) which layers are necessary to keep in order to maintain good task-specific performance while achieving efficiency in transfer learning.\
                       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    en_text_extended_256 = ("Pre-trained Transformer models have achieved state-of-the-art performance on natural language processing tasks and have been adopted as feature extractors for solving downstream tasks such as question answering, natural language inference, and sentiment analysis. The current state-of-the-art Transformer-based pre-trained models consist of dozens of layers and millions of parameters. While deeper and wider models yield better performance, they also need large GPU/TPU memory. For example, BERT-large (Devlin et al., 2019) is trained with 335 million parameters, and requires at least 24 GB of GPU memory to load. The larger size of these models limits their applicability in time- and memory-constrained environments.\
                           Several methods have been proposed to reduce the size of pre-trained models. Notable approaches include pruning parts of the network after training (Michel et al., 2019a, Voita et al., 2019b, McCarley, 2019), reduction through weight factorization and sharing (Lan et al., 2019), compression via knowledge-distillation (Sanh et al., 2019)))aaaaaa")
    en_text_extended_128 = ("Pre-trained Transformer models have achieved state-of-the-art performance on natural language processing tasks and have been adopted as feature extractors for solving downstream tasks such as question answering, natural language inference, and sentiment analysis. The current state-of-the-art Transformer-based pre-trained models consist of dozens of layers and millions of parameters. While deeper and wider models yield better performance, they also need large GPU/TPU memory. For example, BERT-large (Devlin et al., 2019")
    en_text2 = 'King prawns cooked in chili salt and pepper was very much better.'

    #testNLLB()
    #onnx_execution("Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb.", "eng_Latn", "ita_Latn")
    #onnx_execution.onnx_execution_madlad(en_text, 'it', 'onnx/Madlad/Optimized/Optimized2D/Madlad_decoder_2D.onnx')
    #madlad_modifier.conversion_to_2D()
    #madlad_modifier.showInfo('onnx/Madlad/Optimized/Optimized2D/Madlad_decoder_2D.onnx')
    #path = 'onnx/Madlad/Optimized/Optimized2D/Madlad_decoder_2D.onnx'
    #madlad_modifier.replace_unsqueeze(path, path)
    #madlad_modifier.showInfo(path)
    #onnx_execution.onnx_execution_madlad(en_text, 'it', path)
    #onnx_execution.onnx_execution_madlad_cache_test(en_text, "it", "onnx/Madlad/Optimum_Cache/Madlad_decoder_quantized_optimized.onnx")


    path_in = "onnx/Madlad/Optimum_Cache_Optimized/decoder_with_past_model.onnx"
    path_out = "onnx/Madlad/Optimum_Cache_Optimized/Simplified/Madlad_decoder.onnx"
    path_out_dir = "onnx/Madlad/Optimum_Cache_Optimized/Simplified"
    optimized_path_out = "onnx/Madlad/Optimum_Cache_Optimized/OptimizedOptimum/Madlad_decoder.onnx"
    optimized_path_out_dir = "onnx/Madlad/Optimum_Cache_Optimized/OptimizedOptimum"
    #madlad_modifier.simplify_model(path_in, path_out, external_data=True)
    #madlad_modifier.optimize_optimum(path_out_dir, "Madlad_decoder", optimized_path_out_dir)
    #madlad_modifier.quantize_dynamic_madlad(optimized_path_out_dir, "Madlad_decoder_optimized.onnx", optimized_path_out_dir+"/Quantized")

    #onnx_converter.convert_madlad_script_encoder()
    #madlad_modifier.quantize_cache_initializer()

    '''model_name2 = 'jbochi/madlad400-3b-mt'
    model_name = "openai/whisper-small"
    config = AutoConfig.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model2 = T5ForConditionalGeneration.from_pretrained(model_name2)

    print("Whisper encoder:")
    count_parameters(model.model.encoder)
    print('')
    print("Madlad encoder:")
    count_parameters(model2.encoder)'''


    '''models_dir = "onnx/Madlad/Optimum_Cache_Optimized"
    models_dir_out_out = "onnx/Madlad/Optimum_Cache_Optimized/Optimized/QuantizedQuality"
    encoder_name = "encoder_model"
    encoder_dir_out = "onnx/Madlad/Optimum_Cache_Optimized/Optimized"
    encoder_name_out = "Madlad_encoder"
    decoder_name = "decoder_with_past_model"
    decoder_dir_out = "onnx/Madlad/Optimum_Cache_Optimized/Optimized"
    decoder_name_out = "Madlad_decoder"
    #madlad_modifier.quantize_dynamic_madlad_encoder()
    #madlad_modifier.optimize_model(path, path_out)

    madlad_modifier.optimize_onnxruntime_plus(models_dir, encoder_name, encoder_dir_out+"/"+encoder_name_out+".onnx")
    madlad_modifier.optimize_onnxruntime_plus(models_dir, decoder_name, decoder_dir_out+"/"+decoder_name_out+".onnx")
    madlad_modifier.quantize_dynamic_madlad_encoder(encoder_dir_out, encoder_name_out, models_dir_out_out)
    madlad_modifier.quantize_dynamic_madlad(decoder_dir_out, decoder_name_out, models_dir_out_out)

    onnx_execution.onnx_execution_madlad_cache_test(en_text, "it", encoder_path=models_dir_out_out+"/"+encoder_name_out+"_quantized.onnx",
                                                    decoder_path=models_dir_out_out+"/"+decoder_name_out+"_quantized.onnx")'''

    #madlad_modifier.check_whisper_quantization()



    #madlad_modifier.quantize_nllb_4bit()
    #onnx_execution.onnx_execution_nllb_cache_reduced_ram(en_text, "eng_Latn", "ita_Latn", encoder_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit/nllb_encoder_4bit.onnx", decoder_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit/nllb_decoder_4bit.onnx")
    #onnx_execution.compare_models_quality_multi_language()
    
    print("ciao")



    #providers = ['CPUExecutionProvider']
    #initializer_session_quantized = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/Quantized/NLLB_cache_initializer_quantized.onnx"), providers=providers)
    #encoder_session_quantized = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/encoder_model.onnx"), providers=providers)
    #decoder_session_quantized = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/decoder_model.onnx"), providers=providers)
    #embed_and_lm_head_session = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/nllb_embed_and_lm_head_if2.onnx"), providers=providers)
    #result_quantized = onnx_execution.onnx_execution_nllb_cache_reduced_ram("你好啊世界如何看待目前的经济局势失业很严重大家都找不到工作未来会变的好吗.", "zho_Hans", "eng_Latn", encoder_session=encoder_session_quantized, decoder_session=decoder_session_quantized, initializer_session=initializer_session_quantized, embed_and_lm_head_session=embed_and_lm_head_session, log=True)


    #model_path_out = "onnx/WhisperOptimum/decoder_model.onnx"
    #onnx.shape_inference.infer_shapes_path(model_path_out, model_path_out)


    #static_quantizer_madlad.static_quantization_optimum_QLinearOnly_encoder()
    #static_quantizer_madlad.static_quantization_optimum_QLinear()
    #conversione in fp16
    #model_fp16 = float16.convert_float_to_float16_model_path("onnx/Madlad/Optimum_Cache_Optimized/OptimizedOptimum/Madlad_decoder_optimized.onnx", keep_io_types=True)
    #onnx.save(model_fp16, "onnx/Madlad/Optimum_Cache_Optimized/fp16/Madlad_decoder_fp16.onnx", save_as_external_data=True)
    #model_fp16 = float16.convert_float_to_float16_model_path("onnx/Madlad/Optimum_Cache_Optimized/encoder_model.onnx", keep_io_types=True)
    #onnx.save(model_fp16, "onnx/Madlad/Optimum_Cache_Optimized/fp16/Madlad_encoder_fp16.onnx", save_as_external_data=True)

    #update_onnx_opset(Path("onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic/with_past_quality/Madlad_decoder_optimized_quantized.onnx"), 19,
    #                  Path("onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic/with_past_quality/Madlad_decoder_optimized_quantized.onnx"))

    #madlad_modifier.quantize_dynamic_madlad()

    #madlad_modifier.quantize_dynamic_madlad_encoder_int8_int16()

    #esecuzione con encoder e decoder quantizzati
    #onnx_execution.onnx_execution_madlad_cache_test(en_text, "it", "onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic/encoder_model_quantized_int16.onnx", "onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic/with_past_quality/Madlad_decoder_optimized_quantized.onnx")
    #onnx_execution.onnx_execution_madlad_cache(en_text, "it", "onnx/Madlad/Optimum_Cache/DynamicQuantization/Madlad_decoder_quantized.onnx")
    #onnx_execution.onnx_execution_madlad_cache(en_text, "it", "onnx/Madlad/Optimum_Cache_Optimized/decoder_model_merged.onnx")
    #esecuzione con encoder quantizzato e decoder normale
    #onnx_execution.onnx_execution_madlad_cache_test(en_text, "it", decoder_path="onnx/Madlad/Optimum_Cache_Optimized/decoder_with_past_model.onnx")




    #madlad_modifier.adapt_whisper_to_walkie_talkie_mode()
    #madlad_modifier.separate_whisper_pre_ops()


    #onnx_execution.compare_models_quality()

    #modelOnnx = onnx.load_model(Path("onnx/Madlad/Script/StaticQuantization/Test/QLinearOnly/Madlad_decoder_2D_quantized.onnx"))
    #onnx.save_model(modelOnnx, Path("onnx/Madlad/Script/StaticQuantization/Test/QLinearOnly/Madlad_decoder_2D_quantized2.onnx"))
    #onnx.save_model(modelOnnx, Path("onnx/Madlad/Script/StaticQuantization/Test/Madlad_decoder_quantized_static_test.onnx"))


    #onnx_converter.convert_madlad_cache_optimum()
    #madlad_modifier.quantize_dynamic_madlad()
    #model = onnx.load_model("onnx/Madlad/Optimum_Cache/decoder_model_optimized.onnx")
    #model_optimized = onnx.load_model("onnx/Madlad/Optimum_Cache_Optimized/decoder_model_merged.onnx")
    #model_info.print_simplifying_info(model, model_optimized)

    #madlad_modifier.quantize_dynamic_madlad()

    #madlad_modifier.create_madlad_kv_generator("onnx/Madlad/Optimum_Cache_Optimized/decoder_model.onnx",
    #                                           "onnx/Madlad/Optimum_Cache_Optimized/Cache_initializer.onnx")


    #model_name = 'jbochi/madlad400-3b-mt'
    #processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    #tokenizer = SeamlessM4TTokenizer.from_pretrained("facebook/seamless-m4t-v2-large", src_lang="eng", tgt_lang="ita")
    #tokenizer = SeamlessM4TTokenizer.from_pretrained("facebook/hf-seamless-m4t-medium", src_lang="eng", tgt_lang="ita")
    #tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="ita_Latn")
    #tokenizer = T5Tokenizer.from_pretrained(model_name)
    # con torch script = True la conversione funziona
    #model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large",  use_cache=False, torchscript=True)
    #model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium", use_cache=False, torchscript=True)
    #model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-distilled-600M", use_cache=False, torchscript=True)
    #model = T5ForConditionalGeneration.from_pretrained(model_name, use_cache=True)
    #text_inputs = processor(text=en_text, src_lang="eng", return_tensors="pt")
    #inputs = tokenizer(en_text_extended_128, return_tensors="pt")
    #model.eval()
    #input_ids = tokenizer(en_text, return_tensors="pt").input_ids.to(model.device)
    #outputs = model.generate(input_ids=input_ids)





    #onnx_converter.convert_nllb_cache_attention_optimum()
    #model_path_in = "onnx/NLLBOptimum/decoder_model.onnx"
    #onnx.shape_inference.infer_shapes_path(model_path_in, model_path_in)
    #madlad_modifier.create_nllb_cache_initializer(model_path_in, "onnx/NLLBOptimum/Optimized/NLLB_cache_initializer.onnx")
    #madlad_modifier.quantize_dynamic_nllb()

    #onnx_execution.onnx_execution_nllb_cache_test(en_text, "eng_Latn","ita_Latn",
    #                                              encoder_path="onnx/NLLBOptimum/Optimized/Quantized/NLLB_encoder_quantized.onnx",
    #                                              decoder_path="onnx/NLLBOptimum/Optimized/Quantized/NLLB_decoder_quantized.onnx")


    #madlad_modifier.remove_shared_weight("onnx/NLLBOptimum/decoder_with_past_model.onnx", "onnx/NLLBOptimum/Optimized/decoder_model.onnx")
    #madlad_modifier.remove_shared_weight("onnx/NLLBOptimum/encoder_model.onnx", "onnx/NLLBOptimum/Optimized/encoder_model.onnx")
    #model = onnx.load("onnx/NLLBOptimum/encoder_model.onnx")
    #model2=onnx.load("onnx/NLLBOptimum/Optimized/encoder_model.onnx")
    #model_info.print_simplifying_info(model, model2)

    #madlad_modifier.create_nllb_embed_and_lm_head3("onnx/NLLBOptimum/decoder_with_past_model.onnx", "onnx/NLLBOptimum/Optimized/ReducedRAM/nllb_embed_and_lm_head_if3.onnx")
    #madlad_modifier.adapt_nllb_to_embed_and_lm_head("onnx/NLLBOptimum/encoder_model.onnx", "onnx/NLLBOptimum/decoder_with_past_model.onnx", "onnx/NLLBOptimum/Optimized/ReducedRAM/encoder_model.onnx", "onnx/NLLBOptimum/Optimized/ReducedRAM/decoder_model.onnx")
    #onnx_execution.onnx_execution_nllb_cache_reduced_ram(en_text, "eng_Latn", "ita_Latn")

    '''model = onnx.load_model("onnx/NLLBOptimum/decoder_model_merged.onnx")
    for node in model.graph.node:
        del model
        if(node.op_type == "If"):
            attribute = node.attribute[1]
            nodes = attribute.g.node
            for node2 in nodes:
                if(node2.name == "Constant_1375"):
                    print(node2)'''

    #onnx.shape_inference.infer_shapes_path("onnx/NLLBOptimum/encoder_model.onnx", "onnx/NLLBOptimum/encoder_model_shape_infer.onnx")

    '''sess_options = onnxruntime.SessionOptions()
    # Set graph optimization level
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = "onnx/NLLBOptimum/Optimized/ReducedRAM/nllb_embed_and_lm_head_if_optimized.onnx"
    session = onnxruntime.InferenceSession("onnx/NLLBOptimum/Optimized/ReducedRAM/nllb_embed_and_lm_head_if.onnx", sess_options)'''

    #madlad_modifier.quantize_whisper_final()

    #madlad_modifier.create_madlad_embed("onnx/Madlad/Optimum_Cache_Optimized/decoder_with_past_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx")
    #madlad_modifier.adapt_madlad_to_embed("onnx/Madlad/Optimum_Cache_Optimized/encoder_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/decoder_with_past_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx")
    #onnx_execution.onnx_execution_nllb_cache_reduced_ram(en_text, "eng_Latn", "ita_Latn")
    #onnx_execution.onnx_execution_madlad_cache_reduced_ram(en_text, "it")

    #madlad_modifier.quantize_madlad_final()
    #madlad_modifier.get_Attention_MatMul_nodes_madlad("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/PreProcessed/encoder_model.onnx")

    #madlad_modifier.create_whisper_cache_initializer("onnx/WhisperOptimum/decoder_model.onnx", "onnx/WhisperOptimum/ReducedRam/Whisper_cache_initializer_batch.onnx", 2)
    #madlad_modifier.quantize_whisper_final()
    #madlad_modifier.quantize_nllb_final()

    #madlad_modifier.fix_nllb_embed_and_lm_head()
    #madlad_modifier.quantize_nllb_final()




    # aggiornamento di op version
    #update_onnx_opset(Path("onnx/Madlad/Script/StaticQuantization/Test/QDQonly/2D/Madlad_decoder_quantized_static_test.onnx"), 20,
    #                  Path("onnx/Madlad/Script/StaticQuantization/Test/QDQonly/2D/Updated/Madlad_decoder_quantized_static_test.onnx"))
    #model = onnx.load_model("onnx/Madlad/Script/Madlad_decoder_complete.onnx")
    #model.opset_import[0].version = 19
    #onnx.save_model(model, "onnx/Madlad/Script/Madlad_decoder_complete_updated.onnx", save_as_external_data=True)

    # conversione del decoder a fp16
    #model_fp16 = float16.convert_float_to_float16_model_path("onnx/Madlad/Script_updated/Madlad_decoder_complete.onnx", keep_io_types=True)
    #onnx.save(model_fp16, "onnx/Madlad/Script_updated/fp16/Madlad_decoder_fp16.onnx", save_as_external_data=True)

    #test della compatibilita con NNAPI di un modello onnx
    #logger = logging.getLogger("usability_checker")
    #logger.setLevel(logging.DEBUG)
    #usability_checker.analyze_model(model_path=Path("onnx/Seamless/SeamlessV1Medium/Script/quantized/PreProcessed/Seamless_decoder_preProcessed.onnx"),skip_optimize=False, logger=logger)
    #usability_checker.analyze_model(model_path=Path("onnx/Seamless/SeamlessV1Medium/Script/Ort/Seamless_decoder_optimized.onnx"),skip_optimize=False, logger=logger)
    #usability_checker.analyze_model(model_path=Path("onnx/NLLBScript/600M/quantized/Optimum/Static/NLLB_decoder_quantized2.onnx"),skip_optimize=False, logger=logger)
    #usability_checker.analyze_model(model_path=Path("onnx/Madlad/Script_updated/fp16/Madlad_decoder_fp16.onnx"), skip_optimize=False, logger=logger)





    # metodo per eliminare le divisioni in subgraph del modello convertito con dynamo (funziona)
    #convert_onnx_models_to_ort.convert_onnx_models_to_ort(Path("onnx/Madlad/Optimum_Cache/DynamicQuantization/Optimized/Madlad_decoder_quantized2.onnx"),
    #                                                       Path("onnx/Madlad/Optimum_Cache"),
    #                                                       optimization_styles=[OptimizationStyle.Fixed],
    #                                                          target_platform="arm",
    #                                                          save_optimized_onnx_model=True)
    #model = onnx.load_model("onnx/Madlad/Optimum_Cache/decoder_model_merged.onnx")
    #model_optimized = onnx.load_model("onnx/Madlad/Optimum_Cache/Madlad_decoder_quantized_optimized.onnx")
    #model_info.print_simplifying_info(model, model_optimized)

    #quantize_dynamic(Path("onnx/Madlad/Optimum_Cache/decoder_model_optimized_light.onnx"),
    #                 Path("onnx/Madlad/Optimum_Cache/DynamicQuantization/Optimized/Madlad_decoder_quantized2.onnx"),
    #                 extra_options={"EnableSubgraph": True})




    #model = onnx.load_model("onnx/Madlad/Optimum_Cache/Optimized"+"/model.onnx")
    #model_optimized = onnx.load_model("onnx/Madlad/Optimum_Cache/OptimizedOptimum"+"/model_optimized.onnx")
    #model_info.print_simplifying_info(model, model_optimized)

    #madlad_modifier.weight_compression()

    # conversione dell'encoder di seamless in onnx (con dynamo export)
    #kwargs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask}
    #onnx_model = torch.onnx.dynamo_export(model.text_encoder, **kwargs, op_level_debug=True)
    #onnx_model.save("onnx/Seamless/SeamlessV1Medium/Seamless_encoder.onnx")

    # conversione del decoder di seamless in onnx (con dynamo export)
    #export_options = torch.onnx.ExportOptions(dynamic_shapes=False)  #forza a non usare le dynamic_shapes
    #encoder_output_demo = torch.rand((1, 512, 1024), dtype=torch.float32)
    #kwargs = {"input_ids": inputs.input_ids, "encoder_attention_mask": inputs.attention_mask, "encoder_hidden_states": encoder_output_demo}
    #onnx_model = torch.onnx.dynamo_export(model.text_decoder, export_options=export_options, **kwargs)
    #onnx_model.save("onnx/Seamless/SeamlessV1Medium/Seamless_decoder.onnx")

    #conversione dell'encoder di seamless in onnx (con script export)
    #torch.onnx.export(model.text_encoder,  # model being run
    #                  (inputs.input_ids,inputs.attention_mask),  # model input (or a tuple for multiple inputs)
    #                  "onnx/Seamless/SeamlessV1Medium/Script/quantized/4bit/Seamless_encoder_quantized.onnx",  # where to save the model (can be a file or file-like object)
    #                  export_params=True,  # store the trained parameter weights inside the model file
    #                  verbose=True,
    #                  opset_version=11,  # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names=['input_ids']+['attention_mask'],  # the model's input names
    #                  output_names=['last_hidden_state'],  # the model's output names
    #                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    # conversione del decoder di seamless in onnx (con script export)
    #encoder_output_demo = torch.rand((1, 512, 1024), dtype=torch.float32)
    #torch.onnx.export(model.text_decoder,  # model being run
    #                  (inputs.input_ids, inputs.attention_mask, encoder_output_demo, inputs.attention_mask), # model input (or a tuple for multiple inputs)
    #                  "onnx/Seamless/SeamlessV1Medium/Script/Prova/Seamless_decoder.onnx",
                      # where to save the model (can be a file or file-like object)
    #                  export_params=True,  # store the trained parameter weights inside the model file
    #                  verbose=True,
    #                  opset_version=11,  # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names=['input_ids'] + ['attention_mask'] + ['encoder_hidden_states'] + ['encoder_attention_mask'],
    #                  # the model's input names
    #                  output_names=['last_hidden_state'],  # the model's output names
    #                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    # conversione del linear layer finale (lm_head) di seamless in onnx (con script export)
    #decoder_output_demo = torch.rand((1, 512, 1024), dtype=torch.float32)
    #torch.onnx.export(model.lm_head,  # model being run
    #                  (decoder_output_demo),  # model input (or a tuple for multiple inputs)
    #                  "onnx/Seamless/SeamlessV1Medium/Script/Seamless_lm_head.onnx",  # where to save the model (can be a file or file-like object)
    #                  export_params=True,  # store the trained parameter weights inside the model file
    #                  verbose=True,
    #                  opset_version=11,  # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names=['last_hidden_state'],  # the model's input names
    #                  output_names=['logits'],  # the model's output names
    #                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX)


    #preottimizzazioni per la quantizzazione (utile anche senza quantizzazione)
    #shape_inference.quant_pre_process(input_model_path="onnx/Seamless/SeamlessV1Medium/Script/Seamless_encoder.onnx",
    #                                 output_model_path="onnx/Seamless/SeamlessV1Medium/Script/quantized/PreProcessed/Seamless_encoder_preProcessed.onnx",skip_onnx_shape=False,skip_symbolic_shape=False)
    #shape_inference.quant_pre_process(input_model_path="onnx/Seamless/SeamlessV1Medium/Script/Prova/Seamless_decoder.onnx",
    #                                  output_model_path="onnx/Seamless/SeamlessV1Medium/Script/Prova/quantized/PreProcessed/Seamless_decoder_preProcessed.onnx",skip_onnx_shape=False,skip_symbolic_shape=False)

    # unione del decoder (ottimizzato) con lm_head
    #decoder = onnx.load_model(Path("onnx/Seamless/SeamlessV1Medium/Script/Prova/quantized/PreProcessed/Seamless_decoder_preProcessed.onnx"))
    #lm_head = onnx.load_model(Path("onnx/Seamless/SeamlessV1Medium/Script/Seamless_lm_head.onnx"))
    # abbiamo modificato la classe della libreria di onnx usata per convergere 2 modelli eliminando il check finale (lanciava un'eccezione se il modello in memoria superava i 2gb)
    #decoder_complete = compose_modified.merge_models(decoder, lm_head, [("last_hidden_state","last_hidden_state")])
    #rimuoviamo l' ultimo import dalla lista degli imports del modello (opset_import), questo perch`e senn`o abbiamo due import ai.onnx e questo causa il fallimento
    #della quantizzazione.
    #del decoder_complete.opset_import[-1]
    #onnx.save_model(decoder_complete, Path("onnx/Seamless/SeamlessV1Medium/Script/Prova/quantized/PreProcessed/Seamless_decoder_complete_preProcessed.onnx"),
    #                save_as_external_data=True, all_tensors_to_one_file=True, location="Seamless_decoder_complete_data")



    #quantizzazione dinamica a 8 bit
    #quantize_dynamic(Path("onnx/Seamless/SeamlessV1Medium/Script/Prova/quantized/PreProcessed/Seamless_decoder_complete_preProcessed.onnx"),
    #                 Path("onnx/Seamless/SeamlessV1Medium/Script/Prova/quantized/8bit/Seamless_decoder_complete_quantized.onnx"))
    #quantize_dynamic(Path("onnx/Seamless/SeamlessV1Medium/Script/quantized/PreProcessed/Seamless_encoder_preProcessed.onnx"),
    #                 Path("onnx/Seamless/SeamlessV1Medium/Script/quantized/8bit/Seamless_encoder_quantized.onnx"))






    # convert_onnx_models_to_ort.convert_onnx_models_to_ort(Path("onnx/Seamless/SeamlessV1Medium/Script/quantized/8bit/Seamless_decoder_complete_quantized.onnx"),
    #                           Path("onnx/Seamless/SeamlessV1Medium/Script/Ort/Ort_quantized"),
    #                           optimization_styles=[OptimizationStyle.Runtime],
    #                                                      target_platform="arm",
    #                                                      save_optimized_onnx_model=True)





    #conversione di NLLB
    # conversione dell'encoder di NLLB in onnx (con script export)
    #torch.onnx.export(model.model.encoder,  # model being run
    #                  (inputs.input_ids,inputs.attention_mask),  # model input (or a tuple for multiple inputs)
    #                  "onnx/NLLBScript/600M/NLLB_encoder.onnx",  # where to save the model (can be a file or file-like object)
    #                  export_params=True,  # store the trained parameter weights inside the model file
    #                  verbose=True,
    #                  opset_version=11,  # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names=['input_ids']+['attention_mask'],  # the model's input names
    #                  output_names=['last_hidden_state'],  # the model's output names
    #                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    # conversione del decoder di NLLB in onnx (con script export)
    #encoder_output_demo = torch.rand((1, 256, 1024), dtype=torch.float32)
    #torch.onnx.export(model.model.decoder,  # model being run
    #                  (inputs.input_ids, inputs.attention_mask, encoder_output_demo, inputs.attention_mask),
    #                  # model input (or a tuple for multiple inputs)
    #                  "onnx/NLLBScript/600M/NLLB_decoder.onnx",
    #                  # where to save the model (can be a file or file-like object)
    #                  export_params=True,  # store the trained parameter weights inside the model file
    #                  verbose=True,
    #                  opset_version=11,  # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names=['input_ids'] + ['attention_mask'] + ['encoder_hidden_states'] + ['encoder_attention_mask'],
    #                  # the model's input names
    #                  output_names=['last_hidden_state'],  # the model's output names
    #                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    # conversione del linear layer finale (lm_head) di seamless in onnx (con script export)
    #decoder_output_demo = torch.rand((1, 256, 1024), dtype=torch.float32)
    #torch.onnx.export(model.lm_head,  # model being run
    #                  (decoder_output_demo),  # model input (or a tuple for multiple inputs)
    #                  "onnx/NLLBScript/600M/NLLB_lm_head.onnx",  # where to save the model (can be a file or file-like object)
    #                  export_params=True,  # store the trained parameter weights inside the model file
    #                  verbose=True,
    #                  opset_version=11,  # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names=['last_hidden_state'],  # the model's input names
    #                  output_names=['logits'],  # the model's output names
    #                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    # convert_onnx_models_to_ort.convert_onnx_models_to_ort(Path("onnx/Seamless/SeamlessV1Medium/Script/Seamless_decoder.onnx"),
    #                           Path("onnx/Seamless/SeamlessV1Medium/Script/Ort"),
    #                           optimization_styles=[OptimizationStyle.Runtime],
    #                                                      target_platform="arm",
    #                                                      enable_type_reduction=True,
    #                                                      save_optimized_onnx_model=True)

    # preottimizzazioni per la quantizzazione (utile anche senza quantizzazione)
    #shape_inference.quant_pre_process(input_model_path="onnx/NLLBScript/600M/NLLB_encoder.onnx",
    #                                 output_model_path="onnx/NLLBScript/600M/quantized/PreProcessed/NLLB_encoder_preProcessed.onnx",skip_onnx_shape=False,skip_symbolic_shape=False)
    #shape_inference.quant_pre_process(input_model_path="onnx/NLLBScript/600M/NLLB_decoder.onnx",
    #                                  output_model_path="onnx/NLLBScript/600M/quantized/PreProcessed/NLLB_decoder_preProcessed.onnx",skip_onnx_shape=False,skip_symbolic_shape=False)

    # unione del decoder (ottimizzato) con lm_head
    #decoder = onnx.load_model(Path("onnx/NLLBScript/600M/quantized/PreProcessed/NLLB_decoder_preProcessed.onnx"))
    #lm_head = onnx.load_model(Path("onnx/NLLBScript/600M/NLLB_lm_head.onnx"))
    # abbiamo modificato la classe della libreria di onnx usata per convergere 2 modelli eliminando il check finale (lanciava un'eccezione se il modello in memoria superava i 2gb)
    #decoder_complete = compose_modified.merge_models(decoder, lm_head, [("last_hidden_state","last_hidden_state")])
    # rimuoviamo l' ultimo import dalla lista degli imports del modello (opset_import), questo perch`e senn`o abbiamo due import ai.onnx e questo causa il fallimento
    # della quantizzazione.
    #del decoder_complete.opset_import[-1]
    #onnx.save_model(decoder_complete, Path("onnx/NLLBScript/600M/quantized/PreProcessed/NLLB_decoder_complete_preProcessed.onnx"),
    #                save_as_external_data=True, all_tensors_to_one_file=True, location="NLLB_decoder_complete_data")

    # quantizzazione dinamica a 8 bit
    #quantize_dynamic(Path("onnx/NLLBScript/600M/quantized/PreProcessed/NLLB_decoder_complete_preProcessed.onnx"),
    #                 Path("onnx/NLLBScript/600M/quantized//NLLB_decoder_complete_quantized.onnx"))
    #quantize_dynamic(Path("onnx/NLLBScript/600M/quantized/PreProcessed/NLLB_encoder_preProcessed.onnx"),
    #                 Path("onnx/NLLBScript/600M/quantized/NLLB_encoder_complete_quantized.onnx"))

    # conversione del decoder a fp16
    #model_fp16 = float16.convert_float_to_float16_model_path("onnx/NLLBScript/600M/quantized/PreProcessed/NLLB_decoder_complete_preProcessed.onnx", keep_io_types=True)
    #onnx.save(model_fp16, "onnx/NLLBScript/600M/quantized/NLLB_decoder_complete_fp16.onnx")




    #modelOnnx = onnx.load_model("onnx/Madlad/Script/StaticQuantization/Madlad_decoder_quantized2.onnx")
    #modelOnnx.graph.node.appen(make_node())
    #print(modelOnnx)











    #convert(framework="pt", model="facebook/seamless-m4t-v2-large", output=Path("onnx/seamless.onnx"), opset=11,tokenizer=None,use_external_format=False,pipeline_name="feature-extraction",**text_inputs, tgt_lang="ita", generate_speech=False)

    #generate_onnx_representation(pretrained_version="facebook/seamless-m4t-v2-large", model=model)

    #model2 = T5ForConditionalGeneration.from_pretrained("t5-base")
    #T5Converter.generate_onnx_representation(pretrained_version="t5-base",model=model2)

    #pytorch_encoder_params = sum(p.numel() for p in model.text_encoder.parameters())
    #pytorch_decoder_params = sum(p.numel() for p in model.text_decoder.parameters())

    #print(pytorch_encoder_params)
    #print(pytorch_decoder_params)

    #modelOnnx = onnx_models.get_onnx_model("facebook/seamless-m4t-v2-large",quantized=False)
    #input_ids = inputs['input_ids']
    #attention_mask = inputs['attention_mask']
    #output_tokens = modelOnnx.generate(input_ids=input_ids, attention_mask=attention_mask)
    #print(output_tokens)


    #metodo per esportare NLLB in formato Onnx con transformers senza kv cache (per farlo con kv cache cambiare use_cache in True)
    #model_checkpoint = "facebook/nllb-200-distilled-600M"
    #save_directory = "onnx/NLLB_Mono"
    # Load a model from transformers and export it to ONNX
    #ort_model = ORTModelForSeq2SeqLM.from_pretrained(model_checkpoint, export=True, use_cache=False)
    #tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # Save the onnx model and tokenizer
    #ort_model.save_pretrained(save_directory)
    #tokenizer.save_pretrained(save_directory)

    #metodo per usare il modello esportato col metodo sopra
    #tokenizer2 = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    # `from_transformers` will export the model to ONNX on-the-fly 🤯
    #model2 = ORTModelForSeq2SeqLM.from_pretrained("onnx/NLLB", name_or_path="facebook/nllb-200-distilled-600M")
    #onnx_translation = pipeline("translation", model=model2, tokenizer=tokenizer2, src_lang="eng_Latn", tgt_lang="ita_Latn")
    # returns [{'translation_text': 'Mein Name ist Lewis.'}]
    #pred = onnx_translation(en_text)


    #quantizzazione dinamica con optimum (encoder e decoder)
    #model_dir = "onnx/Madlad/Script"
    #quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="Madlad_decoder_complete.onnx")
    #qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)   #QuantFormat.QDQ
    #quantizer.quantize(save_dir="onnx/Madlad/Script/DynamicQuantization", quantization_config=qconfig)
    #quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="Madlad_encoder.onnx")
    #qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)  # QuantFormat.QDQ
    #quantizer.quantize(save_dir="onnx/Madlad/Script/DynamicQuantization", quantization_config=qconfig)

    #quantizzazione dinamica di Madlad
    #quantize_dynamic(Path("onnx/Madlad/Optimum_Cache/decoder_model_merged.onnx"),
    #                 Path("onnx/Madlad/Optimum_Cache/DynamicQuantization/Madlad_decoder_quantized.onnx"), extra_options={"EnableSubgraph": True})
    #quantize_dynamic(Path("onnx/Madlad/Optimum_Cache/encoder_model.onnx"),
    #                 Path("onnx/Madlad/Optimum_Cache/DynamicQuantization/Madlad_encoder_quantized.onnx"))


    #onnx_model = onnx.load_model("onnx/Madlad/Script_INT32/Madlad_decoder_complete.onnx")
    #print('Model :\n\n{}'.format(onnx.helper.printable_graph(onnx_model.graph)))

    #conversione di madlad in onnx con pytorch script (verificare manualmente se `e compatibile con NNAPI)
    # conversione dell'encoder in onnx (con script export)
    '''torch.onnx.export(model.encoder,  # model being run
                      (inputs.input_ids,inputs.attention_mask),  # model input (or a tuple for multiple inputs)
                      "onnx/Madlad/Script_INT32/Madlad_encoder.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      verbose=True,
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input_ids']+['attention_mask'],  # the model's input names
                      output_names=['last_hidden_state'],  # the model's output names
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX)'''

    # conversione del decoder in onnx (con script export)
    '''encoder_output_demo = torch.rand((1, 128, 1024), dtype=torch.float32)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    torch.onnx.export(model.decoder,  # model being run
                      (input_ids, attention_mask, encoder_output_demo, attention_mask),
                      # model input (or a tuple for multiple inputs)
                      "onnx/Madlad/Script_updated/Madlad_decoder.onnx",
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      verbose=True,
                      opset_version=17,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input_ids'] + ['attention_mask'] + ['encoder_hidden_states'] + ['encoder_attention_mask'],
                      # the model's input names
                      output_names=['last_hidden_state'],  # the model's output names
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    # conversione del linear layer finale (lm_head) di seamless in onnx (con script export)
    decoder_output_demo = torch.rand((1, 128, 1024), dtype=torch.float32)
    torch.onnx.export(model.lm_head,  # model being run
                      (decoder_output_demo),  # model input (or a tuple for multiple inputs)
                      "onnx/Madlad/Script_updated/Madlad_lm_head.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      verbose=True,
                      opset_version=17,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['last_hidden_state'],  # the model's input names
                      output_names=['logits'],  # the model's output names
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX)


    # unione del decoder con lm_head
    decoder = onnx.load_model(Path("onnx/Madlad/Script_updated/Madlad_decoder.onnx"))
    lm_head = onnx.load_model(Path("onnx/Madlad/Script_updated/Madlad_lm_head.onnx"))
    # abbiamo modificato la classe della libreria di onnx usata per convergere 2 modelli eliminando il check finale (lanciava un'eccezione se il modello in memoria superava i 2gb)
    decoder_complete = compose_modified.merge_models(decoder, lm_head, [("last_hidden_state","last_hidden_state")])
    # rimuoviamo l' ultimo import dalla lista degli imports del modello (opset_import), questo perch`e senn`o abbiamo due import ai.onnx e questo causa il fallimento
    # della quantizzazione.
    del decoder_complete.opset_import[-1]
    onnx.save_model(decoder_complete, Path("onnx/Madlad/Script_updated/Madlad_decoder_complete.onnx"),
                    save_as_external_data=True, all_tensors_to_one_file=True, location="Madlad_decoder_complete_data")'''


    '''# quantizzazione dinamica con optimum (prova con mode QDQ)
    model_dir = "onnx/Madlad/Optimum_Cache"
    quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="decoder_model_merged.onnx")
    qconfig = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.IntegerOps,  #QLinearOps non funziona
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=True,
        weights_dtype=QuantType.QUInt8,
        weights_symmetric=True,
        per_channel=False,
        reduce_range=False,
        #nodes_to_quantize=[],
        #nodes_to_exclude=[],
        operators_to_quantize=ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION,
    )
    quantizer.quantize(save_dir="onnx/Madlad/Optimum_Cache/DynamicQuantization/Optimum", quantization_config=qconfig)'''





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
