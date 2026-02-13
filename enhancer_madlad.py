import copy
from operator import contains
from os import PathLike
import os
from pathlib import Path
import pprint
import huggingface_hub
import onnx
import onnxruntime
from QuantizationDebug import qdq_loss_debug_updated
import onnx_execution
from onnxruntime.quantization import (
    matmul_nbits_quantizer,
    quant_utils,
    quantize
)
from transformers import T5ForConditionalGeneration
from onnxruntime.quantization import quantize_dynamic, QuantFormat, QuantizationMode, QuantType, quant_pre_process
import optimum.exporters.onnx
from onnx import TensorProto, numpy_helper
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from optimum.exporters.onnx import main_export


en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb — the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."


def create_madlad_final_model(execute_model: bool = False):
    convert_madlad_cache_optimum()
    create_madlad_cache_initializer("onnx/Madlad/Optimum_Cache_Optimized/decoder_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx")
    create_madlad_embed("onnx/Madlad/Optimum_Cache_Optimized/decoder_with_past_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx")
    adapt_madlad_to_embed("onnx/Madlad/Optimum_Cache_Optimized/encoder_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/decoder_with_past_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx")
    quantize_madlad_4bit()
    #quantize_madlad_8bit()
    
    if(execute_model):
        onnx_execution.onnx_execution_madlad_cache_reduced_ram(en_text, "it", encoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_encoder_4bit.onnx",
                                        decoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_decoder_4bit.onnx",
                                        initializer_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_cache_initializer_4bit.onnx",
                                        embed_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/madlad_embed_8bit.onnx", profiling=False)
        
        """onnx_execution.onnx_execution_madlad_cache_reduced_ram(en_text, "it", encoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/QuantizedInt8/madlad_encoder_8bit.onnx",
                                        decoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/QuantizedInt8/madlad_decoder_8bit.onnx",
                                        initializer_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/QuantizedInt8/madlad_cache_initializer_8bit.onnx",
                                        embed_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/QuantizedInt8/madlad_embed_8bit.onnx")"""
                                        
        #onnx_execution.onnx_execution_madlad_cache_reduced_ram(en_text, "it")



def convert_madlad_cache_optimum():
    # Method to export Madlad to Onnx format with optimum and kv cache
    model_name = 'jbochi/madlad400-3b-mt'
    save_directory = "onnx/Madlad/Optimum_Cache_Optimized"

    os.makedirs(save_directory, exist_ok=True)

    if((not Path(save_directory + "/encoder_model.onnx").is_file()) or (not Path(save_directory + "/decoder_with_past_model.onnx").is_file())):
      model = T5ForConditionalGeneration.from_pretrained(model_name, use_cache=True, attn_implementation="eager")
      model.eval()
      optimum.exporters.onnx.onnx_export_from_model(model, Path(save_directory), opset=18, optimize="O1", no_post_process=True)
      


def create_madlad_cache_initializer(model_path, model_path_out):
    directory = os.path.dirname(model_path_out)
    os.makedirs(directory, exist_ok=True)

    if(not Path(model_path_out).is_file()):
        # We extract the decoder with kv_cache without past, to extract the components to be inserted into the initializer.
        model = onnx.load_model(model_path)

        graph = model.graph
        initializers = graph.initializer
        nodes = graph.node


        # we create a dictionary that associates the name of each initializer in the graph with its information
        initializers_dict = {}
        for initializer in initializers:
            initializers_dict[initializer.name] = initializer

        # Let's create a list containing the matrix (initializer) of each MatMul we'll use in the kv_generator
        # and a list containing all the MatMuls we'll use in the kv_generator
        inputs_list = []
        nodes_list = []
        for node in nodes:
            if(("EncDecAttention/k/MatMul" in node.name) or ("EncDecAttention/v/MatMul" in node.name) and node.op_type == "MatMul"):
                nodes_list.append(node)
                for input in node.input:
                    if(input in initializers_dict):
                        inputs_list.append(initializers_dict[input])

        #del model
        #del graph
        #del initializers


        count_output_shape = 0
        node_index = 0
        while(node_index < len(nodes_list)):   #for node in nodes
            node = nodes_list[node_index]

            # cycle to find the block number of this node (this is used to write the transpose output name)
            index = 0
            for i in range(32):
                if("block."+str(i) in node.name):
                    index = i
            # we understand if the current node is a key or value
            if("/k/" in node.name):
                isKey = True
            else:
                isKey = False

            # create and insert the reshape into node_list
            shape_initializer = onnx.helper.make_tensor(   # a tensor is created that represents the shape of the reshape
                                    name="shape_"+str(count_output_shape),
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[4],
                                    vals=[1, -1, 16, 128]   # a fixed batch size of 1 is used so that -1 can be used instead of the dynamic size (since you can't put a dynamic size in a reshape)
            )
            inputs_list.extend([shape_initializer])  # It is added to the graph initializers
            reshape = onnx.helper.make_node(   # the reshape is created
                                    name="Reshape_custom_"+str(count_output_shape),
                                    op_type="Reshape",
                                    inputs=[node.output[0], "shape_"+str(count_output_shape)],
                                    outputs=["Reshape_custom_output_"+str(count_output_shape)],
                                )
            nodes_list.insert(node_index+1, reshape)   # is added to the graph nodes
            node_index = node_index+1

            # create and insert the transpose into node_list
            #perm = onnx.helper.make_attribute(   # an attribute is created that represents the perm of the transpose
            #        key="perm",
            #        value=[0, 2, 1, 3],
            #        attr_type=AttributeType.INTS
            #)
            perm = {"perm": [0, 2, 1, 3]}
            if(isKey):
                output_name = "present."+str(index)+".encoder.key"
            else:
                output_name = "present."+str(index)+".encoder.value"
            transpose = onnx.helper.make_node(   # the transpose is created
                                    name="Transpose_custom_"+str(count_output_shape),
                                    op_type="Transpose",
                                    inputs=["Reshape_custom_output_"+str(count_output_shape)],
                                    outputs=[output_name],
                                    **perm
                                )
            nodes_list.insert(node_index+1, transpose)   # it is added to the graph nodes
            node_index = node_index+1


            count_output_shape = count_output_shape+1
            node_index = node_index+1


        # Create inputs
        encoder_hidden_states = onnx.helper.make_tensor_value_info("encoder_hidden_states",
                                                            onnx.TensorProto.FLOAT,
                                                            [1, "encoder_sequence_length", 1024])
        # Create outputs
        outputs = []
        for i in range(32):
            outputs.append(onnx.helper.make_tensor_value_info("present."+str(i)+".encoder.key",
                                                            onnx.TensorProto.FLOAT,
                                                            [1, 16, "encoder_sequence_length", 128]))

            outputs.append(onnx.helper.make_tensor_value_info("present."+str(i)+".encoder.value",
                                                            onnx.TensorProto.FLOAT,
                                                            [1, 16, "encoder_sequence_length", 128]))

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            nodes=nodes_list,
            name="CacheInitializer",
            inputs=[encoder_hidden_states],  # Graph input
            outputs=outputs,  # Graph output
            initializer=inputs_list,
        )

        # Create the model (ModelProto)
        model_def = onnx.helper.make_model(graph_def, producer_name="nie")
        model_def.opset_import[0].version = 23

        onnx.save_model(model_def, model_path_out, save_as_external_data=False)
        onnx.shape_inference.infer_shapes_path(model_path_out, model_path_out)
        onnx.checker.check_model(model_path_out)



def create_madlad_embed(decoder_path: PathLike, output_path: PathLike):
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)

    if(not Path(output_path).is_file()):
        model = onnx.load(decoder_path)
        graph = model.graph
        initializers = graph.initializer
        nodes = graph.node

        # we create a dictionary that associates the name of each initializer in the graph with its information
        initializers_dict = {}
        for initializer in initializers:
            initializers_dict[initializer.name] = initializer

        #del model
        #del graph
        #del initializers

        embed_nodes = []
        embed_initializers = []
        # Let's add the required nodes and initializers for madlad_embed
        for node in nodes:
            if(node.name == "/decoder/Reshape"):
                embed_nodes.append(node)
                embed_initializers.append(initializers_dict[node.input[1]])
            if(node.name == "/decoder/shared/Gather"):
                node.output.pop()
                node.output.append("embed_matrix")
                embed_nodes.append(node)

        embed_initializers.append(initializers_dict["shared.weight"])


        # Creating the madlad_embed chart
        #create inputs
        input_ids = onnx.helper.make_tensor_value_info("input_ids",
                                                            onnx.TensorProto.INT64,
                                                            ["batch_size", "sequence_length"])
        #create outputs
        embed_matrix = onnx.helper.make_tensor_value_info("embed_matrix",
                                                            onnx.TensorProto.FLOAT,
                                                            ["batch_size", "sequence_length", 1024])

        nllb_embed_graph = onnx.helper.make_graph(
            nodes=embed_nodes,
            name="nllb_embed",
            inputs=[input_ids],  # Graph input
            outputs=[embed_matrix],  # Graph output
            initializer=embed_initializers,
        )

        # Create the model (ModelProto)
        model_def = onnx.helper.make_model(nllb_embed_graph, producer_name="nie")
        model_def.opset_import[0].version = 23

        onnx.save_model(model_def, output_path, save_as_external_data=False)
        onnx.shape_inference.infer_shapes_path(output_path, output_path)
        onnx.checker.check_model(output_path)



def adapt_madlad_to_embed(encoder_path: PathLike, decoder_path: PathLike, encoder_path_out: PathLike, decoder_path_out: PathLike):
    encoderDirectory = os.path.dirname(encoder_path_out)
    decoderDirectory = os.path.dirname(decoder_path_out)
    os.makedirs(encoderDirectory, exist_ok=True)
    os.makedirs(decoderDirectory, exist_ok=True)

    if((not Path(encoder_path_out).is_file()) or (not Path(decoder_path_out).is_file())):
        encoder_model = onnx.load(encoder_path)
        encoder_graph = encoder_model.graph
        encoder_initializers = encoder_graph.initializer
        encoder_nodes = encoder_graph.node

        decoder_model = onnx.load(decoder_path)
        decoder_graph = decoder_model.graph
        decoder_initializers = decoder_graph.initializer
        decoder_nodes = decoder_graph.node

        # we create a dictionary that associates the name of each initializer of the encoder graph with its information
        encoder_initializers_dict = {}
        for initializer in encoder_initializers:
            encoder_initializers_dict[initializer.name] = initializer
        # we create a dictionary that associates the name of each initializer of the decoder graph with its information
        decoder_initializers_dict = {}
        for initializer in decoder_initializers:
            decoder_initializers_dict[initializer.name] = initializer

        # we create a dictionary that associates to each name of each input of the encoder graph a list containing all the nodes (op) that have that input
        encoder_inputs_dict = {}
        for node in encoder_nodes:
            for input in node.input:
                if(input not in encoder_inputs_dict):
                    encoder_inputs_dict[input] = [node]
                elif(node not in encoder_inputs_dict[input]):
                    encoder_inputs_dict[input].append(node)
        # we create a dictionary that associates to each name of each input of the encoder graph a list containing all the nodes (op) that have that input
        decoder_inputs_dict = {}
        for node in decoder_nodes:
            for input in node.input:
                if(input not in decoder_inputs_dict):
                    decoder_inputs_dict[input] = [node]
                elif(node not in decoder_inputs_dict[input]):
                    decoder_inputs_dict[input].append(node)

        #del encoder_model
        #del decoder_model
        #del graph
        #del initializers

        # We remove the nodes exported to madlad_embed from the encoder and rename the input of the other nodes accordingly
        output = "/embed_tokens/Gather_output_0"
        if(output in encoder_inputs_dict):
            for node2 in encoder_inputs_dict[output]:
                index = 0
                for input2 in node2.input:
                    if(input2 == output):
                        break
                    index = index+1
                node2.input[index] = "embed_matrix"

        for node in encoder_nodes:
            if(node.name == "/Reshape"):
                encoder_nodes.remove(node)
        for node in encoder_nodes:
            if(node.name == "/embed_tokens/Gather"):
                encoder_nodes.remove(node)

        # We remove the nodes exported to madlad_embed from the decoder and rename the input of the other nodes accordingly
        output = "/decoder/shared/Gather_output_0"
        if(output in decoder_inputs_dict):
            for node2 in decoder_inputs_dict[output]:
                index = 0
                for input2 in node2.input:
                    if(input2 == output):
                        break
                    index = index+1
                node2.input[index] = "embed_matrix"

        for node in decoder_nodes:
            if(node.name == "/decoder/Reshape"):
                decoder_nodes.remove(node)
        for node in decoder_nodes:
            if(node.name == "/decoder/shared/Gather"):
                decoder_nodes.remove(node)

        # We modify the encoder inputs and remove the embed weights
        embed_matrix = onnx.helper.make_tensor_value_info("embed_matrix",
                                                            onnx.TensorProto.FLOAT,
                                                            ["batch_size", "sequence_length", 1024])
        encoder_graph.input.insert(0, embed_matrix)
        encoder_graph.initializer.remove(encoder_initializers_dict["embed_tokens.weight"])

        # We modify the decoder inputs and outputs and remove the embed weights
        embed_matrix = onnx.helper.make_tensor_value_info("embed_matrix",
                                                            onnx.TensorProto.FLOAT,
                                                            ["batch_size", "sequence_length", 1024])
        decoder_graph.input.insert(0, embed_matrix)
        decoder_graph.initializer.remove(decoder_initializers_dict["shared.weight"])

        # save the updated encoder and decoder
        onnx.save_model(encoder_model, encoder_path_out, save_as_external_data=True)
        onnx.shape_inference.infer_shapes_path(encoder_path_out, encoder_path_out)

        onnx.save_model(decoder_model, decoder_path_out, save_as_external_data=True)
        onnx.shape_inference.infer_shapes_path(decoder_path_out, decoder_path_out)

        onnx.checker.check_model(encoder_path_out)
        onnx.checker.check_model(decoder_path_out)



def quantize_madlad_4bit(qdq = False, quality=False, outputFolder = "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/"):
    accuracy_level = 4
    quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=128, # 2's exponential and >= 16 (128)
        is_symmetric=False, # if true, quantize to Int4. otherwise, quantize to uint4.
        accuracy_level=4, # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35,
        quant_format = quant_utils.QuantFormat.QDQ if qdq else quant_utils.QuantFormat.QOperator,
        op_types_to_quantize={"MatMul"})
    
    quant_config_int8 = copy.deepcopy(quant_config)
    quant_config_int8.bits = 8

    quant_config_hqq = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig()  #op_types_to_quantize={"MatMul", "Gather"} (Gather non è supportato con HQQ)

    os.makedirs(outputFolder, exist_ok=True)

    #quantization of encoder
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx"
    model_int4_path=outputFolder + "madlad_encoder_4bit.onnx"
    if(not Path(model_int4_path).is_file()):
        _quantize_weight_only(model_fp32_path, model_int4_path, quant_config, get_DenseReluDense_nodes(model_fp32_path) if quality else None, accuracy_level, False)
        if(quality):
            _quantize_weight_only(model_int4_path, model_int4_path, quant_config_int8)
        set_model_matmul_accuracy_level(model_int4_path, model_int4_path, accuracy_level)


    #quantization of decoder
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx"
    model_int4_path=outputFolder + "madlad_decoder_4bit.onnx"

    if(not Path(model_int4_path).is_file()):
        _quantize_weight_only(model_fp32_path, model_int4_path, quant_config, get_DenseReluDense_nodes(model_fp32_path) if quality else None, accuracy_level, False)
        if(quality):
            _quantize_weight_only(model_int4_path, model_int4_path, quant_config_int8)
        set_model_matmul_accuracy_level(model_int4_path, model_int4_path, accuracy_level)


    #quantization of cache init
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx"
    model_int4_path=outputFolder + "madlad_cache_initializer_4bit.onnx"

    if(not Path(model_int4_path).is_file()):
        _quantize_weight_only(model_fp32_path, model_int4_path, quant_config, None, accuracy_level)
        set_model_matmul_accuracy_level(model_int4_path, model_int4_path, accuracy_level)


    #quantization of embed (8 bit perché a 4 bit il Gather non viene quantizzato)
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx"
    model_int8_path=outputFolder + "madlad_embed_8bit.onnx"

    if(not Path(model_int8_path).is_file()):
        _quantize_dynamic_int8(model_fp32_path, model_int8_path)
    
    print("\n\nFinal models saved in "+outputFolder)




def _quantize_dynamic_int8(model_fp32_path: str, model_int8_path: str, nodes_to_exclude=None):
    quantize_dynamic(Path(model_fp32_path), Path(model_int8_path),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=nodes_to_exclude,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})
    
def _quantize_weight_only(model_fp32_path: str, model_int_path: str, quant_config, nodes_to_exclude=None, accuracy_level=None, save_external=False):
    model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        accuracy_level=accuracy_level,
        nodes_to_exclude=nodes_to_exclude, # specify a list of nodes to exclude from quantization
        algo_config=quant_config,)
    quant.process()
    quant.model.save_model_to_file(model_int_path, save_external)




def quantize_madlad_8bit(quality = False, weightOnly = False, qdq = False, outputFolder = "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8/"):
    accuracy_level = 4

    quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=128, # 2's exponential and >= 16 (128)
        is_symmetric=False, # if true, quantize to Int4. otherwise, quantize to uint4.
        accuracy_level=accuracy_level, # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35,
        quant_format = quant_utils.QuantFormat.QDQ if qdq else quant_utils.QuantFormat.QOperator,
        op_types_to_quantize={"MatMul"},
        bits=8)

    quant_config_hqq = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig()  #op_types_to_quantize={"MatMul", "Gather"} (Gather non è supportato con HQQ)

    os.makedirs(outputFolder, exist_ok=True)

    #quantization of encoder
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx"
    model_int8_path= outputFolder + "madlad_encoder_8bit.onnx"
    if(not Path(model_int8_path).is_file()):
        nodes_to_exclude = []
        if(quality):
            nodes_to_exclude = get_DenseReluDense_nodes("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx")
        if(not weightOnly):
            _quantize_dynamic_int8(model_fp32_path, model_int8_path, nodes_to_exclude)
        else:
            _quantize_weight_only(model_fp32_path, model_int8_path, quant_config, nodes_to_exclude, accuracy_level)


    #quantization of decoder
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx"
    model_int8_path= outputFolder + "madlad_decoder_8bit.onnx"

    if(not Path(model_int8_path).is_file()):
        if(not weightOnly):
            _quantize_dynamic_int8(model_fp32_path, model_int8_path)
        else:
            _quantize_weight_only(model_fp32_path, model_int8_path, quant_config, None, accuracy_level)


    #quantization of cache init
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx"
    model_int8_path= outputFolder + "madlad_cache_initializer_8bit.onnx"

    if(not Path(model_int8_path).is_file()):
        if(not weightOnly):
            _quantize_dynamic_int8(model_fp32_path, model_int8_path)
        else:
            _quantize_weight_only(model_fp32_path, model_int8_path, quant_config, None, accuracy_level)


    #quantization of embed (8 bit perché a 4 bit il Gather non viene quantizzato)
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx"
    model_int8_path= outputFolder + "madlad_embed_8bit.onnx"

    if(not Path(model_int8_path).is_file()):
        if(not weightOnly):
            _quantize_dynamic_int8(model_fp32_path, model_int8_path)
        else:
            _quantize_weight_only(model_fp32_path, model_int8_path, quant_config, None, accuracy_level)


def set_model_matmul_accuracy_level(input_path: PathLike, output_path: PathLike, accuracy_level: int):
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)

    if(output_path == input_path or (not Path(output_path).is_file())):
        model = onnx.load(input_path)
        graph = model.graph
        nodes = graph.node

        # We add the accuracy_level attribute to all MatMulNBits nodes in the model
        for node in nodes:
            if(("/MatMul_Q4" in node.name) and node.op_type == "MatMulNBits"):
                index = 0
                while (index < len(node.attribute)):  # for node in nodes
                    attr = node.attribute[index]
                    if(attr.name == "accuracy_level"):
                        node.attribute.remove(attr)
                        index = index -1
                    index = index + 1
                accuracy_level_attribute = onnx.helper.make_attribute("accuracy_level", accuracy_level, None, onnx.AttributeProto.AttributeType.INT)
                node.attribute.append(accuracy_level_attribute)
                print("Added accuracy level to the node: "+node.name)

        onnx.save_model(model, output_path, save_as_external_data=False)
        onnx.shape_inference.infer_shapes_path(output_path, output_path)
        onnx.checker.check_model(output_path)



def convert_HQQ_model_to_full_int4(input_path: PathLike, output_path: PathLike): #it is not used because it has the same quality of default quantization (RTN)
    if(output_path == input_path or (not Path(output_path).is_file())):
        model = onnx.load(input_path)
        graph = model.graph
        initializers = graph.initializer
        nodes = graph.node
        
        for initializer in initializers:
            if(("MatMul_" in initializer.name) and ("_zero_points" in initializer.name)):
                #initializer conversion
                # Skip if already packed uint8
                if initializer.data_type == TensorProto.UINT8:
                    continue

                arr = numpy_helper.to_array(initializer)

                # Handle ONLY flattened (1D) float/unpacked zero points
                if arr.ndim != 1 or arr.dtype not in (np.float16, np.float32, np.float64):
                    print(f"Skip {initializer.name}: shape={arr.shape}, dtype={arr.dtype}")
                    continue

                # Round to nearest int, clamp to [0,15] for int4
                zp = np.clip(np.rint(arr), 0, 15).astype(np.uint8)  # shape (L,)

                # Pad to even length (2 values per byte). Pad with default midpoint 8.
                if (zp.size % 2) != 0:
                    zp = np.concatenate([zp, np.array([8], dtype=np.uint8)], axis=0)

                # Pack: byte = (zp0 & 0xF) | ((zp1 & 0xF) << 4)
                low = zp[0::2] & 0x0F
                high = (zp[1::2] & 0x0F) << 4
                packed = (low | high).astype(np.uint8)  # shape (ceil(L/2),)

                # Replace initializer in-place with packed uint8 tensor (same name)
                new_init = numpy_helper.from_array(packed, name=initializer.name)
                initializer.CopyFrom(new_init)

                # Remove from graph inputs if present
                for gi in list(graph.input):
                    if gi.name == initializer.name:
                        graph.input.remove(gi)
                        break
                print("Converted to int4 initializer: "+initializer.name)

        onnx.save_model(model, output_path, save_as_external_data=False)
        onnx.shape_inference.infer_shapes_path(output_path, output_path)
        onnx.checker.check_model(output_path)


def convert_madlad_HQQ_model_to_full_int4():
    convert_HQQ_model_to_full_int4(
        "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/madlad_encoder_4bit.onnx",
        "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_encoder_4bit.onnx"
    )
    convert_HQQ_model_to_full_int4(
        "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/madlad_decoder_4bit.onnx",
        "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_decoder_4bit.onnx"
    )
    convert_HQQ_model_to_full_int4(
        "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/madlad_cache_initializer_4bit.onnx",
        "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_cache_initializer_4bit.onnx"
    )

        
def set_madlad_matmul_accuracy_level(folder: PathLike = "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/", accuracy_level = 4):
    set_model_matmul_accuracy_level(
        folder + "madlad_encoder_4bit.onnx",
        folder + "madlad_encoder_4bit.onnx",
        accuracy_level
    )
    set_model_matmul_accuracy_level(
        folder + "madlad_decoder_4bit.onnx",
        folder + "madlad_decoder_4bit.onnx",
        accuracy_level
    )
    set_model_matmul_accuracy_level(
        folder + "madlad_cache_initializer_4bit.onnx",
        folder + "madlad_cache_initializer_4bit.onnx",
        accuracy_level
    )

def get_DenseReluDense_nodes(path):
    model = onnx.load_model(path)

    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node

    list = []
    for node in nodes:
        if(("DenseReluDense/" in node.name) and ("MatMul" in node.name)):   #DenseReluDense/wo
            list.append(node.name)
            print(node.name)
    print("")
    print("")
    return list
    


if __name__ == '__main__':
    create_madlad_final_model()
    
    '''onnx_execution.compare_models_quality_multi_language(
        initializer_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx",
        encoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx",
        decoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx",
        embed_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx",
        initializer_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/RTNQuality+/madlad_cache_initializer_4bit.onnx",
        encoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/RTNQuality+/madlad_encoder_4bit.onnx",
        decoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/RTNQuality+/madlad_decoder_4bit.onnx",
        embed_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/RTNQuality+/madlad_embed_8bit.onnx",
        modelType = onnx_execution.ModelType.MADLAD, logFile = True, logFileFolder = "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Quality/RTNQuality+/", logFileName = "madlad_quality_Int4"
    )'''