from operator import contains
from os import PathLike
from pathlib import Path
import huggingface_hub
import onnx
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




def create_madlad_final_model(execute_model: bool):
    en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb — the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."
    convert_madlad_cache_optimum()
    create_madlad_cache_initializer("onnx/Madlad/Optimum_Cache_Optimized/decoder_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx")
    create_madlad_embed("onnx/Madlad/Optimum_Cache_Optimized/decoder_with_past_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx")
    adapt_madlad_to_embed("onnx/Madlad/Optimum_Cache_Optimized/encoder_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/decoder_with_past_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx")
    quantize_madlad_4bit()
    #quantize_madlad_8bit()
    #onnx_execution.onnx_execution_nllb_cache_reduced_ram(en_text, "eng_Latn", "ita_Latn")
    
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
    # metodo per esportare Madlad in formato Onnx con optimum e kv cache
    model_name = 'jbochi/madlad400-3b-mt'
    save_directory = "onnx/Madlad/Optimum_Cache_Optimized"

    if((not Path(save_directory + "/encoder_model.onnx").is_file()) or (not Path(save_directory + "/decoder_with_past_model.onnx").is_file())):
      model = T5ForConditionalGeneration.from_pretrained(model_name, use_cache=True, attn_implementation="eager")
      model.eval()
      optimum.exporters.onnx.onnx_export_from_model(model, Path(save_directory), opset=18, optimize="O1")


def create_madlad_cache_initializer(model_path, model_path_out):
    if(not Path(model_path_out).is_file()):
        #preleviamo il decoder con kv_cache senza past, per estrarre le componenti da inserire nell initializer.
        model = onnx.load_model(model_path)

        graph = model.graph
        initializers = graph.initializer
        nodes = graph.node


        # creiamo un dizionario che associa al nome di ogni initializer del grafico le sue informazioni
        initializers_dict = {}
        for initializer in initializers:
            initializers_dict[initializer.name] = initializer

        # creiamo una lista contenente la matrice (initializer) di ogni MatMul che useremo nel kv_generator
        # e una lista contenente tutte le MatMul che useremo nel kv_generator
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

            #for per trovare il numero del block di questo nodo (serve per scrivere il nome dell'output del transpose)
            index = 0
            for i in range(32):
                if("block."+str(i) in node.name):
                    index = i
            #capiamo se il nodo corrente `e un key o value
            if("/k/" in node.name):
                isKey = True
            else:
                isKey = False

            #creaiamo e inseriamo il reshape in node_list
            shape_initializer = onnx.helper.make_tensor(   #si crea una tensore che rappresenta la forma del reshape
                                    name="shape_"+str(count_output_shape),
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[4],
                                    vals=[1, -1, 16, 128]   #si usa una batch size fissa di 1 cosi da poter usare -1 al posto della dimensione dinamica (dato che non si puo inserire una dimensione dinamica in un reshape)
            )
            inputs_list.extend([shape_initializer])  #si aggiunge agli initializer del grafico
            reshape = onnx.helper.make_node(   #si crea il reshape
                                    name="Reshape_custom_"+str(count_output_shape),
                                    op_type="Reshape",
                                    inputs=[node.output[0], "shape_"+str(count_output_shape)],
                                    outputs=["Reshape_custom_output_"+str(count_output_shape)],
                                )
            nodes_list.insert(node_index+1, reshape)   #si aggiunge ai nodi del grafico
            node_index = node_index+1

            #creaiamo e inseriamo il transpose in node_list
            #perm = onnx.helper.make_attribute(   #si crea un attributo che rappresenta la perm del transpose
            #        key="perm",
            #        value=[0, 2, 1, 3],
            #        attr_type=AttributeType.INTS
            #)
            perm = {"perm": [0, 2, 1, 3]}
            if(isKey):
                output_name = "present."+str(index)+".encoder.key"
            else:
                output_name = "present."+str(index)+".encoder.value"
            transpose = onnx.helper.make_node(   #si crea il transpose
                                    name="Transpose_custom_"+str(count_output_shape),
                                    op_type="Transpose",
                                    inputs=["Reshape_custom_output_"+str(count_output_shape)],
                                    outputs=[output_name],
                                    **perm
                                )
            nodes_list.insert(node_index+1, transpose)   #si aggiunge ai nodi del grafico
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
    if(not Path(output_path).is_file()):
        model = onnx.load(decoder_path)
        graph = model.graph
        initializers = graph.initializer
        nodes = graph.node

        # creiamo un dizionario che associa al nome di ogni initializer del grafico le sue informazioni
        initializers_dict = {}
        for initializer in initializers:
            initializers_dict[initializer.name] = initializer

        #del model
        #del graph
        #del initializers

        embed_nodes = []
        embed_initializers = []
        # aggiungiamo i nodi e gli initializers richiesti per nllb_embed
        for node in nodes:
            if(node.name == "/decoder/Reshape"):
                embed_nodes.append(node)
                embed_initializers.append(initializers_dict[node.input[1]])
            if(node.name == "/decoder/shared/Gather"):
                node.output.pop()
                node.output.append("embed_matrix")
                embed_nodes.append(node)

        embed_initializers.append(initializers_dict["shared.weight"])


        # Creazione del grafico nllb_embed
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
    if((not Path(encoder_path_out).is_file()) or (not Path(decoder_path_out).is_file())):
        encoder_model = onnx.load(encoder_path)
        encoder_graph = encoder_model.graph
        encoder_initializers = encoder_graph.initializer
        encoder_nodes = encoder_graph.node

        decoder_model = onnx.load(decoder_path)
        decoder_graph = decoder_model.graph
        decoder_initializers = decoder_graph.initializer
        decoder_nodes = decoder_graph.node

        # creiamo un dizionario che associa al nome di ogni initializer del grafico dell' encoder le sue informazioni
        encoder_initializers_dict = {}
        for initializer in encoder_initializers:
            encoder_initializers_dict[initializer.name] = initializer
        # creiamo un dizionario che associa al nome di ogni initializer del grafico del decoder le sue informazioni
        decoder_initializers_dict = {}
        for initializer in decoder_initializers:
            decoder_initializers_dict[initializer.name] = initializer

        # creiamo un dizionario che associa a ogni nome di ogni input del grafico dell encoder una lista contenente tutti i nodi (op) che possiedono quell'input
        encoder_inputs_dict = {}
        for node in encoder_nodes:
            for input in node.input:
                if(input not in encoder_inputs_dict):
                    encoder_inputs_dict[input] = [node]
                elif(node not in encoder_inputs_dict[input]):
                    encoder_inputs_dict[input].append(node)
        # creiamo un dizionario che associa a ogni nome di ogni input del grafico dell encoder una lista contenente tutti i nodi (op) che possiedono quell'input
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

        #rimuoviamo dall'encoder i nodi esportati in madlad_embed e rinominiamo l'input degli altri nodi di conseguenza
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

        #rimuoviamo dal decoder i nodi esportati in madlad_embed e rinominiamo l'input degli altri nodi di conseguenza
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

        #modifichiamo gli input dell'encoder e rimuoviamo i pesi dell'embed
        embed_matrix = onnx.helper.make_tensor_value_info("embed_matrix",
                                                            onnx.TensorProto.FLOAT,
                                                            ["batch_size", "sequence_length", 1024])
        encoder_graph.input.insert(0, embed_matrix)
        encoder_graph.initializer.remove(encoder_initializers_dict["embed_tokens.weight"])

        #modifichiamo gli input e gli output del decoder e rimuoviamo i pesi dell'embed
        embed_matrix = onnx.helper.make_tensor_value_info("embed_matrix",
                                                            onnx.TensorProto.FLOAT,
                                                            ["batch_size", "sequence_length", 1024])
        decoder_graph.input.insert(0, embed_matrix)
        decoder_graph.initializer.remove(decoder_initializers_dict["shared.weight"])

        #salviamo l' encoder e il decoder aggiornati
        onnx.save_model(encoder_model, encoder_path_out, save_as_external_data=True)
        onnx.shape_inference.infer_shapes_path(encoder_path_out, encoder_path_out)

        onnx.save_model(decoder_model, decoder_path_out, save_as_external_data=True)
        onnx.shape_inference.infer_shapes_path(decoder_path_out, decoder_path_out)

        onnx.checker.check_model(encoder_path_out)
        onnx.checker.check_model(decoder_path_out)



def quantize_madlad_4bit():
    accuracy_level = 4
    quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=16, # 2's exponential and >= 16 (128)
        is_symmetric=False, # if true, quantize to Int4. otherwise, quantize to uint4.
        accuracy_level=4, # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize={"MatMul"})
    
    #quant_config.algorithm = "HQQ"

    quant_config_hqq = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig()  #op_types_to_quantize={"MatMul", "Gather"} (Gather non è supportato con HQQ)


    #quantization of encoder
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx"
    model_int4_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/madlad_encoder_4bit.onnx"
    if(not Path(model_int4_path).is_file()):
        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            accuracy_level=accuracy_level,
            nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
            algo_config=quant_config,)
        quant.process()
        quant.model.save_model_to_file(model_int4_path, False)
        set_model_matmul_accuracy_level(model_int4_path, model_int4_path, accuracy_level)


    #quantization of decoder
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx"
    model_int4_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/madlad_decoder_4bit.onnx"

    if(not Path(model_int4_path).is_file()):
        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            accuracy_level=accuracy_level,
            nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
            algo_config=quant_config,)
        quant.process()
        quant.model.save_model_to_file(model_int4_path, False)
        set_model_matmul_accuracy_level(model_int4_path, model_int4_path, accuracy_level)


    #quantization of cache init
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx"
    model_int4_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/madlad_cache_initializer_4bit.onnx"

    if(not Path(model_int4_path).is_file()):
        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            accuracy_level=accuracy_level,
            nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
            algo_config=quant_config,)
        quant.process()
        quant.model.save_model_to_file(model_int4_path, False)
        set_model_matmul_accuracy_level(model_int4_path, model_int4_path, accuracy_level)


    #quantization of embed (8 bit perché a 4 bit il Gather non viene quantizzato)
    '''model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx"
    model_int4_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/madlad_embed_4bit.onnx"
    
    if(not Path(model_int4_path).is_file()):
        model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            accuracy_level=accuracy_level,
            nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
            algo_config=quant_config,)
        quant.process()
        quant.model.save_model_to_file(model_int4_path, False)
        set_model_matmul_accuracy_level(model_int4_path, model_int4_path, accuracy_level) 
    '''
    
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx"
    model_int8_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/madlad_embed_8bit.onnx"

    if(not Path(model_int8_path).is_file()):
        quantize_dynamic(Path(model_fp32_path), Path(model_int8_path),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})

    
    convert_madlad_HQQ_model_to_full_int4()



def quantize_madlad_8bit(quality = False, weightOnly = False, outputFolder = "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8/"):
    accuracy_level = 4

    quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=128, # 2's exponential and >= 16 (128)
        is_symmetric=False, # if true, quantize to Int4. otherwise, quantize to uint4.
        accuracy_level=accuracy_level, # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize={"MatMul"},
        bits=8)

    quant_config_hqq = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig()  #op_types_to_quantize={"MatMul", "Gather"} (Gather non è supportato con HQQ)


    #quantization of encoder
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx"
    model_int8_path= outputFolder + "madlad_encoder_8bit.onnx"
    if(not Path(model_int8_path).is_file()):
        nodes_to_exclude = []
        if(quality):
            nodes_to_exclude = get_DenseReluDense_nodes("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx")
        if(not weightOnly):
            quantize_dynamic(Path(model_fp32_path), Path(model_int8_path),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=nodes_to_exclude,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})
        else:
            model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model,
                accuracy_level=accuracy_level,
                nodes_to_exclude=nodes_to_exclude, # specify a list of nodes to exclude from quantization
                algo_config=quant_config,)
            quant.process()
            quant.model.save_model_to_file(model_int8_path, False)


    #quantization of decoder
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx"
    model_int8_path= outputFolder + "madlad_decoder_8bit.onnx"

    if(not Path(model_int8_path).is_file()):
        if(not weightOnly):
            quantize_dynamic(Path(model_fp32_path), Path(model_int8_path),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})
        else:
            model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model,
                accuracy_level=accuracy_level,
                nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
                algo_config=quant_config,)
            quant.process()
            quant.model.save_model_to_file(model_int8_path, False)


    #quantization of cache init
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx"
    model_int8_path= outputFolder + "madlad_cache_initializer_8bit.onnx"

    if(not Path(model_int8_path).is_file()):
        if(not weightOnly):
            quantize_dynamic(Path(model_fp32_path), Path(model_int8_path),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})
        else:
            model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model,
                accuracy_level=accuracy_level,
                nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
                algo_config=quant_config,)
            quant.process()
            quant.model.save_model_to_file(model_int8_path, False)


    #quantization of embed (8 bit perché a 4 bit il Gather non viene quantizzato)
    model_fp32_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx"
    model_int8_path= outputFolder + "madlad_embed_8bit.onnx"

    if(not Path(model_int8_path).is_file()):
        if(not weightOnly):
            quantize_dynamic(Path(model_fp32_path), Path(model_int8_path),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})
        else:
            model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model,
                accuracy_level=accuracy_level,
                nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
                algo_config=quant_config,)
            quant.process()
            quant.model.save_model_to_file(model_int8_path, False)


def set_model_matmul_accuracy_level(input_path: PathLike, output_path: PathLike, accuracy_level: int):
    if(output_path == input_path or (not Path(output_path).is_file())):
        model = onnx.load(input_path)
        graph = model.graph
        nodes = graph.node

        # aggiungiamo l'attributo accuracy_level a tutti i nodi MatMulNBits del modello
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



def convert_HQQ_model_to_full_int4(input_path: PathLike, output_path: PathLike):
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
        if(("DenseReluDense/wo" in node.name) and ("MatMul" in node.name)):
            list.append(node.name)
            print(node.name)
    print("")
    print("")
    return list
    


if __name__ == '__main__':
    #set_madlad_matmul_accuracy_level("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/", 4)
    #set_madlad_matmul_accuracy_level("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/RTN/", 4)
    #set_madlad_matmul_accuracy_level("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQ/", 4)

    #onnx_execution.compare_models_quality_multi_language(logFileFolder = "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Quality/HQQPerfAcc0/")

    #quantize_madlad_8bit(quality=False, weightOnly=False, outputFolder="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8/")
    #quantize_madlad_8bit(quality=True, weightOnly=False, outputFolder="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8Quality/")
    #quantize_madlad_8bit(quality=False, weightOnly=True, outputFolder="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8WO/")

    #quantize_madlad_4bit()

    onnx_execution.compare_models_quality_multi_language(
        initializer_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx",
        encoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx",
        decoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx",
        embed_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx",
        initializer_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8WO/madlad_cache_initializer_8bit.onnx",
        encoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/GPTQ/madlad_encoder_4bit.onnx",
        decoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/GPTQ/madlad_decoder_4bit.onnx",
        embed_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8WO/madlad_embed_8bit.onnx",
        modelType = onnx_execution.ModelType.MADLAD, logFile = True, logFileFolder = "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Quality/GPTQ/", logFileName = "madlad_quality_Int4_GPTQ"
    )
    
    '''onnx_execution.compare_models_quality_multi_language(
        initializer_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx",
        encoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx",
        decoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx",
        embed_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx",
        initializer_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8Quality/madlad_cache_initializer_8bit.onnx",
        encoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8Quality/madlad_encoder_8bit.onnx",
        decoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8Quality/madlad_decoder_8bit.onnx",
        embed_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8Quality/madlad_embed_8bit.onnx",
        modelType = onnx_execution.ModelType.MADLAD, logFile = True, logFileFolder = "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Quality/Int8Quality/", logFileName = "madlad_quality_Int8_q"
    )

    onnx_execution.compare_models_quality_multi_language(
        initializer_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx",
        encoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx",
        decoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx",
        embed_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx",
        initializer_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8WO/madlad_cache_initializer_8bit.onnx",
        encoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8WO/madlad_encoder_8bit.onnx",
        decoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8WO/madlad_decoder_8bit.onnx",
        embed_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Int8WO/madlad_embed_8bit.onnx",
        modelType = onnx_execution.ModelType.MADLAD, logFile = True, logFileFolder = "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Quality/Int8WO/", logFileName = "madlad_quality_Int8_wo"
    )'''

    #create_madlad_final_model(True) 