from pathlib import Path
import onnx
import onnx_execution
from onnxruntime.quantization import (
    matmul_nbits_quantizer,
    quant_utils,
    quantize
)

'''
Questa pagina contiene il processo finale che useremo per creare da 0 il modello finale di NLLB.
Completarla quando servirà (per ora non creeremo nuove versioni di NLLB)
'''

def create_nllb_final_model():
    en_text = "Also, unlike in 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb, the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."
    create_nllb_embed_and_lm_head3("onnx/NLLBOptimum/decoder_with_past_model.onnx", "onnx/NLLBOptimum/Optimized/ReducedRAM/nllb_embed_and_lm_head_if3.onnx")
    adapt_nllb_to_embed_and_lm_head("onnx/NLLBOptimum/encoder_model.onnx", "onnx/NLLBOptimum/decoder_with_past_model.onnx", "onnx/NLLBOptimum/Optimized/ReducedRAM/encoder_model.onnx", "onnx/NLLBOptimum/Optimized/ReducedRAM/decoder_model.onnx")
    onnx_execution.onnx_execution_nllb_cache_reduced_ram(en_text, "eng_Latn", "ita_Latn")


def create_nllb_embed_and_lm_head3(decoder_path: str, output_path: str):
    if(not Path(output_path).exists):
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
        lm_head_nodes = []
        lm_head_initializers = []
        embed_and_lm_head_initializers = []

        # aggiungiamo i nodi e gli initializers richiesti per nllb_embed e nllb_lm_head
        attributes = {"value": onnx.helper.make_tensor("logits",
                                                            onnx.TensorProto.FLOAT,
                                                            [0, 1, 256206],
                                                            vals=[])}
        embed_nodes.append(onnx.helper.make_node(name="Constant_1",
                                    op_type="Constant",
                                    inputs=[],
                                    outputs=["logits"],
                                    **attributes))
        attributes = {"value": onnx.helper.make_tensor("embed_matrix",
                                                            onnx.TensorProto.FLOAT,
                                                            [0, 0, 1024],
                                                            vals=[])}
        lm_head_nodes.append(onnx.helper.make_node(name="Constant_2",
                                    op_type="Constant",
                                    inputs=[],
                                    outputs=["embed_matrix"],
                                    **attributes))

        # aggiungiamo i nodi e gli initializers richiesti per nllb_embed
        for node in nodes:
            if(node.name == "/model/decoder/Reshape"):
                embed_nodes.append(node)
                embed_and_lm_head_initializers.append(initializers_dict[node.input[1]])
            if(node.name == "/model/decoder/embed_tokens/Gather"):
                node.output.pop()
                node.output.append("embed_matrix")
                embed_nodes.append(node)

        # aggiungiamo i nodi e gli initializers richiesti per nllb_lm_head
        perm = {"perm": [0, 2, 1]}
        transpose = onnx.helper.make_node(   #si crea il transpose
                                name="Transpose_1",
                                op_type="Transpose",
                                inputs=["pre_logits"],
                                outputs=["Transpose_output_1"],
                                **perm
                            )
        lm_head_nodes.append(transpose)   #si aggiunge ai nodi del grafico
        for node in nodes:
            if(node.name == "/lm_head/MatMul"):
                node.input.pop(0)
                node.input.insert(0, "model.shared.weight")
                node.input.pop(1)
                #node.input.insert(1, "Reshape_output_1")
                node.input.insert(1, "Transpose_output_1")
                lm_head_nodes.append(node)
                node.output.pop(0)
                node.output.append("logits_transposed")

        perm = {"perm": [0, 2, 1]}
        transpose2 = onnx.helper.make_node(   #si crea il transpose
                                name="Transpose_2",
                                op_type="Transpose",
                                inputs=["logits_transposed"],
                                outputs=["logits"],
                                **perm
                            )
        lm_head_nodes.append(transpose2)   #si aggiunge ai nodi del grafico

        embed_and_lm_head_initializers.append(initializers_dict["model.shared.weight"])


        # Creazione del grafico nllb_embed
        embed_matrix = onnx.helper.make_tensor_value_info("embed_matrix",
                                                            onnx.TensorProto.FLOAT,
                                                            ["batch_size", "sequence_length", 1024])
        logits = onnx.helper.make_tensor_value_info("logits",
                                                            onnx.TensorProto.FLOAT,
                                                            ["batch_size", 1, 256206])
        shared_weight = onnx.helper.make_tensor_value_info("model.shared.weight",
                                                            onnx.TensorProto.FLOAT,
                                                            [256206,1024])

        nllb_embed_graph = onnx.helper.make_graph(
            nodes=embed_nodes,
            name="nllb_embed",
            inputs=[],  # Graph input
            outputs=[embed_matrix, logits],  # Graph output
            initializer=embed_initializers,
        )

        # Creazione del grafico nllb_lm_head
        nllb_lm_head_graph = onnx.helper.make_graph(
            nodes=lm_head_nodes,
            name="nllb_lm_head",
            inputs=[],  # Graph input
            outputs=[embed_matrix, logits],  # Graph output
            initializer=lm_head_initializers,
        )

        # Creazione del grafico di nllb_embed_and_lm_head
        attributes = {"then_branch": nllb_lm_head_graph, "else_branch": nllb_embed_graph}
        if_node = onnx.helper.make_node(   #si crea il nodo if
                                    name="optimum::if",
                                    op_type="If",
                                    inputs=["use_lm_head"],
                                    outputs=["embed_matrix", "logits"],
                                    **attributes
                                )
        attributes = {"value": onnx.helper.make_tensor("model.shared.weight",
                                                            onnx.TensorProto.FLOAT,
                                                            [256206, 1024],
                                                            vals=initializers_dict["model.shared.weight"].raw_data, raw=True)}
        '''constant_node = onnx.helper.make_node(name="Constant_3",
                                    op_type="Constant",
                                    inputs=[],
                                    outputs=["model.shared.weight"],
                                    **attributes)'''
        #create inputs
        use_lm_head = onnx.helper.make_tensor_value_info("use_lm_head",
                                                            onnx.TensorProto.BOOL,
                                                            [1])
        input_ids = onnx.helper.make_tensor_value_info("input_ids",
                                                            onnx.TensorProto.INT64,
                                                            ["batch_size", "sequence_length"])
        pre_logits = onnx.helper.make_tensor_value_info("pre_logits",
                                                            onnx.TensorProto.FLOAT,
                                                            ["batch_size", 1, 1024])
        #create outputs
        embed_matrix = onnx.helper.make_tensor_value_info("embed_matrix",
                                                            onnx.TensorProto.FLOAT,
                                                            ["batch_size", "sequence_length", 1024])
        logits = onnx.helper.make_tensor_value_info("logits",
                                                            onnx.TensorProto.FLOAT,
                                                            ["batch_size", 1, 256206])

        nllb_embed_and_lm_head_graph = onnx.helper.make_graph(
            nodes=[if_node],
            name="nllb_embed",
            inputs=[use_lm_head, input_ids, pre_logits],  # Graph input
            outputs=[embed_matrix, logits],  # Graph output
            initializer=embed_and_lm_head_initializers,
        )

        # Create the model (ModelProto)
        model_def = onnx.helper.make_model(nllb_embed_and_lm_head_graph, producer_name="nie")
        model_def.opset_import[0].version = 17
        op = model_def.opset_import
        print(op)

        onnx.save_model(model_def, output_path, save_as_external_data=False)
        onnx.shape_inference.infer_shapes_path(output_path, output_path)
        onnx.checker.check_model(output_path)


def adapt_nllb_to_embed_and_lm_head(encoder_path, decoder_path, encoder_path_out, decoder_path_out):
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

    #del encoder_model
    #del decoder_model
    #del graph
    #del initializers

    #rimuoviamo dall'encoder i nodi esportati in nllb_embed_and_lm_head e rinominiamo l' input di /Mul
    for node in encoder_nodes:
        if(node.name == "/Mul"):
            node.input.remove("/embed_tokens/Gather_output_0")
            node.input.insert(0, "embed_matrix")
            print(node)

    for node in encoder_nodes:
        if(node.name == "/embed_tokens/Gather"):
            encoder_nodes.remove(node)

    #rimuoviamo dal decoder i nodi esportati in nllb_embed_and_lm_head e rinominiamo l' input di /Mul e l' output di /model/decoder/layer_norm/LayerNormalization
    for node in decoder_nodes:
        if(node.name == "/model/decoder/Mul"):
            node.input.remove("/model/decoder/embed_tokens/Gather_output_0")
            node.input.insert(0, "embed_matrix")
        if(node.name == "/model/decoder/layer_norm/LayerNormalization"):
            node.output.remove("/model/decoder/layer_norm/LayerNormalization_output_0")
            node.output.append("pre_logits")

    for node in decoder_nodes:
        if(node.name == "/lm_head/MatMul"):
            decoder_nodes.remove(node)
    for node in decoder_nodes:
        if(node.name == "Transpose_1366"):
            decoder_nodes.remove(node)
    for node in decoder_nodes:
        if(node.name == "/model/decoder/embed_tokens/Gather"):
            decoder_nodes.remove(node)

    #modifichiamo gli input dell'encoder e rimuoviamo i pesi dell'embed
    embed_matrix = onnx.helper.make_tensor_value_info("embed_matrix",
                                                        onnx.TensorProto.FLOAT,
                                                        ["batch_size", "sequence_length", 1024])
    encoder_graph.input.insert(0, embed_matrix)
    encoder_graph.initializer.remove(encoder_initializers_dict["embed_tokens.weight"])

    #modifichiamo gli input e gli output del decoder e rimuoviamo i pesi dell'embed (gli stessi anche dell' lm_head)
    embed_matrix = onnx.helper.make_tensor_value_info("embed_matrix",
                                                        onnx.TensorProto.FLOAT,
                                                        ["batch_size", "sequence_length", 1024])
    decoder_graph.input.insert(0, embed_matrix)
    logits = onnx.helper.make_tensor_value_info("logits",
                                                        onnx.TensorProto.FLOAT,
                                                        ["batch_size", 1, 256206])
    #decoder_graph.output.remove(logits)
    decoder_graph.output.pop(0)
    pre_logits = onnx.helper.make_tensor_value_info("pre_logits",
                                                        onnx.TensorProto.FLOAT,
                                                        ["batch_size", 1, 1024])
    decoder_graph.output.insert(0, pre_logits)

    decoder_graph.initializer.remove(decoder_initializers_dict["model.shared.weight"])

    #salviamo l' encoder e il decoder aggiornati
    onnx.save_model(encoder_model, encoder_path_out, save_as_external_data=False)
    onnx.shape_inference.infer_shapes_path(encoder_path_out, encoder_path_out)

    onnx.save_model(decoder_model, decoder_path_out, save_as_external_data=False)
    onnx.shape_inference.infer_shapes_path(decoder_path_out, decoder_path_out)

    onnx.checker.check_model(encoder_path_out)
    onnx.checker.check_model(decoder_path_out)


def quantize_nllb_4bit():
    #quantization of encoder
    model_fp32_path="onnx/NLLBOptimum/Optimized/ReducedRAM/encoder_model.onnx"
    model_int4_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit_HQQ/nllb_encoder_4bit.onnx"

    quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
      block_size=128, # 2's exponential and >= 16
      is_symmetric=True, # if true, quantize to Int4. otherwise, quantize to uint4.
      accuracy_level=4, # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35,
      quant_format=quant_utils.QuantFormat.QOperator)

    quant_config_hqq = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig()

    model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
      model,
      nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
      algo_config=quant_config_hqq,)
    quant.process()
    quant.model.save_model_to_file(model_int4_path, False)

    #quantization of decoder
    model_fp32_path="onnx/NLLBOptimum/Optimized/ReducedRAM/decoder_model.onnx"
    model_int4_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit_HQQ/nllb_decoder_4bit.onnx"

    model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
      model,
      nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
      algo_config=quant_config_hqq,)
    quant.process()
    quant.model.save_model_to_file(model_int4_path, False)

    #quantization of cache init
    model_fp32_path="onnx/NLLBOptimum/Optimized/NLLB_cache_initializer.onnx"
    model_int4_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit_HQQ/nllb_cache_initializer_4bit.onnx"

    model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
      model,
      nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
      algo_config=quant_config_hqq,)
    quant.process()
    quant.model.save_model_to_file(model_int4_path, False)


    #quantization of embedAndLmHead
    model_fp32_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/PreProcess/nllb_embed_and_lm_head_if2.onnx"   #onnx/NLLBOptimum/Optimized/ReducedRAM/nllb_embed_and_lm_head_fixed.onnx
    model_int4_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit_HQQ/nllb_embed_and_lm_head_4bit.onnx"

    model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
      model,
      nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
      algo_config=quant_config_hqq,)
    quant.process()
    quant.model.save_model_to_file(model_int4_path, False)