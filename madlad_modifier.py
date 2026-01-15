import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy
import onnxoptimizer
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantFormat, QuantizationMode, QuantType, quant_pre_process
from onnxruntime.quantization.matmul_4bits_quantizer import HQQWeightOnlyQuantConfig
from onnxruntime.tools.onnx_model_utils import update_onnx_opset
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.AttributeType import AttributeType
from onnxruntime.transformers import float16
from onnxruntime.transformers.float16 import convert_tensor_float_to_float16
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.tools import symbolic_shape_infer
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer, ORTQuantizableOperator
from optimum.onnxruntime.configuration import AutoOptimizationConfig, OptimizationConfig, QuantizationConfig, \
    ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION, ORTConfig
from optimum.onnxruntime.preprocessors import QuantizationPreprocessor
from optimum.utils.save_utils import maybe_save_preprocessors

import main
import onnx
import onnx_execution
from onnxsim import simplify, model_info

#from Olive.olive.hardware.accelerator import DEFAULT_CPU_ACCELERATOR
#from Olive.olive.passes.onnx import transformer_optimization
from onnx import GraphProto
from onnx import shape_inference
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import onnx
from datasets import Dataset, load_dataset
from packaging.version import Version, parse
from transformers import AutoConfig

from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.transformers.onnx_model import OnnxModel
from onnxruntime.quantization import (
    matmul_4bits_quantizer,
    quant_utils,
    quantize
)


#model_path = "onnx/Madlad/Optimized/Madlad_decoder_optimized.onnx"
#model_path_out = "onnx/Madlad/Optimized/2D/Madlad_decoder_2D.onnx"
#model_path = "onnx/Madlad/Script/Madlad_decoder_complete.onnx"
#model_path = "onnx/Madlad/Simplified/2D/Madlad_decoder_simplified_2D_inf.onnx"
#model_path = "onnx/Madlad/Optimized/2D/Optimized/Madlad_decoder_2D_optimized.onnx"


LOGGER = logging.getLogger(__name__)

def quantize_custom(
        self,
        quantization_config: QuantizationConfig,
        save_dir: Union[str, Path],
        file_suffix: Optional[str] = "quantized",
        calibration_tensors_range: Optional[Dict[str, Tuple[float, float]]] = None,
        use_external_data_format: bool = False,
        preprocessor: Optional[QuantizationPreprocessor] = None,
        has_subgraphs = False
    ) -> Path:
    """
    Quantizes a model given the optimization specifications defined in `quantization_config`.

    Args:
        quantization_config (`QuantizationConfig`):
            The configuration containing the parameters related to quantization.
        save_dir (`Union[str, Path]`):
            The directory where the quantized model should be saved.
        file_suffix (`Optional[str]`, defaults to `"quantized"`):
            The file_suffix used to save the quantized model.
        calibration_tensors_range (`Optional[Dict[str, Tuple[float, float]]]`, defaults to `None`):
            The dictionary mapping the nodes name to their quantization ranges, used and required only when applying static quantization.
        use_external_data_format (`bool`, defaults to `False`):
            Whether to use external data format to store model which size is >= 2Gb.
        preprocessor (`Optional[QuantizationPreprocessor]`, defaults to `None`):
            The preprocessor to use to collect the nodes to include or exclude from quantization.

    Returns:
        The path of the resulting quantized model.
    """
    use_qdq = quantization_config.is_static and quantization_config.format == QuantFormat.QDQ
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if quantization_config.is_static and calibration_tensors_range is None:
        raise ValueError(
            "Requested static quantization in the QuantizationConfig, but no calibration ranges were provided. Please run calibration first using the quantizer fit method, or use dynamic quantization."
        )
    if not quantization_config.is_static:
        if quantization_config.mode != QuantizationMode.IntegerOps:
            LOGGER.warning(
                f"ONNX Runtime dynamic quantization mode should be QuantizationMode.IntegerOps "
                f"(got: {quantization_config.mode})."
            )
        if quantization_config.activations_dtype != QuantType.QUInt8:
            LOGGER.warning(
                f"ONNX Runtime dynamic quantization activations data type should be QuantType.QUInt8 "
                f"(got: {quantization_config.activations_dtype})."
            )

    LOGGER.info(
        f"Creating {'static' if quantization_config.is_static else 'dynamic'} quantizer: {quantization_config}"
    )

    if preprocessor is not None:
        LOGGER.info("Preprocessor detected, collecting nodes to include/exclude")
        nodes_to_quantize, nodes_to_exclude = preprocessor.collect(self.onnx_model_path)

        nodes_to_quantize.update(quantization_config.nodes_to_quantize)
        nodes_to_exclude.update(quantization_config.nodes_to_exclude)

        quantization_config.nodes_to_quantize = list(nodes_to_quantize)
        quantization_config.nodes_to_exclude = list(nodes_to_exclude)

    onnx_model = onnx.load(Path(self.onnx_model_path).as_posix())
    if has_subgraphs:
        if quantization_config.is_static:
            raise NotImplementedError("Static quantization is currently not supported for models with subgraphs.")
        if parse(ort_version) == Version("1.16.0"):
            raise ValueError(
                "ONNX Runtime version v1.16.0 is not compatible with quantization for models with subgraphs, please downgrade to 1.15.1 or upgrade to a higher version. Reference: https://github.com/microsoft/onnxruntime/pull/17651"
            )

    quantizer_factory = QDQQuantizer if use_qdq else ONNXQuantizer

    # The argument `input_qType` has been changed into `activation_qType` from ORT 1.13
    quantizer = quantizer_factory(
        model=onnx_model,
        static=quantization_config.is_static,
        per_channel=quantization_config.per_channel,
        mode=quantization_config.mode,
        weight_qType=quantization_config.weights_dtype,
        activation_qType=quantization_config.activations_dtype,
        tensors_range=calibration_tensors_range,
        reduce_range=quantization_config.reduce_range,
        nodes_to_quantize=quantization_config.nodes_to_quantize,
        nodes_to_exclude=quantization_config.nodes_to_exclude,
        op_types_to_quantize=[
            operator.value if isinstance(operator, ORTQuantizableOperator) else operator
            for operator in quantization_config.operators_to_quantize
        ],
        extra_options={
            "WeightSymmetric": quantization_config.weights_symmetric,
            "ActivationSymmetric": quantization_config.activations_symmetric,
            "EnableSubgraph": has_subgraphs,
            "ForceSymmetric": quantization_config.activations_symmetric
            and quantization_config.weights_symmetric,
            "AddQDQPairToWeight": quantization_config.qdq_add_pair_to_weight,
            "DedicatedQDQPair": quantization_config.qdq_dedicated_pair,
            "QDQOpTypePerChannelSupportToAxis": quantization_config.qdq_op_type_per_channel_support_to_axis,
        },
    )


    LOGGER.info("Quantizing model...")
    quantizer.quantize_model()

    suffix = f"_{file_suffix}" if file_suffix else ""
    quantized_model_path = save_dir.joinpath(f"{self.onnx_model_path.stem}{suffix}").with_suffix(".onnx")
    LOGGER.info(f"Saving quantized model at: {save_dir} (external data format: " f"{use_external_data_format})")
    quantizer.model.save_model_to_file(quantized_model_path.as_posix(), use_external_data_format)

    # Create and save the configuration summarizing all the parameters related to quantization
    ort_config = ORTConfig(quantization=quantization_config, use_external_data_format=use_external_data_format)
    ort_config.save_pretrained(save_dir)

    if self.config is not None:
        self.config.save_pretrained(save_dir)

    maybe_save_preprocessors(self.onnx_model_path.parent, save_dir)

    return Path(save_dir)


def addReshapes_partial(model_path):
    #onnx.shape_inference.infer_shapes_path("onnx/Madlad/Script/Madlad_decoder_complete.onnx",
    #                                       "onnx/Madlad/Script/Madlad_decoder_complete.onnx")
    model = onnx.load_model(model_path)
    #inferred_model = shape_inference.infer_shapes(model)
    #shapes_info = inferred_model.graph.value_info

    #RepeatedCompositeContainer

    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node

    # creiamo un dizionario che associa al nome di ogni output di ogni operatore del grafico la sua dimensione
    shapes_info = model.graph.value_info
    shapes_info_dict = {}
    for info in shapes_info:
        shape = []
        dim = info.type.tensor_type.shape.dim
        for value in dim:
            shape.append(value.dim_value)
        shapes_info_dict[info.name] = shape
    shapes_info_dict["encoder_hidden_states"] = [1,128,1024]  #si aggiunge questo manualmente perche per qualche motivo manca
    shapes_info_dict["logits"] = [1,128,256000]    #si aggiunge questo manualmente perche per qualche motivo manca

    # creiamo un dizionario che associa al nome di ogni initializer del grafico le sue informazioni
    initializers_dict = {}
    for initializer in initializers:
        # Data type:
        # https://github.com/onnx/onnx/blob/rel-1.9.0/onnx/onnx.proto
        #print("Tensor information:")
        #print(f"Tensor Name: {initializer.name}, Data Type: {initializer.data_type}, Shape: {initializer.dims}")
        initializers_dict[initializer.name] = initializer

    # creiamo un dizionario che associa a ogni nome di ogni input del grafico una lista contenente tutti i nodi (op) che possiedono quell'input
    #(volendo posso escludere gli input presenti in initializers, se servir`a velocizzare l' algoritmo)
    inputs_dict = {}
    for node in nodes:
        for input in node.input:
            if(input not in inputs_dict):
                inputs_dict[input] = [node]
            elif(node not in inputs_dict[input]):
                inputs_dict[input].append(node)

    #del model
    #del graph
    #del initializers

    count = 0
    count_incompatibles = 0
    skipped = []            #input di cui non si conosce la dimensione (deve essere vuoto, altrimenti l'algoritmo non funziona)
    incompatible_matmuls = []
    count_output_shape = 0
    node_index = 0
    while(node_index < len(nodes)):   #for node in nodes
        node = nodes[node_index]
        if(node.op_type == "MatMul"):
            compatible = True
            for input in node.input:
                #qui si stabilisce se un MatMul puo essere convertito in 2D o no (cio`e se tutti gli input del MatMul sono 3D)
                if((input not in initializers_dict) and (not input in shapes_info_dict)): skipped.append(input)
                if((input not in initializers_dict) and (input in shapes_info_dict)):  #le matrici costanti sono tutte 2D, quindi controlliamo solo gli input che non sono costanti
                    dims_input = shapes_info_dict[input]
                    if(len(dims_input) > 3):
                        compatible = False
            if(compatible == True):
                # qui si aggiunge al grafico i reshape prima di ogni input 3D del MatMul e dopo il MatMul
                output = node.output[0]
                input_index = 0
                for input in node.input:  #per ogni input del MatMul
                    if(input not in initializers_dict):  # se la matrice di input non `e costante
                        dims_input = shapes_info_dict[input]
                        dims_output = shapes_info_dict[output]
                        if(len(dims_input) == 3 and dims_input[0] == 1):  #se la matrice di input `e 3D e la prima dimensione `e 1
                            # si crea il reshape che ha come nome Reshape_custom_+count_output_shape,
                            # come input questo input, come output un nome generato da noi (Reshape_custom_output_+count_output_shape)
                            # e come dimensione la versione 2D delle dimensioni dell'input (cio`e le stesse ma senza il primo elemento)
                            shape_initializer = onnx.helper.make_tensor(   #si crea una tensore che rappresenta la forma del reshape
                                name="shape_"+str(count_output_shape),
                                data_type=onnx.TensorProto.INT64,
                                dims=[2],
                                vals=[dims_input[1], dims_input[2]]
                            )
                            initializers.extend([shape_initializer])  #si aggiunge agli initializer del grafico
                            reshape = onnx.helper.make_node(   #si crea il reshape
                                name="Reshape_custom_"+str(count_output_shape),
                                op_type="Reshape",
                                inputs=[input, "shape_"+str(count_output_shape)],
                                outputs=["Reshape_custom_output_"+str(count_output_shape)],
                            )
                            nodes.insert(node_index, reshape)   #si aggiunge ai nodi del grafico
                            node_index = node_index+1

                            # si modifica questo input di questo nodo assegnandogli il nome dell'output del reshape appena creato
                            node.input[input_index] = "Reshape_custom_output_"+str(count_output_shape)

                            count_output_shape = count_output_shape+1

                            if(output == "logits"):  #se siamo all'ultimo matmul (che genera l'ouput finale "logits")
                                node.output[0] = "logits_2D"
                                output = node.output[0]
                                # si crea un'altro reshape che ha come nome Reshape_custom_+count_output_shape,
                                # come input l'output di questo MatMul, come output un nome generato da noi (Reshape_custom_output_+count_output_shape)
                                # e come dimensione la versione 3D delle dimensioni dell'output (cio`e le stesse, perche sono gia 3D)
                                shape_initializer2 = onnx.helper.make_tensor(
                                    # si crea una tensore che rappresenta la forma del reshape
                                    name="shape_" + str(count_output_shape),
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[3],
                                    vals=dims_output
                                )
                                initializers.extend([shape_initializer2])  # si aggiunge agli initializer del grafico (nella giusta posizione)
                                reshape2 = onnx.helper.make_node(  # si crea il reshape
                                    name="Reshape_custom_" + str(count_output_shape),
                                    op_type="Reshape",
                                    inputs=[output, "shape_" + str(count_output_shape)],
                                    outputs=["logits"],
                                )
                                nodes.insert(node_index+1, reshape2)  # si aggiunge ai nodi del grafico (nella giusta posizione)
                                node_index = node_index+1
                            else:
                                # si crea un'altro reshape che ha come nome Reshape_custom_+count_output_shape,
                                # come input l'output di questo MatMul, come output un nome generato da noi (Reshape_custom_output_+count_output_shape)
                                # e come dimensione la versione 3D delle dimensioni dell'output (cio`e le stesse, perche sono gia 3D)
                                shape_initializer2 = onnx.helper.make_tensor(
                                    # si crea una tensore che rappresenta la forma del reshape
                                    name="shape_" + str(count_output_shape),
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[3],
                                    vals=dims_output
                                )
                                initializers.extend([shape_initializer2])  # si aggiunge agli initializer del grafico
                                reshape2 = onnx.helper.make_node(  # si crea il reshape
                                    name="Reshape_custom_" + str(count_output_shape),
                                    op_type="Reshape",
                                    inputs=[output, "shape_" + str(count_output_shape)],
                                    outputs=["Reshape_custom_output_" + str(count_output_shape)],
                                )
                                nodes.insert(node_index + 1, reshape2)  # si aggiunge ai nodi del grafico (nella giusta posizione)
                                node_index = node_index + 1

                                # si trovano tutti i nodi che hanno come input l'output di questo MatMul e si modifica il loro input corrispondente
                                # sostituendolo con l'output del reshape creato da ultimo
                                # (vedere se ci sono metodi nativi per cercare un nodo in base a se contiene un input, se non esiste
                                # allora all' inizio dell' algoritmo creare un dizionario che associa a ogni nome di ogni input del grafico
                                # una lista contenente i nomi di tutti i nodi (op) che possiedono quell'input) (devo controllare tutti i nodi, non solo i
                                # MatMul, infatti l'output di un MatMul puo essere usato anche da altri tipi di nodi).
                                if(output in inputs_dict):
                                    for node2 in inputs_dict[output]:
                                        index = 0
                                        for input2 in node2.input:
                                            if(input2 == output):
                                                break
                                            index = index+1
                                        node2.input[index] = "Reshape_custom_output_" + str(count_output_shape)

                                count_output_shape = count_output_shape + 1

                        input_index = input_index+1

            if(compatible == False):
                #qui si printano i MatMul incompatibili e si aggiunge il loro nome a incompatible_matmuls
                print(node.name + " is incompatible")
                for input in node.input:
                    print("input: "+input+f" dims: {shapes_info_dict[input]}")
                count_incompatibles = count_incompatibles+1
                print('')
                incompatible_matmuls.append(node.name)

            #for output in node.output:
                #print(output)
            #print('')
            count = count+1
        node_index = node_index+1

    print("total MatMuls: "+str(count))
    print("MatMul incompatibles: "+str(count_incompatibles))
    print("skipped: "+str(len(skipped)))
    print(skipped)

    # Create inputs
    input_ids = onnx.helper.make_tensor_value_info("input_ids",
                                           onnx.TensorProto.INT64,
                                           [1, 128])
    attention_mask = onnx.helper.make_tensor_value_info("attention_mask",
                                                   onnx.TensorProto.INT64,
                                                   [1, 128])
    encoder_hidden_states = onnx.helper.make_tensor_value_info("encoder_hidden_states",
                                                        onnx.TensorProto.FLOAT,
                                                        [1, 128, 1024])
    encoder_attention_mask = onnx.helper.make_tensor_value_info("encoder_attention_mask",
                                                        onnx.TensorProto.INT64,
                                                        [1, 128])
    # Create output
    logits = onnx.helper.make_tensor_value_info("logits",
                                                       onnx.TensorProto.FLOAT,
                                                       [1, 128, 256000])

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=nodes,
        name="Madlad",
        inputs=[input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask],  # Graph input
        outputs=[logits],  # Graph output
        initializer=initializers,
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="nie")
    model_def.opset_import[0].version = 11

    return [model_def, incompatible_matmuls]


def addReshapes(model_path, model_path_out):
    ret = addReshapes_partial(model_path)
    model_def = ret[0]

    onnx.save_model(model_def, model_path_out, save_as_external_data=True)
    onnx.shape_inference.infer_shapes_path(model_path_out,
                                           model_path_out)
    onnx.checker.check_model(model_path_out)


def showInfo(model_path):
    #onnx.shape_inference.infer_shapes_path(model_path,
    #                                       model_path)
    model = onnx.load_model(model_path)
    # inferred_model = shape_inference.infer_shapes(model)
    # shapes_info = inferred_model.graph.value_info

    # RepeatedCompositeContainer

    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node

    # creiamo un dizionario che associa al nome di ogni output di ogni operatore del grafico la sua dimensione
    shapes_info = model.graph.value_info
    shapes_info_dict = {}
    for info in shapes_info:
        shape = []
        dim = info.type.tensor_type.shape.dim
        for value in dim:
            shape.append(value.dim_value)
        shapes_info_dict[info.name] = shape
    shapes_info_dict["encoder_hidden_states"] = [1, 128, 1024]  # si aggiunge questo manualmente perche per qualche motivo manca
    shapes_info_dict["logits"] = [1, 128, 256000]  # si aggiunge questo manualmente perche per qualche motivo manca

    # creiamo un dizionario che associa al nome di ogni initializer del grafico le sue informazioni
    initializers_dict = {}
    for initializer in initializers:
        # Data type:
        # https://github.com/onnx/onnx/blob/rel-1.9.0/onnx/onnx.proto
        # print("Tensor information:")
        # print(f"Tensor Name: {initializer.name}, Data Type: {initializer.data_type}, Shape: {initializer.dims}")
        initializers_dict[initializer.name] = initializer

    # creiamo un dizionario che associa a ogni nome di ogni input del grafico una lista contenente tutti i nodi (op) che possiedono quell'input
    # (volendo posso escludere gli input presenti in initializers, se servir`a velocizzare l' algoritmo)
    inputs_dict = {}
    for node in nodes:
        for input in node.input:
            if (input not in inputs_dict):
                inputs_dict[input] = [node]
            elif (node not in inputs_dict[input]):
                inputs_dict[input].append(node)

    # del model
    # del graph
    # del initializers

    list_matmul_4d = []
    list_matmul_3d = []
    list_matmul_2d = []
    list_matmul_1d = []
    skipped = []

    op_types = []

    for node in nodes:
        if (node.op_type not in op_types):
            op_types.append(node.op_type)
        if (node.op_type == "MatMul"):
            is4D = False
            is3D = False
            is2D = False
            is1D = False
            dimension = 6
            for input in node.input:
                # qui si stabilisce se un MatMul puo essere convertito in 2D o no (cio`e se tutti gli input del MatMul sono 3D)
                if ((input not in initializers_dict) and (not input in shapes_info_dict)): skipped.append(input)
                if (input in shapes_info_dict):
                    dims_input = shapes_info_dict[input]
                    if (len(dims_input) > 3 and len(dims_input) < dimension):
                        dimension = 4
                    if (len(dims_input) == 3 and len(dims_input) < dimension):
                        dimension = 3
                    if (len(dims_input) == 2 and len(dims_input) < dimension):
                        dimension = 2
                    if (len(dims_input) == 1 and len(dims_input) < dimension):
                        dimension = 1
            if (dimension == 4):
                list_matmul_4d.append(node.name)
            if (dimension == 3):
                list_matmul_3d.append(node.name)
            if (dimension == 2):
                list_matmul_2d.append(node.name)
            if (dimension == 1):
                list_matmul_1d.append(node.name)

    print("1D MatMul:")
    print(list_matmul_1d)
    print('')
    print("2D MatMul:")
    print(list_matmul_2d)
    print('')
    print("3D MatMul:")
    print(list_matmul_3d)
    print('')
    print("4D MatMul:")
    print(list_matmul_4d)
    print('')
    print('')
    print("op types in the model:")
    print(op_types)
    print('')

    print("skipped:")
    print(skipped)


def replace_unsqueeze(model_path, model_path_out):
    model = onnx.load_model(model_path)

    nodes = model.graph.node
    initializers = model.graph.initializer

    node_index = 0
    while (node_index < len(nodes)):  # for node in nodes
        node = nodes[node_index]
        if (node.name == "/Unsqueeze_2"):
            shape_initializer = onnx.helper.make_tensor(  # si crea una tensore che rappresenta la forma del reshape
                name="shape_unsqueeze_2",
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=[1, 1, 1, 128]
            )
            initializers.extend([shape_initializer])  # si aggiunge agli initializer del grafico
            reshape = onnx.helper.make_node(  # si crea il reshape
                name="Reshape_custom_unsqueeze_2",
                op_type="Reshape",
                inputs=["attention_mask", "shape_unsqueeze_2"],
                outputs=["/Unsqueeze_2_output_0"],
            )
            # si sostituisce a /Unsqueeze_2 nei nodi del grafico (non serve aggiornare l'input del nodo successivo perche questo nuovo nodo ha lo stesso nome dell' output del reshape che sostituisce
            nodes.remove(node)
            nodes.insert(node_index, reshape)

        if (node.name == "/Unsqueeze_4"):
            shape_initializer = onnx.helper.make_tensor(  # si crea una tensore che rappresenta la forma del reshape
                name="shape_unsqueeze_4",
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=[1, 1, 1, 128]
            )
            initializers.extend([shape_initializer])  # si aggiunge agli initializer del grafico
            reshape = onnx.helper.make_node(  # si crea il reshape
                name="Reshape_custom_unsqueeze_4",
                op_type="Reshape",
                inputs=["encoder_attention_mask", "shape_unsqueeze_4"],
                outputs=["/Unsqueeze_4_output_0"],
            )
            # si sostituisce a /Unsqueeze_4 nei nodi del grafico (non serve aggiornare l'input del nodo successivo perche questo nuovo nodo ha lo stesso nome dell' output del reshape che sostituisce
            nodes.remove(node)
            nodes.insert(node_index, reshape)

        node_index = node_index+1

    onnx.save_model(model, model_path_out, save_as_external_data=True)



def create_madlad_cache_initializer(model_path, model_path_out):
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
    model_def.opset_import[0].version = 17

    onnx.save_model(model_def, model_path_out, save_as_external_data=False)
    onnx.shape_inference.infer_shapes_path(model_path_out, model_path_out)
    onnx.checker.check_model(model_path_out)


def create_whisper_cache_initializer(model_path, model_path_out, batch_size):
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
        if((("encoder_attn/k_proj/MatMul" in node.name) or ("encoder_attn/v_proj/MatMul" in node.name) or ("encoder_attn/v_proj/Add" in node.name)) and (node.op_type == "MatMul" or node.op_type == "Add")):
            nodes_list.append(node)
            for input in node.input:
                if(input in initializers_dict):
                    inputs_list.append(initializers_dict[input])

    del model
    del graph
    del initializers


    count_output_shape = 0
    node_index = 0
    while(node_index < len(nodes_list)):   #for node in nodes
        node = nodes_list[node_index]

        if(("encoder_attn/k_proj/MatMul" in node.name) or ("encoder_attn/v_proj/Add" in node.name)):
            #for per trovare il numero del block di questo nodo (serve per scrivere il nome dell'output del transpose)
            index = 0
            for i in range(32):
                if("layers."+str(i) in node.name):
                    index = i
            #capiamo se il nodo corrente `e un key o value
            if("/k_proj/" in node.name):
                isKey = True
            else:
                isKey = False

            #creaiamo e inseriamo il reshape in node_list
            shape_initializer = onnx.helper.make_tensor(   #si crea una tensore che rappresenta la forma del reshape
                                    name="shape_"+str(count_output_shape),
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[4],
                                    vals=[batch_size, -1, 12, 64]   #si usa una batch size fissa di 1 cosi da poter usare -1 al posto della dimensione dinamica (dato che non si puo inserire una dimensione dinamica in un reshape)
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
                                                        [batch_size, "encoder_sequence_length", 768])
    # Create outputs
    outputs = []
    for i in range(12):
        outputs.append(onnx.helper.make_tensor_value_info("present."+str(i)+".encoder.key",
                                                        onnx.TensorProto.FLOAT,
                                                        [batch_size, 12, "encoder_sequence_length_out", 64]))

        outputs.append(onnx.helper.make_tensor_value_info("present."+str(i)+".encoder.value",
                                                        onnx.TensorProto.FLOAT,
                                                        [batch_size, 12, "encoder_sequence_length_out", 64]))

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
    model_def.opset_import[0].version = 17

    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, model_path_out, save_as_external_data=False)
    onnx.shape_inference.infer_shapes_path(model_path_out, model_path_out)
    onnx.checker.check_model(model_path_out)



def create_nllb_cache_initializer(model_path, model_path_out):
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
        if((("encoder_attn/k_proj/MatMul" in node.name) or ("encoder_attn/v_proj/MatMul" in node.name) or ("encoder_attn/v_proj/Add" in node.name)) and (node.op_type == "MatMul" or node.op_type == "Add")):
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

        if(("encoder_attn/k_proj/MatMul" in node.name) or ("encoder_attn/v_proj/Add" in node.name)):
            #for per trovare il numero del block di questo nodo (serve per scrivere il nome dell'output del transpose)
            index = 0
            for i in range(32):  #in realt`a il layer massimo `e 12, ma il codice funziona lo stesso
                if("layers."+str(i) in node.name):
                    index = i
            #capiamo se il nodo corrente `e un key o value
            if("/k_proj/" in node.name):
                isKey = True
            else:
                isKey = False

            #creaiamo e inseriamo il reshape in node_list
            shape_initializer = onnx.helper.make_tensor(   #si crea una tensore che rappresenta la forma del reshape
                                    name="shape_"+str(count_output_shape),
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[4],
                                    vals=[1, -1, 16, 64]   #si usa una batch size fissa di 1 cosi da poter usare -1 al posto della dimensione dinamica (dato che non si puo inserire una dimensione dinamica in un reshape)
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
    for i in range(12):
        outputs.append(onnx.helper.make_tensor_value_info("present."+str(i)+".encoder.key",
                                                        onnx.TensorProto.FLOAT,
                                                        [1, 16, "encoder_sequence_length_out", 64]))

        outputs.append(onnx.helper.make_tensor_value_info("present."+str(i)+".encoder.value",
                                                        onnx.TensorProto.FLOAT,
                                                        [1, 16, "encoder_sequence_length_out", 64]))

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
    model_def.opset_import[0].version = 17

    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, model_path_out, save_as_external_data=False)
    onnx.shape_inference.infer_shapes_path(model_path_out, model_path_out)
    onnx.checker.check_model(model_path_out)



def conversion_to_2D():
    en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb, the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."
    model_path = 'onnx/Madlad/Script/Madlad_decoder_complete.onnx'
    model_path_out = 'onnx/Madlad/Optimized/Optimized2D/Madlad_decoder_2D.onnx'
    optimize_model(model_path, model_path_out)
    addReshapes(model_path_out, model_path_out)
    optimize_model(model_path_out, model_path_out)
    replace_unsqueeze(model_path_out, model_path_out)

    showInfo(model_path_out)
    onnx_execution.onnx_execution_madlad(en_text, 'it', model_path_out)




def optimize_onnx_model(model_path, model_path_out):
    model = onnx.load_model(model_path)
    passes = ['eliminate_common_subexpression', 'eliminate_deadend', 'eliminate_duplicate_initializer',
              'eliminate_identity', 'eliminate_if_with_const_cond', 'eliminate_nop_cast', 'eliminate_nop_dropout',
              'eliminate_nop_flatten', 'eliminate_nop_monotone_argmax', 'eliminate_nop_pad', 'eliminate_nop_concat',
              'eliminate_nop_split', 'eliminate_nop_expand', 'eliminate_shape_gather', 'eliminate_nop_transpose', 'eliminate_nop_reshape',
              'eliminate_nop_with_unit', 'eliminate_unused_initializer', 'extract_constant_to_initializer', 'fuse_consecutive_reduce_unsqueeze',
              'fuse_consecutive_transposes', 'fuse_consecutive_unsqueezes']
    optimized_model = onnxoptimizer.optimize(model, None)
    onnx.save(optimized_model, model_path_out, save_as_external_data=True)

    #sess_options = onnxruntime.SessionOptions()
    #sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    #sess_options.optimized_model_filepath = model_path_out
    #onnxruntime.InferenceSession(model_path_out, sess_options=sess_options)
    model_info.print_simplifying_info(model, optimized_model)


def simplify_model(model_path, model_path_out, external_data = False):
    model = onnx.load_model(model_path)
    model_simplyfied, check_ok = simplify(model, include_subgraph=True)
    onnx.save_model(model_simplyfied, model_path_out, save_as_external_data=external_data)
    if check_ok:
        print("Finish! Here is the difference:")
        model_info.print_simplifying_info(model, model_simplyfied)
    else:
        print(
            'Check failed. Please be careful to use the simplified model, or try specifying "--skip-fuse-bn" or "--skip-optimization" (run "onnxsim -h" for details).'
        )
        print("Here is the difference after simplification:")
        model_info.print_simplifying_info(model, model_simplyfied)


def optimize_optimum(model_path_dir, model_path_name, model_path_out):
    # Create the optimizer
    optimizer = ORTOptimizer.from_pretrained(Path(model_path_dir), file_names=[model_path_name+".onnx"])
    # Define the optimization strategy by creating the appropriate configuration
    optimization_config = OptimizationConfig(optimization_level=0,
                                             enable_gelu_approximation=True,
                                             use_multi_head_attention=True)
    # Optimize the model
    optimizer.optimize(save_dir=model_path_out, optimization_config=optimization_config)
    model = onnx.load_model(model_path_dir+"/"+model_path_name+".onnx")
    model_optimized = onnx.load_model(model_path_out+"/"+model_path_name+"_optimized.onnx")
    model_info.print_simplifying_info(model, model_optimized)


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
    config = {"model_type": "t5", "num_heads": 16, "hidden_size": 2048, "optimization_options": {"use_multi_head_attention": True}}
    #optimizer = transformer_optimization.OrtTransformersOptimization(accelerator, config, True)
    #optimizer.run(model, data_root=model_path_dir+"/"+model_path_name+".onnx_data", output_model_path=model_path_out+"/"+model_path_name+"_optimized.onnx")
    optimization_options = FusionOptions("t5")
    optimization_options.use_multi_head_attention = False
    optimization_options.enable_layer_norm = True
    optimization_options.enable_embed_layer_norm = False
    optimization_options.enable_bias_skip_layer_norm = True
    optimization_options.enable_skip_layer_norm = True
    optimization_options.enable_rotary_embeddings = False
    optimization_options.disable_multi_head_attention_bias = True
    optimization_options.enable_bias_skip_layer_norm = False
    optimization_options.enable_bias_gelu = False
    optimizer = optimize_model(model_path_dir+"/"+model_path_name+".onnx", model_type="t5", num_heads=16, hidden_size=2048, optimization_options=optimization_options, only_onnxruntime=True, opt_level=99, verbose=False)
    #optimizer = optimize_by_fusion_mod(model, model_type="t5", num_heads=16, hidden_size=2048, optimization_options=optimization_options)
    # Topologically sort the graph at the end since previous optimizations may have broken it
    optimizer.topological_sort()
    model = onnx.load_model(model_path_dir+"/"+model_path_name+".onnx")
    model_optimized = optimizer.model
    model_info.print_simplifying_info(model, model_optimized)
    onnx.save_model(model_optimized, model_path_out, save_as_external_data=True)
    shape_inference.infer_shapes_path(model_path_out, model_path_out)


def simplify_and_quantize_madlad():
    encoder_path = "onnx/Madlad/Optimum_Cache/encoder_model.onnx"
    encoder_path_out = "onnx/Madlad/Optimum_Cache/Simplified/Madlad_encoder_simplified.onnx"
    decoder_path = "onnx/Madlad/Optimum_Cache/decoder_model_merged.onnx"
    decoder_path_out = "onnx/Madlad/Optimum_Cache/Simplified/Madlad_decoder_simplified.onnx"
    simplify_model(encoder_path, encoder_path_out, external_data=True)
    simplify_model(decoder_path, decoder_path_out, external_data=True)
    quantize_dynamic(Path(encoder_path_out), Path(encoder_path_out))
    quantize_dynamic(Path(decoder_path_out), Path(decoder_path_out), extra_options={"EnableSubgraph": True})
    update_onnx_opset(Path(decoder_path_out), 20, Path(decoder_path_out))
    simplify_model(encoder_path_out, encoder_path_out)
    simplify_model(decoder_path_out, decoder_path_out)




def quantize_dynamic_madlad(decoder_path="onnx/Madlad/Optimum_Cache_Optimized/OptimizedOptimum",
                            decoder_name="Madlad_decoder_optimized",
                            decoder_path_out="onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic/with_past_quality"):
    operators = ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION

    quantize_dynamic(Path(decoder_path+"/"+decoder_name+".onnx"), Path(decoder_path_out+"/"+decoder_name+"_quantized.onnx"),
                     per_channel=False, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=operators,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "DefaultTensorType": onnx.TensorProto.FLOAT,
                                    "MatMulConstBOnly": False})

    '''quantizer = ORTQuantizer.from_pretrained(decoder_path, file_name=decoder_name+'.onnx')
    qconfig = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.IntegerOps,  #QLinearOps non funziona
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=False,
        weights_dtype=QuantType.QUInt8,  #prima era QInt8
        weights_symmetric=False,  #prima era True
        per_channel=False,
        reduce_range=True,
        #nodes_to_quantize=[],
        #nodes_to_exclude=[],
        operators_to_quantize=operators,
    )
    quantizer.quantize(save_dir=decoder_path_out, quantization_config=qconfig, use_external_data_format=False)'''





def quantize_dynamic_madlad_encoder(encoder_path="onnx/Madlad/Optimum_Cache_Optimized",
                            encoder_name="encoder_model",
                            encoder_path_out="onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic"):
    quantizer = ORTQuantizer.from_pretrained(encoder_path, file_name=encoder_name+".onnx")
    operators = ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION

    denseReluNodes = get_DenseReluDense_nodes(encoder_path+"/"+encoder_name+".onnx")
    MatMulNotCostant = get_MatMul_not_costant(encoder_path+"/"+encoder_name+".onnx")

    quantize_dynamic(Path(encoder_path+"/"+encoder_name+".onnx"), Path(encoder_path_out+"/"+encoder_name+"_quantized.onnx"),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=operators,
                     use_external_data_format = True, nodes_to_exclude=denseReluNodes,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "DefaultTensorType": onnx.TensorProto.FLOAT,
                                    "MatMulConstBOnly": False})

    #for i in range(5):
    #    if("/block."+str(i)+"/layer.1/DenseReluDense/wo/MatMul" in denseReluNodes):
    #        denseReluNodes.remove("/block."+str(i)+"/layer.1/DenseReluDense/wo/MatMul")

    '''qconfig = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.IntegerOps,  #QLinearOps non funziona
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=False,
        weights_dtype=QuantType.QUInt8,  #prima era QInt8
        weights_symmetric=False,     #prima era True
        per_channel=True,
        reduce_range=True,
        #nodes_to_quantize=[],
        nodes_to_exclude=denseReluNodes,   #prima era denseReluNodes
        operators_to_quantize=operators,
    )
    quantizer.quantize(save_dir=encoder_path_out, quantization_config=qconfig, use_external_data_format=False)'''

    #convert_encoder_dense_relu_to_fp16(encoder_path_out+"/"+encoder_name+"_quantized.onnx", encoder_path_out+"/"+encoder_name+"_quantized_fp16.onnx")


def quantize_dynamic_madlad_encoder_int8_int16(encoder_path="onnx/Madlad/Optimum_Cache_Optimized",
                                               encoder_name="encoder_model",
                                               encoder_path_out="onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic"):
    #quantizziamo tutto in int8 tranne i MatMul DenseReluDense/wo
    #quantize_dynamic_madlad_encoder(encoder_path, encoder_name, encoder_path_out)

    #quantizziamo i MatMul DenseReluDense/wo in int16
    quantizer = ORTQuantizer.from_pretrained(encoder_path_out, file_name=encoder_name+"_quantized.onnx")
    operators = ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION
    denseReluNodes = get_DenseReluDense_nodes(encoder_path+"/"+encoder_name+".onnx")
    qconfig = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QDQ,
        mode=QuantizationMode.IntegerOps,  #QLinearOps non funziona
        activations_dtype=QuantType.QUInt16,
        activations_symmetric=False,
        weights_dtype=QuantType.QUInt16,  #prima era QInt8
        weights_symmetric=False,     #prima era True
        per_channel=True,
        reduce_range=True,
        nodes_to_quantize=denseReluNodes,
        #nodes_to_exclude=[],   #prima era denseReluNodes
        operators_to_quantize=['MatMul']
    )
    quantizer.quantize(save_dir=encoder_path_out, quantization_config=qconfig, use_external_data_format=False, file_suffix="int16")


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


def get_MatMul_not_costant(path):
    model = onnx.load_model(path)
    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node
    # creiamo un dizionario che associa al nome di ogni initializer del grafico le sue informazioni
    initializers_dict = {}
    for initializer in initializers:
        # Data type:
        # https://github.com/onnx/onnx/blob/rel-1.9.0/onnx/onnx.proto
        #print("Tensor information:")
        #print(f"Tensor Name: {initializer.name}, Data Type: {initializer.data_type}, Shape: {initializer.dims}")
        initializers_dict[initializer.name] = initializer

    list = []
    for node in nodes:
        if(node.op_type == "MatMul"):
            isCostant = False
            for input in node.input:
                if(input in initializers_dict):
                    isCostant = True
            if(not isCostant):
                list.append(node.name)
                print(node.name)

    return list


def convert_encoder_dense_relu_to_fp16(model_path = "onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic/encoder_model_quantized.onnx",
                                       model_path_out = "onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic/encoder_model_quantized_fp16.onnx",
                                       min_positive_val=5.96e-08,
                                       max_finite_val=65504.0):
    model = onnx.load_model(model_path)
    #inferred_model = shape_inference.infer_shapes(model)
    #shapes_info = inferred_model.graph.value_info

    #RepeatedCompositeContainer

    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node

    # creiamo un dizionario che associa al nome di ogni initializer del grafico le sue informazioni
    initializers_dict = {}
    for initializer in initializers:
        # Data type:
        # https://github.com/onnx/onnx/blob/rel-1.9.0/onnx/onnx.proto
        #print("Tensor information:")
        #print(f"Tensor Name: {initializer.name}, Data Type: {initializer.data_type}, Shape: {initializer.dims}")
        initializers_dict[initializer.name] = initializer

    # creiamo un dizionario che associa a ogni nome di ogni input del grafico una lista contenente tutti i nodi (op) che possiedono quell'input
    #(volendo posso escludere gli input presenti in initializers, se servir`a a velocizzare l' algoritmo)
    inputs_dict = {}
    for node in nodes:
        for input in node.input:
            if(input not in inputs_dict):
                inputs_dict[input] = [node]
            elif(node not in inputs_dict[input]):
                inputs_dict[input].append(node)

    #del model
    #del graph
    #del initializers

    count = 0
    count_incompatibles = 0
    skipped = []            #input di cui non si conosce la dimensione (deve essere vuoto, altrimenti l'algoritmo non funziona)
    incompatible_matmuls = []
    count_name = 0
    node_index = 0
    while(node_index < len(nodes)):   #for node in nodes
        node = nodes[node_index]
        if((node.op_type == "MatMul") and ("DenseReluDense/wo" in node.name)):
            # qui si aggiunge al grafico i Cast prima di ogni input del MatMul e dopo il MatMul, e si converte la matrice dei pesi dei MatMul in fp16
            output = node.output[0]
            input_index = 0
            for input in node.input:  #per ogni input del MatMul
                if(input in initializers_dict):  # se la matrice di input `e costante
                    #convertiamo la matrice initializer in fp16
                    tensorFp16 = convert_tensor_float_to_float16(initializers_dict[input], min_positive_val, max_finite_val)
                    initializers.remove(initializers_dict[input])
                    initializers.extend([tensorFp16])
                if(input not in initializers_dict):  # se la matrice di input non `e costante
                    #creiamo il Cast prima dell'input
                    to = {"to": onnx.TensorProto.FLOAT16}
                    cast = onnx.helper.make_node(   #si crea il cast
                                            name="Cast_custom_"+str(count_name),
                                            op_type="Cast",
                                            inputs=[input],
                                            outputs=["Cast_custom_output_" + str(count_name)],
                                            **to
                                        )
                    nodes.insert(node_index, cast)   #si aggiunge ai nodi del grafico
                    node_index = node_index+1

                    # si modifica questo input di questo nodo assegnandogli il nome dell'output del Cast appena creato
                    node.input[input_index] = "Cast_custom_output_" + str(count_name)

                    count_name = count_name + 1

                    #creiamo il Cast dopo l'output
                    to2 = {"to": onnx.TensorProto.FLOAT}
                    cast2 = onnx.helper.make_node(   #si crea il cast
                                            name="Cast_custom_"+str(count_name),
                                            op_type="Cast",
                                            inputs=[output],
                                            outputs=["Cast_custom_output_" + str(count_name)],
                                            **to2
                                        )
                    nodes.insert(node_index + 1, cast2)  # si aggiunge ai nodi del grafico (nella giusta posizione)
                    node_index = node_index + 1

                    # si trovano tutti i nodi che hanno come input l'output di questo MatMul e si modifica il loro input corrispondente
                    # sostituendolo con l'output del Cast creato da ultimo
                    if(output in inputs_dict):
                        for node2 in inputs_dict[output]:
                            index = 0
                            for input2 in node2.input:
                                if(input2 == output):
                                    break
                                index = index+1
                            node2.input[index] = "Cast_custom_output_" + str(count_name)

                    count_name = count_name + 1
                    input_index = input_index+1
            count = count+1
        node_index = node_index+1

    print("total MatMuls: "+str(count))

    # Create inputs
    input_ids = onnx.helper.make_tensor_value_info("input_ids",
                                           onnx.TensorProto.INT64,
                                           ["batch_size", "encoder_sequence_length"])
    attention_mask = onnx.helper.make_tensor_value_info("attention_mask",
                                                   onnx.TensorProto.INT64,
                                                   ["batch_size", "encoder_sequence_length"])
    # Create output
    last_hidden_state = onnx.helper.make_tensor_value_info("last_hidden_state",
                                                        onnx.TensorProto.FLOAT,
                                                        ["batch_size", "encoder_sequence_length", 1024])

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=nodes,
        name="Madlad",
        inputs=[input_ids, attention_mask],  # Graph input
        outputs=[last_hidden_state],  # Graph output
        initializer=initializers,
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="nie")
    model_def.opset_import[0].version = 17

    onnx.save_model(model_def, model_path_out)
    onnx.checker.check_model(model_path_out)


def quantize_dynamic_whisper():
    encoder_path = "onnx/WhisperOptimum/Optimized/encoder_model.onnx"
    path_prep = "onnx/WhisperOptimum/Optimized/DynamicQuantized/PreProcess"
    path_out = "onnx/WhisperOptimum/Optimized/DynamicQuantized"
    encoder_name = "Whisper_encoder"
    quant_pre_process(encoder_path, path_prep+"/"+encoder_name+".onnx")
    #quantize_dynamic(Path(encoder_path_prep), Path(encoder_path_out), per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8)
    quantizer = ORTQuantizer.from_pretrained(path_prep, file_name=encoder_name+".onnx")
    operators = ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION
    qconfig = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.IntegerOps,  #QLinearOps non funziona
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=False,
        weights_dtype=QuantType.QUInt8,  #prima era QInt8
        weights_symmetric=False,     #prima era True
        per_channel=True,
        reduce_range=True,
        #nodes_to_quantize=[],
        #nodes_to_exclude=[],
        operators_to_quantize=operators,
    )
    quantizer.quantize(save_dir=path_out, quantization_config=qconfig, use_external_data_format=False)

    decoder_path = "onnx/WhisperOptimum/Optimized/decoder_with_past_model.onnx"
    decoder_name = "Whisper_decoder"
    quant_pre_process(decoder_path, path_prep+"/"+decoder_name+".onnx")
    #quantize_dynamic(Path(decoder_path_prep), Path(decoder_path_out), per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8)
    quantizer = ORTQuantizer.from_pretrained(path_prep, file_name=decoder_name+".onnx")
    qconfig = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.IntegerOps,  #QLinearOps non funziona
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=False,
        weights_dtype=QuantType.QUInt8,  #prima era QInt8
        weights_symmetric=False,     #prima era True
        per_channel=True,
        reduce_range=True,
        #nodes_to_quantize=[],
        #nodes_to_exclude=[],
        operators_to_quantize=operators,
    )
    quantizer.quantize(save_dir=path_out, quantization_config=qconfig, use_external_data_format=False)


def quantize_dynamic_nllb():
    #quantizzazione dell'encoder
    encoder_path = "onnx/NLLBOptimum/encoder_model.onnx"
    encoder_pre_process_path = "onnx/NLLBOptimum/Optimized/Quantized/PreProcessed/NLLB_encoder_preProcessed.onnx"
    encoder_quantized_path = "onnx/NLLBOptimum/Optimized/Quantized/NLLB_encoder_quantized.onnx"
    quant_pre_process(input_model_path=encoder_path,
                                     output_model_path=encoder_pre_process_path,skip_onnx_shape=False,skip_symbolic_shape=False)
    #dense_relu_nodes = get_DenseReluDense_nodes_nllb(encoder_pre_process_path)
    attention_matmul_nodes = get_Attention_MatMul_nodes_nllb(encoder_pre_process_path)
    quantize_dynamic(Path(encoder_pre_process_path), Path(encoder_quantized_path),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=attention_matmul_nodes,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    #"DefaultTensorType": onnx.TensorProto.FLOAT,
                                    "MatMulConstBOnly": True})

    #quantizzazione del cache initializer
    cache_initializer_path = "onnx/NLLBOptimum/Optimized/NLLB_cache_initializer.onnx"
    cache_initializer_pre_process_path = "onnx/NLLBOptimum/Optimized/Quantized/PreProcessed/NLLB_cache_initializer_preProcessed.onnx"
    cache_initializer_quantized_path = "onnx/NLLBOptimum/Optimized/Quantized/NLLB_cache_initializer_quantized.onnx"
    quant_pre_process(input_model_path=cache_initializer_path,
                                     output_model_path=cache_initializer_pre_process_path,skip_onnx_shape=False,skip_symbolic_shape=False)
    quantize_dynamic(Path(cache_initializer_pre_process_path), Path(cache_initializer_quantized_path),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    #"DefaultTensorType": onnx.TensorProto.FLOAT,
                                    "MatMulConstBOnly": True})

    #quantizzazione del decoder (non si fa il preprocess perch`e supera i 2GB (2.8GB), ma se voglio proprio farla in futuro basta separare l' lm-head del decoder,
    #fare il pre process e poi riunire lm-head al decoder)
    decoder_path = "onnx/NLLBOptimum/decoder_with_past_model.onnx"
    decoder_pre_process_path = "onnx/NLLBOptimum/Optimized/Quantized/PreProcessed/NLLB_decoder_preProcessed.onnx"
    decoder_quantized_path = "onnx/NLLBOptimum/Optimized/Quantized/NLLB_decoder_quantized.onnx"
    #quant_pre_process(input_model_path=decoder_path,
    #                                 output_model_path=decoder_pre_process_path, skip_onnx_shape=False, skip_symbolic_shape=True)
    quantize_dynamic(Path(decoder_path), Path(decoder_quantized_path),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    #"DefaultTensorType": onnx.TensorProto.FLOAT,
                                    "MatMulConstBOnly": True})

def get_DenseReluDense_nodes_nllb(path):
    model = onnx.load_model(path)

    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node

    list = []
    for node in nodes:
        if((("/fc1/" in node.name) or ("/fc2/" in node.name)) and ("MatMul" in node.name)):
            list.append(node.name)
            print(node.name)
    print("")
    print("")
    return list


def get_Attention_MatMul_nodes_nllb(path):
    model = onnx.load_model(path)

    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node

    list = []
    for node in nodes:
        if((("/self_attn/" in node.name) and ("_proj/MatMul" in node.name) and ("out_proj" not in node.name))):
            list.append(node.name)
            print(node.name)
    print("")
    print("")
    return list



def generate_whisper_encoder_initializer():  #(separate_whisper_pre_ops() is equivalent but better)
    olive_whisper_path = "onnx/Whisper/whisper_small_cpu_fp32_multi.onnx"
    path_out = "onnx/WhisperOptimum/Whisper_encoder_initializer.onnx"
    model = onnx.load_model(olive_whisper_path)
    #inferred_model = shape_inference.infer_shapes(model)
    #shapes_info = inferred_model.graph.value_info

    #RepeatedCompositeContainer

    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node

    # creiamo un dizionario che associa al nome di ogni initializer del grafico le sue informazioni
    initializers_dict = {}
    for initializer in initializers:
        initializers_dict[initializer.name] = initializer

    # creiamo un dizionario che associa a ogni nome di ogni input del grafico una lista contenente tutti i nodi (op) che possiedono quell'input
    #(volendo posso escludere gli input presenti in initializers, se servir`a velocizzare l' algoritmo)
    inputs_dict = {}
    for node in nodes:
        for input in node.input:
            if(input not in inputs_dict):
                inputs_dict[input] = [node]
            elif(node not in inputs_dict[input]):
                inputs_dict[input].append(node)

    del model
    del graph
    del initializers

    count = 0
    count_incompatibles = 0
    skipped = []            #input di cui non si conosce la dimensione (deve essere vuoto, altrimenti l'algoritmo non funziona)
    incompatible_matmuls = []
    count_output_shape = 0
    node_index = 0
    nodes_list = []
    for node in nodes:   #for node in nodes
        if(node.name == "BeamSearch_node"):
            break
        else:
            print(node.name+", "+node.op_type)
            nodes_list.append(node)

    # Create inputs
    audio_pcm = onnx.helper.make_tensor_value_info("audio_pcm",
                                                        onnx.TensorProto.FLOAT,
                                                        [1, "sample_len"])
    # Create output
    log_mel = onnx.helper.make_tensor_value_info("log_mel",
                                                       onnx.TensorProto.FLOAT,
                                                       [1, "encode_sequence_length", 3000])

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=nodes_list,
        name="Whisper_encoder_initializer",
        inputs=[audio_pcm],  # Graph input
        outputs=[log_mel],  # Graph output
        initializer=[]
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="nie")
    model_def.opset_import[0].version = 17

    onnx.checker.check_model(model_def)
    model_def = onnx.shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, path_out)





def quantize_cache_initializer():
    decoder_path = "onnx/Madlad/Optimum_Cache_Optimized"
    decoder_path_out = "onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic/cache_initializer"
    #quantize_dynamic(Path(decoder_path), Path(decoder_path_out), extra_options={"EnableSubgraph": True})
    quantizer = ORTQuantizer.from_pretrained(decoder_path, file_name="Cache_initializer.onnx")
    operators = ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION
    qconfig = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.IntegerOps,  #QLinearOps non funziona
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=False,
        weights_dtype=QuantType.QUInt8,  #prima era QInt8
        weights_symmetric=False,  #prima era True
        per_channel=True,
        reduce_range=True,
        #nodes_to_quantize=[],
        #nodes_to_exclude=[],
        operators_to_quantize=operators,
    )
    quantizer.quantize(save_dir=decoder_path_out, quantization_config=qconfig)


def weight_compression():
    import onnx_tool
    modelpath = "onnx/Madlad/Optimum_Cache/decoder_model_merged.onnx"
    m = onnx_tool.Model(modelpath)
    g = m.graph

    def tofloat16():
        for key in g.initials:
            tensor = g.tensormap[key]
            raw = tensor.numpy
            tensor.numpy = raw.astype(numpy.float16)
        m.save_model(m.modelname + '-fp16.onnx')

    def quantize_sym():
        from onnx_tool.quantization import graph_quantize
        for key in g.initials:
            graph_quantize(g, key, block=-1, type='sym', bits=8)
        m.save_model(m.modelname + '-8bits-sym-default.onnx')

    def quantize_asym():
        from onnx_tool.quantization import graph_quantize
        for key in g.initials:
            graph_quantize(g, key, block=-1, type='asym', bits=8)
        m.save_model(m.modelname + '-8bits-asym-default.onnx')

    def quantize_sym_b32():
        from onnx_tool.quantization import graph_quantize
        for key in g.initials:
            graph_quantize(g, key, block=32, type='sym', bits=8)
        m.save_model(m.modelname + '-8bits-sym-b32.onnx')

    def quantize_4bits_sym_b32():
        from onnx_tool.quantization import graph_quantize
        for key in g.initials:
            graph_quantize(g, key, block=32, type='sym', bits=4)
        m.save_model(m.modelname + '-4bits-sym-b32.onnx')

    quantize_4bits_sym_b32()


def quantize_dynamic_whisper_olive():
    encoder_path = "onnx/Whisper/Optimized/whisper_small_cpu_fp32_optimized_multi.onnx"
    path_prep = "onnx/Whisper/Optimized"
    path_out = "onnx/Whisper/OptimumQuantized"
    encoder_name = "whisper_small_cpu_fp32_optimized_multi"

    path = "onnx/Whisper/Optimized/whisper_small_cpu_fp32_optimized_multi.onnx"
    path_out = "onnx/Whisper/OptimumQuantized/whisper_small_cpu_int8_optimized_multi.onnx"
    operators = ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION

    #quant_pre_process(encoder_path, path_prep+"/"+encoder_name+".onnx")
    quantize_dynamic(Path(path), Path(path_out),
                     per_channel=False,
                     reduce_range=False,
                     weight_type=QuantType.QUInt8,
                     #op_types_to_quantize=operators,
                     extra_options={
                         "WeightSymmetric": False,
                         "ActivationSymmetric": False,
                         "EnableSubgraph": True,
                         #"ForceQuantizeNoInputCheck": True,
                         #"MatMulConstBOnly": False
                     })
    '''quantizer = ORTQuantizer.from_pretrained(path_prep, file_name=encoder_name+".onnx")
    qconfig = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.IntegerOps,  #QLinearOps non funziona
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=False,
        weights_dtype=QuantType.QUInt8,  #prima era QInt8
        weights_symmetric=False,     #prima era True
        per_channel=True,
        reduce_range=True,
        #nodes_to_quantize=[],
        #nodes_to_exclude=[],
        operators_to_quantize=operators,
    )
    quantize_custom(quantizer, save_dir=path_out, quantization_config=qconfig, use_external_data_format=False, has_subgraphs=True)'''

def check_whisper_quantization():
    #encoder_path = "onnx/WhisperOptimum/IntelQuantized/int8Dynamic/encoder_model_int8.onnx"
    #decoder_path = "onnx/WhisperOptimum/IntelQuantized/int8Dynamic/decoder_with_past_model_int8.onnx"
    encoder_path = "onnx/WhisperOptimum/IntelQuantized/int8Static/encoder_model_int8_s.onnx"
    decoder_path = "onnx/WhisperOptimum/IntelQuantized/int8Static/decoder_with_past_model.onnx"

    encoder_model = onnx.load_model(encoder_path)
    decoder_model = onnx.load_model(decoder_path)

    encoder_nodes = encoder_model.graph.node
    decoder_nodes = decoder_model.graph.node

    print("Encoder MatMul not quantized:")
    for node in encoder_nodes:
        if node.op_type == "MatMul":
            print(node.name)

    print('')
    print("Decoder MatMul not quantized:")
    for node in decoder_nodes:
        if node.op_type == "MatMul":
            print(node.name)




def adapt_whisper_to_walkie_talkie_mode(model_path="onnx/Whisper/Optimized/Quality/whisper_cpu_int8_quality_fast.onnx", model_path_out = "onnx/Whisper/Optimized/Quality/whisper_cpu_int8_quality_fast_final.onnx", preops_path_out = "onnx/Whisper/Optimized/Quality/whisper_initializer.onnx", post_ops_path_out = "onnx/Whisper/Optimized/Quality/whisper_detokenizer.onnx"):
    model = onnx.load_model(model_path)
    #RepeatedCompositeContainer

    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node

    #del model
    #del graph
    #del initializers

    pre_ops_nodes = separate_whisper_pre_ops(model_path, preops_path_out)
    post_ops_nodes = separate_whisper_bpe_decoder(model_path, post_ops_path_out)

    # Add score output to BeamSearch op
    for node in nodes:
        if(node.op_type == "BeamSearch"):
            output = node.output
            output.append("sequences_scores")
            print(output)

    # Remove all pre ops and post ops nodes
    for node in pre_ops_nodes:
        nodes.remove(node)
    for node in post_ops_nodes:
        nodes.remove(node)

    # Substitute log_mel to pcm_audio input
    inputs = graph.input
    input_index = 0
    while(input_index < len(inputs)):
        input = inputs[input_index]
        if(input.name == "audio_pcm"):
            inputs.remove(input)
            log_mel = onnx.helper.make_tensor_value_info("log_mel",
                                                        onnx.TensorProto.FLOAT,
                                                        ["batch_size", "encode_sequence_length", 3000])
            inputs.insert(input_index, log_mel)
        input_index = input_index+1

    # Set outputs (sostitute str with sequences and add sequences_scores output)
    sequences = onnx.helper.make_tensor_value_info("sequences",
                                                        onnx.TensorProto.INT32,
                                                        ["batch_size", "num_return_sequences", "sequence_length"])
    sequence_score = onnx.helper.make_tensor_value_info("sequences_scores",
                                                        onnx.TensorProto.FLOAT,
                                                        ["batch_size", "num_return_sequences"])
    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=nodes,
        name="Whisper",
        inputs=inputs,  # Graph input
        outputs=[sequences, sequence_score],  # Graph output
        initializer=graph.initializer
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="nie")
    model_def.opset_import[0].version = 17

    onnx.save_model(model_def, model_path_out, save_as_external_data=False)



#the model is slower than the model with separated preops, so I will use that
def adapt_whisper_to_walkie_talkie_mode_with_integrated_preops(model_path="onnx/Whisper/Optimized/Quality/whisper_cpu_int8_quality_fast.onnx", model_path_out = "onnx/Whisper/Optimized/Quality/whisper_cpu_int8_quality_fast_final.onnx", preops_path_out = "onnx/Whisper/Optimized/Quality/whisper_initializer.onnx", post_ops_path_out = "onnx/Whisper/Optimized/Quality/whisper_detokenizer.onnx"):
    model = onnx.load_model(model_path)
    #RepeatedCompositeContainer

    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node

    #del model
    #del graph
    #del initializers

    post_ops_nodes = separate_whisper_bpe_decoder(model_path, post_ops_path_out)

    # Add score output to BeamSearch op
    for node in nodes:
        if(node.op_type == "BeamSearch"):
            output = node.output
            output.append("sequences_scores")
            print(output)

    # Remove all post ops nodes
    for node in post_ops_nodes:
        nodes.remove(node)

    # Insert beam_size in pcm_audio input
    inputs = graph.input
    input_index = 0
    while(input_index < len(inputs)):
        input = inputs[input_index]
        if(input.name == "audio_pcm"):
            inputs.remove(input)
            log_mel = onnx.helper.make_tensor_value_info("audio_pcm",
                                                        onnx.TensorProto.FLOAT,
                                                        ["batch_size", "sample_len"])
            inputs.insert(input_index, log_mel)
        input_index = input_index+1

    # Set outputs (sostitute str with sequences and add sequences_scores output)
    sequences = onnx.helper.make_tensor_value_info("sequences",
                                                        onnx.TensorProto.INT32,
                                                        ["batch_size", "num_return_sequences", "sequence_length"])
    sequence_score = onnx.helper.make_tensor_value_info("sequences_scores",
                                                        onnx.TensorProto.FLOAT,
                                                        ["batch_size", "num_return_sequences"])
    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=nodes,
        name="Whisper",
        inputs=inputs,  # Graph input
        outputs=[sequences, sequence_score],  # Graph output
        initializer=graph.initializer,
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="nie")
    model_def.opset_import[0].version = 17

    onnx.save_model(model_def, model_path_out, save_as_external_data=False)



def separate_whisper_pre_ops(model_path="onnx/Whisper/Optimized/Quality/whisper_cpu_int8_quality_fast.onnx", model_path_out = "onnx/Whisper/Optimized/Quality/whisper_pre_ops.onnx"):
    model = onnx.load_model(model_path)
    #inferred_model = shape_inference.infer_shapes(model)
    #shapes_info = inferred_model.graph.value_info

    #RepeatedCompositeContainer

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
    nodes_to_remove = []
    initializers_to_keep = []

    for node in nodes:
        if(node.op_type == "BeamSearch"):
            nodes_to_remove.append(node)
        if(node.op_type == "Cast"):
            nodes_to_remove.append(node)
        if(node.name == "BpeDecoder_1"):
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        nodes.remove(node)

    for node in nodes:
        for input in node.input:
            if(input in initializers_dict):
                if(input not in initializers_to_keep):
                    initializers_to_keep.append(initializers_dict[input])

    # Set input
    audio_pcm = onnx.helper.make_tensor_value_info("audio_pcm",
                                                        onnx.TensorProto.FLOAT,
                                                        [1, "sample_len"])   #"batch_size" al posto di 1
    # Add sequences_scores output and sostitute str with sequences
    log_mel = onnx.helper.make_tensor_value_info("log_mel",
                                                        onnx.TensorProto.FLOAT,
                                                        [1, "encode_sequence_length", 3000])
    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=nodes,
        name="WhisperModelInitializer",
        inputs=[audio_pcm],  # Graph input
        outputs=[log_mel],  # Graph output
        initializer=initializers_to_keep,
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="nie")
    model_def.opset_import[0].version = 17

    onnx.save_model(model_def, model_path_out, save_as_external_data=False)
    onnx.shape_inference.infer_shapes_path(model_path_out,
                                           model_path_out) #strict_mode=True, data_prop=True
    onnx.checker.check_model(model_path_out)

    return nodes


def separate_whisper_bpe_decoder(model_path="onnx/Whisper/Optimized/Quality/whisper_cpu_int8_quality_fast.onnx", model_path_out = "onnx/Whisper/Optimized/Quality/whisper_detokenizer.onnx"):
    model = onnx.load_model(model_path)
    #inferred_model = shape_inference.infer_shapes(model)
    #shapes_info = inferred_model.graph.value_info

    #RepeatedCompositeContainer

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

    nodes_to_add = []
    initializers_to_keep = []

    for node in nodes:
        if(node.name == "BpeDecoder_1"):
            nodes_to_add.append(node)
        if(node.op_type == "Cast"):
            nodes_to_add.append(node)

    for node in nodes_to_add:
        for input in node.input:
            if(input in initializers_dict):
                if(input not in initializers_to_keep):
                    initializers_to_keep.append(initializers_dict[input])

    # Set input
    sequences = onnx.helper.make_tensor_value_info("sequences",
                                                        onnx.TensorProto.INT32,
                                                        [1, 1, "sequence_length"])
    # Set output
    str = onnx.helper.make_tensor_value_info("str",
                                                        onnx.TensorProto.STRING,
                                                        [1, "text"])
    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=nodes_to_add,
        name="WhisperDetokenizer",
        inputs=[sequences],  # Graph input
        outputs=[str],  # Graph output
        initializer=initializers_to_keep,
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="nie")
    model_def.opset_import[0].version = 17

    onnx.save_model(model_def, model_path_out, save_as_external_data=False)

    return nodes_to_add

def has_same_value(val_one,val_two):
    if val_one.raw_data == val_two.raw_data:
        return True
    else:
        return False



def remove_shared_weight(model_path, output_path):
    model=onnx.load(model_path)
    onnx_model=OnnxModel(model)

    count = len(model.graph.initializer)
    print("number of initializers: " + str(count))
    same = [-1] * count
    for i in range(count - 1):
        if same[i] >= 0:
            continue

        if("onnx::MatMul_" in model.graph.initializer[i].name):
            initializer = model.graph.initializer[i]
            print("checking " + model.graph.initializer[i].name)

        for j in range(i+1, count):
            if has_same_value(model.graph.initializer[i], model.graph.initializer[j]):
                same[j] = i

    for i in range(count):
        if same[i] >= 0:
            print("removing node: " + str(model.graph.initializer[same[i]].name))
            onnx_model.replace_input_of_all_nodes(model.graph.initializer[i].name, model.graph.initializer[same[i]].name)

    onnx_model.update_graph()

    onnx_model.save_model_to_file(output_path)

    model=onnx.load(model_path)
    model2=onnx.load(output_path)
    model_info.print_simplifying_info(model, model2)



def create_nllb_embed_and_lm_head(decoder_path, output_path):
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
    for node in nodes:
        # aggiungiamo i nodi e gli initializers richiesti per nllb_embed
        if(node.name == "/model/decoder/Reshape"):
            embed_nodes.append(node)
            embed_and_lm_head_initializers.append(initializers_dict[node.input[1]])
        if(node.name == "/model/decoder/embed_tokens/Gather"):
            node.output.pop()
            node.output.append("embed_matrix")
            embed_nodes.append(node)

        # aggiungiamo i nodi e gli initializers richiesti per nllb_lm_head
        if(node.name == "Transpose_1366"):
            lm_head_nodes.append(node)
        if(node.name == "/lm_head/MatMul"):
            node.input.pop(0)
            node.input.insert(0, "pre_logits")
            lm_head_nodes.append(node)

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
    #onnx.shape_inference.infer_shapes_path(output_path, output_path)
    onnx.checker.check_model(output_path)



def create_nllb_embed_and_lm_head2(decoder_path, output_path):
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

    embed_and_lm_head_nodes = []
    embed_and_lm_head_initializers = []

    # aggiungiamo i nodi e gli initializers richiesti per nllb_embed e nllb_lm_head
    for node in nodes:
        # aggiungiamo i nodi e gli initializers richiesti per nllb_embed
        if(node.name == "/model/decoder/Reshape"):
            embed_and_lm_head_nodes.append(node)
            embed_and_lm_head_initializers.append(initializers_dict[node.input[1]])
        if(node.name == "/model/decoder/embed_tokens/Gather"):
            node.output.pop()
            node.output.append("embed_matrix")
            embed_and_lm_head_nodes.append(node)

        # aggiungiamo i nodi e gli initializers richiesti per nllb_lm_head
        if(node.name == "Transpose_1366"):
            embed_and_lm_head_nodes.append(node)
        if(node.name == "/lm_head/MatMul"):
            node.input.pop(0)
            node.input.insert(0, "pre_logits")
            embed_and_lm_head_nodes.append(node)

    #embed_and_lm_head_initializers.append(initializers_dict["model.shared.weight"])
    attributes = {"value": onnx.helper.make_tensor("model.shared.weight",
                                                          onnx.TensorProto.FLOAT,
                                                          [256206, 1024],
                                                           vals=initializers_dict["model.shared.weight"].raw_data, raw=True)}
    constant_node = onnx.helper.make_node(name="Constant_3",
                                op_type="Constant",
                                inputs=[],
                                outputs=["model.shared.weight"],
                                **attributes)
    embed_and_lm_head_nodes.insert(0, constant_node)

    # Creazione del grafico di nllb_embed_and_lm_head
    #create inputs
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
        nodes=embed_and_lm_head_nodes,
        name="nllb_embed",
        inputs=[input_ids, pre_logits],  # Graph input
        outputs=[embed_matrix, logits],  # Graph output
        initializer=embed_and_lm_head_initializers,
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(nllb_embed_and_lm_head_graph, producer_name="nie")
    model_def.opset_import[0].version = 17
    op = model_def.opset_import
    print(op)

    onnx.save_model(model_def, output_path, save_as_external_data=False)
    #onnx.shape_inference.infer_shapes_path(output_path, output_path)
    onnx.checker.check_model(output_path)


def create_nllb_embed_and_lm_head3(decoder_path, output_path):
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



def fix_nllb_embed_and_lm_head():
    model = onnx.load_model("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/nllb_embed_and_lm_head_if2.onnx")
    initializers = model.graph.initializer
    nodes = model.graph.node
    embed_nodes = None

    # creiamo un dizionario che associa al nome di ogni initializer del grafico le sue informazioni
    initializers_dict = {}
    for initializer in initializers:
        initializers_dict[initializer.name] = initializer
        print(initializer.name)

    print('')

    embed_and_lm_head_nodes = []
    embed_and_lm_head_initializers = []

    #inseriamo in embed il transpose per model.shared.weight_transposed
    for node in nodes:
        if(node.op_type == "If"):
            embed_nodes = node.attribute[0].g.node
            perm = {"perm": [1, 0]}
            transpose2 = onnx.helper.make_node(   #si crea il transpose
                                    name="Transpose_1",
                                    op_type="Transpose",
                                    inputs=["model.shared.weight_transposed_quantized"],
                                    outputs=["model.shared.weight_quantized"],
                                    **perm
                                )
            embed_nodes.insert(0, transpose2)   #si aggiunge ai nodi del grafico

    #spostiamo l' initializer model.shared.weight_transposed da lm_head al modello padre
    for node in nodes:
        if(node.op_type == "If"):
            lm_head_graph = node.attribute[1].g
            lm_head_initializers = node.attribute[1].g.initializer
            initializers_lm_head_dict = {}
            for initializer in lm_head_initializers:
                initializers_lm_head_dict[initializer.name] = initializer
                print(initializer.name)
            initializers.append(initializers_lm_head_dict["model.shared.weight_transposed_quantized"])
            lm_head_initializers.remove(initializers_lm_head_dict["model.shared.weight_transposed_quantized"])

    initializers.remove(initializers_dict["model.shared.weight_quantized"])

    onnx_model = OnnxModel(model)
    onnx_model.update_graph()

    out_path = "onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/nllb_embed_and_lm_head_fixed_quantized.onnx"
    onnx_model.save_model_to_file(out_path)
    onnx.checker.check_model(out_path)
    onnx.shape_inference.infer_shapes_path(out_path, out_path)





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


def quantize_nllb_final():
    #optimization
    '''sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = "onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/Optimized/encoder_model.onnx"
    onnxruntime.InferenceSession("onnx/NLLBOptimum/Optimized/ReducedRAM/encoder_model.onnx", sess_options)

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = "onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/Optimized/decoder_model.onnx"
    onnxruntime.InferenceSession("onnx/NLLBOptimum/Optimized/ReducedRAM/decoder_model.onnx", sess_options)'''

    '''sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = "onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/Optimized/nllb_embed_and_lm_head_fixed.onnx"
    onnxruntime.InferenceSession("onnx/NLLBOptimum/Optimized/ReducedRAM/nllb_embed_and_lm_head_fixed.onnx", sess_options)'''

    #pre processing
    '''quant_pre_process(input_model_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/Optimized/encoder_model.onnx",
                                      output_model_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/PreProcess/encoder_model.onnx", skip_symbolic_shape=True)
    quant_pre_process(input_model_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/Optimized/decoder_model.onnx",
                                      output_model_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/PreProcess/decoder_model.onnx", skip_symbolic_shape=True)'''
    '''quant_pre_process(input_model_path="onnx/NLLBOptimum/Optimized/ReducedRAM/nllb_embed_and_lm_head_fixed.onnx",   #onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/Optimized/nllb_embed_and_lm_head_if.onnx
                                      output_model_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/PreProcess/nllb_embed_and_lm_head_fixed.onnx", skip_symbolic_shape=True, skip_optimization=True)   # save_as_external_data=True, all_tensors_to_one_file=True'''

    #quantization
    '''nodes_to_exclude = get_Attention_MatMul_nodes_nllb("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/PreProcess/encoder_model.onnx")
    quantize_dynamic(Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/PreProcess/encoder_model.onnx"), Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/encoder_model_quality.onnx"),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=nodes_to_exclude,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})'''

    '''quantize_dynamic(Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/PreProcess/decoder_model.onnx"), Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/decoder_model.onnx"),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})'''

    quantize_dynamic(Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/PreProcess/nllb_embed_and_lm_head_fixed.onnx"), Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/nllb_embed_and_lm_head_fixed.onnx"),
                     per_channel=False, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=['MatMul', 'Conv', 'Reshape', 'Transpose', 'Gather'],   #['MatMul', 'Conv', 'Reshape', 'Transpose', 'Gather']
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": True,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": False})

def quantize_nllb_4bit():
    #quantization of encoder
    model_fp32_path="onnx/NLLBOptimum/Optimized/ReducedRAM/encoder_model.onnx"
    model_int4_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit_HQQ/nllb_encoder_4bit.onnx"

    quant_config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
      block_size=128, # 2's exponential and >= 16
      is_symmetric=True, # if true, quantize to Int4. otherwise, quantize to uint4.
      accuracy_level=4, # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35,
      quant_format=quant_utils.QuantFormat.QOperator)

    quant_config_hqq = matmul_4bits_quantizer.HQQWeightOnlyQuantConfig()

    model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
      model,
      nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
      algo_config=quant_config_hqq,)
    quant.process()
    quant.model.save_model_to_file(model_int4_path, False)

    #quantization of decoder
    model_fp32_path="onnx/NLLBOptimum/Optimized/ReducedRAM/decoder_model.onnx"
    model_int4_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit_HQQ/nllb_decoder_4bit.onnx"

    model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
      model,
      nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
      algo_config=quant_config_hqq,)
    quant.process()
    quant.model.save_model_to_file(model_int4_path, False)

    #quantization of cache init
    model_fp32_path="onnx/NLLBOptimum/Optimized/NLLB_cache_initializer.onnx"
    model_int4_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit_HQQ/nllb_cache_initializer_4bit.onnx"

    model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
      model,
      nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
      algo_config=quant_config_hqq,)
    quant.process()
    quant.model.save_model_to_file(model_int4_path, False)


    #quantization of embedAndLmHead
    model_fp32_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/PreProcess/nllb_embed_and_lm_head_if2.onnx"   #onnx/NLLBOptimum/Optimized/ReducedRAM/nllb_embed_and_lm_head_fixed.onnx
    model_int4_path="onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit_HQQ/nllb_embed_and_lm_head_4bit.onnx"

    model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
      model,
      nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
      algo_config=quant_config_hqq,)
    quant.process()
    quant.model.save_model_to_file(model_int4_path, False)

    quantize_dynamic(Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit_HQQ/nllb_embed_and_lm_head_4bit.onnx"), Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/4bit_HQQ/nllb_embed_and_lm_head_4bit.onnx"),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=['Gather'],   #['MatMul', 'Conv', 'Reshape', 'Transpose', 'Gather']
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": True,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": False})



def quantize_whisper_final():
    #optimization
    '''sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = "onnx/WhisperOptimum/ReducedRam/Quantized/Optimized/Whisper_encoder.onnx"
    onnxruntime.InferenceSession("onnx/WhisperOptimum/ReducedRam/Whisper_encoder.onnx", sess_options)

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = "onnx/WhisperOptimum/ReducedRam/Quantized/Optimized/Whisper_decoder.onnx"
    onnxruntime.InferenceSession("onnx/WhisperOptimum/ReducedRam/Whisper_decoder.onnx", sess_options)

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = "onnx/WhisperOptimum/ReducedRam/Quantized/Optimized/Whisper_cache_initializer.onnx"
    onnxruntime.InferenceSession("onnx/WhisperOptimum/ReducedRam/Whisper_cache_initializer.onnx", sess_options)'''

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = "onnx/WhisperOptimum/ReducedRam/Quantized/Optimized/Whisper_cache_initializer_batch.onnx"
    onnxruntime.InferenceSession("onnx/WhisperOptimum/ReducedRam/Whisper_cache_initializer_batch.onnx", sess_options)

    #pre processing
    '''quant_pre_process(input_model_path="onnx/WhisperOptimum/ReducedRam/Quantized/Optimized/Whisper_encoder.onnx",
                                      output_model_path="onnx/WhisperOptimum/ReducedRam/Quantized/PreProcessed/Whisper_encoder.onnx", skip_symbolic_shape=False)
    quant_pre_process(input_model_path="onnx/WhisperOptimum/ReducedRam/Quantized/Optimized/Whisper_decoder.onnx",
                                      output_model_path="onnx/WhisperOptimum/ReducedRam/Quantized/PreProcessed/Whisper_decoder.onnx", skip_symbolic_shape=False)
    quant_pre_process(input_model_path="onnx/WhisperOptimum/ReducedRam/Quantized/Optimized/Whisper_cache_initializer.onnx", 
                                      output_model_path="onnx/WhisperOptimum/ReducedRam/Quantized/PreProcessed/Whisper_cache_initializer.onnx")'''
    quant_pre_process(input_model_path="onnx/WhisperOptimum/ReducedRam/Quantized/Optimized/Whisper_cache_initializer_batch.onnx",
                                      output_model_path="onnx/WhisperOptimum/ReducedRam/Quantized/PreProcessed/Whisper_cache_initializer_batch.onnx")

    #quantization
    '''attention_nodes = get_Attention_MatMul_nodes_whisper("onnx/WhisperOptimum/ReducedRam/Quantized/PreProcessed/Whisper_encoder.onnx")
    quantize_dynamic(Path("onnx/WhisperOptimum/ReducedRam/Quantized/PreProcessed/Whisper_encoder.onnx"), Path("onnx/WhisperOptimum/ReducedRam/Quantized/Whisper_encoder.onnx"),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,    #["MatMul", "Gemm", "Gather", "Attention", "LSTM", "Transpose", "EmbedLayerNormalization" ],
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})

    quantize_dynamic(Path("onnx/WhisperOptimum/ReducedRam/Quantized/PreProcessed/Whisper_decoder.onnx"), Path("onnx/WhisperOptimum/ReducedRam/Quantized/Whisper_decoder.onnx"),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})

    quantize_dynamic(Path("onnx/WhisperOptimum/ReducedRam/Quantized/PreProcessed/Whisper_cache_initializer.onnx"), Path("onnx/WhisperOptimum/ReducedRam/Quantized/Whisper_cache_initializer.onnx"),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})'''

    quantize_dynamic(Path("onnx/WhisperOptimum/ReducedRam/Quantized/PreProcessed/Whisper_cache_initializer_batch.onnx"), Path("onnx/WhisperOptimum/ReducedRam/Quantized/Whisper_cache_initializer_batch.onnx"),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})


def get_Attention_MatMul_nodes_whisper(path):
    model = onnx.load_model(path)
    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node

    list = []
    for node in nodes:
        if((("/self_attn/" in node.name) and ("_proj/MatMul" in node.name) and ("out_proj" not in node.name))):
            list.append(node.name)
            print(node.name)
    print("")
    print("")
    return list


def create_madlad_embed(decoder_path, output_path):
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
    model_def.opset_import[0].version = 17

    onnx.save_model(model_def, output_path, save_as_external_data=False)
    onnx.shape_inference.infer_shapes_path(output_path, output_path)
    onnx.checker.check_model(output_path)



def adapt_madlad_to_embed(encoder_path, decoder_path, encoder_path_out, decoder_path_out):
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


def quantize_madlad_final():
    #optimization and pre processing (replicated by me)
    #symbolic_shape_infer.shape_inference.infer_shapes_path("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx", "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx")
    #symbolic_shape_infer.SymbolicShapeInference.infer_shapes()

    '''optimizer = optimize_model("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx", model_type="t5", num_heads=16, hidden_size=2048, optimization_options=None, only_onnxruntime=True, opt_level=2, verbose=False)
    optimizer.topological_sort() # Topologically sort the graph at the end since previous optimizations may have broken it
    model_optimized = optimizer.model
    onnx.save_model(model_optimized, "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/PreProcessed/encoder_model.onnx", save_as_external_data=True)

    optimizer = optimize_model("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx", model_type="t5", num_heads=16, hidden_size=2048, optimization_options=None, only_onnxruntime=True, opt_level=2, verbose=False)
    optimizer.topological_sort() # Topologically sort the graph at the end since previous optimizations may have broken it
    model_optimized = optimizer.model
    onnx.save_model(model_optimized, "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/PreProcessed/decoder_model.onnx", save_as_external_data=True)

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Optimized/madlad_embed.onnx"
    onnxruntime.InferenceSession("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx", sess_options)'''

    #pre processing
    '''shape_inference.infer_shapes_path("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/PreProcessed/encoder_model.onnx")
    shape_inference.infer_shapes_path("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/PreProcessed/decoder_model.onnx")
    quant_pre_process(input_model_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Optimized/madlad_embed.onnx",   #onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/Optimized/nllb_embed_and_lm_head_if.onnx
                                      output_model_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/PreProcessed/madlad_embed.onnx")'''

    #quantization
    nodes_to_exclude = get_DenseReluDense_nodes("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/PreProcessed/encoder_model.onnx")
    quantize_dynamic(Path("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/PreProcessed/encoder_model.onnx"), Path("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/encoder_model_quality2.onnx"),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,    #["MatMul", "Gemm", "Gather", "Attention", "LSTM", "Transpose", "EmbedLayerNormalization" ],
                     use_external_data_format = False, nodes_to_exclude=nodes_to_exclude,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})

    '''quantize_dynamic(Path("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/PreProcessed/decoder_model.onnx"), Path("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/decoder_model.onnx"),
                     per_channel=False, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})

    quantize_dynamic(Path("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/PreProcessed/madlad_embed.onnx"), Path("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/madlad_embed.onnx"),
                     per_channel=True, reduce_range=True, weight_type=QuantType.QUInt8, op_types_to_quantize=None,
                     use_external_data_format = False, nodes_to_exclude=None,
                     extra_options={"EnableSubgraph": False,
                                    "ActivationSymmetric": False,
                                    "WeightSymmetric": False,
                                    "MatMulConstBOnly": True})'''


def get_Attention_MatMul_nodes_madlad(path):
    model = onnx.load_model(path)
    graph = model.graph
    initializers = graph.initializer
    nodes = graph.node

    list = []
    for node in nodes:
        if((("/SelfAttention/" in node.name) and ("/k/" in node.name or "/q/" in node.name or "/v/" in node.name) and (node.op_type == "MatMul"))):
            list.append(node.name)
            print(node.name)
    print("")
    print("")
    return list


