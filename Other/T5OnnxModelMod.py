import logging
from typing import Optional, Union

from onnxruntime.transformers.fusion_attention import AttentionMask, FusionAttention
from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization
from onnxruntime.transformers.fusion_simplified_layernorm import FusionSimplifiedLayerNormalization, \
    FusionSkipSimplifiedLayerNormalization
from onnxruntime.transformers.fusion_skiplayernorm import FusionSkipLayerNormalization
from onnxruntime.transformers.fusion_utils import NumpyHelper
from onnxruntime.transformers.onnx_model import OnnxModel
#from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from onnxruntime.transformers.onnx_model_t5 import FusionRelativePositionBiasBlock

from BertOnnxModelMod import BertOnnxModel
from onnx import NodeProto, TensorProto, helper
import numpy as np

logger = logging.getLogger(__name__)


class FusionT5Attention(FusionAttention):
    """
    Fuse T5 Attention subgraph into one Attention node.
    """

    def __init__(
            self,
            model: OnnxModel,
            hidden_size: int,
            num_heads: int,
            attention_mask: AttentionMask,
    ):
        super().__init__(
            model,
            hidden_size,
            num_heads,
            attention_mask,
            use_multi_head_attention=False,
            search_op_types=["SkipSimplifiedLayerNormalization", "Add"],
        )
        self.static_kv = 1

    def create_attention_node(
            self,
            mask_index: str,
            q_matmul: NodeProto,
            k_matmul: NodeProto,
            v_matmul: NodeProto,
            num_heads: int,
            hidden_size: int,
            input: str,
            output: str,
            add_qk_str: str,
            scale: Optional[float] = None,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.
        Args:
            mask_index (str): mask input
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for K
            v_matmul (NodeProto): MatMul node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name
        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])

        if q_weight is None:
            print(
                f"{q_matmul.input[1]} is not an initializer. "
                "Please set do_constant_folding=True in torch.onnx.export to unblock attention fusion"
            )
            return None

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        # assert q and k have same shape as expected
        assert qw.shape == kw.shape

        qw_in_size = qw.shape[0]
        kw_in_size = kw.shape[0]
        vw_in_size = vw.shape[0]

        assert qw_in_size == kw_in_size == vw_in_size

        if hidden_size > 0 and hidden_size != qw_in_size:
            logger.warning(
                f"Input hidden size ({hidden_size}) is not same as weight matrix dimension of q,k,v ({qw_in_size}). "
                "Please provide a correct input hidden size or pass in 0"
            )

        qw_out_size = np.prod(qw.shape[1:])
        qkv_weight = np.stack((qw, kw, vw), axis=1)
        qkv_weight_dim = 3 * qw_out_size

        attention_node_name = self.model.create_node_name("Attention")

        weight = helper.make_tensor(
            name=attention_node_name + "_qkv_weight",
            data_type=TensorProto.FLOAT,
            dims=[qw_in_size, qkv_weight_dim],
            vals=qkv_weight.tobytes(),
            raw=True,
        )

        self.model.add_initializer(weight, self.this_graph_name)

        attention_inputs = [
            input,
            attention_node_name + "_qkv_weight",
            "",
        ]
        if mask_index is not None:
            attention_inputs.append(mask_index)
        else:
            attention_inputs.append("")

        if add_qk_str is not None:
            attention_inputs.append("")  # no past
            attention_inputs.append(add_qk_str)

        attention_node = helper.make_node(
            "Attention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        if scale is not None:
            attention_node.attribute.extend([helper.make_attribute("scale", scale)])

        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        return attention_node

    def create_mha_node(
            self,
            query: str,
            key: str,
            value: str,
            mask_index: str,
            res_pos_bias: str,
            past_key: str,
            past_value: str,
            output: str,
            present_key: str,
            present_value: str,
            num_heads: int,
            hidden_size: int,
    ) -> Union[NodeProto, None]:
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        attention_node_name = self.model.create_node_name("MultiHeadAttention")
        attention_inputs = [
            query,
            "" if key is None else key,  # key
            "" if value is None else value,  # value
            "",  # bias
        ]
        if mask_index is not None:
            attention_inputs.append(mask_index)
        else:
            attention_inputs.append("")

        if res_pos_bias is not None:
            attention_inputs.append(res_pos_bias)
        else:
            attention_inputs.append("")

        if past_key is not None:
            assert past_value is not None
            attention_inputs.append(past_key)
            attention_inputs.append(past_value)


class T5OnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionT5Attention(self, self.hidden_size, self.num_heads, self.attention_mask)
        self.layer_norm_fusion = FusionLayerNormalization(self)  #prima era FusionSimplifiedLayerNormalization
        self.skip_layer_norm_fusion = FusionSkipLayerNormalization(self)   #prima era FusionSkipSimplifiedLayerNormalization
        # TODO: consider retrive max_distance from model.
        # math.log(max_distance / (num_buckets // 2))
        self.rpb_fusion = FusionRelativePositionBiasBlock(self, 128)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def fuse_layer_norm(self):
        self.layer_norm_fusion.apply()

    def fuse_skip_layer_norm(self):
        self.skip_layer_norm_fusion.apply()

    # Remove get_extended_attention_mask() since it generates all zeros.
    def remove_extended_mask_decoder_init(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Add":
                extended_mask_nodes = self.match_parent_path(
                    node,
                    [
                        "Mul",
                        "Sub",
                        "Mul",
                        "Unsqueeze",
                        "Cast",
                        "LessOrEqual",
                        "Tile",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                    ],
                    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                )
                if extended_mask_nodes is None:
                    continue

                rpb_nodes = self.match_parent_path(node, ["RelativePositionBias"], [0])
                if rpb_nodes is None:
                    continue

                rpb_node = rpb_nodes[0]
                rpb_node.output[0] = node.output[0]

                nodes_to_remove.extend(extended_mask_nodes)
                nodes_to_remove.append(node)
                self.remove_nodes(nodes_to_remove)

    def remove_extended_mask_decoder(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Add":
                extended_mask_nodes = self.match_parent_path(
                    node,
                    [
                        "Mul",
                        "Sub",
                        "Mul",
                        "Unsqueeze",
                        "Concat",
                        "Cast",
                        "LessOrEqual",
                        "Tile",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                    ],
                    [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                )
                if extended_mask_nodes is None:
                    continue

                rpb_nodes = self.match_parent_path(node, ["Slice", "RelativePositionBias"], [0, 0])
                if rpb_nodes is None:
                    continue

                rpb_node = rpb_nodes[0]
                rpb_node.output[0] = node.output[0]

                nodes_to_remove.extend(extended_mask_nodes)
                nodes_to_remove.append(node)
                self.remove_nodes(nodes_to_remove)

    def preprocess(self):
        self.adjust_reshape_and_expand()
        self.rpb_fusion.apply()

    def postprocess(self):
        # remove get_extended_attention_mask() since it generates all zeros.
        self.remove_extended_mask_decoder_init()
        self.remove_extended_mask_decoder()

        self.prune_graph()
