import gc
import os
import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer, HunYuanDenseV1ForCausalLM
from torch.export import Dim

os.environ["TORCH_LOGS"] = "+dynamic,+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"


model_name = "tencent/HY-MT1.5-1.8B"                           # Set the folder path where the Hunyuan-MT-1.5 whole project downloaded.
STOP_TOKEN = [120020, 127960]                                             # The stop_id in Hunyuan-MT-1.5-1.8B is"120020"; 127960 for 7B
MAX_SEQ_LEN = 4096                                                        # The max context length.
sentence = "May the force be with you"                                    # The test sentence after the export process.
original_language = "English"                                             # Source language of the text to translate. Accepts: English/Chinese/Abbreviation (case-insensitive). See get_language() for all supported languages.
target_language = "Chinese"                                               # Target language for translation. Accepts: English/Chinese/Abbreviation (case-insensitive). See get_language() for all supported languages.


def get_language(language_input):
    """
    Accepts a language identifier (full Chinese name, abbreviation, or full English name)
    and returns the standardized full English and full Chinese names.

    The function is case-insensitive for English names and abbreviations.

    Args:
        language_input (str): The language identifier to look up. e.g., "中文", "zh", "chinese", "French"

    Returns:
        tuple[str, str] or tuple[None, None]: A tuple containing the
        standard full English name and the full Chinese name.
        Returns (None, None) if the language is not found.
    """

    # 1. CANONICAL DATA STORE
    # The primary source of truth, keyed by the unique language abbreviation.
    # This structure prevents data duplication.
    # Columns are aligned for clear reading.
    LANGUAGE_DATA = {
        # Abbr:     { English Name,              Chinese Name }
        "ar":      {"english_name": "Arabic",                  "chinese_name": "阿拉伯语"},
        "bn":      {"english_name": "Bengali",                 "chinese_name": "孟加拉语"},
        "my":      {"english_name": "Burmese",                 "chinese_name": "缅甸语"},
        "yue":     {"english_name": "Cantonese",               "chinese_name": "粤语"},
        "zh":      {"english_name": "Chinese",                 "chinese_name": "中文"},
        "zh-Hant": {"english_name": "Chinese (Traditional)",   "chinese_name": "繁体中文"},
        "cs":      {"english_name": "Czech",                   "chinese_name": "捷克语"},
        "nl":      {"english_name": "Dutch",                   "chinese_name": "荷兰语"},
        "en":      {"english_name": "English",                 "chinese_name": "英语"},
        "tl":      {"english_name": "Filipino",                "chinese_name": "菲律宾语"},
        "fr":      {"english_name": "French",                  "chinese_name": "法语"},
        "de":      {"english_name": "German",                  "chinese_name": "德语"},
        "gu":      {"english_name": "Gujarati",                "chinese_name": "古吉拉特语"},
        "he":      {"english_name": "Hebrew",                  "chinese_name": "希伯来语"},
        "hi":      {"english_name": "Hindi",                   "chinese_name": "印地语"},
        "id":      {"english_name": "Indonesian",              "chinese_name": "印尼语"},
        "it":      {"english_name": "Italian",                 "chinese_name": "意大利语"},
        "ja":      {"english_name": "Japanese",                "chinese_name": "日语"},
        "kk":      {"english_name": "Kazakh",                  "chinese_name": "哈萨克语"},
        "km":      {"english_name": "Khmer",                   "chinese_name": "高棉语"},
        "ko":      {"english_name": "Korean",                  "chinese_name": "韩语"},
        "ms":      {"english_name": "Malay",                   "chinese_name": "马来语"},
        "mr":      {"english_name": "Marathi",                 "chinese_name": "马拉地语"},
        "mn":      {"english_name": "Mongolian",               "chinese_name": "蒙古语"},
        "fa":      {"english_name": "Persian",                 "chinese_name": "波斯语"},
        "pl":      {"english_name": "Polish",                  "chinese_name": "波兰语"},
        "pt":      {"english_name": "Portuguese",              "chinese_name": "葡萄牙语"},
        "ru":      {"english_name": "Russian",                 "chinese_name": "俄语"},
        "es":      {"english_name": "Spanish",                 "chinese_name": "西班牙语"},
        "ta":      {"english_name": "Tamil",                   "chinese_name": "泰米尔语"},
        "te":      {"english_name": "Telugu",                  "chinese_name": "泰卢固语"},
        "th":      {"english_name": "Thai",                    "chinese_name": "泰语"},
        "bo":      {"english_name": "Tibetan",                 "chinese_name": "藏语"},
        "tr":      {"english_name": "Turkish",                 "chinese_name": "土耳其语"},
        "uk":      {"english_name": "Ukrainian",               "chinese_name": "乌克兰语"},
        "ur":      {"english_name": "Urdu",                    "chinese_name": "乌尔都语"},
        "ug":      {"english_name": "Uyghur",                  "chinese_name": "维吾尔语"},
        "vi":      {"english_name": "Vietnamese",              "chinese_name": "越南语"},
    }

    # 2. ALIAS MAP GENERATION
    # Create a comprehensive lookup map from all possible inputs to the canonical abbreviation.
    # This map is built dynamically from LANGUAGE_DATA to ensure consistency.
    LANGUAGE_ALIAS_MAP = {}
    for abbr, data in LANGUAGE_DATA.items():
        # Map from abbreviation (e.g., "zh")
        LANGUAGE_ALIAS_MAP[abbr.lower()] = abbr
        # Map from English name (e.g., "chinese")
        LANGUAGE_ALIAS_MAP[data["english_name"].lower()] = abbr
        # Map from Chinese name (e.g., "中文")
        LANGUAGE_ALIAS_MAP[data["chinese_name"]] = abbr

    # Add other common aliases.
    EXTRA_ALIASES = {
        "tagalog": "tl",
        "farsi": "fa",
        "myanmar": "my",
        "uighur": "ug",
        "traditional chinese": "zh-Hant",
        "cambodian": "km",
    }
    LANGUAGE_ALIAS_MAP.update(EXTRA_ALIASES)

    # 3. LOOKUP LOGIC
    # Normalize the input to handle whitespace and case variations.
    lookup_key = str(language_input).strip().lower()

    # Find the canonical abbreviation using the alias map.
    canonical_abbr = LANGUAGE_ALIAS_MAP.get(lookup_key)

    if canonical_abbr:
        # Retrieve the definitive language data.
        lang_data = LANGUAGE_DATA.get(canonical_abbr)
        return lang_data["english_name"], lang_data["chinese_name"]
    else:
        # Return None if no match is found.
        return None, None


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


def rotate_half(x, head_dim_half, dim):
    x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
    return torch.cat((-x2, x1), dim=dim)


def repeat_k(kv_states, num_key_value_groups, head_dim, num_heads):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, head_dim, -1)


def repeat_v(kv_states, num_key_value_groups, head_dim, num_heads):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, -1, head_dim)


class HUNYUAN(torch.nn.Module):
    def __init__(self, hunyuan, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers):
        super(HUNYUAN, self).__init__()
        self.hunyuan = hunyuan
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = float(1e-6)

        scale_factor = float(head_dim ** -0.25)
        for i in range(num_layers):
            self.hunyuan.model.layers._modules[f'{i}'].self_attn.query_layernorm.weight.data *= scale_factor
            self.hunyuan.model.layers._modules[f'{i}'].self_attn.key_layernorm.weight.data *= scale_factor

        data = self.hunyuan.model.embed_tokens.weight.data
        self.zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
        self.scale = ((torch.max(data, dim=1)[0] - self.zero_point[:, 0]) / 255.0).unsqueeze(1)
        self.embed_data = quantize_to_uint8(data, 1.0 / self.scale, self.zero_point)

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        idx_theta = position_ids * self.hunyuan.model.rotary_emb.inv_freq
        cos_rotary_pos_emb = torch.cos(idx_theta)
        sin_rotary_pos_emb = torch.sin(idx_theta)
        self.cos_rotary_pos_emb = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).half()
        self.sin_rotary_pos_emb = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).half()

        self.attention_mask = (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

    def forward(self, *all_inputs):
        # Inputs (same ordering as your original export):
        # all_inputs[0..L-1]           -> past keys   (kvh, 1, hd, hist)
        # all_inputs[L..2L-1]          -> past values (kvh, 1, hist, hd)
        # all_inputs[-1]               -> input_ids   (1, ids_len)
        input_ids = all_inputs[0]  # keep your original convention

        ids_len = input_ids.shape[1]
        history_len = all_inputs[1].shape[3]   # past key: (kvh,1,hd,hist)
        kv_seq_len = history_len + ids_len

        attention_bias = self.attention_mask[:, :ids_len, :kv_seq_len].float()  # (1, ids_len, kv_seq_len)

        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()  # (1, ids_len, hd)
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()  # (1, ids_len, hd)

        # Embeddings: (1, ids_len, hidden)
        hidden_states = self.embed_data[input_ids] * self.scale[input_ids] + self.zero_point[input_ids]

        save_key = [None] * self.num_layers
        save_value = [None] * self.num_layers

        for i, layer in enumerate(self.hunyuan.model.layers):
            hidden_states_norm = layer.input_layernorm.weight * (
                hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            )

            # q_proj: (1, ids_len, heads*hd) -> (1, ids_len, heads, hd) -> (heads, ids_len, hd)
            q = layer.self_attn.q_proj(hidden_states_norm)
            q = q.reshape(1, ids_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).squeeze(0)

            # k_proj: (1, ids_len, kvh*hd) -> (1, ids_len, kvh, hd) -> (kvh, 1, ids_len, hd)
            k = layer.self_attn.k_proj(hidden_states_norm)
            k = k.reshape(1, ids_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)

            # v_proj: (1, ids_len, kvh*hd) -> (1, ids_len, kvh, hd) -> (kvh, 1, ids_len, hd)
            v = layer.self_attn.v_proj(hidden_states_norm)
            v = v.reshape(1, ids_len, self.num_key_value_heads, self.head_dim).permute(0, 2, 1, 3)

            # Rotary (broadcast works: q is (h, s, d), cos/sin are (1, s, d); k is (1, kvh, s, d))
            q = q * rotary_pos_emb_cos + rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin
            k = k * rotary_pos_emb_cos.unsqueeze(1) + rotate_half(k, self.head_dim_half, -1) * rotary_pos_emb_sin.unsqueeze(1)

            # Q/K layernorms
            q = layer.self_attn.query_layernorm.weight * (
                q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            )

            # Key LN over last dim (hd) while k is (1, kvh, s, d), then transpose to (kvh,1,d,s) like your original
            k = layer.self_attn.key_layernorm.weight * (
                k / torch.sqrt(k.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            )
            k = k.permute(1, 0, 3, 2)  # (kvh, 1, hd, ids_len)

            # Concat cache
            k = torch.cat((all_inputs[1+i], k), dim=-1)  # (kvh,1,hd,hist+ids)
            v = torch.cat((all_inputs[1+i + self.num_layers], v.permute(1, 0, 2, 3)), dim=2)  # (kvh,1,hist+ids,hd)

            save_key[i] = k
            save_value[i] = v

            # Repeat kv heads -> heads
            k_rep = repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads)  # (heads, hd, T)
            v_rep = repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads)  # (heads, T, hd)

            # Attention
            attn = torch.nn.functional.softmax(torch.matmul(q, k_rep) + attention_bias, dim=-1, dtype=torch.float32)  # (heads, ids_len, T)

            attn_out = layer.self_attn.o_proj(
                torch.matmul(attn, v_rep).transpose(0, 1).contiguous().view(1, -1, layer.self_attn.o_proj.in_features)
            )
            hidden_states = hidden_states + attn_out

            # MLP block
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.weight * (
                hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            )
            hidden_states = layer.mlp.down_proj(
                layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) * layer.mlp.up_proj(hidden_states)
            )
            hidden_states = hidden_states + residual

        # If you want logits for the *last* token only (like your original), keep this:
        last = hidden_states[:, -1]
        last = self.hunyuan.model.norm.weight * (
            last / torch.sqrt(last.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        )
        logits = self.hunyuan.lm_head(last)  # (1, vocab)

        return logits, *save_key, *save_value

        


def build_dynamic_shapes(num_layers: int, max_seq_len: int):
    ids = Dim("ids_len", min=1, max=max_seq_len)
    hist = Dim("history_len", min=0, max=max_seq_len-1)

    dyn_flat = []
    dyn_flat.append({1: ids})  # input_ids
    dyn_flat.extend([{3: hist} for _ in range(num_layers)])  # keys
    dyn_flat.extend([{2: hist} for _ in range(num_layers)])  # values

    return (tuple(dyn_flat),)  # wrap because forward has *all_inputs


def export_model(onnx_model_folder = 'onnx/HY-MT/CustomExport/', onnx_model_file = "HY-MT.onnx"):
    os.makedirs(onnx_model_folder, exist_ok=True)
    onnx_model_path = onnx_model_folder + onnx_model_file
    print('Export start ...')
    with torch.inference_mode():
        # Load the original model
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
        head_dim = model.config.head_dim
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        num_key_value_heads = model.config.num_key_value_heads

        # Build an optimized model
        model = HUNYUAN(model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers)

        # Generate dummies for torch.onnx.export()
        input_ids = torch.ones((1, 1), dtype=torch.int32)
        past_keys = torch.rand((num_key_value_heads, 1, head_dim, 1), dtype=torch.float32)
        past_values = torch.rand((num_key_value_heads, 1, 1, head_dim), dtype=torch.float32)

        # Prepare input and output names
        all_inputs = []
        input_names = []
        output_names = []
        dynamic_axes = {'input_ids': {1: 'ids_len'}}
        dynamic_shapes = build_dynamic_shapes(num_layers=num_layers, max_seq_len=MAX_SEQ_LEN)
        input_names.append('input_ids')
        all_inputs.append(input_ids)
        output_names.append('logits')
        for i in range(num_layers):
            name = f'in_key_{i}'
            input_names.append(name)
            all_inputs.append(past_keys)
            dynamic_axes[name] = {3: 'history_len'}
            name = f'out_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {3: 'history_len_plus_ids_len'}
        for i in range(num_layers):
            name = f'in_value_{i}'
            input_names.append(name)
            all_inputs.append(past_values)
            dynamic_axes[name] = {2: 'history_len'}
            name = f'out_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {2: 'history_len_plus_ids_len'}

        torch.onnx.export(
            model,
            tuple(all_inputs),
            onnx_model_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            #dynamic_shapes=dynamic_shapes,
            do_constant_folding=False,
            opset_version=17,
            dynamo=False,
            external_data=True,
            verbose=True,
        )
    print('\nExport done!\n')


def export_model2():
    en_text_extended_128 = ("Pre-trained Transformer models have achieved state-of-the-art performance on natural language processing tasks and have been adopted as feature extractors for solving downstream tasks such as question answering, natural language inference, and sentiment analysis. The current state-of-the-art Transformer-based pre-trained models consist of dozens of layers and millions of parameters. While deeper and wider models yield better performance, they also need large GPU/TPU memory. For example, BERT-large (Devlin et al., 2019")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # con torch script = True la conversione funziona
    model = HunYuanDenseV1ForCausalLM.from_pretrained(model_name, use_cache=True)
    model.eval()
    messages = [
        {"role": "user", "content": "Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house."},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )

    # conversione (da fare)








