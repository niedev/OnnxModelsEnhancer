from collections import OrderedDict
import enum
import hashlib
import os
from pathlib import Path
import pickle
import time

import datasets
import onnxruntime
import torch
from transformers import NllbTokenizer, T5Tokenizer
from strsimpy.normalized_levenshtein import NormalizedLevenshtein


class TranslationCache:
    def __init__(self, max_items=1000, ttl_s=24*3600, cache_file="Cache/translation_cache.pkl", save_cache_to_file=True):
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._data = OrderedDict()  # key -> (timestamp, value)
        self.cache_file = cache_file
        self.save_cache_to_file = save_cache_to_file
        self._load_from_file()

    def get(self, key):
        item = self._data.get(key)
        if not item:
            return None
        ts, val = item
        if time.time() - ts > self.ttl_s:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return val

    def set(self, key, value):
        self._data[key] = (time.time(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)
        if(self.save_cache_to_file): self._save_to_file()

    def make_cache_key(self, text, tgt_lang, encoder_path, decoder_path, initializer_path, embed_path):
        payload = "\n".join([
            "v1",  # bump when you change logic
            tgt_lang,
            encoder_path, decoder_path, initializer_path, embed_path,
            text
        ]).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    

    # private helpers
    def _load_from_file(self):
        if not os.path.exists(self.cache_file):
            return

        try:
            with open(self.cache_file, "rb") as f:
                data = pickle.load(f)

            # Only accept if it looks valid
            if isinstance(data, OrderedDict):
                self._data = data

            # Clean expired items on load
            now = time.time()
            expired = [k for k, (ts, _) in self._data.items()
                       if now - ts > self.ttl_s]

            for k in expired:
                del self._data[k]

        except Exception:
            # If anything goes wrong (corrupt file, etc.), start fresh
            self._data = OrderedDict()

    def _save_to_file(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._data, f)
        except Exception as e:
            # You might want logging here in a real app
            print(f"Warning: could not save cache: {e}")



translation_cache_madlad = TranslationCache(max_items=50000, ttl_s=float('inf'), save_cache_to_file=False)

def onnx_execution(text, src_lang, tgt_lang):
    # caricamento encoder session
    providers = ['CPUExecutionProvider']  # 'ROCMExecutionProvider',
    encoder_session = onnxruntime.InferenceSession('onnx/NLLBScript/600M/quantized/NLLB_encoder_quantized.onnx', providers=providers)
    # caricamento decoder session
    #decoder_session = onnxruntime.InferenceSession('onnx/NLLBScript/600M/quantized/Optimum/Static/NLLB_decoder_quantized2.onnx', providers=providers)
    #decoder_session = onnxruntime.InferenceSession('onnx/NLLBScript/600M/quantized/NLLB_decoder_complete_quantized.onnx', providers=providers)
    decoder_session = onnxruntime.InferenceSession('onnx/NLLBScript/600M/quantized/Optimum/Static/Test/NLLB_decoder_quantized2.onnx', providers=providers)
    #prepariamo gli input dell'encoder
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang, tgt_lang=tgt_lang)
    inputEncoder = tokenizer(text, max_length=256, padding='max_length', return_tensors='pt')
    encoder_input_ids = inputEncoder.input_ids.numpy()
    encoder_attention_mask = inputEncoder.attention_mask.numpy()
    #esecuzione dell'encoder
    encoderOuput = encoder_session.run(["last_hidden_state"], {"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask})
    # prepariamo gli input del decoder
    decoder_input_ids_pt = torch.cat([torch.tensor([[2, tokenizer.convert_tokens_to_ids(tgt_lang)]], dtype=torch.int64), torch.ones(1, 256 - 2, dtype=torch.int64)], dim=1)
    decoder_input_ids = decoder_input_ids_pt.numpy()
    decoder_attention_mask = torch.cat([torch.tensor([[1,1]], dtype=torch.int64), torch.zeros(1,256-2, dtype=torch.int64)], dim=1).numpy()
    out = -1
    i = 1
    result = []
    while(out != 2):
        #esecuzione del decoder
        decoderOutput = decoder_session.run(["logits"],
                                           {"input_ids": decoder_input_ids,
                                                    "attention_mask": decoder_attention_mask,
                                                    "encoder_hidden_states": encoderOuput[0],
                                                    "encoder_attention_mask": encoder_attention_mask})
        test = decoderOutput[0][0][i, :]
        out = test.argmax()
        result.append(out)
        #prepariamo gli input del decoder per la prossima iterazione
        decoder_input_ids[0][1+i] = out  #si modifica il primo padding inserendo il nuovo valore
        decoder_attention_mask[0][1 + i] = 1  # si modifica il primo valore 0 inserendo 1 al suo posto (in pratica allunghiamo di 1 i valori a cui dobbiamo stare attenti)
        print(tokenizer.decode(result))  # _convert_id_to_token(out)
        i = i+1


def onnx_execution_madlad(text, tgt_lang, decoder_path):
    text = "<2"+tgt_lang+"> "+text
    model_name = 'jbochi/madlad400-3b-mt'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # caricamento encoder session
    providers = ['CPUExecutionProvider']  # 'ROCMExecutionProvider',
    #encoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Script/DynamicQuantization/Madlad_encoder_quantized.onnx"))
    encoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Optimum/encoder_model.onnx"), providers=providers)
    # caricamento decoder session
    #decoder_session = onnxruntime.InferenceSession('onnx/NLLBScript/600M/quantized/Optimum/Static/NLLB_decoder_quantized2.onnx', providers=providers)
    #decoder_session = onnxruntime.InferenceSession('onnx/NLLBScript/600M/quantized/NLLB_decoder_complete_quantized.onnx', providers=providers)
    #decoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Script/DynamicQuantization/Madlad_decoder_quantized.onnx"))
    #decoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Script/StaticQuantization/Test/Madlad_decoder_quantized_static_test.onnx"), providers=providers)
    #decoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Optimized/2D/Optimized/Madlad_decoder_2D_optimized.onnx"), providers=providers)
    #decoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Optimized/Madlad_decoder_optimized.onnx"), providers=providers)
    decoder_session = onnxruntime.InferenceSession(Path(decoder_path), providers=providers)
    #prepariamo gli input dell'encoder
    inputEncoder = tokenizer(text, max_length=128, padding='max_length', return_tensors='pt')
    encoder_input_ids = inputEncoder.input_ids.numpy()
    encoder_attention_mask = inputEncoder.attention_mask.numpy()
    #esecuzione dell'encoder
    encoderOuput = encoder_session.run(["last_hidden_state"], {"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask})
    # prepariamo gli input del decoder
    decoder_input_ids_pt = torch.cat([torch.tensor([[0]], dtype=torch.int64), torch.ones(1, 128 - 1, dtype=torch.int64)], dim=1)
    decoder_input_ids = decoder_input_ids_pt.numpy()
    decoder_attention_mask = torch.cat([torch.tensor([[1]], dtype=torch.int64), torch.zeros(1,128-1, dtype=torch.int64)], dim=1).numpy()
    out = -1
    i = 1
    result = []
    while(out != 2):
        #esecuzione del decoder
        decoderOutput = decoder_session.run(["logits"],
                                           {"input_ids": decoder_input_ids,
                                                    "attention_mask": decoder_attention_mask,
                                                    "encoder_hidden_states": encoderOuput[0],
                                                    "encoder_attention_mask": encoder_attention_mask})
        test = decoderOutput[0][0][i, :]
        out = test.argmax()
        result.append(out)
        #prepariamo gli input del decoder per la prossima iterazione
        decoder_input_ids[0][1+i] = out  #si modifica il primo padding inserendo il nuovo valore
        decoder_attention_mask[0][1 + i] = 1  # si modifica il primo valore 0 inserendo 1 al suo posto (in pratica allunghiamo di 1 i valori a cui dobbiamo stare attenti)
        print(tokenizer.decode(result))  # _convert_id_to_token(out)
        i = i+1

    #per ora testa solo la prima parola (quel che basta per capire se il modello funziona), eventualmente poi se servir`a fargli generare tutto e poi tradurlo in testo col tokenizer



def onnx_execution_madlad_cache(text, tgt_lang, decoder_path):
    text = "<2"+tgt_lang+"> "+text
    model_name = 'jbochi/madlad400-3b-mt'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # caricamento encoder session
    providers = ['CPUExecutionProvider']  # 'ROCMExecutionProvider',
    #encoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Script/DynamicQuantization/Madlad_encoder_quantized.onnx"))
    #encoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Optimum_Cache/DynamicQuantization/Madlad_encoder_quantized.onnx"), providers=providers)
    encoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Optimum/encoder_model.onnx"), providers=providers)
    # caricamento decoder session
    decoder_session = onnxruntime.InferenceSession(Path(decoder_path), providers=providers)
    #prepariamo gli input dell'encoder
    inputEncoder = tokenizer(text, return_tensors='pt')
    encoder_input_ids = inputEncoder.input_ids.numpy()
    encoder_attention_mask = inputEncoder.attention_mask.numpy()
    #esecuzione dell'encoder
    encoderOuput = encoder_session.run(["last_hidden_state"], {"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask})

    # prepariamo gli input del decoder
    decoder_input_ids = torch.tensor([[0]], dtype=torch.int64).numpy()
    out = -1
    result = []
    encoder_input_ids_length = len(encoder_input_ids[0])

    input_feed = {"input_ids": decoder_input_ids,
                 "encoder_hidden_states": encoderOuput[0],
                 "encoder_attention_mask": encoder_attention_mask,
                 "use_cache_branch": torch.tensor([False], dtype=torch.bool).numpy()}
    for i in range(32):
        input_feed["past_key_values."+str(i)+".decoder.key"] = torch.zeros(1, 16, 0, 128, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".decoder.value"] = torch.zeros(1, 16, 0, 128, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".encoder.key"] = torch.zeros(1, 16, encoder_input_ids_length, 128, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".encoder.value"] = torch.zeros(1, 16, encoder_input_ids_length, 128, dtype=torch.float32).numpy()

    output_names = ["logits"]
    for i in range(32):
        output_names.append("present."+str(i)+".decoder.key")
        output_names.append("present."+str(i)+".decoder.value")
        output_names.append("present."+str(i)+".encoder.key")
        output_names.append("present."+str(i)+".encoder.value")

    # esecuzione del decoder senza cache
    initialDecoderOutput = decoder_session.run(output_names, input_feed)
    decoderOutput = initialDecoderOutput

    test = initialDecoderOutput[0][0][0]
    out = test.argmax()
    result.append(out)
    #prepariamo gli input del decoder per la prossima iterazione
    decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  #si usa il risultato come unico input_id della prossima iterazione
    print(tokenizer.decode(result))  # _convert_id_to_token(out)

    while(out != 2):
        input_feed = {"input_ids": decoder_input_ids,
                     "encoder_hidden_states": encoderOuput[0],
                     "encoder_attention_mask": encoder_attention_mask,
                     "use_cache_branch": torch.tensor([True], dtype=torch.bool).numpy()}
        count = 1
        for i in range(32):
            input_feed["past_key_values."+str(i)+".decoder.key"] = decoderOutput[count]
            count = count+1
            input_feed["past_key_values."+str(i)+".decoder.value"] = decoderOutput[count]
            count = count+1
            input_feed["past_key_values."+str(i)+".encoder.key"] = initialDecoderOutput[count]
            count = count+1
            input_feed["past_key_values."+str(i)+".encoder.value"] = initialDecoderOutput[count]
            count = count+1

        # esecuzione del decoder senza cache
        decoderOutput = decoder_session.run(output_names, input_feed)

        test = decoderOutput[0][0][0]
        out = test.argmax()
        result.append(out)
        #prepariamo gli input del decoder per la prossima iterazione
        decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  #si usa il risultato come unico input_id della prossima iterazione
        print(tokenizer.decode(result))  # _convert_id_to_token(out)



def onnx_execution_madlad_cache_test(text, tgt_lang, encoder_path="onnx/Madlad/Optimum_Cache_Optimized/encoder_model.onnx",
                                     decoder_path="onnx/Madlad/Optimum_Cache_Optimized/decoder_with_past_model.onnx",
                                     encoder_session=None, decoder_session=None, initializer_session=None, log=True):
    text = "<2"+tgt_lang+"> "+text
    model_name = 'jbochi/madlad400-3b-mt'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # caricamento encoder session
    providers = ['CPUExecutionProvider']  # 'ROCMExecutionProvider',
    if(encoder_session is None):
        #encoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Script/Madlad_encoder.onnx"))
        #encoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Optimum/encoder_model.onnx"), providers=providers)
        #encoder_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic/Madlad_encoder_fp16_quantized.onnx"), providers=providers)
        encoder_session = onnxruntime.InferenceSession(Path(encoder_path), providers=providers)
    # caricamento decoder session
    if(decoder_session is None):
        decoder_session = onnxruntime.InferenceSession(Path(decoder_path), providers=providers)
    # caricamento cache initializer session
    if(initializer_session is None):
        initializer_session = onnxruntime.InferenceSession(Path("onnx/Madlad/Optimum_Cache_Optimized/QuantizeDynamic/cache_initializer/Cache_initializer_quantized.onnx"), providers=providers)
    #prepariamo gli input dell'encoder
    inputEncoder = tokenizer(text, max_length=128, padding='max_length', return_tensors='pt')
    encoder_input_ids = inputEncoder.input_ids.numpy()
    encoder_attention_mask = inputEncoder.attention_mask.numpy()
    #esecuzione dell'encoder
    encoderOuput = encoder_session.run(["last_hidden_state"], {"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask})

    # prepariamo gli input del decoder
    decoder_input_ids = torch.tensor([[0]], dtype=torch.int64).numpy()
    out = -1
    result = []
    encoder_input_ids_length = len(encoder_input_ids[0])

    input_feed = {"input_ids": decoder_input_ids,
                 "encoder_hidden_states": encoderOuput[0],
                 "encoder_attention_mask": encoder_attention_mask,
                 "use_cache_branch": torch.tensor([False], dtype=torch.bool).numpy()}
    for i in range(32):
        input_feed["past_key_values."+str(i)+".decoder.key"] = torch.zeros(1, 16, 0, 128, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".decoder.value"] = torch.zeros(1, 16, 0, 128, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".encoder.key"] = torch.zeros(1, 16, encoder_input_ids_length, 128, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".encoder.value"] = torch.zeros(1, 16, encoder_input_ids_length, 128, dtype=torch.float32).numpy()

    output_names = []
    for i in range(32):
        output_names.append("present."+str(i)+".encoder.key")
        output_names.append("present."+str(i)+".encoder.value")

    # esecuzione del cache initializer
    initialDecoderOutput = initializer_session.run(output_names, {"encoder_hidden_states": encoderOuput[0]})

    #decoderOutput = initialDecoderOutput

    #prima esecuzione usando solo il decoder con past
    output_names = ["logits"]
    for i in range(32):
        output_names.append("present."+str(i)+".decoder.key")
        output_names.append("present."+str(i)+".decoder.value")

    input_feed = {"input_ids": decoder_input_ids,
                     "encoder_hidden_states": encoderOuput[0],
                     "encoder_attention_mask": encoder_attention_mask}
    count = 0
    for i in range(32):
        input_feed["past_key_values."+str(i)+".decoder.key"] = torch.zeros(1, 16, 0, 128, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".decoder.value"] = torch.zeros(1, 16, 0, 128, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".encoder.key"] = initialDecoderOutput[count]
        count = count+1
        input_feed["past_key_values."+str(i)+".encoder.value"] = initialDecoderOutput[count]
        count = count+1

    decoderOutput = decoder_session.run(output_names, input_feed)

    test = decoderOutput[0][0][0]
    out = test.argmax()
    result.append(out)
    #prepariamo gli input del decoder per la prossima iterazione
    decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  #si usa il risultato come unico input_id della prossima iterazione
    if(log):
        print(tokenizer.decode(result))  # _convert_id_to_token(out)

    while(out != 2):
        input_feed = {"input_ids": decoder_input_ids,
                     "encoder_hidden_states": encoderOuput[0],
                     "encoder_attention_mask": encoder_attention_mask}
        count = 1
        initializer_count = 0
        for i in range(32):
            input_feed["past_key_values."+str(i)+".decoder.key"] = decoderOutput[count]
            count = count+1
            input_feed["past_key_values."+str(i)+".decoder.value"] = decoderOutput[count]
            count = count+1
            input_feed["past_key_values."+str(i)+".encoder.key"] = initialDecoderOutput[initializer_count]
            initializer_count = initializer_count+1
            input_feed["past_key_values."+str(i)+".encoder.value"] = initialDecoderOutput[initializer_count]
            initializer_count = initializer_count+1

        # esecuzione del decoder con cache
        decoderOutput = decoder_session.run(output_names, input_feed)

        test = decoderOutput[0][0][0]
        out = test.argmax()
        result.append(out)
        #prepariamo gli input del decoder per la prossima iterazione
        decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  #si usa il risultato come unico input_id della prossima iterazione
        if(log):
            print(tokenizer.decode(result))  # _convert_id_to_token(out)

    return tokenizer.decode(result)


def onnx_execution_madlad_cache_reduced_ram(text, tgt_lang, encoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx",
                                     decoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx",
                                     initializer_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx",
                                     embed_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx",
                                     encoder_session:onnxruntime.InferenceSession|None=None,
                                     decoder_session:onnxruntime.InferenceSession|None=None,
                                     initializer_session:onnxruntime.InferenceSession|None=None,
                                     embed_session:onnxruntime.InferenceSession|None=None,
                                     log=True, profiling=False, cacheResults=True):
    init_time = time.time()
    embed_time_count = 0
    encoder_time_count = 0
    decoder_time_count = 0
    initializer_time_count = 0
    text = "<2"+tgt_lang+"> "+text
    model_name = 'jbochi/madlad400-3b-mt'
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    if(encoder_session is not None):
        encoder_path = encoder_session._model_path

    if(decoder_session is not None):
        decoder_path = decoder_session._model_path

    if(initializer_session is not None):
        initializer_path = initializer_session._model_path

    if(embed_session is not None):
        embed_path = embed_session._model_path

    if(cacheResults):
        key = translation_cache_madlad.make_cache_key(text, tgt_lang, encoder_path, decoder_path, initializer_path, embed_path)

        cached = translation_cache_madlad.get(key)
        if cached is not None:
            return cached
    
    # caricamento encoder session
    providers = ['CPUExecutionProvider']  # 'ROCMExecutionProvider',
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED    #ORT_DISABLE_ALL
    sess_options.enable_profiling = profiling
    if(embed_session is None):
        embed_session = onnxruntime.InferenceSession(Path(embed_path), sess_options=sess_options, providers=providers)
    if(encoder_session is None):
        encoder_session = onnxruntime.InferenceSession(Path(encoder_path), sess_options=sess_options, providers=providers)
    # caricamento decoder session
    if(decoder_session is None):
        decoder_session = onnxruntime.InferenceSession(Path(decoder_path), sess_options=sess_options, providers=providers)
    # caricamento cache initializer session
    if(initializer_session is None):
        initializer_session = onnxruntime.InferenceSession(Path(initializer_path), sess_options=sess_options, providers=providers)
    #prepariamo gli input dell'encoder
    inputEncoder = tokenizer(text, max_length=128, padding='max_length', return_tensors='pt')
    encoder_input_ids = inputEncoder.input_ids.numpy()
    encoder_attention_mask = inputEncoder.attention_mask.numpy()
    #esecuzione dell'encoder
    embed_time = time.time()
    embedOuput = embed_session.run(["embed_matrix"], {"input_ids": encoder_input_ids})
    embed_time_count = time.time() - embed_time
    encoder_time = time.time()
    encoderOuput = encoder_session.run(["last_hidden_state"], {"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask, "embed_matrix": embedOuput[0]})
    encoder_time_count = time.time() - encoder_time

    # prepariamo gli input del decoder
    decoder_input_ids = torch.tensor([[0]], dtype=torch.int64).numpy()
    out = -1
    result = []

    output_names = []
    for i in range(32):
        output_names.append("present."+str(i)+".encoder.key")
        output_names.append("present."+str(i)+".encoder.value")

    # esecuzione del cache initializer
    initializer_time = time.time()
    initialDecoderOutput = initializer_session.run(output_names, {"encoder_hidden_states": encoderOuput[0]})  #
    initializer_time_count = time.time() - initializer_time
    #decoderOutput = initialDecoderOutput

    #prima esecuzione usando solo il decoder con past
    embed_time = time.time()
    embedOuput = embed_session.run(["embed_matrix"], {"input_ids": decoder_input_ids})
    embed_time_count = embed_time_count + (time.time() - embed_time)
    output_names = ["logits"]
    for i in range(32):
        output_names.append("present."+str(i)+".decoder.key")
        output_names.append("present."+str(i)+".decoder.value")

    input_feed = {"input_ids": decoder_input_ids,
                  "embed_matrix": embedOuput[0],
                  #"encoder_hidden_states": encoderOuput[0],
                  "encoder_attention_mask": encoder_attention_mask}
    count = 0
    for i in range(32):
        input_feed["past_key_values."+str(i)+".decoder.key"] = torch.zeros(1, 16, 0, 128, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".decoder.value"] = torch.zeros(1, 16, 0, 128, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".encoder.key"] = initialDecoderOutput[count]
        count = count+1
        input_feed["past_key_values."+str(i)+".encoder.value"] = initialDecoderOutput[count]
        count = count+1

    decoder_time = time.time()
    decoderOutput = decoder_session.run(output_names, input_feed)
    decoder_time_count = time.time() - decoder_time

    test = decoderOutput[0][0][0]
    out = test.argmax()
    result.append(out)
    #prepariamo gli input del decoder per la prossima iterazione
    decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  #si usa il risultato come unico input_id della prossima iterazione
    if(log):
        print(tokenizer.decode(result))  # _convert_id_to_token(out)

    while(out != 2):
        if(len(result) >= len(encoder_input_ids[0])*30):
            print("Decoder timeout")
            raise Exception("Decoder timeout")
        
        embed_time = time.time()
        embedOuput = embed_session.run(["embed_matrix"], {"input_ids": decoder_input_ids})
        embed_time_count = embed_time_count + (time.time() - embed_time)
        input_feed = {"input_ids": decoder_input_ids,
                      "embed_matrix": embedOuput[0],
                      #"encoder_hidden_states": encoderOuput[0],
                      "encoder_attention_mask": encoder_attention_mask}
        count = 1
        initializer_count = 0
        for i in range(32):
            input_feed["past_key_values."+str(i)+".decoder.key"] = decoderOutput[count]
            count = count+1
            input_feed["past_key_values."+str(i)+".decoder.value"] = decoderOutput[count]
            count = count+1
            input_feed["past_key_values."+str(i)+".encoder.key"] = initialDecoderOutput[initializer_count]
            initializer_count = initializer_count+1
            input_feed["past_key_values."+str(i)+".encoder.value"] = initialDecoderOutput[initializer_count]
            initializer_count = initializer_count+1

        # esecuzione del decoder con cache
        decoder_time = time.time()
        decoderOutput = decoder_session.run(output_names, input_feed)
        decoder_time_count = decoder_time_count + (time.time() - decoder_time)

        test = decoderOutput[0][0][0]
        out = test.argmax()
        result.append(out)
        #prepariamo gli input del decoder per la prossima iterazione
        decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  #si usa il risultato come unico input_id della prossima iterazione
        if(log):
            print(tokenizer.decode(result))  # _convert_id_to_token(out)

    if(log):
        print("Execution done in: " + str(time.time() - init_time) + " s")
        print("Execution embed done in: " + str(embed_time_count) + " s")
        print("Execution encoder done in: " + str(encoder_time_count) + " s")
        print("Execution decoder done in: " + str(decoder_time_count) + " s")
        print("Execution cache initializer done in: " + str(initializer_time_count) + " s")
    
    out_text = tokenizer.decode(result)

    if(cacheResults):
        translation_cache_madlad.set(key, out_text)

    return out_text


def onnx_execution_nllb_cache_test(text, src_lang, tgt_lang, encoder_path="onnx/NLLBOptimum/encoder_model.onnx",
                                     decoder_path="onnx/NLLBOptimum/decoder_with_past_model.onnx",
                                     encoder_session=None, decoder_session=None, initializer_session=None, log=True):
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang, tgt_lang=tgt_lang)
    # encoder session loading
    providers = ['CPUExecutionProvider']  # 'ROCMExecutionProvider',
    if(encoder_session is None):
        encoder_session = onnxruntime.InferenceSession(Path(encoder_path), providers=providers)
    # decoder session loading
    if(decoder_session is None):
        decoder_session = onnxruntime.InferenceSession(Path(decoder_path), providers=providers)
    # cache initializer session loading
    if(initializer_session is None):
        #initializer_session = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/NLLB_cache_initializer.onnx"), providers=providers)
        initializer_session = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/Quantized/NLLB_cache_initializer_quantized.onnx"), providers=providers)
    #
    inputEncoder = tokenizer(text, return_tensors='pt')
    encoder_input_ids = inputEncoder.input_ids.numpy()
    encoder_attention_mask = inputEncoder.attention_mask.numpy()
    #esecuzione dell'encoder
    encoderOuput = encoder_session.run(["last_hidden_state"], {"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask})

    # we prepare the encoder inputs
    decoder_input_ids = torch.tensor([[2]], dtype=torch.int64).numpy()
    out = -1
    result = []
    encoder_input_ids_length = len(encoder_input_ids[0])

    input_feed = {"input_ids": decoder_input_ids,
                 "encoder_hidden_states": encoderOuput[0],
                 "encoder_attention_mask": encoder_attention_mask,
                 "use_cache_branch": torch.tensor([False], dtype=torch.bool).numpy()}
    for i in range(12):
        input_feed["past_key_values."+str(i)+".decoder.key"] = torch.zeros(1, 16, 0, 64, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".decoder.value"] = torch.zeros(1, 16, 0, 64, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".encoder.key"] = torch.zeros(1, 16, encoder_input_ids_length, 64, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".encoder.value"] = torch.zeros(1, 16, encoder_input_ids_length, 64, dtype=torch.float32).numpy()

    output_names = []
    for i in range(12):
        output_names.append("present."+str(i)+".encoder.key")
        output_names.append("present."+str(i)+".encoder.value")

    # cache initializer execution
    initialDecoderOutput = initializer_session.run(output_names, {"encoder_hidden_states": encoderOuput[0]})

    #first execution of the decoder with input eos
    output_names = ["logits"]
    for i in range(12):
        output_names.append("present."+str(i)+".decoder.key")
        output_names.append("present."+str(i)+".decoder.value")

    input_feed = {"input_ids": decoder_input_ids,
                     "encoder_attention_mask": encoder_attention_mask}
    count = 0
    for i in range(12):
        input_feed["past_key_values."+str(i)+".decoder.key"] = torch.zeros(1, 16, 0, 64, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".decoder.value"] = torch.zeros(1, 16, 0, 64, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".encoder.key"] = initialDecoderOutput[count]
        count = count+1
        input_feed["past_key_values."+str(i)+".encoder.value"] = initialDecoderOutput[count]
        count = count+1

    decoderOutput = decoder_session.run(output_names, input_feed)

    #second execution of the decoder with input the language code
    decoder_input_ids = torch.tensor([[tokenizer.convert_tokens_to_ids(tgt_lang)]], dtype=torch.int64).numpy()
    input_feed = {"input_ids": decoder_input_ids,
                     "encoder_attention_mask": encoder_attention_mask}
    count = 1
    initializer_count = 0
    for i in range(12):
        input_feed["past_key_values."+str(i)+".decoder.key"] = decoderOutput[count]
        count = count+1
        input_feed["past_key_values."+str(i)+".decoder.value"] = decoderOutput[count]
        count = count+1
        input_feed["past_key_values."+str(i)+".encoder.key"] = initialDecoderOutput[initializer_count]
        initializer_count = initializer_count+1
        input_feed["past_key_values."+str(i)+".encoder.value"] = initialDecoderOutput[initializer_count]
        initializer_count = initializer_count+1

    decoderOutput = decoder_session.run(output_names, input_feed)


    test = decoderOutput[0][0][0]
    out = test.argmax()
    result.append(out)
    #second execution of the decoder with input the language code
    decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  #we use the result as the only input_id of the next iteration
    if(log):
        print(tokenizer.decode(result))

    while(out != 2):
        input_feed = {"input_ids": decoder_input_ids,
                     "encoder_attention_mask": encoder_attention_mask}
        count = 1
        initializer_count = 0
        for i in range(12):
            input_feed["past_key_values."+str(i)+".decoder.key"] = decoderOutput[count]
            count = count+1
            input_feed["past_key_values."+str(i)+".decoder.value"] = decoderOutput[count]
            count = count+1
            input_feed["past_key_values."+str(i)+".encoder.key"] = initialDecoderOutput[initializer_count]
            initializer_count = initializer_count+1
            input_feed["past_key_values."+str(i)+".encoder.value"] = initialDecoderOutput[initializer_count]
            initializer_count = initializer_count+1

        # esecuzione del decoder con cache
        decoderOutput = decoder_session.run(output_names, input_feed)

        test = decoderOutput[0][0][0]
        out = test.argmax()
        result.append(out)
        #prepariamo gli input del decoder per la prossima iterazione
        decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  #si usa il risultato come unico input_id della prossima iterazione
        if(log):
            print(tokenizer.decode(result))  # _convert_id_to_token(out)

    return tokenizer.decode(result)


def onnx_execution_nllb_cache_reduced_ram(text, src_lang, tgt_lang, encoder_path="onnx/NLLBOptimum/ReducedRam/Quantized/4bit_HQQ/nllb_encoder_4bit.onnx",
                                     decoder_path="onnx/NLLBOptimum/ReducedRam/Quantized/4bit_HQQ/nllb_decoder_4bit.onnx",
                                     initializer_path="onnx/NLLBOptimum/ReducedRam/Quantized/4bit_HQQ/nllb_cache_initializer_4bit.onnx",
                                     embed_path="onnx/NLLBOptimum/ReducedRam/Quantized/4bit_HQQ/nllb_embed_and_lm_head_4bit.onnx",
                                     encoder_session=None, decoder_session=None, initializer_session=None, embed_and_lm_head_session=None, log=True):
    #tokenizer loading
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang, tgt_lang=tgt_lang)
    # encoder session loading
    providers = ['CPUExecutionProvider']
    if(encoder_session is None):
        encoder_session = onnxruntime.InferenceSession(Path(encoder_path), providers=providers)
    # decoder session loading
    if(decoder_session is None):
        decoder_session = onnxruntime.InferenceSession(Path(decoder_path), providers=providers)
    # cache initializer session loading
    if(initializer_session is None):
        #initializer_session = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/NLLB_cache_initializer.onnx"), providers=providers)
        #initializer_session = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/Quantized/NLLB_cache_initializer_quantized.onnx"), providers=providers)
        initializer_session = onnxruntime.InferenceSession(Path(initializer_path), providers=providers)
    # embed_and_lm_head session loading
    if(embed_and_lm_head_session is None):
        #embed_and_lm_head_session = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/ReducedRAM/nllb_embed_and_lm_head_if3.onnx"), providers=providers)
        #embed_and_lm_head_session = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/ReducedRAM/Quantized/nllb_embed_and_lm_head_if2.onnx"), providers=providers)
        embed_and_lm_head_session = onnxruntime.InferenceSession(Path(embed_path), providers=providers)

    # we prepare the embed inputs
    inputEncoder = tokenizer(text, return_tensors='pt')
    encoder_input_ids = inputEncoder.input_ids.numpy()
    empty_pre_logits = torch.zeros(0, 1, 1024, dtype=torch.float32).numpy()
    empty_input_ids = torch.zeros(0, 4, dtype=torch.int64).numpy()
    # embed execution
    embedOuput = embed_and_lm_head_session.run(["embed_matrix"], {"use_lm_head": torch.tensor([False], dtype=torch.bool).numpy(),
                                                                  "input_ids": encoder_input_ids, "pre_logits": empty_pre_logits})

    # we prepare the encoder inputs
    encoder_attention_mask = inputEncoder.attention_mask.numpy()
    # encoder execution
    encoderOuput = encoder_session.run(["last_hidden_state"], {"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask,
                                                               "embed_matrix": embedOuput[0]})

    # we prepare the decoder inputs
    decoder_input_ids = torch.tensor([[2]], dtype=torch.int64).numpy()
    out = -1
    result = []

    output_names = []
    for i in range(12):
        output_names.append("present."+str(i)+".encoder.key")
        output_names.append("present."+str(i)+".encoder.value")

    # cache initializer execution
    initialDecoderOutput = initializer_session.run(output_names, {"encoder_hidden_states": encoderOuput[0]})

    # first execution of the decoder with input eos
    output_names = ["pre_logits"]
    for i in range(12):
        output_names.append("present."+str(i)+".decoder.key")
        output_names.append("present."+str(i)+".decoder.value")

    embedOuput = embed_and_lm_head_session.run(["embed_matrix"], {"use_lm_head": torch.tensor([False], dtype=torch.bool).numpy(),
                                                                  "input_ids": decoder_input_ids, "pre_logits": empty_pre_logits})

    input_feed = {"input_ids": decoder_input_ids,
                  "embed_matrix": embedOuput[0],
                  #"encoder_hidden_states": encoderOuput[0],
                  "encoder_attention_mask": encoder_attention_mask}
    count = 0
    for i in range(12):
        input_feed["past_key_values."+str(i)+".decoder.key"] = torch.zeros(1, 16, 0, 64, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".decoder.value"] = torch.zeros(1, 16, 0, 64, dtype=torch.float32).numpy()
        input_feed["past_key_values."+str(i)+".encoder.key"] = initialDecoderOutput[count]
        count = count+1
        input_feed["past_key_values."+str(i)+".encoder.value"] = initialDecoderOutput[count]
        count = count+1

    decoderOutput = decoder_session.run(output_names, input_feed)

    # second execution of the decoder with input the language code
    decoder_input_ids = torch.tensor([[tokenizer.convert_tokens_to_ids(tgt_lang)]], dtype=torch.int64).numpy()

    embedOuput = embed_and_lm_head_session.run(["embed_matrix"], {"use_lm_head": torch.tensor([False], dtype=torch.bool).numpy(),
                                                                  "input_ids": decoder_input_ids, "pre_logits": empty_pre_logits})

    input_feed = {"input_ids": decoder_input_ids,
                  "embed_matrix": embedOuput[0],
                  #"encoder_hidden_states": encoderOuput[0],
                  "encoder_attention_mask": encoder_attention_mask}
    count = 1
    initializer_count = 0
    for i in range(12):
        input_feed["past_key_values."+str(i)+".decoder.key"] = decoderOutput[count]
        count = count+1
        input_feed["past_key_values."+str(i)+".decoder.value"] = decoderOutput[count]
        count = count+1
        input_feed["past_key_values."+str(i)+".encoder.key"] = initialDecoderOutput[initializer_count]
        initializer_count = initializer_count+1
        input_feed["past_key_values."+str(i)+".encoder.value"] = initialDecoderOutput[initializer_count]
        initializer_count = initializer_count+1

    decoderOutput = decoder_session.run(output_names, input_feed)
    logitsOutput = embed_and_lm_head_session.run(["logits"], {"use_lm_head": torch.tensor([True], dtype=torch.bool).numpy(),
                                                                  "input_ids": empty_input_ids, "pre_logits": decoderOutput[0]})


    test = logitsOutput[0][0][0]
    out = test.argmax()
    result.append(out)
    # we prepare the decoder inputs for the next iteration
    decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  # we use the result as the only input_id of the next iteration (previous results are useless with kv cache)
    if(log):
        print(tokenizer.decode(result))

    while(out != 2):
        embedOuput = embed_and_lm_head_session.run(["embed_matrix"], {"use_lm_head": torch.tensor([False], dtype=torch.bool).numpy(),
                                                                  "input_ids": decoder_input_ids, "pre_logits": empty_pre_logits})
        input_feed = {"input_ids": decoder_input_ids,
                      "embed_matrix": embedOuput[0],
                      "encoder_attention_mask": encoder_attention_mask}
        count = 1
        initializer_count = 0
        for i in range(12):
            input_feed["past_key_values."+str(i)+".decoder.key"] = decoderOutput[count]
            count = count+1
            input_feed["past_key_values."+str(i)+".decoder.value"] = decoderOutput[count]
            count = count+1
            input_feed["past_key_values."+str(i)+".encoder.key"] = initialDecoderOutput[initializer_count]
            initializer_count = initializer_count+1
            input_feed["past_key_values."+str(i)+".encoder.value"] = initialDecoderOutput[initializer_count]
            initializer_count = initializer_count+1

        # execution of the decoder with cache
        decoderOutput = decoder_session.run(output_names, input_feed)
        # execution of the lm_head
        logitsOutput = embed_and_lm_head_session.run(["logits"], {"use_lm_head": torch.tensor([True], dtype=torch.bool).numpy(),
                                                                  "input_ids": empty_input_ids, "pre_logits": decoderOutput[0]})

        test = logitsOutput[0][0][0]
        out = test.argmax()
        result.append(out)
        # we prepare the decoder inputs for the next iteration
        decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  # we use the result as the only input_id of the next iteration (previous results are useless with kv cache)
        if(log):
            print(tokenizer.decode(result))

    return tokenizer.decode(result)


class ModelType(enum.Enum):
    NLLB = 1
    MADLAD = 2

def compare_models_quality(
        initializer_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx",
        encoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx",
        decoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx",
        embed_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx",
        initializer_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_cache_initializer_4bit.onnx",
        encoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_encoder_4bit.onnx",
        decoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_decoder_4bit.onnx",
        embed_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_embed_8bit.onnx",
        data_dir = "en-it", src_lan="eng_Latn", tgt_lan="it", modelType = ModelType.MADLAD, logFile = False, logFileFolder = "/Quality/HQQPerf/", logFileName = "madlad_quality_HQQPerf"
    ):
    
    dataset = datasets.load_dataset("opus100", split="validation[150:250]", data_dir=data_dir)

    providers = ['CPUExecutionProvider']

    initializer_session = onnxruntime.InferenceSession(Path(initializer_path), providers=providers)
    encoder_session = onnxruntime.InferenceSession(Path(encoder_path), providers=providers)
    decoder_session = onnxruntime.InferenceSession(Path(decoder_path), providers=providers)
    embed_session = onnxruntime.InferenceSession(Path(embed_path), providers=providers)

    initializer_session_quantized = onnxruntime.InferenceSession(Path(initializer_quant_path), providers=providers)
    encoder_session_quantized = onnxruntime.InferenceSession(Path(encoder_quant_path), providers=providers)
    decoder_session_quantized = onnxruntime.InferenceSession(Path(decoder_quant_path), providers=providers)
    embed_session_quantized = onnxruntime.InferenceSession(Path(embed_quant_path), providers=providers)

    src = data_dir.split('-')[0]

    levenshtein = NormalizedLevenshtein()
    sentencesEqual = 0
    sentencesDifferent = 0
    similarity_score_avg = 0
    for dataComplete in dataset:
        data = dataComplete['translation']
        if("íhVÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿþÿÿÿReceive Time OutðËìv ÿÿÿÿÿÿÿÿÿÿÿÿ" in data[src]): continue  #used to skip a problematic input
        try:
            if(modelType == ModelType.NLLB):
                #esecuzione con encoder e decoder normali
                result = onnx_execution_nllb_cache_reduced_ram(data[src], src_lan, tgt_lan, encoder_session=encoder_session, decoder_session=decoder_session, initializer_session=initializer_session, embed_and_lm_head_session=embed_session, log=False)
                #esecuzione con encoder e decoder quantizzati
                result_quantized = onnx_execution_nllb_cache_reduced_ram(data[src], src_lan, tgt_lan, encoder_session=encoder_session_quantized, decoder_session=decoder_session_quantized, initializer_session=initializer_session_quantized, embed_and_lm_head_session=embed_session_quantized, log=False)
            else:
                #esecuzione con encoder e decoder normali
                result = onnx_execution_madlad_cache_reduced_ram(data[src], tgt_lan, encoder_session=encoder_session, decoder_session=decoder_session, initializer_session=initializer_session, embed_session=embed_session, log=False, cacheResults=True)
                #esecuzione con encoder e decoder quantizzati
                result_quantized = onnx_execution_madlad_cache_reduced_ram(data[src], tgt_lan, encoder_session=encoder_session_quantized, decoder_session=decoder_session_quantized, initializer_session=initializer_session_quantized, embed_session=embed_session_quantized, log=False, cacheResults=False)
            
            similarity_score = 1
            if(result != result_quantized):
                print('')
                print("The sentence |"+data[src]+"| is different")
                print("Normal: "+result)
                print("Quantized: "+result_quantized)
                sentencesDifferent = sentencesDifferent+1
                similarity_score = levenshtein.similarity(result, result_quantized)
                print("Similarity Score: "+str(similarity_score))
                if(logFile):
                    with open(logFileFolder + logFileName + "_" + data_dir + ".txt", 'a+') as f:
                        f.write("\n")
                        f.write("The sentence |"+data[src]+"| is different\n")
                        f.write("Normal: "+result+"\n")
                        f.write("Quantized: "+result_quantized+"\n")
                        f.write("Similarity Score: "+str(similarity_score)+"\n")
            else:
                #print("The sentence |"+result+"| is equal")
                sentencesEqual = sentencesEqual+1

            similarity_score_avg = similarity_score_avg + similarity_score
        except:
            print("Error translating: "+data[src])

    similarity_score_avg = similarity_score_avg/len(dataset)

    print('')
    print('')
    print("Sentences equals: "+str(sentencesEqual))
    print("Sentences different: "+str(sentencesDifferent))
    print("Similarity score avg: "+str(similarity_score_avg))
    
    if(logFile):
        with open(logFileFolder + logFileName + "_" + data_dir + ".txt", 'a+') as f:
            f.write("\n\n")
            f.write("Sentences equals: "+str(sentencesEqual)+"\n")
            f.write("Sentences different: "+str(sentencesDifferent)+"\n")
            f.write("Similarity score avg: "+str(similarity_score_avg)+"\n")
    
    return similarity_score_avg


def compare_models_quality_multi_language(
        initializer_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx",
        encoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx",
        decoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx",
        embed_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx",
        initializer_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_cache_initializer_4bit.onnx",
        encoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_encoder_4bit.onnx",
        decoder_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_decoder_4bit.onnx",
        embed_quant_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/HQQPerf/madlad_embed_8bit.onnx",
        modelType = ModelType.MADLAD, logFile = True, logFileFolder = "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Quality/HQQPerf/", logFileName = "madlad_quality_HQQPerf"
    ):
    
    if(logFile):
        os.makedirs(logFileFolder, exist_ok=True)
    
    if(modelType == ModelType.NLLB):
        similarity_score_en_it = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="en-it", src_lan="eng_Latn", tgt_lan="ita_Latn", modelType=ModelType.NLLB,
                                                        logFile=logFile, logFileFolder=logFileFolder, logFileName=logFileName)  #english to italian
        similarity_score_en_zh = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="en-zh", src_lan="eng_Latn", tgt_lan="zho_Hans", modelType=ModelType.NLLB,
                                                        logFile=logFile, logFileFolder=logFileFolder, logFileName=logFileName)  #english to chinese (simplified)
        similarity_score_en_ja = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="en-ja", src_lan="eng_Latn", tgt_lan="jpn_Jpan", modelType=ModelType.NLLB,
                                                        logFile=logFile, logFileFolder=logFileFolder, logFileName=logFileName)  #english to japanese
        similarity_score_en_es = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="en-es", src_lan="eng_Latn", tgt_lan="spa_Latn", modelType=ModelType.NLLB,
                                                        logFile=logFile, logFileFolder=logFileFolder, logFileName=logFileName)  #english to spanish
        similarity_score_en_fr = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="en-fr", src_lan="eng_Latn", tgt_lan="fra_Latn", modelType=ModelType.NLLB,
                                                        logFile=logFile, logFileFolder=logFileFolder, logFileName=logFileName)  #english to french
        similarity_score_de_en = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="de-en", src_lan="deu_Latn", tgt_lan="eng_Latn", modelType=ModelType.NLLB,
                                                        logFile=logFile, logFileFolder=logFileFolder, logFileName=logFileName)  #german to english
    else:
        similarity_score_en_it = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="en-it", tgt_lan="it", logFile=logFile,
                                                        logFileFolder=logFileFolder, logFileName=logFileName)   #english to italian
        similarity_score_en_zh = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="en-zh", tgt_lan="zh", logFile=logFile,
                                                        logFileFolder=logFileFolder, logFileName=logFileName)  #english to chinese (simplified)
        similarity_score_en_ja = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="en-ja", tgt_lan="jp", logFile=logFile,
                                                        logFileFolder=logFileFolder, logFileName=logFileName)  #english to japanese
        similarity_score_en_es = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="en-es", tgt_lan="es", logFile=logFile,
                                                        logFileFolder=logFileFolder, logFileName=logFileName)  #english to spanish
        similarity_score_en_fr = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="en-fr", tgt_lan="fr", logFile=logFile,
                                                        logFileFolder=logFileFolder, logFileName=logFileName)  #english to french
        similarity_score_de_en = compare_models_quality(initializer_path, encoder_path, decoder_path, embed_path, initializer_quant_path,
                                                        encoder_quant_path, decoder_quant_path, embed_quant_path,
                                                        data_dir="de-en", tgt_lan="en", logFile=logFile,
                                                        logFileFolder=logFileFolder, logFileName=logFileName)  #german to english

    similarity_score_avg = (similarity_score_en_it + similarity_score_en_zh + similarity_score_en_ja + similarity_score_en_es + similarity_score_en_fr + similarity_score_de_en)/6

    print("TOTAL Similarity score avg: "+str(similarity_score_avg))
    if(logFile):
        with open(logFileFolder + logFileName + ".txt", 'a+') as f:
            f.write("TOTAL Similarity score avg: "+str(similarity_score_avg)+"\n")



