from typing import Dict, List
from onnxruntime.quantization import CalibrationDataReader
import numpy as np
import torch
from transformers import T5Tokenizer
import onnxruntime
from pathlib import Path



class MadladEncoderCalibrationDataReader(CalibrationDataReader):
    def __init__(self, samples: List[Dict[str, str]], targetLanguages: List[str], tokenizer_name="google/madlad400-3b-mt", embed_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx", max_source_length=128):
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED    #ORT_ENABLE_EXTENDED
        embed_session = onnxruntime.InferenceSession(Path(embed_path), sess_options=sess_options, providers=['CPUExecutionProvider'])

        self.encoded = []
        for s in samples:
            trgLang = targetLanguages.pop()
            text = f"<2{trgLang}> {s['text']}" 
            inputEncoder = tokenizer(text, max_length=max_source_length, padding='max_length', return_tensors='pt')
            encoder_input_ids = inputEncoder.input_ids.numpy()
            encoder_attention_mask = inputEncoder.attention_mask.numpy()
            embedOuput = embed_session.run(["embed_matrix"], {"input_ids": encoder_input_ids})
            self.encoded.append({
                "input_ids": encoder_input_ids,
                "attention_mask": encoder_attention_mask,
                "embed_matrix": embedOuput[0]})
        self._idx = 0


    def get_next(self):
        if self._idx >= len(self.encoded):
            return None
        x = self.encoded[self._idx]
        self._idx += 1
        return x
    
    def reset(self):
        self._idx = 0
    

class MadladDecoderCalibrationDataReader(CalibrationDataReader):
    def __init__(self, samples: List[Dict[str, str]], targetLanguages: List[str],
                tokenizer_name="google/madlad400-3b-mt", embed_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/madlad_embed.onnx",
                encoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx",
                decoder_path="onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx",
                initializer_path="onnx/Madlad/Optimum_Cache_Optimized/cache_initializer.onnx", max_source_length=128):
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED    #ORT_ENABLE_EXTENDED
        embed_session = onnxruntime.InferenceSession(Path(embed_path), sess_options=sess_options, providers=['CPUExecutionProvider'])
        encoder_session = onnxruntime.InferenceSession(Path(encoder_path), sess_options=sess_options, providers=['CPUExecutionProvider'])
        decoder_session = onnxruntime.InferenceSession(Path(decoder_path), sess_options=sess_options, providers=['CPUExecutionProvider'])
        initializer_session = onnxruntime.InferenceSession(Path(initializer_path), sess_options=sess_options, providers=['CPUExecutionProvider'])

        self.encoded = []
        for s in samples:
            trgLang = targetLanguages.pop()
            text = f"<2{trgLang}> {s['text']}" 

            #prepariamo gli input dell'encoder
            inputEncoder = tokenizer(text, max_length=max_source_length, padding='max_length', return_tensors='pt')
            encoder_input_ids = inputEncoder.input_ids.numpy()
            encoder_attention_mask = inputEncoder.attention_mask.numpy()

            #esecuzione dell'encoder
            embedOuput = embed_session.run(["embed_matrix"], {"input_ids": encoder_input_ids})
            encoderOuput = encoder_session.run(["last_hidden_state"], {"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask, "embed_matrix": embedOuput[0]})

            # prepariamo gli input del decoder
            decoder_input_ids = torch.tensor([[0]], dtype=torch.int64).numpy()
            out = -1
            result = []

            output_names = []
            for i in range(32):
                output_names.append("present."+str(i)+".encoder.key")
                output_names.append("present."+str(i)+".encoder.value")

            # esecuzione del cache initializer
            initialDecoderOutput = initializer_session.run(output_names, {"encoder_hidden_states": encoderOuput[0]})
            #prima esecuzione usando solo il decoder con past
            embedOuput = embed_session.run(["embed_matrix"], {"input_ids": decoder_input_ids})
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

            decoderOutput = decoder_session.run(output_names, input_feed)

            test = decoderOutput[0][0][0]
            out = test.argmax()
            result.append(out)
            #prepariamo gli input del decoder per la prossima iterazione
            decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  #si usa il risultato come unico input_id della prossima iterazione

            while(out != 2):
                embedOuput = embed_session.run(["embed_matrix"], {"input_ids": decoder_input_ids})
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

                if(len(result) >= len(encoder_input_ids[0])*0.5):
                    self.encoded.append(input_feed)  #forniamo gli input per la calibrazione
                    break
                # esecuzione del decoder con cache
                decoderOutput = decoder_session.run(output_names, input_feed)

                test = decoderOutput[0][0][0]
                out = test.argmax()
                result.append(out)

                if(out == 2):
                    self.encoded.append(input_feed)  #forniamo gli input per la calibrazione
                    break

                #prepariamo gli input del decoder per la prossima iterazione
                decoder_input_ids = torch.tensor([[out]], dtype=torch.int64).numpy()  #si usa il risultato come unico input_id della prossima iterazione


        self._idx = 0


    def get_next(self):
        if self._idx >= len(self.encoded):
            return None
        x = self.encoded[self._idx]
        self._idx += 1
        return x
    
    def reset(self):
        self._idx = 0