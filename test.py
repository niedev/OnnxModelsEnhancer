import torch
from pathlib import Path
import onnxruntime
from transformers import NllbTokenizer

def translate(text, src_lang, tgt_lang, encoder_path="onnx/NLLBOptimum/Optimized/ReducedRAM/encoder_model.onnx",
                                     decoder_path="onnx/NLLBOptimum/Optimized/ReducedRAM/decoder_model.onnx",
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
        initializer_session = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/Quantized/NLLB_cache_initializer_quantized.onnx"), providers=providers)
    # embed_and_lm_head session loading
    if(embed_and_lm_head_session is None):
        embed_and_lm_head_session = onnxruntime.InferenceSession(Path("onnx/NLLBOptimum/Optimized/ReducedRAM/nllb_embed_and_lm_head_if3.onnx"), providers=providers)

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