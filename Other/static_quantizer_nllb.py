import functools
import os

import numpy
import torch

import onnx
import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType, write_calibration_table
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QDQ, default_quantization_parameters, \
    QuantizationConfig, AutoCalibrationConfig, ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QOPS
from transformers import T5Tokenizer, NllbTokenizer, BatchEncoding


def preprocess_fn_NLLB(ex, tokenizer: NllbTokenizer, encoder_session: onnxruntime.InferenceSession):
    decoder_input_ids_list = []
    decoder_attention_mask_list = []
    encoder_hidden_states_list = []
    encoder_attention_mask_list = []
    #inseriamo l' input parola per parola della seguente frase demo
    en_text = "Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb."
    it_text = "Inoltre, a differenza del 2014, non ci sono quasi tante lacune. Non si può semplicemente acquistare una lampadina a 150 watt o una lampadina a tre vie."
    # prepariamo gli input dell'encoder
    inputEncoder = tokenizer(en_text, max_length=256, padding='max_length', return_tensors='pt')
    encoder_input_ids = inputEncoder.input_ids.numpy()
    encoder_attention_mask = inputEncoder.attention_mask.numpy()
    # esecuzione dell'encoder
    encoderOuput = encoder_session.run(["last_hidden_state"],{"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask})
    # prepariamo gli input del decoder
    inputDecoder = tokenizer(text_target=it_text, max_length=255, padding='max_length', return_tensors='pt')
    eos_position = (inputDecoder.input_ids[0] == 2).nonzero(as_tuple=True)[0][0] + 1
    # si modificano gli input_ids e la attention _mask prodotti dal tokenizer per renderli compatibili con il decoder
    decoder_input_ids_pt = torch.cat([torch.tensor([[2]]), inputDecoder.input_ids], dim=1)  #aggiungiamo un <eos> all' inizio
    decoder_input_ids_pt[0][eos_position] = 1  # sostituiamo l' ultimo 2 con il pad (1)
    decoder_attention_mask_pt = torch.cat([inputDecoder.attention_mask, torch.tensor([[0]])],dim=1)  # aggiungiamo uno 0 alla fine

    i = 0
    while(i < eos_position-1):
        decoder_input_ids_cut = torch.clone(decoder_input_ids_pt)
        decoder_attention_mask_cut = torch.clone(decoder_attention_mask_pt)
        j = 0
        while(2+i+j < eos_position):
            decoder_input_ids_cut[0][2+i+j] = 1  # sostituiamo il token alla posizione 2+i+j col pad (1)
            decoder_attention_mask_cut[0][2+i+j] = 0  # sostituiamo l' 1 alla posizione 2+i+j con 0 (dato che sopra abbiamo "eliminato" il token alla posizione 2+i+j)
            j = j+1
        decoder_input_ids = decoder_input_ids_cut.numpy()
        decoder_attention_mask = decoder_attention_mask_cut.numpy()
        decoder_input_ids_list.append(decoder_input_ids[0].tolist())
        decoder_attention_mask_list.append(decoder_attention_mask[0].tolist())
        encoder_hidden_states_list.append(encoderOuput[0][0].tolist())
        encoder_attention_mask_list.append(encoder_attention_mask[0].tolist())
        i = i+1

    #inseriamo il resto delle frasi (provenienti dal dataset)
    i = 38
    while (i < len(ex["translation"])):
        # prepariamo gli input dell'encoder
        text_en = ex["translation"][i]["en"]
        text_it = ex["translation"][i]["it"]
        inputEncoder = tokenizer(text_en, max_length=256, padding='max_length', return_tensors='pt')
        encoder_input_ids = inputEncoder.input_ids.numpy()
        encoder_attention_mask = inputEncoder.attention_mask.numpy()
        # esecuzione dell'encoder
        encoderOuput = encoder_session.run(["last_hidden_state"],
                                           {"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask})
        # prepariamo gli input del decoder
        inputDecoder = tokenizer(text_target=text_it, max_length=255, padding='max_length', return_tensors='pt')
        eos_position = (inputDecoder.input_ids[0] == 2).nonzero(as_tuple=True)[0][0] + 1
        # si modificano gli input_ids e la attention _mask prodotti dal tokenizer per renderli compatibili con il decoder
        decoder_input_ids_pt = torch.cat([torch.tensor([[2]]), inputDecoder.input_ids],dim=1)  # aggiungiamo un <eos> all' inizio
        decoder_input_ids_pt[0][eos_position] = 1  # sostituiamo l' ultimo 2 con il pad (1)
        decoder_attention_mask_pt = torch.cat([inputDecoder.attention_mask, torch.tensor([[0]])],dim=1)  # aggiungiamo uno 0 alla fine
        if (eos_position > 4):
            decoder_input_ids_pt[0][
                eos_position - 1] = 1  # sostituiamo l'ultimo token col pad (1) (cosi il modello non deve indovinare sempre <eos>)
            decoder_input_ids_pt[0][
                eos_position - 2] = 1  # sostituiamo il penultimo token col pad (1) (cosi il modello non deve indovinare spesso un punto (.)))
            decoder_attention_mask_pt[0][
                eos_position - 1] = 0  # sostituiamo l'ultimo 1 con 0 (dato che sopra abbiamo "eliminato" l'ultimo token)
            decoder_attention_mask_pt[0][
                eos_position - 2] = 0  # sostituiamo il penultimo 1 con 0 (dato che sopra abbiamo "eliminato" il penultimo token)
        decoder_input_ids = decoder_input_ids_pt.numpy()
        decoder_attention_mask = decoder_attention_mask_pt.numpy()
        # si aggiungono gli input del decoder nelle loro liste
        decoder_input_ids_list.append(decoder_input_ids[0].tolist())
        decoder_attention_mask_list.append(decoder_attention_mask[0].tolist())
        encoder_hidden_states_list.append(encoderOuput[0][0].tolist())
        encoder_attention_mask_list.append(encoder_attention_mask[0].tolist())
        i = i + 1

    result = BatchEncoding({"input_ids": decoder_input_ids_list,
            "attention_mask": decoder_attention_mask_list,
            "encoder_hidden_states": encoder_hidden_states_list,
            "encoder_attention_mask": encoder_attention_mask_list}, n_sequences=1)  #, tensor_type="np"

    return result
