import functools
import os
from pathlib import Path

import numpy
import torch

import madlad_modifier
import onnx
import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType, write_calibration_table
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QDQ, default_quantization_parameters, \
    QuantizationConfig, AutoCalibrationConfig, ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QOPS, \
    ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION
from transformers import T5Tokenizer, BatchEncoding


def preprocess_fn_Madlad(ex, tokenizer: T5Tokenizer, encoder_session: onnxruntime.InferenceSession):
    decoder_input_ids_list = []
    decoder_attention_mask_list = []
    encoder_hidden_states_list = []
    encoder_attention_mask_list = []
    #inseriamo l' input parola per parola della seguente frase demo
    en_text = "<2it> Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb."
    it_text = "Inoltre, a differenza del 2014, non ci sono quasi tante lacune. Non si può semplicemente acquistare una lampadina a 150 watt o una lampadina a tre vie."
    # prepariamo gli input dell'encoder
    inputEncoder = tokenizer(en_text, max_length=128, padding='max_length', return_tensors='pt')
    encoder_input_ids = inputEncoder.input_ids.numpy()
    encoder_attention_mask = inputEncoder.attention_mask.numpy()
    # esecuzione dell'encoder
    encoderOuput = encoder_session.run(["last_hidden_state"],{"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask})
    # prepariamo gli input del decoder
    inputDecoder = tokenizer(it_text, max_length=127, padding='max_length', return_tensors='pt')
    eos_position = (inputDecoder.input_ids[0] == 2).nonzero(as_tuple=True)[0][0] + 1
    # si modificano gli input_ids e la attention _mask prodotti dal tokenizer per renderli compatibili con il decoder
    decoder_input_ids_pt = torch.cat([torch.tensor([[0]]), inputDecoder.input_ids], dim=1)  #aggiungiamo uno 0 all' inizio
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
        decoder_input_ids = decoder_input_ids_cut.numpy().astype(numpy.int32)
        decoder_attention_mask = decoder_attention_mask_cut.numpy().astype(numpy.int32)
        decoder_input_ids_list.append(decoder_input_ids[0].tolist())
        decoder_attention_mask_list.append(decoder_attention_mask[0].tolist())
        encoder_hidden_states_list.append(encoderOuput[0][0].tolist())
        encoder_attention_mask_list.append(encoder_attention_mask[0].astype(numpy.int32).tolist())
        i = i+1

    #inseriamo il resto delle frasi (provenienti dal dataset)
    i = 44
    while (i < len(ex["translation"])):
        # prepariamo gli input dell'encoder
        text_en = "<2it> "+ex["translation"][i]["en"]
        text_it = ex["translation"][i]["it"]
        inputEncoder = tokenizer(text_en, max_length=128, padding='max_length', return_tensors='pt')
        encoder_input_ids = inputEncoder.input_ids.numpy()
        encoder_attention_mask = inputEncoder.attention_mask.numpy()
        # esecuzione dell'encoder
        encoderOuput = encoder_session.run(["last_hidden_state"],
                                           {"input_ids": encoder_input_ids, "attention_mask": encoder_attention_mask})
        # prepariamo gli input del decoder
        inputDecoder = tokenizer(text_it, max_length=127, padding='max_length', return_tensors='pt')
        eos_position = (inputDecoder.input_ids[0] == 2).nonzero(as_tuple=True)[0][0] + 1
        # si modificano gli input_ids e la attention _mask prodotti dal tokenizer per renderli compatibili con il decoder
        decoder_input_ids_pt = torch.cat([torch.tensor([[0]]), inputDecoder.input_ids],dim=1)  # aggiungiamo uno 0 all' inizio
        decoder_input_ids_pt[0][eos_position] = 1  # sostituiamo l' ultimo 2 con il pad (1)
        decoder_attention_mask_pt = torch.cat([inputDecoder.attention_mask, torch.tensor([[0]])],dim=1)  # aggiungiamo uno 0 alla fine
        if (eos_position > 4):
            decoder_input_ids_pt[0][eos_position - 1] = 1  # sostituiamo l'ultimo token col pad (1) (cosi il modello non deve indovinare sempre <eos>)
            decoder_input_ids_pt[0][eos_position - 2] = 1  # sostituiamo il penultimo token col pad (1) (cosi il modello non deve indovinare spesso un punto (.)))
            decoder_attention_mask_pt[0][eos_position - 1] = 0  # sostituiamo l'ultimo 1 con 0 (dato che sopra abbiamo "eliminato" l'ultimo token)
            decoder_attention_mask_pt[0][eos_position - 2] = 0  # sostituiamo il penultimo 1 con 0 (dato che sopra abbiamo "eliminato" il penultimo token)
        decoder_input_ids = decoder_input_ids_pt.numpy().astype(numpy.int32)
        decoder_attention_mask = decoder_attention_mask_pt.numpy().astype(numpy.int32)
        # si aggiungono gli input del decoder nelle loro liste
        decoder_input_ids_list.append(decoder_input_ids[0].tolist())
        decoder_attention_mask_list.append(decoder_attention_mask[0].tolist())
        encoder_hidden_states_list.append(encoderOuput[0][0].tolist())
        encoder_attention_mask_list.append(encoder_attention_mask[0].astype(numpy.int32).tolist())
        i = i + 1

    result = BatchEncoding({"input_ids": decoder_input_ids_list,
            "attention_mask": decoder_attention_mask_list,
            "encoder_hidden_states": encoder_hidden_states_list,
            "encoder_attention_mask": encoder_attention_mask_list}, n_sequences=1)  #, tensor_type="np"

    return result


def preprocess_fn_encoder(ex, tokenizer: T5Tokenizer):
    input_ids_list = []
    attention_mask_list = []
    #inseriamo l' input parola per parola della seguente frase demo
    en_text = "<2it> Also unlike 2014, there aren’t nearly as many loopholes. You can’t just buy a 150-watt incandescent or a three-way bulb, the ban covers any normal bulb that generates less than 45 lumens per watt, which pretty much rules out both incandescent and halogen tech in their entirety."
    # prepariamo gli input dell'encoder
    inputEncoder = tokenizer(en_text, max_length=128, padding='max_length', return_tensors='pt')
    encoder_input_ids = inputEncoder.input_ids.numpy()
    encoder_attention_mask = inputEncoder.attention_mask.numpy()
    # si aggiungono gli input dell'encoder nelle loro liste
    input_ids_list.append(encoder_input_ids[0].tolist())
    attention_mask_list.append(encoder_attention_mask[0].tolist())

    #inseriamo il resto delle frasi (provenienti dal dataset)
    i = 1
    while (i < len(ex["translation"])):
        # prepariamo gli input dell'encoder
        text_en = "<2it> "+ex["translation"][i]["en"]
        inputEncoder = tokenizer(text_en, max_length=128, padding='max_length', return_tensors='pt')
        encoder_input_ids = inputEncoder.input_ids.numpy()
        encoder_attention_mask = inputEncoder.attention_mask.numpy()
        # si aggiungono gli input dell'encoder nelle loro liste
        input_ids_list.append(encoder_input_ids[0].tolist())
        attention_mask_list.append(encoder_attention_mask[0].tolist())
        i = i + 1

    result = BatchEncoding({"input_ids": input_ids_list,
            "attention_mask": attention_mask_list}, n_sequences=1)  #, tensor_type="np"

    return result



def static_quantization_optimum_QLinear_only():
    model_out_name = "Madlad_decoder_2D.onnx"
    model_dir = "onnx/Madlad/Optimized/Optimized2D"
    model_name = 'jbochi/madlad400-3b-mt'
    quantizer = ORTQuantizer.from_pretrained(model_dir, file_name=model_out_name)
    all=ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QOPS
    all.remove('Add')
    all.remove('Softmax')
    all.remove('Mul')
    format, mode, operators_to_quantize = default_quantization_parameters(True, operators_to_quantize=all)
    #nodes_to_exclude = []
    #for i in range(0,11):
        #nodes_to_exclude.append("/layers."+str(i)+"/self_attn/Reshape_7")
        #nodes_to_exclude.append("/layers."+str(i)+"/encoder_attn/Reshape_7")

    qconfig = QuantizationConfig(
        is_static=True,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.QLinearOps,
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=True,
        weights_dtype=QuantType.QUInt8,
        weights_symmetric=True,
        per_channel=False,
        reduce_range=False,
        operators_to_quantize=operators_to_quantize,
    )
    #qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
    # caricamento tokenizer
    #tokenizerNLLBen = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn",tgt_lang="ita_Latn")
    tokenizerEn = T5Tokenizer.from_pretrained(model_name)
    # caricamento encoder session
    providers = ['CPUExecutionProvider']  # 'ROCMExecutionProvider',
    encoder_session = onnxruntime.InferenceSession("onnx/Madlad/Optimum/encoder_model.onnx", providers=providers)

    # Create the calibration dataset
    calibration_samples = 120   #38 per il preprocess_fn_test
    calibration_dataset = quantizer.get_calibration_dataset(
        "opus100",
        dataset_config_name="en-it",
        preprocess_function=functools.partial(preprocess_fn_Madlad, tokenizer=tokenizerEn,encoder_session=encoder_session),
        num_samples=calibration_samples,
        dataset_split="train",
        # preprocess_batch=False
    )

    # Create the calibration configuration containing the parameters related to calibration.
    #calibration_config = AutoCalibrationConfig.percentiles(calibration_dataset, percentile=99.999)
    #calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
    calibration_config = AutoCalibrationConfig.entropy(calibration_dataset)

    # liberare la ram da tutte le risorse inutili (ad esempio la sessione dell'encoder), misurare quanta `e occupata in questo punto (e capire quanta posso liberarne)
    del encoder_session

    # Perform the calibration step: computes the activations quantization ranges (RAM optimized)
    shards = 4   #in quanti partial_fit dividiamo il calibration_dataset
    for i in range(shards):
        shard = calibration_dataset.shard(shards, i)
        quantizer.partial_fit(
            dataset=shard,
            calibration_config=calibration_config,
            operators_to_quantize=qconfig.operators_to_quantize,
            batch_size=1,   #calibration_samples//shards
            use_external_data_format=True,
        )
    ranges = quantizer.compute_ranges()

    listofComponentsOutOfRange = []
    listofComponentsMildOutOfRange = []
    listofComponentsNotPerfect = []
    namesofComponentsOutOfRange = []
    namesofComponentsMildOutOfRange = []
    namesofComponentsNotPerfect = []
    keys = list(ranges)
    for key in keys:
        data = ranges[key]
        highest = data.highest         #data["highest"]
        lowest = data.lowest           #data["lowest"]
        if(highest > 100000 or lowest < -100000):
            listofComponentsOutOfRange.append({key: [lowest, highest]})
            namesofComponentsOutOfRange.append(key)
        elif(highest > 900 or lowest < -900):
            listofComponentsMildOutOfRange.append({key: [lowest, highest]})
            namesofComponentsMildOutOfRange.append(key)
        elif (highest > 255 or lowest < -255):
            listofComponentsNotPerfect.append({key: [lowest, highest]})
            namesofComponentsNotPerfect.append(key)

    print("Components Out of Range:")
    print(namesofComponentsOutOfRange)
    for component in listofComponentsOutOfRange:
        print(component)
    print('')
    print("Components Mildly Out of Range:")
    print(namesofComponentsMildOutOfRange)
    for component in listofComponentsMildOutOfRange:
        print(component)
    print('')
    print("Components Not Perfect:")
    print(namesofComponentsNotPerfect)
    for component in listofComponentsNotPerfect:
        print(component)
    print('')

    try:
        write_calibration_table(ranges.data, dir=model_dir)
        print("Calibration is done. Calibration cache is saved to calibration.json")
    except:
        print("Failed saving calibration data")

    # remove temp augmented model again
    os.remove("augmented_model.onnx")

    model_quantized_path = quantizer.quantize(
        save_dir="onnx/Madlad/Script/StaticQuantization/Test/QLinearOnly",
        calibration_tensors_range=ranges,
        quantization_config=qconfig,
        use_external_data_format=False
    )

    modelOnnx = onnx.load_model(Path(str(model_quantized_path)+"/Madlad_decoder_2D_quantized.onnx"))
    onnx.save_model(modelOnnx, Path(str(model_quantized_path)+"/Madlad_decoder_2D_quantized2.onnx"))
    onnx.save_model(modelOnnx, Path("onnx/Madlad/Script/StaticQuantization/Test/Madlad_decoder_quantized_static_test.onnx"))


def static_quantization_optimum_QDQ():
    model_out_name = "Madlad_decoder_2D_quantized.onnx"
    model_dir = "onnx/Madlad/Script/StaticQuantization/Test/QLinearGather"
    model_name = 'jbochi/madlad400-3b-mt'
    quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="Madlad_decoder_2D_quantized.onnx")
    operators_to_quantize_in = ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QDQ
    #operators_to_quantize_in.remove('Where')
    #operators_to_quantize_in.remove('Reshape')
    #operators_to_quantize_in.remove('Add')   (non presente)
    operators_to_quantize_in.remove('Softmax')

    operators_to_quantize_in.remove('Gather')  #(gia quantizzato)
    #operators_to_quantize_in.remove('Mul')   (non presente)
    #operators_to_quantize_in.remove('Sub')   (non presente)
    #operators_to_quantize_in.remove('Neg')   (non presente)
    #operators_to_quantize_in.remove('Log')   (non presente)
    #operators_to_quantize_in.remove('Div')   (non presente)
    #operators_to_quantize_in.remove('Cast')   (non presente)
    #operators_to_quantize_in.remove('Transpose')
    #operators_to_quantize_in.remove('Min')   (non presente)
    #operators_to_quantize_in.remove('Gather')

    #operators_to_quantize_in.remove('Unsqueeze')
    #operators_to_quantize_in.remove('Squeeze')

    #operators_to_quantize_in.remove('Cast')   (non presente)
    #operators_to_quantize_in.remove('Reshape')   (non presente)
    format, mode, operators_to_quantize = default_quantization_parameters(True,operators_to_quantize=operators_to_quantize_in)
    #nodes_to_exclude = []
    #for i in range(0,11):
        #nodes_to_exclude.append("/layers."+str(i)+"/self_attn/Reshape_7")
        #nodes_to_exclude.append("/layers."+str(i)+"/encoder_attn/Reshape_7")

    ret = madlad_modifier.addReshapes_partial(model_dir + "/" + model_out_name)
    excluded_matmuls = ret[1]

    qconfig = QuantizationConfig(
        is_static=True,
        format=QuantFormat.QDQ,
        mode=QuantizationMode.QLinearOps,
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=True,
        weights_dtype=QuantType.QUInt8,
        weights_symmetric=True,
        per_channel=False,
        reduce_range=False,
        nodes_to_exclude=excluded_matmuls,
        operators_to_quantize=operators_to_quantize,
    )
    #qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
    # caricamento tokenizer
    #tokenizerNLLBen = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn",tgt_lang="ita_Latn")
    tokenizerEn = T5Tokenizer.from_pretrained(model_name)
    # caricamento encoder session
    providers = ['CPUExecutionProvider']  # 'ROCMExecutionProvider',
    encoder_session = onnxruntime.InferenceSession("onnx/Madlad/Optimum/encoder_model.onnx",providers=providers)

    # Create the calibration dataset
    calibration_samples = 120   #38 per il preprocess_fn_test
    calibration_dataset = quantizer.get_calibration_dataset(
        "opus100",
        dataset_config_name="en-it",
        preprocess_function=functools.partial(preprocess_fn_Madlad, tokenizer=tokenizerEn,encoder_session=encoder_session),
        num_samples=calibration_samples,
        dataset_split="train",
        # preprocess_batch=False
    )

    # Create the calibration configuration containing the parameters related to calibration.
    #calibration_config = AutoCalibrationConfig.percentiles(calibration_dataset, percentile=99.999)
    #calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
    calibration_config = AutoCalibrationConfig.entropy(calibration_dataset)

    # liberare la ram da tutte le risorse inutili (ad esempio la sessione dell'encoder), misurare quanta `e occupata in questo punto (e capire quanta posso liberarne)
    del encoder_session

    # Perform the calibration step: computes the activations quantization ranges (RAM optimized)
    shards = 4   #in quanti partial_fit dividiamo il calibration_dataset
    for i in range(shards):
        shard = calibration_dataset.shard(shards, i)
        quantizer.partial_fit(
            dataset=shard,
            calibration_config=calibration_config,
            operators_to_quantize=qconfig.operators_to_quantize,
            batch_size=1,   #calibration_samples//shards
            use_external_data_format=True,
        )
    ranges = quantizer.compute_ranges()

    listofComponentsOutOfRange = []
    listofComponentsMildOutOfRange = []
    listofComponentsNotPerfect = []
    namesofComponentsOutOfRange = []
    namesofComponentsMildOutOfRange = []
    namesofComponentsNotPerfect = []
    keys = list(ranges)
    for key in keys:
        data = ranges[key]
        highest = data.highest         #data["highest"]
        lowest = data.lowest           #data["lowest"]
        if(highest > 100000 or lowest < -100000):
            listofComponentsOutOfRange.append({key, data})
            namesofComponentsOutOfRange.append(key)
        elif(highest > 900 or lowest < -900):
            listofComponentsMildOutOfRange.append({key, data})
            namesofComponentsMildOutOfRange.append(key)
        elif (highest > 255 or lowest < -255):
            listofComponentsNotPerfect.append({key, data})
            namesofComponentsNotPerfect.append(key)

    print("Components Out of Range:")
    print(namesofComponentsOutOfRange)
    print("Components Mildly Out of Range:")
    print(namesofComponentsMildOutOfRange)
    print("Components Not Perfect:")
    print(namesofComponentsNotPerfect)

    try:
        write_calibration_table(ranges.data, dir=model_dir)
        print("Calibration is done. Calibration cache is saved to calibration.json")
    except:
        print("Failed saving calibration data")

    # remove temp augmented model again
    os.remove("augmented_model.onnx")

    model_quantized_path = quantizer.quantize(
        save_dir="onnx/Madlad/Script/StaticQuantization/Test/QLinearAndQDQGather",
        calibration_tensors_range=ranges,
        quantization_config=qconfig,
        use_external_data_format=True
    )

    modelOnnx = onnx.load_model(str(model_quantized_path)+"/Madlad_decoder_2D_quantized_quantized.onnx")
    onnx.save(modelOnnx, "onnx/Madlad/Script/StaticQuantization/Test/Madlad_decoder_quantized_static_test.onnx")
    onnx.save(modelOnnx, str(model_quantized_path)+"/Madlad_decoder_quantized_static_test.onnx")


def static_quantization_optimum_QDQ_only():
    model_out_name = "Madlad_decoder_2D.onnx"
    model_dir = "onnx/Madlad/Optimized/Optimized2D"
    model_name = 'jbochi/madlad400-3b-mt'
    quantizer = ORTQuantizer.from_pretrained(model_dir, file_name=model_out_name)
    operators_to_quantize_in = ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QDQ
    #operators_to_quantize_in.remove('Where')
    #operators_to_quantize_in.remove('Reshape')
    #operators_to_quantize_in.remove('Add')   (non presente)
    operators_to_quantize_in.remove('Softmax')

    #operators_to_quantize_in.remove('MatMul')  #(gia quantizzato)
    #operators_to_quantize_in.remove('Conv')    #(gia quantizzato)
    #operators_to_quantize_in.remove('Mul')   (non presente)
    #operators_to_quantize_in.remove('Sub')   (non presente)
    #operators_to_quantize_in.remove('Neg')   (non presente)
    #operators_to_quantize_in.remove('Log')   (non presente)
    #operators_to_quantize_in.remove('Div')   (non presente)
    #operators_to_quantize_in.remove('Cast')   (non presente)
    #operators_to_quantize_in.remove('Transpose')
    #operators_to_quantize_in.remove('Min')   (non presente)
    #operators_to_quantize_in.remove('Gather')

    #operators_to_quantize_in.remove('Unsqueeze')
    #operators_to_quantize_in.remove('Squeeze')

    #operators_to_quantize_in.remove('Cast')   (non presente)
    #operators_to_quantize_in.remove('Reshape')   (non presente)
    format, mode, operators_to_quantize = default_quantization_parameters(True,operators_to_quantize=operators_to_quantize_in)
    #nodes_to_exclude = []
    #for i in range(0,11):
        #nodes_to_exclude.append("/layers."+str(i)+"/self_attn/Reshape_7")
        #nodes_to_exclude.append("/layers."+str(i)+"/encoder_attn/Reshape_7")

    ret = madlad_modifier.addReshapes_partial(model_dir+"/"+model_out_name)
    excluded_matmuls = ret[1]

    qconfig = QuantizationConfig(
        is_static=True,
        format=QuantFormat.QDQ,
        mode=QuantizationMode.QLinearOps,
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=True,
        weights_dtype=QuantType.QUInt8,
        weights_symmetric=True,
        per_channel=False,
        reduce_range=False,
        nodes_to_exclude = excluded_matmuls,   #si escludono i MatMul non convertiti in 2D (perche erano 4D)
        operators_to_quantize=operators_to_quantize,
    )
    #qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
    # caricamento tokenizer
    #tokenizerNLLBen = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn",tgt_lang="ita_Latn")
    tokenizerEn = T5Tokenizer.from_pretrained(model_name)
    # caricamento encoder session
    providers = ['CPUExecutionProvider']  # 'ROCMExecutionProvider',
    encoder_session = onnxruntime.InferenceSession("onnx/Madlad/Optimum/encoder_model.onnx",providers=providers)

    # Create the calibration dataset
    calibration_samples = 120   #38 per il preprocess_fn_test
    calibration_dataset = quantizer.get_calibration_dataset(
        "opus100",
        dataset_config_name="en-it",
        preprocess_function=functools.partial(preprocess_fn_Madlad, tokenizer=tokenizerEn,encoder_session=encoder_session),
        num_samples=calibration_samples,
        dataset_split="train",
        # preprocess_batch=False
    )

    # Create the calibration configuration containing the parameters related to calibration.
    #calibration_config = AutoCalibrationConfig.percentiles(calibration_dataset, percentile=99.999)
    #calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
    calibration_config = AutoCalibrationConfig.entropy(calibration_dataset)

    # liberare la ram da tutte le risorse inutili (ad esempio la sessione dell'encoder), misurare quanta `e occupata in questo punto (e capire quanta posso liberarne)
    del encoder_session

    # Perform the calibration step: computes the activations quantization ranges (RAM optimized)
    shards = 4   #in quanti partial_fit dividiamo il calibration_dataset
    for i in range(shards):
        shard = calibration_dataset.shard(shards, i)
        quantizer.partial_fit(
            dataset=shard,
            calibration_config=calibration_config,
            operators_to_quantize=qconfig.operators_to_quantize,
            batch_size=1,   #calibration_samples//shards
            use_external_data_format=True,
        )
    ranges = quantizer.compute_ranges()

    listofComponentsOutOfRange = []
    listofComponentsMildOutOfRange = []
    listofComponentsNotPerfect = []
    namesofComponentsOutOfRange = []
    namesofComponentsMildOutOfRange = []
    namesofComponentsNotPerfect = []
    keys = list(ranges)
    for key in keys:
        data = ranges[key]
        highest = data.highest         #data["highest"]
        lowest = data.lowest           #data["lowest"]
        if(highest > 100000 or lowest < -100000):
            listofComponentsOutOfRange.append({key, data})
            namesofComponentsOutOfRange.append(key)
        elif(highest > 900 or lowest < -900):
            listofComponentsMildOutOfRange.append({key, data})
            namesofComponentsMildOutOfRange.append(key)
        elif (highest > 255 or lowest < -255):
            listofComponentsNotPerfect.append({key, data})
            namesofComponentsNotPerfect.append(key)

    print("Components Out of Range:")
    print(namesofComponentsOutOfRange)
    print("Components Mildly Out of Range:")
    print(namesofComponentsMildOutOfRange)
    print("Components Not Perfect:")
    print(namesofComponentsNotPerfect)

    try:
        write_calibration_table(ranges.data, dir=model_dir)
        print("Calibration is done. Calibration cache is saved to calibration.json")
    except:
        print("Failed saving calibration data")

    # remove temp augmented model again
    os.remove("augmented_model.onnx")

    model_quantized_path = quantizer.quantize(
        save_dir="onnx/Madlad/Script/StaticQuantization/Test/QDQonly/2D",
        calibration_tensors_range=ranges,
        quantization_config=qconfig,
        use_external_data_format=True
    )

    modelOnnx = onnx.load_model(str(model_quantized_path)+"/Madlad_decoder_2D_quantized.onnx")
    onnx.save(modelOnnx, "onnx/Madlad/Script/StaticQuantization/Test/Madlad_decoder_quantized_static_test.onnx")
    onnx.save(modelOnnx, str(model_quantized_path)+"/Madlad_decoder_quantized_static_test.onnx")


def static_quantization_optimum_QOps_only_with_cache():   #da fare
    model_out_name = "Madlad_decoder_2D.onnx"
    model_dir = "onnx/Madlad/Optimized/Optimized2D"
    model_name = 'jbochi/madlad400-3b-mt'
    quantizer = ORTQuantizer.from_pretrained(model_dir, file_name=model_out_name)
    operators_to_quantize_in = ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QDQ
    #operators_to_quantize_in.remove('Where')
    #operators_to_quantize_in.remove('Reshape')
    #operators_to_quantize_in.remove('Add')   (non presente)
    operators_to_quantize_in.remove('Softmax')

    #operators_to_quantize_in.remove('MatMul')  #(gia quantizzato)
    #operators_to_quantize_in.remove('Conv')    #(gia quantizzato)
    #operators_to_quantize_in.remove('Mul')   (non presente)
    #operators_to_quantize_in.remove('Sub')   (non presente)
    #operators_to_quantize_in.remove('Neg')   (non presente)
    #operators_to_quantize_in.remove('Log')   (non presente)
    #operators_to_quantize_in.remove('Div')   (non presente)
    #operators_to_quantize_in.remove('Cast')   (non presente)
    #operators_to_quantize_in.remove('Transpose')
    #operators_to_quantize_in.remove('Min')   (non presente)
    #operators_to_quantize_in.remove('Gather')

    #operators_to_quantize_in.remove('Unsqueeze')
    #operators_to_quantize_in.remove('Squeeze')

    #operators_to_quantize_in.remove('Cast')   (non presente)
    #operators_to_quantize_in.remove('Reshape')   (non presente)
    format, mode, operators_to_quantize = default_quantization_parameters(True,operators_to_quantize=operators_to_quantize_in)
    #nodes_to_exclude = []
    #for i in range(0,11):
        #nodes_to_exclude.append("/layers."+str(i)+"/self_attn/Reshape_7")
        #nodes_to_exclude.append("/layers."+str(i)+"/encoder_attn/Reshape_7")

    ret = madlad_modifier.addReshapes_partial(model_dir+"/"+model_out_name)
    excluded_matmuls = ret[1]

    qconfig = QuantizationConfig(
        is_static=True,
        format=QuantFormat.QDQ,
        mode=QuantizationMode.QLinearOps,
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=True,
        weights_dtype=QuantType.QUInt8,
        weights_symmetric=True,
        per_channel=False,
        reduce_range=False,
        nodes_to_exclude = excluded_matmuls,   #si escludono i MatMul non convertiti in 2D (perche erano 4D)
        operators_to_quantize=operators_to_quantize,
    )
    #qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
    # caricamento tokenizer
    #tokenizerNLLBen = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn",tgt_lang="ita_Latn")
    tokenizerEn = T5Tokenizer.from_pretrained(model_name)
    # caricamento encoder session
    providers = ['CPUExecutionProvider']  # 'ROCMExecutionProvider',
    encoder_session = onnxruntime.InferenceSession("onnx/Madlad/Optimum/encoder_model.onnx",providers=providers)

    # Create the calibration dataset
    calibration_samples = 120   #38 per il preprocess_fn_test
    calibration_dataset = quantizer.get_calibration_dataset(
        "opus100",
        dataset_config_name="en-it",
        preprocess_function=functools.partial(preprocess_fn_Madlad, tokenizer=tokenizerEn,encoder_session=encoder_session),
        num_samples=calibration_samples,
        dataset_split="train",
        # preprocess_batch=False
    )

    # Create the calibration configuration containing the parameters related to calibration.
    #calibration_config = AutoCalibrationConfig.percentiles(calibration_dataset, percentile=99.999)
    #calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
    calibration_config = AutoCalibrationConfig.entropy(calibration_dataset)

    # liberare la ram da tutte le risorse inutili (ad esempio la sessione dell'encoder), misurare quanta `e occupata in questo punto (e capire quanta posso liberarne)
    del encoder_session

    # Perform the calibration step: computes the activations quantization ranges (RAM optimized)
    shards = 4   #in quanti partial_fit dividiamo il calibration_dataset
    for i in range(shards):
        shard = calibration_dataset.shard(shards, i)
        quantizer.partial_fit(
            dataset=shard,
            calibration_config=calibration_config,
            operators_to_quantize=qconfig.operators_to_quantize,
            batch_size=1,   #calibration_samples//shards
            use_external_data_format=True,
        )
    ranges = quantizer.compute_ranges()

    listofComponentsOutOfRange = []
    listofComponentsMildOutOfRange = []
    listofComponentsNotPerfect = []
    namesofComponentsOutOfRange = []
    namesofComponentsMildOutOfRange = []
    namesofComponentsNotPerfect = []
    keys = list(ranges)
    for key in keys:
        data = ranges[key]
        highest = data.highest         #data["highest"]
        lowest = data.lowest           #data["lowest"]
        if(highest > 100000 or lowest < -100000):
            listofComponentsOutOfRange.append({key, data})
            namesofComponentsOutOfRange.append(key)
        elif(highest > 900 or lowest < -900):
            listofComponentsMildOutOfRange.append({key, data})
            namesofComponentsMildOutOfRange.append(key)
        elif (highest > 255 or lowest < -255):
            listofComponentsNotPerfect.append({key, data})
            namesofComponentsNotPerfect.append(key)

    print("Components Out of Range:")
    print(namesofComponentsOutOfRange)
    print("Components Mildly Out of Range:")
    print(namesofComponentsMildOutOfRange)
    print("Components Not Perfect:")
    print(namesofComponentsNotPerfect)

    try:
        write_calibration_table(ranges.data, dir=model_dir)
        print("Calibration is done. Calibration cache is saved to calibration.json")
    except:
        print("Failed saving calibration data")

    # remove temp augmented model again
    os.remove("augmented_model.onnx")

    model_quantized_path = quantizer.quantize(
        save_dir="onnx/Madlad/Script/StaticQuantization/Test/QDQonly/2D",
        calibration_tensors_range=ranges,
        quantization_config=qconfig,
        use_external_data_format=True
    )

    modelOnnx = onnx.load_model(str(model_quantized_path)+"/Madlad_decoder_2D_quantized.onnx")
    onnx.save(modelOnnx, "onnx/Madlad/Script/StaticQuantization/Test/Madlad_decoder_quantized_static_test.onnx")
    onnx.save(modelOnnx, str(model_quantized_path)+"/Madlad_decoder_quantized_static_test.onnx")


def static_quantization_optimum_QLinearOnly_encoder():
    model_out_name = "Madlad_encoder.onnx"
    model_dir = "onnx/Madlad/Script"
    model_name = 'jbochi/madlad400-3b-mt'
    quantizer = ORTQuantizer.from_pretrained(model_dir, file_name=model_out_name)
    #all=ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QDQ
    all = ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION
    #all.remove('Softmax')
    #all.remove('Mul')

    #all.remove('Add')
    #all.remove('Cast')
    format, mode, operators_to_quantize = default_quantization_parameters(True, operators_to_quantize=all)

    qconfig = QuantizationConfig(
        is_static=True,
        format=QuantFormat.QDQ,
        mode=QuantizationMode.QLinearOps,
        activations_dtype=QuantType.QUInt8,
        activations_symmetric=False,
        weights_dtype=QuantType.QInt8,
        weights_symmetric=True,
        per_channel=False,
        reduce_range=False,
        operators_to_quantize=operators_to_quantize,
    )
    #qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
    # caricamento tokenizer
    #tokenizerNLLBen = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn",tgt_lang="ita_Latn")
    tokenizerEn = T5Tokenizer.from_pretrained(model_name)

    # Create the calibration dataset
    calibration_samples = 1   #38 per il preprocess_fn_test (120 per la calibrazione vera)
    calibration_dataset = quantizer.get_calibration_dataset(
        "opus100",
        dataset_config_name="en-it",
        preprocess_function=functools.partial(preprocess_fn_encoder, tokenizer=tokenizerEn),
        num_samples=calibration_samples,
        dataset_split="train",
        # preprocess_batch=False
    )

    # Create the calibration configuration containing the parameters related to calibration.
    #calibration_config = AutoCalibrationConfig.percentiles(calibration_dataset, percentile=99.999)
    #calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
    calibration_config = AutoCalibrationConfig.entropy(calibration_dataset)

    # Perform the calibration step: computes the activations quantization ranges (RAM optimized)
    shards = 1   #in quanti partial_fit dividiamo il calibration_dataset (4)
    for i in range(shards):
        shard = calibration_dataset.shard(shards, i)
        quantizer.partial_fit(
            dataset=shard,
            calibration_config=calibration_config,
            operators_to_quantize=qconfig.operators_to_quantize,
            batch_size=1,   #calibration_samples//shards
            use_external_data_format=True,
        )
    ranges = quantizer.compute_ranges()

    listofComponentsOutOfRange = []
    listofComponentsMildOutOfRange = []
    listofComponentsNotPerfect = []
    namesofComponentsOutOfRange = []
    namesofComponentsMildOutOfRange = []
    namesofComponentsNotPerfect = []
    keys = list(ranges)
    for key in keys:
        data = ranges[key]
        highest = data.highest         #data["highest"]
        lowest = data.lowest           #data["lowest"]
        if(highest > 100000 or lowest < -100000):
            listofComponentsOutOfRange.append({key: [lowest, highest]})
            namesofComponentsOutOfRange.append(key)
        elif(highest > 900 or lowest < -900):
            listofComponentsMildOutOfRange.append({key: [lowest, highest]})
            namesofComponentsMildOutOfRange.append(key)
        elif (highest > 255 or lowest < -255):
            listofComponentsNotPerfect.append({key: [lowest, highest]})
            namesofComponentsNotPerfect.append(key)

    print("Components Out of Range:")
    print(namesofComponentsOutOfRange)
    for component in listofComponentsOutOfRange:
        print(component)
    print('')
    print("Components Mildly Out of Range:")
    print(namesofComponentsMildOutOfRange)
    for component in listofComponentsMildOutOfRange:
        print(component)
    print('')
    print("Components Not Perfect:")
    print(namesofComponentsNotPerfect)
    for component in listofComponentsNotPerfect:
        print(component)
    print('')

    try:
        write_calibration_table(ranges.data, dir=model_dir)
        print("Calibration is done. Calibration cache is saved to calibration.json")
    except:
        print("Failed saving calibration data")

    # remove temp augmented model again
    os.remove("augmented_model.onnx")

    model_quantized_path = quantizer.quantize(
        save_dir="onnx/Madlad/Optimum_Cache_Optimized/QuantizedStatic",
        calibration_tensors_range=ranges,
        quantization_config=qconfig,
        use_external_data_format=True
    )

    #modelOnnx = onnx.load_model(Path("onnx/Madlad/Script_INT32/StaticQuantization/Madlad_decoder_complete_quantized.onnx"))
    #onnx.save(modelOnnx, "onnx/Madlad/Script_INT32/StaticQuantization/Madlad_decoder_quantized_static_test.onnx")