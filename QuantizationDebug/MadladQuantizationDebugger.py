

import json
import os
from pathlib import Path
import pickle
import pprint
import random
from CalibrationData.DatasetData import getDatasetData
from CalibrationData.MadladCalibrationDataReader import MadladDecoderCalibrationDataReader, MadladEncoderCalibrationDataReader
import qdq_loss_debug_updated

def show_quant_error_weight(model_path: str, model_qdq_quantized_path: str, encoder = True, save_result_folder=None):
    weight_error_file_name = 'weight_error_'+ ('encoder' if encoder else 'decoder') +'.json'
    weight_error_ordered_file_name = 'weight_error_ordered_'+ ('encoder' if encoder else 'decoder') +'.json'

    qdq_weight_cmp = qdq_loss_debug_updated.create_weight_matching(model_path,
                                          model_qdq_quantized_path)
    
    error_dict = qdq_loss_debug_updated.compute_weight_error(qdq_weight_cmp, err_func=qdq_loss_debug_updated.compute_signal_to_quantization_relative_norm_percentage)

    if(save_result_folder is not None):
        os.makedirs(save_result_folder, exist_ok=True)
        with open(save_result_folder+weight_error_file_name, 'w') as f:
                json.dump(error_dict, f)

    error_list = []
    for key, value in error_dict.items():
        error_list.append((key, value))
    error_list.sort(key=lambda x: x[1], reverse=True)
    with open(save_result_folder+weight_error_ordered_file_name, 'w') as f:
        json.dump(error_list, f)

    pprint.pp(error_list, indent=2)


def show_quant_error_activations(model_folder: str, model_name: str, model_qdq_quantized_path: str, encoder = True, save_results_folder=None):
    quant_debug_folder = model_folder+"QuantDebug"
    model_augmented_path = quant_debug_folder+"/"+model_name+".onnx"
    model_qdq_augmented_path = quant_debug_folder+"/"+model_name+"_q.onnx"
    activation_error_file_name = 'activation_error_'+ ('encoder' if encoder else 'decoder') + '.json'
    activation_error_odered_file_name = 'activation_error_ordered_'+ ('encoder' if encoder else 'decoder') + '.json'

    os.makedirs(quant_debug_folder, exist_ok=True)

    if(not Path(model_augmented_path).is_file()):
        qdq_loss_debug_updated.modify_model_output_intermediate_tensors(model_folder+model_name+".onnx", model_augmented_path, ["MatMul"], save_as_external_data=True)
        print("Created augmented float model in: "+model_augmented_path)
    if(not Path(model_qdq_augmented_path).is_file()):
        qdq_loss_debug_updated.modify_model_output_intermediate_tensors(model_qdq_quantized_path, model_qdq_augmented_path, ["MatMul"], save_as_external_data=False)
        print("Created augmented quantized model in: "+model_qdq_augmented_path)

    #generation and saving of activation_error
    activation_error: dict[str, float] | None = None
    if(save_results_folder is None or (not Path(save_results_folder+activation_error_file_name).is_file())):
        #creation of data reader
        rng = random.Random()
        f = open("hf_token.txt")
        dataset = getDatasetData(20, hf_token=f.read())
        targetLanguages = []
        for data in dataset:
            targetLanguages.append(data["lang"])
        rng.shuffle(targetLanguages)
        if(encoder):
            calibrationDataReader = MadladEncoderCalibrationDataReader(dataset, targetLanguages)
        else:
            calibrationDataReader = MadladDecoderCalibrationDataReader(dataset, targetLanguages)
        
        print("Loaded input data")

        #generation of activation error dict
        model_activations = qdq_loss_debug_updated.collect_activations(model_augmented_path, calibrationDataReader)
        print("Collected float model activations")
        calibrationDataReader.reset()
        model_qdq_activations = qdq_loss_debug_updated.collect_activations(model_qdq_augmented_path, calibrationDataReader)
        print("Collected quantized model activations")

        activation_matching = qdq_loss_debug_updated.create_activation_matching_updated(model_qdq_activations, model_activations)
        print("Created activations matching")

        activation_error = qdq_loss_debug_updated.compute_activation_error_clean(activation_matching, err_func=qdq_loss_debug_updated.compute_signal_to_quantization_relative_norm_percentage)
        print("Computed activations error")

        if(save_results_folder is not None):
            os.makedirs(save_results_folder, exist_ok=True)
            with open(save_results_folder+activation_error_file_name, 'w') as f:
                json.dump(activation_error, f)
                print("Saved activations error")
    elif(Path(save_results_folder+activation_error_file_name).is_file()):
        with open(save_results_folder+activation_error_file_name) as f:
            activation_error = json.load(f)

    activation_error_list = []
    for key, value in activation_error.items():
        activation_error_list.append((key, value))
    activation_error_list.sort(key=lambda x: x[1], reverse=True)
    with open(save_results_folder+activation_error_odered_file_name, 'w') as f:
        json.dump(activation_error_list, f)
    
    pprint.pp(activation_error_list, indent=2)

    ''' 
    1) Sistemare la struttura di DatasetData e di MadladCalibrationDataReader spostandoli in una cartella dedicata all'esterno di GPTQ,
       in modo da renderli generici e poterli usare anche qui (fatto)
    2) Usando i due file sorgenti citati nel punto 1) eseguire collect_activation su entrambi i modelli (fatto)
    3) Usare i risultati dei collect_activation per chiamare create_activation_matching (fatto)
    4) Usare il risultato di create_activation_matching per chiamare compute_activation_error (fatto)
    5) Salvare il risultato di compute_activation_error nella save_results_folder (se non è None) (fatto)
    6) Creare la funzione, in qdq_loss_debug_updated.py, che fa la media di tutti i batch del risultato di compute_activation_error, (fatto)
       dopodiché chiamarla e salvare il suo risultato nella save_results_folder (se non è None) (fatto)
    7) Implementare il salvataggio del risultato anche in show_quant_error_weight (fatto)
    8) Aggiungere al debug delle attivazione dei log per capire il progresso.
    '''





if __name__ == "__main__":
    show_quant_error_weight("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/encoder_model.onnx", 
                            "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/RTNqdq/madlad_encoder_4bit.onnx",
                            True, "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Quality/RTN/QuantDebug/")
    show_quant_error_weight("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/decoder_model.onnx", 
                            "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/RTNqdq/madlad_decoder_4bit.onnx",
                            False, "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Quality/RTN/QuantDebug/")

    show_quant_error_activations("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/",
                                 "encoder_model",
                                 "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/RTNqdq/madlad_encoder_4bit.onnx",
                                 True, "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Quality/RTN/QuantDebug/")
    show_quant_error_activations("onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/",
                                 "decoder_model",
                                 "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/RTNqdq/madlad_decoder_4bit.onnx",
                                 False, "onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/Quality/RTN/QuantDebug/")
    
