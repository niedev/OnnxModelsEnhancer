# OnnxModelsEnhancer

This is a tool used to optimize ONNX models; it is used for all the models integrated in [RTranslator](https://github.com/niedev/RTranslator).

This is the first version of this tool, so it's pretty barebones. The models officially supported are the following:
- Madlad 400 MT 3B
- HY-MT 1.5 1.8B
- TranslateGemma

New models will be added in the future.

## How to use

### Installation

This guide is for an Ubuntu environment. If you use Windows, you can try to replicate the logic with equivalent Windows commands, or use WSL.

First of all, clone this repo:

```
git clone https://github.com/niedev/OnnxModelsEnhancer

cd OnnxModelsEnhancer
```

After that, create a Python virtual environment and install the requirements.txt inside it:

```
sudo apt update

sudo apt install python3-venv

python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt
```

Let's see in detail how to use this tool and what it does for each of the supported models.
<br/><br/>

### Madlad 400

After the installation, to generate an optimized version of Madlad 400 MT 3B, you just need to execute the file "enchancer_madlad.py".

```
python enhancer_madlad.py
```

After the script has finished, the location of the generated files will be indicated in the terminal (missing folders will be created automatically):

"Final models saved in onnx/Madlad/Optimum_Cache_Optimized/ReducedRam/Quantized/"
<br/>

#### Execution:

You can now notice that the generated model files are different than normal. In fact, the generated model has a custom architecture (we'll see why below). Consequently, its execution also occurs differently than normal. Model inference for this architecture is implemented in "[onnx_execution.py](https://github.com/niedev/OnnxModelsEnhancer/blob/main/onnx_execution.py)", in the "onnx_execution_madlad_cache_reduced_ram" method.
So, to run it in Python, simply copy the code or replicate its logic for execution in other languages ​​and platforms supported by onnxruntime.

#### Performance and quality:

The generated model supports kv-cache, and it is a lot smaller in size compared to the standard Madlad with kv-cache, obtained from optimum conversion.
In that case, two copies of the decoder are created: one that generates the kv-cache on the first iteration, and the other (with_past) that has the kv-cache among its inputs and is used for subsequent iterations. Latest versions of Optimum resolve this with decoder merging, but for Madlad, it doesn't work.

My version, however, uses a custom architecture, that has a single decoder copy, and a small model dedicated to the initial kv-cache creation. Plus, since the encoder and decoder in Madlad have the same embedding matrix, the embedding is separated in a dedicated model, to avoid the embedding matrix duplication.

With these techniques, the non quantized model drops its size from 20GB to 12GB.

Finally, the generated models are quantized in int4, with various optimizations to achieve very good performance and quality. So the final model size is 1.7GB and can be deployed even in mobile environments.
<br/><br/>

### HY-MT 1.5

After the installation, to generate an optimized version of HY-MT 1.5 1.8B, you just need to execute the file "enhancer_hy_mt_1.5.py".

```
python enhancer_hy_mt_1.5.py
```

After the script has finished, the location of the generated file will be indicated in the terminal (missing folders will be created automatically).
<br/>

#### Execution:

Since this model is decoder-only, the custom architecture optimizations done for Madlad can't be replicated here, so the model is a single .onnx file (plus its data) and it can be executed normally. Model inference is implemented in "[onnx_execution.py](https://github.com/niedev/OnnxModelsEnhancer/blob/main/onnx_execution.py)", in the "onnx_execution_hy_cache" method.
So, to run it in Python, simply copy the code or replicate its logic for execution in other languages ​​and platforms supported by onnxruntime.

#### Performance and quality:

The generated model supports kv-cache, and it is quantized in int8, with various optimizations to achieve very good performance and quality. So the final model size is 2GB and can be deployed even in mobile environments.
This model has not been quantized in int4 because it loses too much quality.
<br/><br/>

### Translate Gemma

After the installation, to generate an optimized version of Translate Gemma 4B, you just need to execute the file "enhancer_gemma3.py".

```
python enhancer_gemma3.py
```

After the script has finished, the location of the generated file will be indicated in the terminal (missing folders will be created automatically).
<br/>

#### Execution:

Since this model is decoder-only, like hy-mt, the custom architecture optimizations done for Madlad can't be replicated here, so the model is a single .onnx file (plus its data) and it can be executed normally. Model inference is implemented in "[onnx_execution.py](https://github.com/niedev/OnnxModelsEnhancer/blob/main/onnx_execution.py)", in the "onnx_execution_hy_cache" method.
So, to run it in Python, simply copy the code or replicate its logic for execution in other languages ​​and platforms supported by onnxruntime.

#### Performance and quality:

The generated model supports kv-cache, and it is quantized in int8. However, since this model will not be used in RTranslator (because HY-MT 1.5 is similar but with much better translation quality), several optimizations have not yet been made.


