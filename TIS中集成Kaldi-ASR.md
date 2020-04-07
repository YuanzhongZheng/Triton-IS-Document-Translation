# TIS中集成Kaldi-ASR

TIS提供了基于自定义的Kaldi-ASR后端，它可以将高性能云推理应用在Kaldi-ASR模型，其中包括TIS和客户端之间的gRPC通信，以及推理请求的批处理。代码库由NVIDIA测试和维护。

## 目录

- [目录](#目录)
- [解决方案概述](#解决方案概述)
   * [参考模型](#参考模型)
   * [默认配置](#默认配置)
- [安装](#安装)
  * [必需项](#必需项)
- [快速上手](#快速上手)
- [高级玩法](#高级玩法)
   * [参数](#参数)
      * [模型路径](#模型路径)
      * [模型配置](#模型配置)
      * [推理引擎配置](#推理引擎配置)
  * [推理过程](#推理过程)
  * [客户端命令行参数](#客户端命令行参数)
   * [输入/输出](#输入输出)
     * [输入](#输入)
     * [输出](#输出)
   * [解析自定义Kaldi-ASR的模型](#解析自定义Kaldi-ASR的模型)
- [性能](#性能)
  * [指标](#指标)
  * [结果](#结果)
- [版本说明](#版本说明)
  * [变更日志](#变更日志)
  * [已知缺陷](#已知缺陷)

## 解决方案概述

This repository provides a wrapper around the online GPU-accelerated ASR pipeline from the paper [GPU-Accelerated Viterbi Exact Lattice Decoder for Batched Online and Offline Speech Recognition](https://arxiv.org/abs/1910.10032). That work includes a high-performance implementation of a GPU HMM Decoder, a low-latency Neural Net driver, fast Feature Extraction for preprocessing, and new ASR pipelines tailored for GPUs. These different modules have been integrated into the Kaldi ASR framework.

This repository contains a TensorRT Inference Server custom backend for the Kaldi ASR framework. This custom backend calls the high-performance online GPU pipeline from the Kaldi ASR framework. This TensorRT Inference Server integration provides ease-of-use to Kaldi ASR inference: gRPC streaming server, dynamic sequence batching, and multi-instances support. A client connects to the gRPC server, streams audio by sending chunks to the server, and gets back the inferred text as an answer (see [Input/Output](#input-output)). More information about the TensorRT Inference Server can be found [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/).  

This TensorRT Inference Server integration is meant to be used with the LibriSpeech model for demonstration purposes. We include a pre-trained version of this model to allow you to easily test this work (see [Quick Start Guide](#quick-start-guide)). Both the TensorRT Inference Server integration and the underlying Kaldi ASR online GPU pipeline are a work in progress and will support more functionalities in the future. This includes online iVectors not currently supported in the Kaldi ASR GPU online pipeline and being replaced by a zero vector (see [Known issues](#known-issues)). Support for a custom Kaldi model is experimental (see [Using a custom Kaldi model](#using-custom-kaldi-model)).

### 参考模型

A reference model is used by all test scripts and benchmarks presented in this repository to illustrate this solution. We are using the Kaldi ASR `LibriSpeech` recipe, available [here](https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5). It was trained by NVIDIA and is delivered as a pre-trained model.

### 默认配置

Details about parameters can be found in the [Parameters](#parameters) section.

* `model path`: Configured to use the pretrained LibriSpeech model.
* `beam`: 10
* `lattice_beam`: 7
* `max_active`: 10,000
* `frame_subsampling_factor`: 3
* `acoustic_scale`: 1.0
* `num_worker_threads`: 20
* `max_execution_batch_size`: 256
* `max_batch_size`: 4096
* `instance_group.count`: 2

## 安装

### 必需项 

This repository contains Dockerfiles which extends the Kaldi and TensorRT Inference Server NVIDIA GPU Cloud (NGC) containers and encapsulates some dependencies. Aside from these dependencies, ensure you have [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) installed.


For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)


## 快速上手

1. Clone the repository.
 
```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/Kaldi/SpeechRecognition
```

2. Build the client and server containers.
 
`scripts/docker/build.sh`

3. Download and set up the pre-trained model and eval dataset.

`scripts/docker/launch_download.sh`

The model and dataset are downloaded in the `data/` folder.

4. Start the server.

`scripts/docker/launch_server.sh`

Once you see the line `Starting Metrics Service at 0.0.0.0:8002`, the server is ready to be used. You can then start the client.

Currently, multi-GPU is not supported. By default GPU 0 is used. You can use a specific GPU by using `NVIDIA_VISIBLE_DEVICES`:

`NVIDIA_VISIBLE_DEVICES=<GPUID> scripts/docker/launch_server.sh`

5. Start the client.

The following command will stream 1000 parallel streams to the server. The `-p` option prints the inferred `TEXT` sent back from the server. 

`scripts/docker/launch_client.sh -p`


## 高级玩法

### 参数

The configuration is done through the `config.pbtxt` file available in `model-repo/` directory. It allows you to specify the following:

####  模型路径

The following parameters can be modified if you want to use your own Kaldi model. 

* `mfcc_filename`
* `ivector_filename`
* `nnet3_rxfilename`
* `fst_rxfilename`
* `word_syms_rxfilename`

#### 模型配置

The model configuration parameters are passed to the model and have  an impact on both accuracy and performance. The model parameters are usually Kaldi ASR parameters, meaning, if they are, you can reuse the values that are currently being used in the CPU Kaldi ASR pipeline. 

* `beam`
* `lattice_beam`
* `max_active`
* `frame_subsampling_factor`
* `acoustic_scale`

#### 推理引擎配置

The inference engine configuration parameters configure the inference engine. They impact performance, but not accuracy.

* `max_batch_size`: The maximum number of inference channels opened at a given time. If set to `4096`, then one instance will handle at most 4096 concurrent users.
* `num_worker_threads`: The number of CPU threads for the postprocessing CPU tasks, such as lattice determinization and text generation from the lattice.
* `max_execution_batch_size`: The size of one execution batch on the GPU. This parameter should be set as large as necessary to saturate the GPU, but not bigger. Larger batches will lead to a higher throughput, smaller batches to lower latency. 
* `input.WAV_DATA.dims`: The maximum number of samples per chunk. The value must be a multiple of `frame_subsampling_factor * chunks_per_frame`.

### 推理过程

Inference is done through simulating concurrent users. Each user is attributed to one utterance from the LibriSpeech dataset. It streams that utterance by cutting it into chunks and gets the final `TEXT` output once the final chunk has been sent. A parameter sets the number of active users being simulated in parallel.  

### 客户端命令行参数

The client can be configured through a set of parameters that define its behavior. To see the full list of available options and their descriptions, use the `-h` command-line option. The parameters are:

```
    -v
    -i <Number of iterations on the dataset>
    -c <Number of parallel audio channels>
    -a <Path to the scp dataset file>
    -l <Maximum number of samples per chunk. Must correspond to the server config>
    -u <URL for inference service and its gRPC port>
    -o : Only feed each channel at realtime speed. Simulates online clients.
    -p : Print text outputs

```

### 输入、输出

The API is currently experimental.

#### 输入

The server execpts chunks of audio each containing up to `input.WAV_DATA.dims` samples. Per default, this corresponds to 510ms of audio per chunk. The last chunk can send a partial chunk smaller than this maximum value. 

The chunk is made of a float array set in the input `WAV_DATA`, with the input `WAV_DATA_DIM` containing the number of samples contained in that chunk. Flags can be set to declare a chunk as a first chunk or last chunk for a sequence. Finally, each chunk from a given sequence is associated with a `CorrelationID`. Every chunk belonging to the same sequence must be given the same `CorrelationID`. 

#### 输出

Once the server receives the final chunk for a sequence (with the `END` flag set), it will generate the output associated with that sequence, and send it back to the client. The end of the sequencing procedure is:

1. Process the last chunk.
2. Flush and process the Neural Net context. 
3. Generate the full lattice for the sequence.
4. Determinize the lattice.
5. Find the best path in the lattice.
6. Generate the text output for that best path.
7. Send the text back to the client.

Even if only the best path is used, we are still generating a full lattice for benchmarking purposes. Partial results (generated after each timestep) are currently not available but will be added in a future release. 

### 解析自定义Kaldi-ASR的模型

Support for Kaldi ASR models that are different from the provided LibriSpeech model is experimental. However, it is possible to modify the [Model Path](#model-path) section of the config file `model-repo/kaldi_online/config.pbtxt` to set up your own model. 

The models and Kaldi allocators are currently not shared between instances. This means that if your model is large, you may end up with not enough memory on the GPU to store two different instances. If that's the case, you can set `count` to `1` in the `instance_group` section of the config file.

## 性能


### 指标

Throughput is measured using the RTFX metric. It is defined such as : `RTFX = (number of seconds of audio inferred) / (compute time in seconds)`. It is the inverse of the RTF (Real Time Factor) metric, such as `RTFX = 1/RTF`.

Latency is defined as the delay between the availability of the last chunk of audio and the reception of the inferred text. More precisely, it is defined such as :

1. *Client:* Last audio chunk available
2. ***t0** <- Current time*
3. *Client:* Send last audio chunk
4. *Server:* Compute inference of last chunk
5. *Server:* Generate the raw lattice for the full utterance
6. *Server:* Determinize the raw lattice
7. *Server:* Generate the text output associated with the best path in the determinized lattice
8. *Client:* Receive text output
9. *Client:* Call callback with output
10. ***t1** <- Current time*  

The latency is defined such as `latency = t1 - t0`.

### 结果

Our results were obtained by:

1. Building and starting the server as described in [Quick Start Guide](#quick-start-guide).
2. Running  `scripts/run_inference_all_v100.sh` and  `scripts/run_inference_all_t4.sh`

| GPU | Realtime I/O | Number of parallel audio channels | Throughput (RTFX) | Latency | | | |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |------ |
| | | | | 90% | 95% | 99% | Avg |
| V100 | No | 2000 | 1769.8 | N/A | N/A | N/A | N/A |
| V100 | Yes | 1500 |  1220 | 0.424 | 0.473 | 0.758 | 0.345 |
| V100 | Yes | 1000 |  867.4 | 0.358 | 0.405 | 0.707 | 0.276 |
| V100 | Yes | 800 |  647.8 | 0.304 | 0.325 | 0.517 | 0.238 |
| T4 | No | 1000 | 906.7 | N/A | N/A | N/A| N/A |
| T4 | Yes | 700 | 629.6 | 0.629 | 0.782 | 1.01 | 0.463 |
| T4 | Yes | 400 | 373.7 | 0.417 | 0.441 | 0.690 | 0.349 |

## 版本说明

### 变更日志

2020年1月
* 首版发布

### 已知缺陷

虽然基准测试脚本中使用的参考模型需要mfcc和iVector才能达到最佳准确率，但是目前只支持mfcc特征。未来的版本中将添加对iVector的支持。