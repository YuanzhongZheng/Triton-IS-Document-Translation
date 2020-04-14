# TIS中集成Kaldi-ASR

TIS提供了基于Kaldi-ASR功能的自定义后端，它可以在Kaldi-ASR模型应用高性能云推理功能，自定义后端包括了TIS和客户端之间的gRPC通信功能，以及推理请求的批处理功能。NVIDIA团队负责开发维护代码库。

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
   * [输入、输出](#输入输出)
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

这个项目基于论文（[GPU-Accelerated Viterbi Exact Lattice Decoder for Batched Online and Offline Speech Recognition](https://arxiv.org/abs/1910.10032)）实现了一个可利用线上GPU加速的ASR算法容器。容器内部含有一个基于GPU的高性能HMM解码器，一个低延迟的神经网络驱动，一个用于预处理数据的快速提取特征的模块，以及一套基于GPU的ASR算法。上述模块已经集成进Kaldi-ASR的TIS框架中。

这个项目基于Kaldi-ASR实现了一个应用于TIS框架的自定义后端。这个自定义后端可以在Kaldi-ASR框架中调用GPU来实现性能的大幅提升。TIS为Kaldi ASR推理提供了gRPC流式服务，动态序列批处理，多实例支持的功能。同时，项目中还提供了一个客户端用来实现连接gRPC服务器，发送音频流数据到服务器，接收推理后的结果的功能。(参考[输入、输出](#输入、输出))。有兴趣的用户可以通过此[链接](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/)获取更多细节和信息。

NVIDIA团队在这个项目中提供了一个基于`LibriSpeech`的预训练的模型，这使得用户可以轻松上手和测试(参考[快速上手](#快速上手))。目前TIS的集成工作和Kaldi-ASR在线GPU流程模型工作还处于开发阶段，未来NVIDIA会支持更多功能。比如这个版本并不支持i-vector在Kaldi-ASR在线GPU流程中解析，目前版本中的i-vector会被零向量所暂时替换(参考[已知缺陷](#已知缺陷))。此外，是否支持用户自己的其它Kaldi模型还在试验中 (参考 [解析自定义Kaldi-ASR的模型](#解析自定义Kaldi-ASR的模型)).

### 参考模型

这个项目提供的所有测试脚本和参考基准都基于`LibriSpeech`的Kaldi-ASR模型[详见此链接](https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5)，NVIDIA团队完成训练工作的同时，也发布了一个预训练的模型。

### 默认配置

关于参数的详细信息请见[参数](#参数) 部分.

* `model path`: 利用预训练好的ibriSpeech模型进行配置
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

项目中包含两个镜像文件，它们封装了Kaldi，TIS和一些依赖项。另外，请确保已安装[NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)。

如果想更快上手NGC容器，可以参考官方NGC文档：
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)


## 快速上手

1. 从云端复制代码。
 
```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/Kaldi/SpeechRecognition
```

2. 编译客户端与服务端的容器。
 
`scripts/docker/build.sh`

3. 下载和安装预训练模型和数据集。

`scripts/docker/launch_download.sh`

模型和数据集被下载到 `data/` 文件夹.

4. 启动服务端.

`scripts/docker/launch_server.sh`

当看到 `Starting Metrics Service at 0.0.0.0:8002`时，说明服务端已成功启动，接下来就可以启动客户端。

目前, 项目并不支持多GPU模式，默认用GPU 0。用户也可以通过`NVIDIA_VISIBLE_DEVICES`来指定具体GPU:

`NVIDIA_VISIBLE_DEVICES=<GPUID> scripts/docker/launch_server.sh`

5. 启动客户端.

下面这个脚本会向服务器发出1000个并行流请求，`-p`选项会打印从服务器发回的推理结果(`TEXT`)：

`scripts/docker/launch_client.sh -p`


## 高级玩法

### 参数

通过`model-repo/`文件夹的`config.pbtxt`来进行参数的配置，用户可以自定义以下选项：

####  模型路径

下列参数文件都可以替换成用户自己的Kaldi模型：

* `mfcc_filename`
* `ivector_filename`
* `nnet3_rxfilename`
* `fst_rxfilename`
* `word_syms_rxfilename`

#### 模型配置

下列模型配置参数用来传输给模型，它们会对模型的准确率和性能产生影响。模型参数通常是Kaldi-ASR的参数，这意味着用户可以复用Kaldi-ASR在CPU版本上的参数。

* `beam`
* `lattice_beam`
* `max_active`
* `frame_subsampling_factor`
* `acoustic_scale`

#### 推理引擎配置

推理引擎参数用来配置TIS中的推理引擎，它们只影响性能，不影响准确率。

* `max_batch_size`: 给定时间内的最大推理通道数。假如值为`4096`，那么一个实例将最多处理4096个并发请求。
* `num_worker_threads`: CPU进行后处理时的线程数，比如生成原始lattice结构以及从lattice结构中生成文本。
* `max_execution_batch_size`: GPU上的单个批尺寸大小。该参数的设定应满足恰好使GPU满负荷。批尺寸越大，吞吐量越高，但同时延迟也越高。
* `input.WAV_DATA.dims`: 每个语音数据块包含的最大采样点，必须是`frame_subsampling_factor * chunks_per_frame`的整数倍。

### 推理过程

在LibriSpeech数据集中，每个用户请求是一句语音数据流，TIS通过模拟多用户请求来完成推理加速过程。TIS将语音数据流分割成语音块，当发送完最后一个语音块后，TIS将会返回最终`TEXT` 结果。TIS可以通过一个参数来设置并行模拟的活动用户数。  

### 客户端命令行参数

客户端可以通过一系列参数定义来完成配置。用户可以通过命令行选项`-h`来查看所有参数选项以及它们的具体作用。这些参数是：
```
    -v
    -i <在数据集上的迭代次数>
    -c <并行音频通道数>
    -a <数据库中的scp文件路径>
    -l <每个语音块的最大采样点数，必须与服务端配置保持一致>
    -u <推理服务的URL以及它的gRPC端口>
    -o : 实时给每个通道传输数据，用来模拟在线客户端
    -p : 打印输出文本结果

```

### 输入、输出

API还处于实验阶段……

#### 输入

服务端希望发送过来的每个语音数据块可以包含`input.WAV_DATA.dims`个采样数据点，默认情况下对应每个语音块的510ms长度。最后一个语音块比较特殊，可以少于这个值。

The chunk is made of a float array set in the input `WAV_DATA`, with the input `WAV_DATA_DIM` containing the number of samples contained in that chunk. Flags can be set to declare a chunk as a first chunk or last chunk for a sequence. Finally, each chunk from a given sequence is associated with a `CorrelationID`. Every chunk belonging to the same sequence must be given the same `CorrelationID`. 

#### 输出

当服务端接收到了序列语句中的最后一个语音块（由`END`来标识），它将生成该序列的结果同时将结果返回给客户端。最后的处理步骤是：

1. 处理最后一个语音数据块。
2. 在神经网络中处理整个流数据。 
3. 对这句话生成lattice全路径。
4. 确定lattice。
5. 在lattice中找到最佳路径。
6. 通过最佳路径生成输出文本。
7. 向客户端发送文本结果。

即使只输出最佳路径，TIS目前仍然会生成用于基准测试的完整lattice。当前版本不支持用户获取每步生成的分步结果，但这个功能会在将来版本中实现。

### 解析自定义Kaldi-ASR的模型

Support for Kaldi ASR models that are different from the provided LibriSpeech model is experimental. However, it is possible to modify the [Model Path](#model-path) section of the config file `model-repo/kaldi_online/config.pbtxt` to set up your own model. 

The models and Kaldi allocators are currently not shared between instances. This means that if your model is large, you may end up with not enough memory on the GPU to store two different instances. If that's the case, you can set `count` to `1` in the `instance_group` section of the config file.

## 性能


### 指标

吞吐量使用RTFX来衡量，RTFX的定义为: `RTFX = (已推断的音频时长) / (服务计算耗费的时间)`. 它是RTF的倒数： `RTFX = 1/RTF`.

最后一个可用音频块和推断文本的接收之间的时间差被定义为延迟，更详细地表述如下：

1. *客户端:* 最后一个可用音频块
2. ***t0** <- Current time*
3. *客户端:* 发送最后一个可用音频块到服务端
4. *服务端:* 对最后一个可用音频块进行计算推理
5. *服务端:* 对整句话生成原始lattice结构
6. *服务端:* 确定原始lattice结构
7. *服务端:* 在原始lattice结构中生成最佳输出文本
8. *客户端:* 接收文本输出
9. *客户端:* 对输出结果进行回调
10. ***t1** <- Current time*  

延迟定义为： `latency = t1 - t0`.

### 结果

1. 参考[快速上手](#快速上手)进行编译和运行服务；
2. 运行  `scripts/run_inference_all_v100.sh` 和  `scripts/run_inference_all_t4.sh`。

| GPU | Realtime I/O | 音频并行通道数 | 吞吐量 (RTFX) | 延迟 | | | |
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

P.S. 貌似最新版本已经支持iVector了，但代码还没有合入官方