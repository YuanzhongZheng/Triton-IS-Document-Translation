# 快速Transformer网络

这个项目提供了全套脚本和代码来对一个高度优化的Transformer网络进行推理，这个项目由NVIDIA官方团队来开发测试和维护。

## 目录
- [网络模型概述](#网络模型概述)
    * [V1版快速Transformer](#V1版快速Transformer)
    * [V2版快速Transformer](#V2版快速Transformer)
    * [模型对比](#模型对比)
- [版本日志](#版本日志)
    * [更新日志](#更新日志)
    * [已知缺陷](#已知缺陷)


## 网络模型概述

### V1版快速Transformer

FasterTransformer V1 provides a highly optimized BERT equivalent Transformer layer for inference, including C++ API, TensorFlow op and TensorRT plugin. The experiments show that FasterTransformer V1 can provide 1.3 ~ 2 times speedup on NVIDIA Tesla T4 and NVIDIA Tesla V100 for inference. 

### V2版快速Transformer

FastTransformer V2 adds a highly optimized OpenNMT-tf based decoder and decoding for inference in FasterTransformer V1, including C++ API and TensorFlow op. The experiments show that FasterTransformer V2 can provide 1.5 ~ 11 times speedup on NVIDIA Telsa T4 and NVIDIA Tesla V 100 for inference.

### 模型对比

下表展示了两个版本的快速Transformer网络模型的区别：

| 框架               | 编码            |解码             |
|-----------------------|--------------------------|---------------|
|V1版快速Transformer  |  Yes |No |
|V2版快速Transformer  |  Yes |Yes |


## 版本日志
2020年6月将不再支持V1版快速Transformer

### 更新日志

2020年3月：
- 在快速Transformer 2.0版本中：
  - 修复解码器的最大序列长度不能超过128的BUG;
  - Add `translate_sample.py` to demonstrate how to translate a sentence by restoring the pretrained model of OpenNMT-tf.
  - Fix the bug that decoding does not check finish or not after each step. 
  - Fix the bug of decoder about max_seq_len.
  - Modify the decoding model structure to fit the OpenNMT-tf decoding model. 
    - Add a layer normalization layer after decoder.
    - Add a normalization for inputs of decoder
    
February 2020
 * Release the FasterTransformer 2.0
 * Provide a highly optimized OpenNMT-tf based decoder and decoding, including C++ API and TensorFlow OP.
 * Refine the sample codes of encoder.
 * Add dynamic batch size feature into encoder op.

July 2019
 * Release the FasterTransformer 1.0
 * Provide a highly optimized bert equivalent transformer layer, including C++ API, TensorFlow OP and TensorRT plugin.
 

## 已知缺陷

目前还没有已知问题。
