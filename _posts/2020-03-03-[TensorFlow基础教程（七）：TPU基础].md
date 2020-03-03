---
layout: post
title: '2020-03-03-[TensorFlow基础教程（七）：TPU基础]'
date: 2020-03-03
author: Qizhiguo
cover: 
tags: TensorFlow
---

# TPU使用指南

TPU（张量处理器：tensor processing unit）是谷歌一种加速训练的芯片可以实现对模型的训练，专为机器学习打造的 ASIC，广泛应用于翻译、相册、搜索、助理和 Gmail 等诸多 Google 产品。本教程使用的TensorFlow版本是2.0，与1.x有一点不一样，接下来将详细介绍TPU的使用。官方网址：https://tensorflow.google.cn/guide/tpu?hl=en。

```python
from __future__ import absolute_import, division, print_function, unicode_literals
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

import os
import tensorflow_datasets as tfds

```

```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
```

```python
def create_model():
  # return tf.keras.Sequential(
  #     [tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
  #      tf.keras.layers.Flatten(),
  #      tf.keras.layers.Dense(128, activation='relu'),
  #      tf.keras.layers.Dense(10)])
  return tf.keras.applications.vgg16.VGG16(weights=None, input_shape=(32, 32, 3))
```

其中有一个大坑就是一定注意数据集的数据类型，一般数据为float32，标签为int64，有些数据类型TPU不支持，还有就是仅支持model的fit方法，所以你需要将你的数据转化成tf.data.Dataset的格式然后进行数据喂入，数据增强自己写个函数即可配合tf.data.Dataset的map函数进行处理，非常不建议将所有的数据加载到内存。

```python
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
  model = create_model()
  model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001),
         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
         metrics=['sparse_categorical_accuracy'])


(train, train_labels), (test, test_labels) = tf.keras.datasets.cifar10.load_data()
mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

train = train.reshape([train.shape[0], 32, 32, 3])
train = train.astype(np.float32)
train_labels = train_labels.astype(np.int64)

# test = test.astype(np.float32) / 225.0
# test = test.reshape([test.shape[0], 32, 32, 3])
# test_labels = test_labels.astype(np.int64)



# 数据增强
def pro(x):
  x = tf.cast(x, dtype=tf.float32) / 225.0
  x = (x - mean) / std
  x = tf.image.random_flip_up_down(x)
  return x


train = tf.data.Dataset.from_tensor_slices(train).map(pro)
train_labels = tf.data.Dataset.from_tensor_slices(train_labels)

data = tf.data.Dataset.zip((train, train_labels)).batch(200)
print(data)

model.fit(data,
    epochs=100
    # validation_data=(test, test_labels)
)
```

参考文章：

https://tensorflow.google.cn/guide/tpu?hl=en

https://blog.csdn.net/big91987/article/details/87898100



