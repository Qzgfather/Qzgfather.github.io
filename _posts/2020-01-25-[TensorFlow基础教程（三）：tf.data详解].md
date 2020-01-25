---
layout: post
title: '[TensorFlow基础教程（三）：tf.data详解]'
date: 2020-01-25
author: Qizhiguo
cover: 
tags: TensorFlow
---

# [TensorFlow基础教程（三）：tf.data详解]

   借助 tf.data API，可以根据简单的可重用片段构建复杂的输入管道。例如，图片模型的管道可能会汇聚分布式文件系统中的文件中的数据、对每个图片应用随机扰动，并将随机选择的图片合并成用于训练的批次。文本模型的管道可能包括从原始文本数据中提取符号、根据对照表将其转换为嵌入标识符，以及将不同长度的序列组合成批次数据。使用 tf.data API 可以轻松处理大量数据、不同的数据格式以及复杂的转换。

**我们还是建议使用tf,data对数据进行处理，结合map函数对数据进行处理和操作。**

tf.data API 在 TensorFlow 中引入了两个新的抽象类：

- tf.data.Dataset 表示一系列元素，其中每个元素包含一个或多个 Tensor 对象。例如，在图像管道中，元素可能是单个训练样本，具有一对表示图像数据和标签的张量，可以通过两种不同的方式来创建数据集：

 - 创建来源（例如 **Dataset.from_tensor_slices()**，）通过一个或多个 tf.Tensor 对象构建数据集。

 - 应用转换（例如 **Dataset.batch()**），以通过一个或多个 tf.data.Dataset 对象构建数据集。对数据打包形成一个批次。

- tf.data.Iterator 提供了从数据集中提取元素的主要方法。Iterator.get_next() 返回的操作会在执行时生成 Dataset 的下一个元素，并且此操作通常充当输入管道代码和模型之间的接口。最简单的迭代器是“单次迭代器”，它与特定的 Dataset 相关联，并对其进行一次迭代。要实现更复杂的用途，您可以通过 Iterator.initializer 操作使用不同的数据集重新初始化和参数化迭代器，这样一来，您就可以在同一个程序中对训练和验证数据进行多次迭代（举例而言）。


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 2s 0us/step

```python
# 模型构造
inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = layers.Dense(64, activation='relu')(inputs)
h1 = layers.Dense(64, activation='relu')(h1)
outputs = layers.Dense(10, activation='softmax')(h1)
model = keras.Model(inputs, outputs)
# keras.utils.plot_model(model, 'net001.png', show_shapes=True)

model.compile(optimizer=keras.optimizers.RMSprop(),
             loss=keras.losses.SparseCategoricalCrossentropy(),
             metrics=[keras.metrics.SparseCategoricalAccuracy()])

# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 训练模型
history = model.fit(x_train, y_train, batch_size=64, epochs=3,
         validation_data=(x_val, y_val))
print('history:')
print(history.history)
```

    Train on 50000 samples, validate on 10000 samples
    Epoch 1/3
    50000/50000 [==============================] - 7s 135us/sample - loss: 0.3430 - sparse_categorical_accuracy: 0.9015 - val_loss: 0.1782 - val_sparse_categorical_accuracy: 0.9474
    Epoch 2/3
    50000/50000 [==============================] - 5s 94us/sample - loss: 0.1632 - sparse_categorical_accuracy: 0.9514 - val_loss: 0.1290 - val_sparse_categorical_accuracy: 0.9637
    Epoch 3/3
    50000/50000 [==============================] - 5s 99us/sample - loss: 0.1195 - sparse_categorical_accuracy: 0.9645 - val_loss: 0.1485 - val_sparse_categorical_accuracy: 0.9554
    history:
    {'loss': [0.34302246375560763, 0.1631989517867565, 0.11950971954107284], 'sparse_categorical_accuracy': [0.9015, 0.95138, 0.96446], 'val_loss': [0.17818026667833328, 0.12904182977378367, 0.14847243195772172], 'val_sparse_categorical_accuracy': [0.9474, 0.9637, 0.9554]}


## 基本机制
​		本指南的这一部分介绍了创建不同种类的 Dataset 和 Iterator 对象的基础知识，以及如何从这些对象中提取数据。

​		要启动输入管道，您必须定义来源。例如，要通过内存中的某些张量构建 Dataset，您可以使用 tf.data.Dataset.from_tensors() 或 tf.data.Dataset.from_tensor_slices()。如果输入数据是 TFRecord 格式存储在磁盘上，那么你可以构建 tf.data.TFRecordDataset，读取数据。**而且我建议使用TFRecord 格式文件来进行储存，比如10个TFRecord 格式文件要比一万张图片操作起来要简单的多。**

一旦有了 Dataset 对象，你就可以更方便的进行操作，比如使用map函数，对数据集中的图像进行裁剪、旋转等操作。


## 1.读取Numpy数据
如果您的所有输入数据都适合存储在内存中，则根据输入数据创建 Dataset 的最简单方法是将它们转换为 tf.Tensor 对象，并使用 Dataset.from_tensor_slices()，将其转换为dateset对象，进行下一步的操作。


```python
import tensorflow as tf
import numpy as np

# 随机生成numpy数据。
x_train = np.random.rand(500, 3)
x_labels = np.random.rand(500, 1)
dataset = tf.data.Dataset.from_tensor_slices((x_train, x_labels))
print(dataset)
```

**请注意：上面的代码段会将 features 和 labels 数组作为 tf.constant() 指令嵌入在 TensorFlow 图中。这样非常适合小型数据集，但会浪费内存，因为会多次复制数组的内容，并可能会达到 tf.GraphDef 协议缓冲区的 2GB 上限。**

## 2.读取 TFRecord 数据
tf.data API 支持多种文件格式，因此可以处理那些不适合存储在内存中的大型数据集。例如，TFRecord 文件格式是一种面向记录的简单**二进制格式**，很多 TensorFlow 应用采用此格式来训练数据。通过 tf.data.TFRecordDataset 类，可以将一个或多个 TFRecord 文件的内容作为输入管道的一部分进行流式传输。


```python
# Creates a dataset that reads all of the examples from two files.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
```

TFRecordDataset **初始化程序的 filenames 参数可以是字符串、字符串列表，也可以是字符串 tf.Tensor。**具体如何生成 TFRecord 数据我们在数据结构化中详细介绍。

## 3.读取文本数据
​		很多数据集都是作为一个或多个文本文件分布的。tf.data.TextLineDataset 提供了一种从一个或多个文本文件中提取行的简单方法。给定一个或多个文件名，**TextLineDataset 会为这些文件的每行生成一个字符串值元素。**像 TFRecordDataset 一样，TextLineDataset 将接受 filenames（作为 tf.Tensor）。


```python
filenames = ["./1.txt", "./2.txt"]
dataset = tf.data.TextLineDataset(filenames)
print(dataset)
```

## 4.预设api读取数据

TensorFlow2.0内置了常用数据集的下载以及数据处理脚本，我们只需要一行代码就可以下载=处理并返回处理好的数据，数据集有：boston_housing、cifar10、import cifar100、fashion_mnist。

以cifar10数据集为例，我们使用tensorflow.keras.datasets.mnist.load_data()就可以返回训练集及其标签和测试集以及其标签，返回的是int8型的numpy数组数据。

```python
from tensorflow.keras import datasets
(train, train_labels), (test, test_labels) = datasets.mnist.load_data()
print('train_shape: {0},'
      'train_labels_shape: {1}, '
      'test_shape: {2}, '
      'test_labels_shap:{3}.'.format(train.shape,  train_labels.shape, test.shape,  test_labels.shape))
```

```python
train_shape: (60000, 28, 28),
train_labels_shape: (60000,),
test_shape: (10000, 28, 28), 
test_labels_shap:(10000,).
```

## 5.使用 Dataset.map() 预处理数据
**Dataset.map(f) 转换通过将指定函数 f 应用于输入数据集的每个元素来生成新数据集。此转换基于 map() 函数**（通常应用于函数式编程语言中的列表和其他结构）。函数 f 会接受表示输入中单个元素的 tf.Tensor 对象，并返回表示新数据集中单个元素的 tf.Tensor 对象。此函数的实现使用标准的 TensorFlow 指令将一个元素转换为另一个元素。

在读取图片作为训练数据的时候，可以建立一个文件包含所有图片的路径的字符串集合，使用


```python
## 常用处理
# 读入图片数据，转换大小
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label

# A vector of filenames.
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
```

## 处理多个周期
tf.data API 提供了两种主要方式来处理同一数据的多个周期。要迭代数据集多个周期，最简单的方法是使用 Dataset.repeat() 转换。例如，要创建一个将其输入重复 10 个周期的数据集。官方网站上，阐述了 repeat 在 shuffle 之前使用可以有效提高性能，但是模糊了数据样本的 epoch 实际上，可以这样理解shuffle取之前已经重置了源数据集， 即先repeat，后shuffle。tf会将数据集乘以repeat次数，然后整个打乱一次，把它当作一个数据集。


```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat(10)
dataset = dataset.batch(32)
```

## 小结
- 1.读取Numpy数据        **tf.data.Dataset.from_tensor_slices((features, labels))**
- 2.读取 TFRecord 数据     **tf.data.TFRecordDataset(images_name)**
- 3.读取文本数据         **tf.data.TextLineDataset(filenames)**
- 4.预设api读取数据
