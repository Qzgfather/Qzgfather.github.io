---
layout: post
title: '[TensorFlow基础教程（一）：基础概念与操作]'
date: 2020-01-23
author: Qizhiguo
cover: 
tags: TensorFlow
---

# [TensorFlow基础教程（一）：基础概念与操作]

## 1、Tensorflow介绍

   TensorFlow 2.0 网站将该项目描述为“端到端开源机器学习平台”。实际上 TensorFlow 已进化成为一个更全面的“工具、库和社区资源生态系统”，可帮助研究人员构建和部署人工智能助力的应用。
TensorFlow 2.0 有四大组成部分：

- TensorFlow 核心，一个用于开发和训练机器学习模型的开源库；
- TensorFlow.js，一个用于在浏览器和 Node.js 上训练和部署模型的 JavaScript 库；
- TensorFlow Lite，一个轻量级库，用于在移动和嵌入式设备上部署模型；
- TensorFlow Extended，一个在大型生产环境中准备数据、训练、验证和部署模型的平台。

TensorFlow 2.0 生态系统包括对 Python、JavaScript 和 Swift 的支持，以及对云、浏览器和边缘设备的部署支持。TensorBoard（可视化）和 TensorFlow Hub（模型库）都是很有用的工具。TensorFlow Extended（TFX）则支持端到端生产流水线。[参考](http://baijiahao.baidu.com/sid=1640756159447567798&wfr=spider&for=pc)
## 2、Keras有什么优点：

### 2.1更简洁

   keras原本是独立于Tensorflow的高级API可以实现跨平台的模型部署，其后端可以是tensorflow。谷歌将Keras收购后，作为其官方的API并入到Tensorflow2.0中，tf.keras与keras并不完全相同（import keras ≠ from tensorflow import keras）独立的keras在以后将逐渐淡出人们的视野。

### 2.2兼容Keras
​        最新版 TensorFlow 中的 tf.keras 版本可能与 PyPI 中的最新 keras 版本不同。请查看 tf.keras.version。保存模型的权重时，tf.keras 默认采用检查点格式。请传递 save_format='h5' 以使用 HDF5，请在保存模型的时候使用h5格式的文件来保存模型。[官方介绍](https://tensorflow.google.cn/guide/keras)


### 2.3更灵活
​		如果接触过pytorch，相比pytorch，发现tensorflow 1.x版本过于复杂，比如计算图的定义，会话的定义。在tensorflow 2.0版本中更加方便的应用，TensorFlow2.x可以更快速的搭建神经网络，你只需要调用几个api就可以搭建一个模型，同时如果你想自定义其中的组成部分的时候，也可以快速的构建自己的优化器、损失函数等。


```python
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)

variable1 = tf.constant(3)
variable2 = tf.constant(2)

result = tf.add(variable1, variable2)
print(result)
```

```python
tf.Tensor(5, shape=(), dtype=int32)
```


## 3、基础结构
TensorFlow在定义模型的时候有很多种方法，第一种是堆叠式，就是把要用的神经网络层，直接堆叠起来，模型自动计算输出tensor的形状，并进行计算。  

**1. 堆叠式**

**2. 函数式**

**3. 自定义式**

堆叠式模型一般分为五个步骤：

- **数据处理**

- **网络的定义**
- **模型的编译**
- **喂入数据**
- **定义其他操作（模型的保存，模型参数的可视化）**

 ## 2.1数据处理
​        深度学习需要大量的数据，在进行模型搭建之前一定要对数据进行充分的分析，并对数据进行相关操作然后再去建立合适的模型进行训练，这里我们生成（1000,72）的随机数矩阵作为训练集，同时生成一个（1000,10）的训练集标签，以此类推生成测试集，我们为了简单生成随机数当做训练集，在现实情况中，数据一般都是有规律的，我们就是使用TensorFlow寻找这个规律。


```python
import numpy as np

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))

val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))
```

## 2.2模型训练
在 Keras 中，您可以通过组合层来构建模型。模型（通常）是由层构成的图。最常见的模型类型是层的堆叠：tf.keras.Sequential 模型。


```python
# 1 定义网络
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 2.模型编译——使用compile
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=[tf.keras.metrics.categorical_accuracy])

# # 3.喂入数据
model.fit(train_x, train_y, epochs=10, batch_size=100,
          validation_data=(val_x, val_y))

# 4.其他操作——可视化模型结构
model.summary()
```

    Train on 1000 samples, validate on 200 samples
    Epoch 1/10
    1000/1000 [==============================] - 1s 988us/sample - loss: 12.3978 - categorical_accuracy: 0.1110 - val_loss: 12.7143 - val_categorical_accuracy: 0.1050
    Epoch 2/10
    1000/1000 [==============================] - 0s 69us/sample - loss: 13.1638 - categorical_accuracy: 0.1120 - val_loss: 13.9713 - val_categorical_accuracy: 0.1000
    Epoch 3/10
    1000/1000 [==============================] - 0s 69us/sample - loss: 14.8748 - categorical_accuracy: 0.1070 - val_loss: 16.4407 - val_categorical_accuracy: 0.1100
    Epoch 4/10
    1000/1000 [==============================] - 0s 67us/sample - loss: 18.1452 - categorical_accuracy: 0.1130 - val_loss: 20.9342 - val_categorical_accuracy: 0.0950
    Epoch 5/10
    1000/1000 [==============================] - 0s 55us/sample - loss: 23.2897 - categorical_accuracy: 0.1150 - val_loss: 26.6265 - val_categorical_accuracy: 0.0900
    Epoch 6/10
    1000/1000 [==============================] - 0s 63us/sample - loss: 28.8893 - categorical_accuracy: 0.1050 - val_loss: 32.5985 - val_categorical_accuracy: 0.1150
    Epoch 7/10
    1000/1000 [==============================] - 0s 63us/sample - loss: 34.9628 - categorical_accuracy: 0.1010 - val_loss: 38.9556 - val_categorical_accuracy: 0.1000
    Epoch 8/10
    1000/1000 [==============================] - ETA: 0s - loss: 37.6530 - categorical_accuracy: 0.100 - 0s 67us/sample - loss: 41.3707 - categorical_accuracy: 0.0970 - val_loss: 46.3646 - val_categorical_accuracy: 0.1150
    Epoch 9/10
    1000/1000 [==============================] - 0s 70us/sample - loss: 49.9655 - categorical_accuracy: 0.1010 - val_loss: 57.5191 - val_categorical_accuracy: 0.1050
    Epoch 10/10
    1000/1000 [==============================] - 0s 65us/sample - loss: 62.8709 - categorical_accuracy: 0.0960 - val_loss: 73.0761 - val_categorical_accuracy: 0.0950
    Model: "sequential_35"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_105 (Dense)            multiple                  2336      
    _________________________________________________________________
    dense_106 (Dense)            multiple                  1056      
    _________________________________________________________________
    dense_107 (Dense)            multiple                  330       
    =================================================================
    Total params: 3,722
    Trainable params: 3,722
    Non-trainable params: 0
    _________________________________________________________________


## 2.3API介绍
### tf.keras.Sequential():初始化模型，可以传入参数，也可以不传入，传入的参数是一个包含layer的列表。
### tf.keras.layers.Dense():定义全连接层，运算逻辑：Y = W·X+b，  
- **activation**：要使用的激活函数。如果您未指定任何内容，则不会应用任何激活（即“线性”激活：）a(x) = x，
- **use_bias**：布尔值，层是否使用偏差矢量。即是否在进行W·X后加上偏置b。
- **kernel_initializer**：权重矩阵的初始化程序。
- **bias_initializer**：偏置向量的初始化程序。
- **kernel_regularizer**：正则化函数应用于kernel权重矩阵，L1或者L2正则化。。
- **bias_regularizer**：正则化函数应用于偏差向量，L1或者L2正则化。

### tf.keras.Model.compile() ：编译模型，tf.keras.Model一般是你定义的模型名字。
- **optimizer**：此对象会指定训练过程。从 tf.keras.optimizers 模块向其传递优化器实例，例如 tf.keras.optimizers.Adam、tf.keras.optimizers.SGD,
- **loss**：要在优化期间最小化的函数。常见选择包括均方误差 (mean_squared_error：mse)、categorical_crossentropy 和 binary_crossentropy。损失函数由名称或通过从 tf.keras.losses 模块传递可调用对象来指定。比如：tf.keras.losses.categorical_crossentropy（交叉熵准确率）。
- **metrics**：用于监控训练。它们是 tf.keras.metrics 模块中的字符串名称或可调用对象。
- **optimizer列举：**

算法|字符表示|对象表示
:-|:-:|:-:
Adam|'adam'|tf.keras.optimizers.Adam
SGD|'SGD'|tf.keras.optimizers.SGD

- **loss列举:**

算法|字符表示|对象表示
:-|:-:|:-:
交叉熵 |'categorical_crossentropy'|tf.keras.losses.categorical_crossentropy
二分类交叉熵|'binary_crossentropy'|tf.keras.losses.binary_crossentropy
MSE|'mse'|tf.keras.losses.mean_squared_error,tf.keras.losses.mse,tf.keras.metrics.MSE
mae|'mae'|tf.losses.MAE,tf.losses.mae,tf.losses.mean_absolute_error

### tf.keras.Model.fit()：喂入数据。

```python
 fit(self,
          model,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_freq=1,
          **kwargs):
```



- **epochs**：以周期为单位进行训练。一个周期是对整个输入数据的一次迭代（以较小的批次完成迭代）。
- **batch_size**：当传递 NumPy 数据时，模型将数据分成较小的批次，并在训练期间迭代这些批次。此整数指定每个批次的大小。请注意，如果样本总数不能被批次大小整除，则最后一个批次可能更小。
- **validation_data**：在对模型进行原型设计时，可以轻松监控该模型在某些验证数据上达到的效果。输入此参数（输入和标签元组）可以让该模型在每个周期结束时显示所传递数据的损失和指标。
- **validation_split**：对训练集进行划分取值0到1，比如取0.2的时候将20%的数据当作测试集。
- **callbacks**：一个比较重要的参数，如果想可视化或者保存模型，需要构建一个callbacks列表，然后传入fit中。
- **shuffle**：是否打乱数据，默认是打乱数据。

### 其他喂入数据的方法：tf.data.Dataset

```python
import tensorflow as tf
from tensorflow import keras

# 1 定义网络
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 2.模型编译——使用compile
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=[tf.keras.metrics.categorical_accuracy])

# # 3.喂入数据_2

dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)) # 生成一个数据
dataset = dataset.batch(100)
# repeat()将数据集进行重复，如果不传入参数将不停的进行重复
dataset = dataset.repeat()
print(dataset)

model.fit(dataset, epochs=10, steps_per_epoch=10)



# # 4.其他操作——可视化模型结构
model.summary()
```

    <RepeatDataset shapes: ((None, 72), (None, 10)), types: (tf.float64, tf.float64)>
    WARNING:tensorflow:Layer sequential_34 is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    Train for 10 steps
    Epoch 1/10
    10/10 [==============================] - 1s 94ms/step - loss: 11.9120 - categorical_accuracy: 0.1050
    Epoch 2/10
    10/10 [==============================] - 0s 4ms/step - loss: 12.2187 - categorical_accuracy: 0.1040
    Epoch 3/10
    10/10 [==============================] - 0s 4ms/step - loss: 13.3768 - categorical_accuracy: 0.1030
    Epoch 4/10
    10/10 [==============================] - 0s 5ms/step - loss: 15.4902 - categorical_accuracy: 0.1030
    Epoch 5/10
    10/10 [==============================] - 0s 4ms/step - loss: 18.1487 - categorical_accuracy: 0.1030
    Epoch 6/10
    10/10 [==============================] - 0s 5ms/step - loss: 20.2402 - categorical_accuracy: 0.1030
    Epoch 7/10
    10/10 [==============================] - 0s 4ms/step - loss: 22.2389 - categorical_accuracy: 0.1080
    Epoch 8/10
    10/10 [==============================] - 0s 5ms/step - loss: 26.5437 - categorical_accuracy: 0.1090
    Epoch 9/10
    10/10 [==============================] - 0s 5ms/step - loss: 32.6566 - categorical_accuracy: 0.1160
    Epoch 10/10
    10/10 [==============================] - 0s 5ms/step - loss: 39.2358 - categorical_accuracy: 0.1090
    Model: "sequential_34"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_102 (Dense)            multiple                  2336      
    _________________________________________________________________
    dense_103 (Dense)            multiple                  1056      
    _________________________________________________________________
    dense_104 (Dense)            multiple                  330       
    =================================================================
    Total params: 3,722
    Trainable params: 3,722
    Non-trainable params: 0
    _________________________________________________________________


### 注意：一般在进行批次训练时经常有三个参数容易弄混：
- 批大小：batch_size,默认32
- 训练轮数：epochs
- 批个数：steps_per_epoch，steps_per_epoch=总样本数（all_samples) / 批大小(batch_size）

现在用的优化器SGD是stochastic gradient descent的缩写，但不代表是一个样本就更新一回，还是基于mini-batch的。
那 batch、epoch、 iteration代表什么呢？

- batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
- iteration：1个iteration等于使用batchsize个样本训练一次；
- epoch：1个epoch等于使用训练集中的全部样本训练一次，通俗的讲epoch的值就是整个数据集被轮几次。

比如训练集有500个样本，batchsize = 10 ，那么训练完整个样本集：iteration=50，epoch=1.

官方网站上，阐述了 repeat 在 shuffle 之前使用可以有效提高性能，但是模糊了数据样本的 epoch 实际上，可以这样理解shuffle取之前已经重置了源数据集， 即先repeat，后shuffle。tf会将数据集乘以repeat次数，然后整个打乱一次，把它当作一个数据集。

## 2.4模型的评估与预测
训练好模型后可以对模型进行评估,使用的API时是：**model.evaluate()**


```python
# 模型的评估
test_x = np.random.random((1000, 72))
test_y = np.random.random((1000, 10))
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()
model.evaluate(test_data, steps=30)
```

    30/30 [==============================] - 0s 9ms/step - loss: 72.8322 - categorical_accuracy: 0.0979
    
    [72.8322021484375, 0.09791667]


```python
# 模型的预测
result = model.predict(test_x, batch_size=32)
print(result)
```

    [[3.8827932e-04 4.6541311e-07 1.7521141e-21 ... 8.7908869e-13
      4.3893911e-10 1.7103927e-04]
     [3.1108255e-04 2.0661598e-07 1.6624059e-21 ... 4.2849480e-13
      5.3313892e-10 2.3800629e-04]
     [1.3693042e-04 2.1717564e-07 2.8686385e-22 ... 2.6706458e-13
      3.3628944e-10 1.3130548e-04]
     ...
     [5.6340022e-04 2.2013110e-06 5.8089653e-19 ... 3.2290354e-11
      7.3468160e-09 3.4577397e-04]
     [5.0389755e-04 1.0036985e-06 1.3823169e-19 ... 7.2727219e-12
      4.2529877e-09 5.0028716e-04]
     [6.1346550e-04 7.8577131e-07 3.4149053e-19 ... 7.8434672e-12
      5.0080353e-09 5.5151404e-04]]
