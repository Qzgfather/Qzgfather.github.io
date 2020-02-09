---
layout: post
title: '[TensorFlow基础教程（五）：Fashion MNIST]'
date: 2020-02-10
author: Qizhiguo
cover: 
tags: TensorFlow
---

# [TensorFlow基础教程（五）：Fashion MNIST]

Fashion MNIST的目标是作为经典MNIST数据的替换——通常被用作计算机视觉机器学习程序的“Hello, World”。MNIST数据集包含手写数字(0,1,2等)的图像，格式与我们将在这里使用的衣服相同使用了时尚MNIST的多样性，因为它是一个比常规MNIST稍微更具挑战性的问题。这两个数据集都相对较小，用于验证算法是否按预期工作。它们是测试和调试代码的好起点我们将使用6万张图片来训练网络和1万张图片来评估网络对图片的分类有多精确。


```python
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
```


```python
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

```


```python
train_images = train_images.reshape(60000, 784).astype('float32') / 225


model = keras.Sequential()
model.add(keras.layers.Dense(784,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

model.fit(train_images,train_labels,batch_size=32,epochs=10)
model.summary()
```

    Train on 60000 samples
    Epoch 1/10
    60000/60000 [==============================] - 48s 796us/sample - loss: 2.3028 - sparse_categorical_accuracy: 0.0995 - loss: 2.3028 - sparse
    Epoch 2/10
    60000/60000 [==============================] - 47s 777us/sample - loss: 2.3027 - sparse_categorical_accuracy: 0.0993
    Epoch 3/10
    60000/60000 [==============================] - 47s 785us/sample - loss: 2.3028 - sparse_categorical_accuracy: 0.0975
    Epoch 4/10
    60000/60000 [==============================] - 47s 786us/sample - loss: 2.3028 - sparse_categorical_accuracy: 0.0990
    Epoch 5/10
    60000/60000 [==============================] - 47s 791us/sample - loss: 2.3028 - sparse_categorical_accuracy: 0.0989
    Epoch 6/10
    60000/60000 [==============================] - 48s 804us/sample - loss: 2.3028 - sparse_categorical_accuracy: 0.0989
    Epoch 7/10
    60000/60000 [==============================] - 47s 776us/sample - loss: 2.3028 - sparse_categorical_accuracy: 0.0980 - 
    Epoch 8/10
    60000/60000 [==============================] - 48s 795us/sample - loss: 2.3027 - sparse_categorical_accuracy: 0.1011 - l
    Epoch 9/10
    60000/60000 [==============================] - 47s 790us/sample - loss: 2.3028 - sparse_categorical_accuracy: 0.0986
    Epoch 10/10
    60000/60000 [==============================] - 48s 799us/sample - loss: 2.3028 - sparse_categorical_accuracy: 0.0986- ETA: 6s - los - ETA: 0s - loss: 2.3028 - sparse_categorical_
    Model: "sequential_7"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_17 (Dense)             multiple                  615440    
    _________________________________________________________________
    dense_18 (Dense)             multiple                  78500     
    _________________________________________________________________
    dense_19 (Dense)             multiple                  10100     
    _________________________________________________________________
    dense_20 (Dense)             multiple                  1010      
    =================================================================
    Total params: 705,050
    Trainable params: 705,050
    Non-trainable params: 0
    _________________________________________________________________



```python
train_images = train_images.reshape(60000, 784).astype('float32') / 225
inputs = keras.Input(shape=(784))
h1 = keras.layers.Dense(100,activation='relu')(inputs)
h2 = keras.layers.Dense(100,activation='relu')(h1)
pred = keras.layers.Dense(10,activation='softmax')(h2)

model = keras.Model(inputs=inputs,outputs=pred)

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()]
             )
    
model.fit(train_images,train_labels)
```

    Train on 60000 samples
    60000/60000 [==============================] - 11s 180us/sample - loss: 0.4988 - sparse_categorical_accuracy: 0.8211





    <tensorflow.python.keras.callbacks.History at 0x1b7aee16748>



## 小结
到目前为止我们介绍深度学习的“Hello World”项目（mnist、fashion mnist）时采用的是多层感知机模型，这种模型每一层都是全连接层，接下来我们将介绍卷积神经网络。
