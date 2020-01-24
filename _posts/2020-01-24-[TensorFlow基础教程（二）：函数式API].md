---
layout: post
title: '[TensorFlow基础教程（二）：函数式API]'
date: 2020-01-24
author: Qizhiguo
cover: 
tags: TensorFlow
---

# [TensorFlow基础教程（二）：函数式API]

# 高级使用方法

## 1.函数式API
tf.keras.Sequential 模型是层的简单堆叠，无法表示任意模型。使用 Keras 函数式 API 可以构建复杂的模型拓扑，例如：

**多输入模型，**

**多输出模型，**

**具有共享层的模型（同一层被调用多次），**

**具有非序列数据流的模型（例如，残差连接）。**





**使用函数式 API 构建的模型具有以下特征：**

层实例可调用并返回张量。
输入张量和输出张量用于定义 tf.keras.Model 实例。
此模型的训练方式和 Sequential 模型一样。


```python
import tensorflow as tf
from tensorflow import keras
import json
```


```python
import numpy as np

# train_x：训练数据
# train_y：训练标签 因为随机生成，训练的精度很差。
train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))
val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))
```


```python
input_x = tf.keras.Input(shape=(72,))
hidden1 = tf.keras.layers.Dense(32, activation='relu')(input_x) # 在定义好得层后面加上括号就可以对括号里的内容进行处理
hidden2 = tf.keras.layers.Dense(16, activation='relu')(hidden1)
pred = tf.keras.layers.Dense(10, activation='softmax')(hidden2)

model = tf.keras.Model(inputs=input_x, outputs=pred) # 重点，必须要有的
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=32, epochs=5)
```

    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.iter
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_1
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_2
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.decay
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.learning_rate
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'm' for (root).layer-0.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'v' for (root).layer-0.kernel
    WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/alpha/guide/checkpoints#loading_mechanics for details.
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer
    WARNING:tensorflow:Unresolved object in checkpoint: (root).layer-0.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.iter
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_1
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_2
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.decay
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.learning_rate
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'm' for (root).layer-0.kernel
    WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'v' for (root).layer-0.kernel
    WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/alpha/guide/checkpoints#loading_mechanics for details.
    Train on 1000 samples
    Epoch 1/5
    1000/1000 [==============================] - 1s 904us/sample - loss: 13.5606 - accuracy: 0.0890
    Epoch 2/5
    1000/1000 [==============================] - 0s 120us/sample - loss: 21.9375 - accuracy: 0.0910
    Epoch 3/5
    1000/1000 [==============================] - 0s 128us/sample - loss: 39.1652 - accuracy: 0.0880
    Epoch 4/5
    1000/1000 [==============================] - 0s 132us/sample - loss: 67.1671 - accuracy: 0.1080
    Epoch 5/5
    1000/1000 [==============================] - 0s 144us/sample - loss: 115.7851 - accuracy: 0.1040

    <tensorflow.python.keras.callbacks.History at 0x206eac393c8>

## 2.模型子类化
通过对 tf.keras.Model 进行子类化并定义您自己的前向传播来构建完全可自定义的模型。在 init 方法中创建层并将它们设置为类实例的属性。在 call 方法中定义前向传播


```python
class MyModel(tf.keras.Model): # 继承tf.keras.Model类，进行自定义
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # 定义你的神经网络层
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(num_classes, activation='sigmoid')
    def call(self, inputs):
        #使用你在__init__定义的层来构建多层的神经网络。
        x = self.dense_1(inputs)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
```


```python
# 模型实例化
model = MyModel(num_classes=10)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(train_x, train_y,batch_size=32, epochs=5)
```

    Train on 1000 samples
    Epoch 1/5
    WARNING:tensorflow:From c:\python36\lib\site-packages\tensorflow_core\python\ops\math_grad.py:1394: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    1000/1000 [==============================] - 1s 822us/sample - loss: 11.5143 - accuracy: 0.0960
    Epoch 2/5
    1000/1000 [==============================] - 0s 104us/sample - loss: 11.4683 - accuracy: 0.1130
    Epoch 3/5
    1000/1000 [==============================] - 0s 139us/sample - loss: 11.4607 - accuracy: 0.1190
    Epoch 4/5
    1000/1000 [==============================] - 0s 131us/sample - loss: 11.4568 - accuracy: 0.1250
    Epoch 5/5
    1000/1000 [==============================] - 0s 122us/sample - loss: 11.4529 - accuracy: 0.1280

    <tensorflow.python.keras.callbacks.History at 0x206ecde8588>

## 3.自定义层
通过对 tf.keras.layers.Layer 进行子类化并实现以下方法来创建自定义层：
- **_init_（）**：在成员变量中保存配置,获取神经元的个数。决定着权重矩阵的列和张量的形状
- build：创建层的权重。使用 add_weight 方法添加权重。
- call：定义前向传播。
- compute_output_shape：指定在给定输入形状的情况下如何计算层的输出形状。或者，可以通过实现 get_config 方法和 from_config 类方法序列化层。

### 为什么需要自定义层？
在一些卷积神经网络中，一部分卷积层需要自己设计，并重复调用，所以网络层的自定义是一个很实用的功能。


```python
"""
这一部分只对Desne进行重新定义，对于Desne的计算原理在这不在阐述。
"""
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # 为这层网路创建权重
        self.kernel = self.add_weight(name='kernel',
                                  shape=shape,
                                  initializer='uniform',
                                  trainable=True)
    # Be sure to call this at the end
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

# 以下内容不便于理解，上述代码可以实现自定义功能
#     计算输出维度 [b,72]
#     def compute_output_shape(self, input_shape):
#         shape = tf.TensorShape(input_shape).as_list()
#         shape[-1] = self.output_dim
#         return tf.TensorShape(shape)

#     def get_config(self):
#         base_config = super(MyLayer, self).get_config()
#         base_config['output_dim'] = self.output_dim
#         return base_config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
```


```python
# 使用自己定义的层来搭建模型。
model = tf.keras.Sequential([
    MyLayer(10),
    tf.keras.layers.Activation('softmax')])

# 设置编译数据
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 喂入数据进行训练
model.fit(train_x, train_y, batch_size=32, epochs=5)

```

    Train on 1000 samples
    Epoch 1/5
    1000/1000 [==============================] - 1s 595us/sample - loss: 11.6531 - accuracy: 0.0950
    Epoch 2/5
    1000/1000 [==============================] - 0s 93us/sample - loss: 11.6529 - accuracy: 0.0930
    Epoch 3/5
    1000/1000 [==============================] - 0s 96us/sample - loss: 11.6534 - accuracy: 0.0920
    Epoch 4/5
    1000/1000 [==============================] - 0s 114us/sample - loss: 11.6534 - accuracy: 0.0950
    Epoch 5/5
    1000/1000 [==============================] - 0s 124us/sample - loss: 11.6520 - accuracy: 0.0950

## 4回调
回调是传递给模型的对象，用于在训练期间自定义该模型并扩展其行为。您可以编写自定义回调，也可以使用包含以下方法的内置 tf.keras.callbacks：
- tf.keras.callbacks.ModelCheckpoint：定期保存模型的检查点。
- tf.keras.callbacks.LearningRateScheduler：动态更改学习速率。
- tf.keras.callbacks.EarlyStopping：在验证效果不再改进时中断训练，就是训练效果不再改善的时候提前终止训练。
- tf.keras.callbacks.TensorBoard：使用 TensorBoard 监控模型的行为。打开终端输入 tensorboard --logdir== .\log即可。

要使用 tf.keras.callbacks.Callback，请将其传递给模型的 fit 方法


```python
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='.\logs')
]
model.fit(train_x, train_y, batch_size=32, epochs=5, callbacks=callbacks)
```

    Train on 1000 samples
    WARNING:tensorflow:Model failed to serialize as JSON. Ignoring... Layers with arguments in `__init__` must override `get_config`.
    Epoch 1/5
     448/1000 [============>.................] - ETA: 0s - loss: 11.5190 - accuracy: 0.0915WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    1000/1000 [==============================] - 0s 116us/sample - loss: 11.4905 - accuracy: 0.0890
    Epoch 2/5
     608/1000 [=================>............] - ETA: 0s - loss: 11.4970 - accuracy: 0.0839WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    1000/1000 [==============================] - 0s 96us/sample - loss: 11.4942 - accuracy: 0.0900
    Epoch 3/5
     544/1000 [===============>..............] - ETA: 0s - loss: 11.4443 - accuracy: 0.0827    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    1000/1000 [==============================] - 0s 97us/sample - loss: 11.4907 - accuracy: 0.0860
    Epoch 4/5
     576/1000 [================>.............] - ETA: 0s - loss: 11.4966 - accuracy: 0.1007WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    1000/1000 [==============================] - 0s 99us/sample - loss: 11.4900 - accuracy: 0.0880
    Epoch 5/5
     576/1000 [================>.............] - ETA: 0s - loss: 11.6530 - accuracy: 0.0903WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
    1000/1000 [==============================] - 0s 98us/sample - loss: 11.4933 - accuracy: 0.0880

## 5.模型的保存与读取
将模型训练好以后，我们需要将训练好的参数进行保存。主要的方式有：
### 5.1仅保存权重
使用 tf.keras.Model.save_weights 保存并加载模型的权重,默认情况下，会以 TensorFlow 检查点文件格式保存模型的权重。权重也可以另存为 Keras HDF5 格式（Keras 多后端实现的默认格式）.


```python
model.save_weights('./models')
```


```python
!cd ./models && dir
```

     驱动器 C 中的卷没有标签。
     卷的序列号是 8E03-0E05
    
     C:\Users\Administrator\tensorflow2.0\models 的目录
    
    2019/09/30  10:55    <DIR>          .
    2019/09/30  10:55    <DIR>          ..
    2019/09/30  10:54                69 checkpoint
    2019/09/30  10:54             9,554 models.data-00000-of-00001
    2019/09/30  10:54               605 models.index
                   3 个文件         10,228 字节
                   2 个目录  8,211,222,528 可用字节



```python
# 新建一个模型
new_model = tf.keras.Sequential([
    MyLayer(10),
    tf.keras.layers.Activation('softmax')])
# 进行编译
new_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 导入模型参数
new_model.load_weights('./models')
val_x = np.random.random((100, 72))
val_y = np.random.random((100, 10))
# 开始验证
result = new_model.evaluate(val_x, val_y, batch_size=10)
print(result)
```

    100/1 [========================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================] - 0s 2ms/sample - loss: 11.4490 - accuracy: 0.0600
    [11.448968887329102, 0.06]


默认情况下，会以 TensorFlow 检查点文件格式保存模型的权重。权重也可以另存为 Keras HDF5 格式（Keras 多后端实现的默认格式）：


```python
model.save_weights('my_model.h5', save_format='h5')

model.load_weights('my_model.h5')
```

### 5.2仅保存配置
可以保存模型的配置，此操作会对模型架构（不含任何权重）进行序列化。即使没有定义原始模型的代码，保存的配置也可以重新创建并初始化相同的模型。Keras 支持 JSON 和 YAML 序列化格式


```python
# 转换为json数据。
json_string = model.to_json()
json_string
```


    '{"class_name": "Model", "config": {"name": "model_1", "layers": [{"name": "input_2", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 72], "dtype": "float32", "sparse": false, "name": "input_2"}, "inbound_nodes": []}, {"name": "dense_5", "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"name": "dense_6", "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"name": "dense_7", "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_6", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_7", 0, 0]]}, "keras_version": "2.2.4-tf", "backend": "tensorflow"}'


```python
# 进行结构化
import json
import pprint
pprint.pprint(json.loads(json_string))
```

    {'backend': 'tensorflow',
     'class_name': 'Model',
     'config': {'input_layers': [['input_2', 0, 0]],
                'layers': [{'class_name': 'InputLayer',
                            'config': {'batch_input_shape': [None, 72],
                                       'dtype': 'float32',
                                       'name': 'input_2',
                                       'sparse': False},
                            'inbound_nodes': [],
                            'name': 'input_2'},
                           {'class_name': 'Dense',
                            'config': {'activation': 'relu',
                                       'activity_regularizer': None,
                                       'bias_constraint': None,
                                       'bias_initializer': {'class_name': 'Zeros',
                                                            'config': {}},
                                       'bias_regularizer': None,
                                       'dtype': 'float32',
                                       'kernel_constraint': None,
                                       'kernel_initializer': {'class_name': 'GlorotUniform',
                                                              'config': {'seed': None}},
                                       'kernel_regularizer': None,
                                       'name': 'dense_5',
                                       'trainable': True,
                                       'units': 32,
                                       'use_bias': True},
                            'inbound_nodes': [[['input_2', 0, 0, {}]]],
                            'name': 'dense_5'},
                           {'class_name': 'Dense',
                            'config': {'activation': 'relu',
                                       'activity_regularizer': None,
                                       'bias_constraint': None,
                                       'bias_initializer': {'class_name': 'Zeros',
                                                            'config': {}},
                                       'bias_regularizer': None,
                                       'dtype': 'float32',
                                       'kernel_constraint': None,
                                       'kernel_initializer': {'class_name': 'GlorotUniform',
                                                              'config': {'seed': None}},
                                       'kernel_regularizer': None,
                                       'name': 'dense_6',
                                       'trainable': True,
                                       'units': 16,
                                       'use_bias': True},
                            'inbound_nodes': [[['dense_5', 0, 0, {}]]],
                            'name': 'dense_6'},
                           {'class_name': 'Dense',
                            'config': {'activation': 'softmax',
                                       'activity_regularizer': None,
                                       'bias_constraint': None,
                                       'bias_initializer': {'class_name': 'Zeros',
                                                            'config': {}},
                                       'bias_regularizer': None,
                                       'dtype': 'float32',
                                       'kernel_constraint': None,
                                       'kernel_initializer': {'class_name': 'GlorotUniform',
                                                              'config': {'seed': None}},
                                       'kernel_regularizer': None,
                                       'name': 'dense_7',
                                       'trainable': True,
                                       'units': 10,
                                       'use_bias': True},
                            'inbound_nodes': [[['dense_6', 0, 0, {}]]],
                            'name': 'dense_7'}],
                'name': 'model_1',
                'output_layers': [['dense_7', 0, 0]]},
     'keras_version': '2.2.4-tf'}


从 json 重新创建模型（刚刚初始化）。如果储存为yaml格式，只需要将model.to_json(),tf.keras.models.model_from_json()替换为model.to_yaml()、tf.keras.models.model_from_yaml.


```python
fresh_model = tf.keras.models.model_from_json(json_string)
```

### 5.3整个模型
整个模型可以保存到一个文件中，其中包含权重值、模型配置乃至优化器配置。这样，您就可以对模型设置检查点并稍后从完全相同的状态继续训练，而无需访问原始代码。


```python
#存储为 HDF5 格式文件
model.save('my_model.h5')

# 读入模型数据
model = tf.keras.models.load_model('my_model.h5')
```

### 小结
- 只保存权重： model.save_weights()/model.load_weights
- 只保存配置：model.to_json(),tf.keras.models.model_from_json()/model.toyaml(),tf.keras.models.model_from_yaml()
- 全部保存： model.save('*****.h5')/model = tf.keras.models.load_model('*****h5')
