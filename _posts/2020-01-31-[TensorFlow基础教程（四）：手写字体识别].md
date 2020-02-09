# 1.MNIST数据集介绍
MNIST 数据集来自美国国家标准与技术研究所, National Institute of Standards and Technology (NIST). 训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员. 测试集(test set) 也是同样比例的手写数字数据.

这一节我将介绍不同的实现方法，以后遇到的不同的代码风格可以进行分辨。


```python
import tensorflow as tf
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```


```python
x_train = x_train.reshape(60000,784) / 225
print(x_train.shape)
print(y_train.shape)
```

    (60000, 784)
    (60000,)



```python
# 直接堆叠方式1
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(784,activation="relu"))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

model.compile(
    optimizer= keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

x_val = x_train[-10000:]
y_val = y_train[-10000:]

model.fit(x_train,y_train,batch_size=64,epochs=10, validation_data=(x_val, y_val))

model.summary()
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/10
    60000/60000 [==============================] - 25s 409us/sample - loss: 0.2003 - sparse_categorical_accuracy: 0.9391 - val_loss: 0.1028 - val_sparse_categorical_accuracy: 0.9672
    Epoch 2/10
    60000/60000 [==============================] - 23s 382us/sample - loss: 0.0840 - sparse_categorical_accuracy: 0.9744 - val_loss: 0.0582 - val_sparse_categorical_accuracy: 0.9832
    Epoch 3/10
    60000/60000 [==============================] - 23s 383us/sample - loss: 0.0580 - sparse_categorical_accuracy: 0.9823 - val_loss: 0.0399 - val_sparse_categorical_accuracy: 0.9878
    Epoch 4/10
    60000/60000 [==============================] - 23s 385us/sample - loss: 0.0437 - sparse_categorical_accuracy: 0.9871 - val_loss: 0.0230 - val_sparse_categorical_accuracy: 0.9927ss: 0.0439 - sparse_categorical_accura
    Epoch 5/10
    60000/60000 [==============================] - 23s 379us/sample - loss: 0.0352 - sparse_categorical_accuracy: 0.9898 - val_loss: 0.0167 - val_sparse_categorical_accuracy: 0.9952
    Epoch 6/10
    60000/60000 [==============================] - 24s 395us/sample - loss: 0.0264 - sparse_categorical_accuracy: 0.9919 - val_loss: 0.0135 - val_sparse_categorical_accuracy: 0.9954
    Epoch 7/10
    60000/60000 [==============================] - 23s 383us/sample - loss: 0.0220 - sparse_categorical_accuracy: 0.9934 - val_loss: 0.0231 - val_sparse_categorical_accuracy: 0.9939
    Epoch 8/10
    60000/60000 [==============================] - 23s 383us/sample - loss: 0.0189 - sparse_categorical_accuracy: 0.9947 - val_loss: 0.0121 - val_sparse_categorical_accuracy: 0.9966
    Epoch 9/10
    60000/60000 [==============================] - 25s 418us/sample - loss: 0.0149 - sparse_categorical_accuracy: 0.9956 - val_loss: 0.0144 - val_sparse_categorical_accuracy: 0.9960
    Epoch 10/10
    60000/60000 [==============================] - 25s 415us/sample - loss: 0.0135 - sparse_categorical_accuracy: 0.9961 - val_loss: 0.0104 - val_sparse_categorical_accuracy: 0.9972
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                multiple                  615440    
    _________________________________________________________________
    dense_1 (Dense)              multiple                  50240     
    _________________________________________________________________
    dense_2 (Dense)              multiple                  650       
    =================================================================
    Total params: 666,330
    Trainable params: 666,330
    Non-trainable params: 0
    _________________________________________________________________



```python
# 直接堆叠方式2
model = tf.keras.Sequential(
    [
    tf.keras.layers.Dense(784,activation="relu"),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
    ]
)

model.compile(
    optimizer= keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

x_val = x_train[-10000:]
y_val = y_train[-10000:]

model.fit(x_train,y_train,batch_size=64,epochs=3)

model.summary()
```

    Train on 60000 samples
    Epoch 1/3
    60000/60000 [==============================] - 24s 398us/sample - loss: 0.2034 - sparse_categorical_accuracy: 0.9374 - loss: 0.2495 - sparse_categorical_accuracy: 0. - E - ETA: 5s - loss: 0.2254  - ETA: 2s - loss: 0.2143 - sparse_categorical_accu - ETA: 1s - loss: 0.2114 - spar
    Epoch 2/3
    60000/60000 [==============================] - 23s 382us/sample - loss: 0.0836 - sparse_categorical_accuracy: 0.9748
    Epoch 3/3
    60000/60000 [==============================] - 22s 372us/sample - loss: 0.0582 - sparse_categorical_accuracy: 0.9823
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_3 (Dense)              multiple                  615440    
    _________________________________________________________________
    dense_4 (Dense)              multiple                  50240     
    _________________________________________________________________
    dense_5 (Dense)              multiple                  650       
    =================================================================
    Total params: 666,330
    Trainable params: 666,330
    Non-trainable params: 0
    _________________________________________________________________



```python
# 函数式API
inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = keras.layers.Dense(64, activation='relu')(inputs)
h1 = keras.layers.Dense(64, activation='relu')(h1)
outputs = keras.layers.Dense(10, activation='softmax')(h1)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer= keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

x_val = x_train[-10000:]
y_val = y_train[-10000:]

model.fit(x_train,y_train,batch_size=64,epochs=10)

model.summary()
```

    Train on 60000 samples
    Epoch 1/10
    60000/60000 [==============================] - 5s 82us/sample - loss: 0.3105 - sparse_categorical_accuracy: 0.9109
    Epoch 2/10
    60000/60000 [==============================] - 4s 70us/sample - loss: 0.1393 - sparse_categorical_accuracy: 0.9575
    Epoch 3/10
    60000/60000 [==============================] - 4s 72us/sample - loss: 0.1029 - sparse_categorical_accuracy: 0.9685
    Epoch 4/10
    60000/60000 [==============================] - 4s 70us/sample - loss: 0.0827 - sparse_categorical_accuracy: 0.9753
    Epoch 5/10
    60000/60000 [==============================] - 4s 69us/sample - loss: 0.0697 - sparse_categorical_accuracy: 0.9794
    Epoch 6/10
    60000/60000 [==============================] - 4s 66us/sample - loss: 0.0606 - sparse_categorical_accuracy: 0.9819
    Epoch 7/10
    60000/60000 [==============================] - 4s 68us/sample - loss: 0.0529 - sparse_categorical_accuracy: 0.9845
    Epoch 8/10
    60000/60000 [==============================] - 4s 69us/sample - loss: 0.0478 - sparse_categorical_accuracy: 0.9865
    Epoch 9/10
    60000/60000 [==============================] - 4s 68us/sample - loss: 0.0418 - sparse_categorical_accuracy: 0.9878
    Epoch 10/10
    60000/60000 [==============================] - 4s 67us/sample - loss: 0.0378 - sparse_categorical_accuracy: 0.9887
    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    mnist_input (InputLayer)     [(None, 784)]             0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 64)                50240     
    _________________________________________________________________
    dense_9 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_10 (Dense)             (None, 10)                650       
    =================================================================
    Total params: 55,050
    Trainable params: 55,050
    Non-trainable params: 0
    _________________________________________________________________



```python
# 高级自定义模型
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = keras.layers.Dense(784, activation='relu')
        self.d1 = keras.layers.Dense(64, activation='relu')
        self.d2 = keras.layers.Dense(10, activation='softmax')
    def call(self, x):
        x = self.dense(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()
model.compile(
    optimizer= keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

model.fit(x_train,y_train,batch_size=64,epochs=10)

model.summary()
```

    Train on 60000 samples
    Epoch 1/10
    60000/60000 [==============================] - 23s 385us/sample - loss: 0.2151 - sparse_categorical_accuracy: 0.9346
    Epoch 2/10
    60000/60000 [==============================] - 23s 379us/sample - loss: 0.0847 - sparse_categorical_accuracy: 0.9747
    Epoch 3/10
    60000/60000 [==============================] - 23s 378us/sample - loss: 0.0577 - sparse_categorical_accuracy: 0.9824 - loss: 0.0580 - sparse_categorical_accuracy: 0.
    Epoch 4/10
    60000/60000 [==============================] - 22s 372us/sample - loss: 0.0442 - sparse_categorical_accuracy: 0.9869
    Epoch 5/10
    60000/60000 [==============================] - 23s 379us/sample - loss: 0.0344 - sparse_categorical_accuracy: 0.9900
    Epoch 6/10
    60000/60000 [==============================] - 23s 386us/sample - loss: 0.0272 - sparse_categorical_accuracy: 0.9921 - loss:
    Epoch 7/10
    60000/60000 [==============================] - 22s 374us/sample - loss: 0.0231 - sparse_categorical_accuracy: 0.9934
    Epoch 8/10
    60000/60000 [==============================] - 23s 385us/sample - loss: 0.0178 - sparse_categorical_accuracy: 0.9952 - loss: 
    Epoch 9/10
    60000/60000 [==============================] - 23s 375us/sample - loss: 0.0168 - sparse_categorical_accuracy: 0.9952
    Epoch 10/10
    60000/60000 [==============================] - 22s 372us/sample - loss: 0.0131 - sparse_categorical_accuracy: 0.9959
    Model: "my_model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_11 (Dense)             multiple                  615440    
    _________________________________________________________________
    dense_12 (Dense)             multiple                  50240     
    _________________________________________________________________
    dense_13 (Dense)             multiple                  650       
    =================================================================
    Total params: 666,330
    Trainable params: 666,330
    Non-trainable params: 0
    _________________________________________________________________



```python
   
```
