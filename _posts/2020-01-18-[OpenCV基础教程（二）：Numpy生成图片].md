---
layout: post
title: '[OpenCV基础教程（二）：Numpy生成图片]'
date: 2020-01-18
author: Qizhiguo
cover: 
tags: OpenCV
---

# [OpenCV基础教程（二）：Numpy生成图片]

在某些时候我们需要生成一张照片，比如全黑或者全白，因为经过OpenCV读取到的图片是numpy的数组格式，所以我们可以使用numpy来生成一张我们需要的图片。

## 1.生成黑色图片

根据BGR色彩空间，我们知道黑色对应的值是0，所以我们使用numpy的numpy.zeros(shape=[]),可以快速进行创建，其中shape参数代表着图片的形状以及通道数，如果我们生成三通道的大小为高640，宽480，通道数为3的图片的代码为：

```python
import cv2
import numpy as np

image = np.zeros([640, 480, 3], dtype=np.uint8)
cv2.imshow('picture', image)
print(image.shape)
cv2.waitKey(0)
```

## 2.生成白色图片

在RGB色彩空间中白的编码是[255, 255, 255],所以收纳柜fill()函数进行填充，填充255.

```python
import cv2
import numpy as np

image = np.zeros([640, 480, 3], dtype=np.uint8)
# 使用fill填充255
image.fill(255)
cv2.imshow('white',image)
cv2.waitKey(0)

# 你也可以这样对BGR每个通道进行赋值
image[:, :, 0] = 255 # 蓝色
image[:, :, 1] = 255 # 绿色
image[:, :, 2] = 255 # 红色
# 所以最终三元色合成为白色。
```

##  3.生成任意颜色

我们可以对每个通道进行赋值，所以我们如果知道某个颜色的BGR值尽可以进行赋值，生成我们需要的纯色，后面将会有更简单的通道操作。

