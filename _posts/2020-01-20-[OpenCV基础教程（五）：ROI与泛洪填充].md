---
layout: post
title: '[OpenCV基础教程（五）：ROI与泛洪填充]'
date: 2020-01-20
author: Qizhiguo
cover: 
tags: OpenCV
---

# [OpenCV基础教程（五）：ROI与泛洪填充]

## 1.ROI

​        ROI是我们感兴趣的区域，比如脸部等，选取ROI后可以对ROI进行分析和下一步操作，把研究的范围进一步缩小，比如当进行人脸识别的时候，我们只对图中的脸部区域感兴趣，所以我们将包含脸部的图片截取出来，因为我们操作的是numpy数组，所以使用数组切片就可以很简单的将脸部截取出来形成ROI。

```python
import cv2
import numpy as np


image = 'D:/1.jpg'
image = cv2.imread(image)
image2 = image/copy()
face = image2[12:92, 198:274, ]
cv2.imshow('face', face)
cv2.waiKey(0)
```

![face](https://raw.githubusercontent.com/Qzgfather/Qzgfather.github.io/master/assets/img/girl_face.jpg)

   所以我们通过切片操作获得了脸部的ROI然后可以进行下一步操作。

## 2.Mask详解

掩膜（mask）

​        在有些图像处理的函数中有的参数里面会有mask参数，即此函数支持掩膜操作，首先何为掩膜以及有什么用，如下：

​        字图像处理中的掩膜的概念是借鉴于PCB制版的过程，在半导体制造中，许多芯片工艺步骤采用光刻技术，用于这些步骤的图形“底片”称为掩膜（也称作“掩模”），其作用是：在硅片上选定的区域中对一个不透明的图形模板遮盖，继而下面的腐蚀或扩散将只影响选定的区域以外的区域。
​        图像掩膜与其类似，用选定的图像、图形或物体，对处理的图像（全部或局部）进行遮挡，来控制图像处理的区域或处理过程。 数字图像处理中,掩模为二维矩阵数组,有时也用多值图像，图像掩模主要用于：

1. 提取感兴趣区,用预先制作的感兴趣区掩模与待处理图像相乘,得到感兴趣区图像,感兴趣区内图像值保持不变,而区外图像值都为0。 
2. 屏蔽作用,用掩模对图像上某些区域作屏蔽,使其不参加处理或不参加处理参数的计算,或仅对屏蔽区作处理或统计。 
3. 结构特征提取,用相似性变量或图像匹配方法检测和提取图像中与掩模相似的结构特征。 特殊形状图像的制作。 

**在所有图像基本运算的操作函数中，凡是带有掩膜（mask）的处理函数，其掩膜都参与运算（输入图像运算完之后再与掩膜图像或矩阵运算）。**[参考](https://blog.csdn.net/gavinmiaoc/article/details/80856246)

## 3.泛洪填充

​        泛洪填充类似于画图中的油漆桶，对一定的区域进行填充，使用的算法是泛洪算法，关于泛洪算法的细节请参考：https://blog.csdn.net/lion19930924/article/details/54293661。

在OpenCV中使用floodFill（）函数即可完成泛洪填充。

```python
'''floodFill(image, mask, seedPoint, newVal, loDiff=None, upDiff=None, flags=None)'''
## 1.image，是传入的图片
## 2.mask为掩码，如果想对整幅图片进行处理，在构建mask的时候，mask不能与原图片一样大，要加2.
## 3.seedPoint直译为随机点，格式为（x,y），代表要处理点的坐标，所以是两个数字。
## 4.newVal表示要填充新的数值，（B,G,R）格式。
## 5.loDiff表示最低值，即填充时的最低像素值，格式（B,G,R）,在使用的时候，是（x，y）点的三通道像素值减去
##   loDiff。
## 6.upDiff表示最高值，即填充时的最高像素值，格式（B,G,R）,在使用的时候，是（x，y）点的三通道像素值加上
##   upDiff
## 7.flags代表处理方法，是否结合mask进行填充flag：(1)当为CV_FLOODFILL_FIXED_RANGE时，待处理的像素点 ##  与种子点作比较，在范围之内，则填充此像素 。（改变图像）------(2)CV_FLOODFILL_MASK_ONLY 此位设置填 ##  充的对像， 若设置此位，则mask不能为空，此时，函数不填充原始图像image，而是填充掩码图像。 mask的指定的 ##  位置为零时才填充，不为零不填充。

import cv2
import numpy as np


def fill_color_demo(image):
    copyimage = image.copy()
    h, w = image.shape[:2]
    # floodFill，配合cv2.FLOODFILL_FIXED_RANGE,mask不为一的时候填充
    mask = np.ones([h + 2, w + 2], np.uint8)
    cv2.floodFill(copyimage, 
                  mask, (150, 220), 
                  (0, 255, 255), 
                  (80, 80, 80), 
                  (10, 10, 10), 
                  cv2.FLOODFILL_FIXED_RANGE)
    cv2.imshow('demo', copyimage)


src = cv2.imread('./lenna.jpg')
cv2.imshow('images', src)
fill_color_demo(src)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

我们设置了一个点（150， 220）大概在肩部的位置，我们进行填充得到以下的效果。

![foodfill](https://raw.githubusercontent.com/Qzgfather/Qzgfather.github.io/master/assets/img/floodfill.jpg)

**结合mask的应用**

```python
# 填充新建的mask
import cv2
import numpy as np


def fill_binary():
    image = np.zeros([400, 400, 3], np.uint8)
    image[100: 300, 100: 300] = 255
    cv2.imshow('fill_biary', image)

    mask = np.ones([402, 402, 1], np.uint8)
    mask[101: 301, 101: 301] = 0
    # floodFill()填充， 把，mask初始化为1，想要填充的地方设置成0
    cv2.floodFill(image, mask, (200, 200), (0, 0, 255), cv2.FLOODFILL_MASK_ONLY)
    cv2.imshow('fill binary', image)


fill_binary()
cv2.waitKey(0)
cv2.destroyAllWindows()

```

<img src="https://raw.githubusercontent.com/Qzgfather/Qzgfather.github.io/master/assets/img/mask.jpg" alt="mask" style="zoom:50%;" />

经过填充后变成：

<img src="https://raw.githubusercontent.com/Qzgfather/Qzgfather.github.io/master/assets/img/fill_mask.jpg" style="zoom:50%;" />

)