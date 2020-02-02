---
layout: post
title: 'Paper Note 1 <SSD: Single Shot MultiBox Detector>'
date: 2020-01-20
author: Qizhiguo
cover: 
tags: PaperNote
---

# Paper Note 1：<SSD: Single Shot MultiBox Detector>

## 1.Notable drawbacks of existing technology


hypothesize bounding boxes

resample pixels or features for each box

apply a high-quality classifier.

## 2.The advantages or contributions of Our model

### **advantages：**

 eliminating bounding box proposals 

does not resample pixels or features for bounding box hypotheses

using separate predictors (filters) for different aspect ratio detections

detection at multiple scales.

### **contributions**：

- We introduce SSD, a single-shot detector for multiple categories that is **faster than**
  **the previous state-of-the-art for single shot detectors (YOLO)**, and significantly
  more accurate, in fact as accurate as slower techniques that perform explicit region
  proposals and pooling (including Faster R-CNN).
- The core of SSD is predicting category scores and box offsets for a fixed set of
  default bounding boxes **using small convolutional filters applied to feature maps.**
- **To achieve high detection accuracy we produce predictions of different scales from**
  **feature maps of different scales, and explicitly separate predictions by aspect ratio.**
- These design features lead to simple end-to-end training and high accuracy, even
  on low resolution input images, further improving the speed vs accuracy trade-off.
- Experiments include timing and accuracy analysis on models with varying input
  size evaluated on PASCAL VOC, COCO, and ILSVRC and are compared to a
  range of recent state-of-the-art approaches.

## 3.Structure of Fast RCNN

### 

![image-20200201202132544](https://raw.githubusercontent.com/Qzgfather/Qzgfather.github.io/master/assets/img/ssd-net.png)



### 

## 4.Training details

4.1Choosing scales and aspect ratios for default boxes

the number of default box:

38 * 38 *4 +19 * 19 *6 +10 * 10 * 10 *6 + 5 * 5 * 6  + 3 * 3  * 4+ 1* 1 * 4 = 8732

the function of default box:
$$
s_{k}=s_{\min }+\frac{s_{\max }-s_{\min }}{m-1}(k-1), \quad k \in[1, m]
$$
sk = [0.2,0.34,0.48,0.62,0.76,0.9]

a_r = {1,2,3,1/2,1/3} 

weight：
$$
w_{k}^{a}=s_{k} \sqrt{a_{r}}
$$
height：
$$
h_{k}^{a}=s_{k} / \sqrt{a_{r}}
$$
4.2Matching strategy

threshold is 0.5

4.3Loss function

The overall objective loss function is a weighted sum of the localization loss (loc) and the confidence loss (conf):
$$
L(x, c, l, g)=\frac{1}{N}\left(L_{\operatorname{con} f}(x, c)+\alpha L_{\operatorname{loc}}(x, l, g)\right)
$$
localization loss:
$$
\begin{aligned}
L_{l o c}(x, l, g)=& \sum_{i \in P o s} \sum_{m \in\{c x, c y, w, h\}} x_{i j}^{k} \operatorname{smooth}_{L 1}\left(l_{i}^{m}-\hat{g}_{j}^{m}\right) \\
\hat{g}_{j}^{c x}=\left(g_{j}^{c x}-d_{i}^{c x}\right) / d_{i}^{w} & \hat{g}_{j}^{c y}=\left(g_{j}^{c y}-d_{i}^{c y}\right) / d_{i}^{h} \\
\hat{g}_{j}^{w}=\log \left(\frac{g_{j}^{w}}{d_{i}^{w}}\right) & \hat{g}_{j}^{h}=\log \left(\frac{g_{j}^{h}}{d_{i}^{h}}\right)
\end{aligned}
$$
The confidence loss is the softmax loss over multiple classes confidences (c):
$$
L_{c o n f}(x, c)=-\sum_{i \in P o s}^{N} x_{i j}^{p} \log \left(\hat{c}_{i}^{p}\right)-\sum_{i \in N e g} \log \left(\hat{c}_{i}^{0}\right) \quad \text { where } \quad \hat{c}_{i}^{p}=\frac{\exp \left(c_{i}^{p}\right)}{\sum_{p} \exp \left(c_{i}^{p}\right)}
$$
4.4Hard negative mining

we sort them using the highest confidence loss for each default box and pick the top ones so that the ratio **between the negatives and positives is at most 3:1.** We found that this leads to faster optimization and a more stable training.

4.5Data augmentation

- Use the entire original input image.
- Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3,
  0.5, 0.7, or 0.9.
- Randomly sample a patch.

## 5.Other Information(e.g. mAP on COCO or Voc)

![image-20200201203502407](https://raw.githubusercontent.com/Qzgfather/Qzgfather.github.io/master/assets/img/ssd-map.png)