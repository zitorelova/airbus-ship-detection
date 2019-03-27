# Airbus Ship Detection 

This is the source code for my solution to the [Airbus Ship Detection challenge](https://www.kaggle.com/c/airbus-ship-detection).

### Overview

The challenge is an [image segmentation](https://en.wikipedia.org/wiki/Image_segmentation) problem that ran on the Kaggle platform from August to November 2018. It was hosted by Airbus, a European aerospace corporation that designs, manufactures, and sells civil and military aerospace products worldwide. In this challenge, competitors are tasked to create models that detect ships in satellite images as quickly as possible. With the growth of ship traffic, there is also an increase in the chances of infractions at sea. The work done in this competition is able to address problems such as environmentally devastating ship accidents, piracy, illegal fishing, drug trafficking, and illegal cargo movement.

### Solution

The final solution consists of two main parts, a classifier for detecting the presence of ships in images and a mask predictor for ship localization. Since a majority of the images do not have ships in them, they first pass through the classifier where the presence of ships is predicted. Images that are predicted as having ships in them are then passed through the mask predictor where a mask is returned for that specific image. Images predicted as not having ships are given an empty mask.

### Setup 

After cloning the repository, place the competition data into the **data** folder. Run **classifier.py** to train the first ship classifier. Finally run **segment.py** to train the mask predictor.

### References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
- [The Lovasz-Sofmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790)
- [Hypercolumns for Object Segmentation and Fine-grained Localization](https://arxiv.org/abs/1411.5752)
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- [Effective Use of Dilated Convolutions for Segmenting Small Object Instances in Remote Sensing Imagery](https://arxiv.org/abs/1709.00179)
- [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
