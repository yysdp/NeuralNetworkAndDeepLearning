from tensorflow.keras.datasets import cifar10

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

import numpy as np

def rgb2gray(rgb):
  #把彩色图转化为灰度图，如果当前像素点为[r,g,b],那么对应的灰度点为0.299*r+0.587*g+0.114*b
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

(x_train, _),(x_test, _) = cifar10.load_data()

img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]

#将100张彩色原图集合在一起显示
imgs = x_test[: 100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])

plt.figure()
plt.axis('off')
plt.title('Original color images')
plt.imshow(imgs, interpolation = 'none')
plt.show()

#将图片灰度化后显示出来
x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)
imgs = x_test_gray[: 100]
imgs = imgs.reshape((10, 10, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('gray images')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.show()

#将彩色图片和灰度图正规化,也就是把像素点值设置到[0,1]之间
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_gray = x_train_gray.astype('float32') / 255
x_test_gray = x_test_gray.astype('float32') / 255

'''
将二维图片集合压扁为一维向量[num *row * col * 3],
num 是图片数量
'''
print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
print(x_train.shape)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
print(x_train_gray.shape)
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols,
                                   1)
print(x_train_gray.shape)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)
batch_size = 32
kernel_size = 3
# 由于图片编码后需要保持图片物体与颜色信息，因此编码后的一维向量维度要变大
latent_dim = 256
layer_filters = [64, 128, 256]

inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for filters in layer_filters:
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=2,
               activation='relu', padding='same')(x)

