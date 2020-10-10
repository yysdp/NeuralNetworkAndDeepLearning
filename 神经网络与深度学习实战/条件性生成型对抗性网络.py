from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
import math
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.python.keras.layers import Activation

#from google.colab import drive

#drive.mount('/content/gdrive')

# %%

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import RMSprop
import os

# build_and_train_models()
#
# # %%
#
from tensorflow.keras.models import load_model
import os
#
# generator = load_model('/content/gdrive/My Drive/dcgan_mnist.h5')
# # 构造一批随机初始化的一维向量让生成者网络创造图片
# noise = np.random.randint(-1.0, 1.0, size=[16, 100])
# plot_images(generator, noise_input=noise,
#             show=True, model_name="test_image")
#
#
#
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import to_categorical


def build_cgan_discriminator(inputs, y_labels, image_size):

    '''
    识别图片，并将图片与输入的one-hot-vector关联起来
    '''
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]
    x = inputs
    print(x.shape,y_labels.shape,y_labels[0])
    y = Dense(image_size * image_size)(y_labels)
    print(x.shape, y.shape,y[0])
    y = Reshape((image_size, image_size, 1))(y)
    print(x.shape, y.shape)
    # 把图片数据与one-hot-vector拼接起来,这里是唯一与前面代码不同之处
    x = concatenate([x, y])
    print(x.shape, y.shape)

    for filters in layer_filters:
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model([inputs, y_labels], x,
                          name='discriminator')
    print(discriminator.summary())
    return discriminator


# %%

def build_cgan_generator(inputs, y_labels, image_size):
    '''
    生成者网络在构造图片时，需要将输入向量与对应的one-hot-vector结合在一起考虑
    '''
    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]
    # 将输入向量与One-hot-vector结合在一起
    x = concatenate([inputs, y_labels], axis=1)
    x = Dense(image_resize * image_resize * layer_filters[0])(x)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)
    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)
    x = Activation('sigmoid')(x)
    generator = Model([inputs, y_labels], x, name='generator')
    print(generator.summary())
    return generator


def train_cgan(models, data, params):
    generator, discriminator, adversarial = models
    # 获取图片数据以及图片对应数字的one-hot-vector
    x_train, y_train = data
    batch_size, latent_size, train_steps, num_labels, model_name = params
    save_interval = 500
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    '''
    np.eye产生对角矩阵,例如np.eye(3) = [[1,0,0], [0,1,0], [0,0,1]],
    于是np.eye(3)[2, 3, 1] = [[0,1,0], [0,0,1], [1,0,0]]
    '''
    noise_class = np.eye(num_labels)[np.arange(0, 16) % num_labels]
    train_size = x_train.shape[0]
    print(model_name, "Labels for generated images: ", np.argmax(noise_class, 1))
    for i in range(train_steps):
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]
        # 增加图片对应的one-hot-vector
        real_labels = y_train[rand_indexes]

        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        # 增加构造图片对应的one-hot-vector
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]
        fake_images = generator.predict([noise, fake_labels])
        # 把真实图片和虚假图片连接起来
        x = np.concatenate((real_images, fake_images))
        # 将真实图片对应的one-hot-vecotr和虚假图片对应的One-hot-vector连接起来
        y_labels = np.concatenate((real_labels, fake_labels))

        y = np.ones([2 * batch_size, 1])
        # 上半部分图片为真，下半部分图片为假
        y[batch_size:, :] = 0.0
        # 先训练识别者网络，这里需要将图片及对应的one-hot-vector输入
        loss, acc = discriminator.train_on_batch([x, y_labels], y)
        log = "%d: [discriminator loss : %f, acc: %f]" % (i, loss, acc)
        '''
        冻结识别者网络，构造随机一维向量以及指定数字的one-hot-vector输入生成者
        网络进行训练
        '''
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]
        y = np.ones([batch_size, 1])
        loss, acc = adversarial.train_on_batch([noise, fake_labels], y)
        log = "%s [adversarial loss :%f, acc: %f]" % (log, loss, acc)
        if (i + 1) % save_interval == 0:
            print(log)
            if (i + 1) == train_steps:
                show = True
            else:
                show = False
            plot_images_cgan(generator,
                             noise_input=noise_input,
                             noise_class=noise_class,
                             show=show,
                             step=(i + 1),
                             model_name=model_name)
            generator.save(model_name + ".h5")


# %%

def build_and_train_models_cgan():
    (x_train, y_train), (_, _) = mnist.load_data()
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    # 获得要生成数字的最大值
    num_labels = np.amax(y_train) + 1
    # 转换为one-hot-vector
    y_train = to_categorical(y_train)

    model_name = "/content/gdrive/My Drive/cgan_mnist"
    latent_size = 100
    batch_size = 64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size, 1)
    label_shape = (num_labels,)
    inputs = Input(shape=input_shape, name='discriminator_input')
    labels = Input(shape=label_shape, name='class_labels')
    # 构建识别者网络时要传入图片对应的One-hot-vector
    discriminator = build_cgan_discriminator(inputs, labels, image_size)
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    input_shape = (latent_size,)
    inputs = Input(shape=input_shape, name='z_input')
    # 构造生成者时也要传入one-hot-vector
    generator = build_cgan_generator(inputs, labels, image_size)
    return
    optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)
    # 将生成者和识别者连接起来时要冻结识别者
    discriminator.trainable = False
    outputs = discriminator([generator([inputs, labels]), labels])
    adversarial = Model([inputs, labels], outputs, name=model_name)
    adversarial.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train_cgan(models, data, params)


# %%

def plot_images_cgan(generator,
                     noise_input,
                     noise_class,
                     show=False,
                     step=0,
                     model_name=''):
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict([noise_input, noise_class])
    print(model_name, "labels for generated images: ", np.argmax(noise_class,
                                                                 axis=1))
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')

    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')

from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
import math
from tensorflow.keras.layers import Activation, Dense, Input

build_and_train_models_cgan()