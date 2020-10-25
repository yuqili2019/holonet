# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:46:04 2019
@author: Yuqi Li
"""
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import time
import scipy.io
import h5py
import hdf5storage
import keras.backend as K
import tensorflow as tf
from numpy import random
import math
from keras import losses
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, Multiply, Conv2D, ConvLSTM2D, Conv3D, Activation, \
    BatchNormalization, AveragePooling2D, Add, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, \
    Concatenate, Lambda, Reshape, Conv2DTranspose, Flatten, Dense
from tensorflow.python.keras.layers import Lambda
from keras.layers import GaussianNoise
from keras import regularizers
from keras.constraints import NonNeg, UnitNorm
from mylayer2 import MyLayer
from PIL import Image
import collections as cl
import numpy as np

num_sample = 500
num_test = 100
Nz = 5
Nxy = 64
iter_num = 20


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))


class HOLONET():
    def __init__(self, img_rows=Nxy, img_cols=Nxy, img_ranks=Nz):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_ranks = img_ranks

    def residual_block(self, layer_input, filters):  # denoising network
        """Denoiser Residual Block, here  """
        # complex
        d = layer_input
        d_real = tf.keras.layers.Lambda(lambda x: tf.math.real(x))(d)
        d_imag = tf.keras.layers.Lambda(lambda x: tf.math.imag(x))(d)

        d_real = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 3, 3]))(d_real)
        d_imag = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 3, 3]))(d_imag)

        d_real = tf.keras.layers.Lambda(
            lambda x: x[:, :, Nxy - 4:2 * Nxy + 4, Nxy - 4:2 * Nxy + 4])(d_real)
        d_imag = tf.keras.layers.Lambda(
            lambda x: x[:, :, Nxy - 4:2 * Nxy + 4, Nxy - 4:2 * Nxy + 4])(d_imag)

        Conv1 = Conv2D(filters, kernel_size=3, strides=1, activation='relu', use_bias=True,
                       padding='valid', data_format='channels_first', kernel_initializer='zeros')
        Conv2 = Conv2D(filters, kernel_size=3, strides=1, activation='relu', use_bias=True,
                       padding='valid', data_format='channels_first', kernel_initializer='zeros')

        d_real = Conv2(Conv1(d_real))
        d_imag = Conv2(Conv1(d_imag))

        d_real_sym = d_real
        d_imag_sym = d_imag

        d_abs = tf.keras.layers.Lambda(lambda x: tf.math.abs(tf.complex(x[0], x[1])))(
            [d_real, d_imag])
        minus_d = MyLayer((Nz, Nxy + 4, Nxy + 4))(d_abs)

        minus_d = Activation('relu')(minus_d)

        d_scale = tf.keras.layers.Lambda(lambda x: tf.sign(x))(minus_d)

        d_real = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([d_scale, d_real])
        d_imag = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))([d_scale, d_imag])

        Conv3 = Conv2D(filters, kernel_size=3, strides=1, activation='relu', use_bias=True,
                       padding='valid', data_format='channels_first', kernel_initializer='zeros')
        Conv4 = Conv2D(filters, kernel_size=3, strides=1, use_bias=True, padding='valid',
                       data_format='channels_first', kernel_initializer='zeros')

        d_real = Conv4(Conv3(d_real))
        d_imag = Conv4(Conv3(d_imag))

        d_real_sym = Conv4(Conv3(d_real_sym))
        d_imag_sym = Conv4(Conv3(d_imag_sym))

        d_real_sym_diff = tf.keras.layers.Lambda(lambda x: tf.abs(tf.math.subtract(x[0], x[1])))(
            [d_real_sym, tf.keras.layers.Lambda(lambda x: tf.math.real(x))(d)])
        d_imag_sym_diff = tf.keras.layers.Lambda(lambda x: tf.abs(tf.math.subtract(x[0], x[1])))(
            [d_imag_sym, tf.keras.layers.Lambda(lambda x: tf.math.imag(x))(d)])

        d_sym_diff = tf.keras.layers.Lambda(lambda x: tf.math.add(x[0], x[1]))(
            [d_real_sym_diff, d_imag_sym_diff])

        d = tf.keras.layers.Lambda(lambda x: tf.complex(x[0], x[1]))([d_real, d_imag])

        d = Add()([d, layer_input])
        return d, d_sym_diff

    def get_holonet(self, train_bn=True):

        # INPUTS
        inputsdata = tf.keras.layers.Input(shape=(Nz, Nxy, Nxy), dtype=tf.float32)

        inputskernel = tf.keras.layers.Input(shape=(Nz, Nxy, Nxy), dtype=tf.complex64)

        pinhole = tf.keras.layers.Input(shape=(Nz, Nxy, Nxy), dtype=tf.float32)

        def fftshift2d_tf(a_tensor):
            input_shape = a_tensor.shape.as_list()
            numel = len(input_shape)
            new_tensor = a_tensor
            for axis in range(numel - 2, numel):
                split = (input_shape[axis] + 1) // 2
                mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
                new_tensor = tf.gather(new_tensor, mylist, axis=axis)
            return new_tensor

        def ifftshift2d_tf(a_tensor):
            input_shape = a_tensor.shape.as_list()
            numel = len(input_shape)

            new_tensor = a_tensor
            for axis in range(numel - 2, numel):
                n = input_shape[axis]
                split = n - (n + 1) // 2
                mylist = np.concatenate((np.arange(split, n), np.arange(split)))
                new_tensor = tf.gather(new_tensor, mylist, axis=axis)
            return new_tensor

        def A_func(input_img, kernel, pinhole, type):

            realimg = input_img
            input_img = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.complex64))(input_img)
            kernel = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.complex64))(kernel)
            pinhole = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.complex64))(pinhole)

            ft_pinhole = tf.keras.layers.Lambda(
                lambda x: ifftshift2d_tf(tf.fft2d(fftshift2d_tf(x))))(pinhole)

            conj_kernel = tf.keras.layers.Lambda(lambda x: tf.math.conj(x))(kernel)
            point_field = tf.keras.layers.Lambda(lambda x: tf.math.multiply(x[0], x[1]))(
                [conj_kernel, ft_pinhole])
            plane_wave = tf.keras.layers.Lambda(
                lambda x: ifftshift2d_tf(tf.ifft2d(fftshift2d_tf(x))))(point_field)

            vol_field = tf.keras.layers.Lambda(lambda x: tf.math.multiply(x[0], x[1]))(
                [input_img, plane_wave])
            vol_field_ft = tf.keras.layers.Lambda(
                lambda x: ifftshift2d_tf(tf.fft2d(fftshift2d_tf(x))))(vol_field)

            plane_field_ft = tf.keras.layers.Lambda(lambda x: tf.math.multiply(x[0], x[1]))(
                [vol_field_ft, kernel])
            plane_field_ft = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-3))(
                plane_field_ft)

            field2d = tf.keras.layers.Lambda(lambda x: ifftshift2d_tf(tf.ifft2d(fftshift2d_tf(x))))(
                plane_field_ft)

            if type == 0:
                return tf.keras.layers.Lambda(lambda x: 2 * tf.math.real(x))(field2d)
            else:
                return Add()([tf.keras.layers.Lambda(lambda x: 2 * tf.math.real(x))(field2d),
                              tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x * x, axis=-3))(
                                  realimg)])

        def At_func(input_img, kernel, pinhole):
            input_img = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.complex64))(input_img)
            kernel = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.complex64))(kernel)
            pinhole = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.complex64))(pinhole)

            conj_input_img = tf.keras.layers.Lambda(lambda x: tf.math.conj(x))(input_img)
            conj_input_img = tf.keras.layers.Lambda(
                lambda x: ifftshift2d_tf(tf.ifft2d(fftshift2d_tf(x))))(conj_input_img)
            plane_field_ft = tf.keras.layers.Lambda(lambda x: tf.math.conj(x))(conj_input_img)

            plane_field_ft = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(plane_field_ft)
            plane_field_ft = tf.keras.layers.Lambda(lambda x: tf.tile(x, (1, Nz, 1, 1)))(
                plane_field_ft)

            conj_kernel = tf.keras.layers.Lambda(lambda x: tf.math.conj(x))(kernel)
            vol_field_ft = tf.keras.layers.Lambda(lambda x: tf.math.multiply(x[0], x[1]))(
                [conj_kernel, plane_field_ft])

            ft_pinhole = tf.keras.layers.Lambda(
                lambda x: ifftshift2d_tf(tf.fft2d(fftshift2d_tf(x))))(pinhole)
            point_field = tf.keras.layers.Lambda(lambda x: tf.math.multiply(x[0], x[1]))(
                [conj_kernel, ft_pinhole])

            point_field_ft = tf.keras.layers.Lambda(
                lambda x: ifftshift2d_tf(tf.ifft2d(fftshift2d_tf(x))))(point_field)

            volume_field = tf.keras.layers.Lambda(
                lambda x: ifftshift2d_tf(tf.fft2d(fftshift2d_tf(tf.math.conj(x)))))(vol_field_ft)

            conj_point_field_ft = tf.keras.layers.Lambda(lambda x: tf.math.conj(x))(point_field_ft)
            conj_volume_field = tf.keras.layers.Lambda(lambda x: tf.math.conj(x))(volume_field)

            field3d = tf.keras.layers.Lambda(lambda x: tf.math.multiply(x[0], x[1]))(
                [conj_point_field_ft, conj_volume_field])

            return field3d


                        
        def scalelayer(inputimg):
            # complex
            img_real = tf.keras.layers.Lambda(lambda x: tf.real(x))(inputimg)
            img_imag = tf.keras.layers.Lambda(lambda x: tf.imag(x))(inputimg)
            # scale = tf.keras.layers.DepthwiseConv2D(1, use_bias=False, kernel_initializer= 'zeros', data_format='channels_first')
            # temp_real = scale(img_real)
            # temp_imag = scale(img_imag)
            
            up_inputimg_real = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(img_real)
            up_inputimg_imag = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(img_imag)
            scale = tf.keras.layers.Conv3D(1, 1, use_bias=False,  kernel_initializer= 'zeros', data_format='channels_last')
            scaled_inputimg_real = scale(up_inputimg_real)
            scaled_inputimg_imag = scale(up_inputimg_imag)
            temp_real = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, -1))(scaled_inputimg_real)
            temp_imag = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, -1))(scaled_inputimg_imag)

            return tf.keras.layers.Lambda(lambda x: tf.complex(x[0], x[1]))([temp_real, temp_imag])

        def denoiseblock(x, isnonlocal=False):
            temp, symloss1 = self.residual_block(x, Nz)
            #			temp,symloss2 = self.residual_block(temp, Nz)
            return temp, symloss1  # tf.keras.layers.Lambda(lambda x: tf.add(x[0],x[1]))([

        # symloss1,symloss2])

        def stage(prex, x, v, I, kernel, pinhole):
            squarefftres = A_func(x, kernel, pinhole, 0)

            diff = tf.keras.layers.Lambda(lambda x: tf.math.subtract(x[0], x[1]))([squarefftres, I])

            diff = At_func(diff, kernel, pinhole)

            diff = scalelayer(diff)

            diff_xv = tf.keras.layers.Lambda(lambda x: tf.math.subtract(x[0], x[1]))([v, x])
            diff_xv = scalelayer(diff_xv)

            diff_xx = tf.keras.layers.Lambda(lambda x: tf.math.subtract(x[0], x[1]))([x, prex])
            diff_xx = scalelayer(diff_xx)

            x_next = tf.keras.layers.Lambda(lambda x: tf.math.add(x[0], x[1]))([x, diff_xv])
            x_next = tf.keras.layers.Lambda(lambda x: tf.math.add(x[0], x[1]))([x_next, diff_xx])
            x_next = tf.keras.layers.Lambda(lambda x: tf.math.subtract(x[0], x[1]))([x_next, diff])

            v_next, symloss = denoiseblock(x_next)

            return x_next, v_next, symloss

        # calculate response images
        response = A_func(inputsdata, inputskernel, pinhole, 1)

        # get initial guess
        tempx = At_func(response, inputskernel, pinhole)

        tempv = tempx
        prex = tempx
        lastx = tempx

        # start reconstruction
        tempx, tempv, tempsymloss = stage(prex, lastx, tempv, response, inputskernel, pinhole)
        prex = lastx
        lastx = tempx
        symloss = tempsymloss

        for i in range(iter_num - 1):
            tempx, tempv, tempsymloss = stage(prex, lastx, tempv, response, inputskernel, pinhole)
            prex = lastx
            lastx = tempx
            symloss = tf.keras.layers.Lambda(lambda x: tf.add(x[0], x[1]))([symloss, tempsymloss])

        tempv_real = tf.keras.layers.Lambda(lambda x: tf.math.real(x))(tempv)
        tempv_imag = tf.keras.layers.Lambda(lambda x: tf.math.imag(x))(tempv)

        # tempv_real = Activation('sigmoid')(tempv_real)#Activation('relu')(tempv_real)#
        # tempv_imag = Activation('sigmoid')(tempv_imag)#Activation('relu')(tempv_real)#

        tempv_real = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0, 1))(tempv_real)
        tempv_imag = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0, 1))(tempv_imag)
        # reconstructed results -- tempv
        tempv = tf.keras.layers.Lambda(lambda x: tf.complex(x[0], x[1]))([tempv_real, tempv_imag])

        def Myloss(symloss):
            def m_loss(y_true, y_pred):
                tempreal = tf.math.abs(tf.real(y_true) - tf.real(y_pred))
                tempimag = tf.math.abs(tf.imag(y_true) - tf.imag(y_pred))
                tempdiff = tf.math.reduce_mean(tempreal) + tf.math.reduce_mean(
                    tempimag) + tf.math.reduce_mean(symloss) / 1000 / iter_num
                return tempdiff

            return m_loss

        # Setup the model inputs / outputs
        model = Model(inputs=[inputsdata, inputskernel, pinhole], outputs=tempv)

        model.compile(
            optimizer=Adam(lr=0.0001),
            loss=Myloss(symloss),
            metrics=['accuracy']
        )
        model.summary()
        return model

    def load_data(self, testtime):

        if testtime == True:
            trainX = []
            testX = np.empty((num_test, Nxy, Nxy, Nz))
            for i in range(num_sample + 0, num_sample + num_test):
                mat = h5py.File("./sim_train_data/" + str(i + 1) + ".mat")
                temp_testX = np.transpose(mat['label']).astype(np.float32)
                testX[i - num_sample, :, :, :] = temp_testX
            testX = np.transpose(testX, (0, 3, 1, 2))
            print(np.max(testX))
            print(np.min(testX))

        else:
            testX = []
            trainX = np.empty((num_sample, Nxy, Nxy, Nz))
            for i in range(num_sample):
                mat = h5py.File("./sim_train_data/" + str(i + 1) + ".mat")
                temp_trainX = np.transpose(mat['label']).astype(np.float32)
                trainX[i, :, :, :] = temp_trainX
            trainX = np.transpose(trainX, (0, 3, 1, 2))

            print(np.max(trainX))
            print(np.min(trainX))
        return trainX, testX

    def load_kernel(self):
        mat = hdf5storage.loadmat("./sim_train_data/otf3d.mat")
        otf3d = mat['otf3d']
        otf3d = np.transpose(otf3d, (2, 0, 1))
        otf3d = np.expand_dims(otf3d, axis=0)

        ifftotf3d = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(otf3d, axes=(2, 3))), axes=(2, 3))
        ifftkernel = np.flip(ifftotf3d, (2, 3))
        return otf3d, ifftkernel

    def train(self):

        print("loading data")
        testtime = False
        imgs_train, imgs_test = self.load_data(testtime)
        print("loading data done")
        otf3d, ifftkernel = self.load_kernel()
        pinhole = np.ones((1, Nz, Nxy, Nxy))

        model = self.get_holonet()

        model.load_weights('holonet.hdf5')
        model_checkpoint = ModelCheckpoint('holonet.hdf5',
										   monitor='loss',
										   verbose=1,
                                           save_best_only=True)

        print(imgs_train.shape)
        history = model.fit([imgs_train, np.tile(otf3d, (num_sample, 1, 1, 1)),
                             np.tile(pinhole, (num_sample, 1, 1, 1))],
							imgs_train,
							batch_size=32,
                            epochs=500,
							verbose=1,
							validation_split=0.2,
							shuffle=True,
                            callbacks=[model_checkpoint])

        img_predict = model.predict([imgs_train, np.tile(otf3d, (num_sample, 1, 1, 1)),
                                     np.tile(pinhole, (num_sample, 1, 1, 1))],
									batch_size=1,
                                    verbose=1)

        print(np.sum(np.sum(img_predict)))
        print(np.sum(np.sum(imgs_train)))
        print(np.sum(np.max(img_predict)))
        print(np.sum(np.max(imgs_train)))

        scipy.io.savemat('predict.mat', {'predict': img_predict})
        scipy.io.savemat('gt.mat', {'gt': imgs_train})


if __name__ == '__main__':
    m_holonet = HOLONET()
    timer = ElapsedTimer()
    m_holonet.train()
    timer.elapsed_time()
