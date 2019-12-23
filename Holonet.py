# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:46:04 2019

@author: Yuqi Li
"""
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import time
import scipy.io
import h5py
import hdf5storage
import keras.backend as K
import tensorflow as tf
from numpy import random
import math
from tensorflow.examples.tutorials.mnist import input_data
from keras import losses 
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, Multiply,  Conv2D, ConvLSTM2D, Conv3D,Activation, BatchNormalization, AveragePooling2D, Add, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Concatenate,  Lambda, Reshape, Conv2DTranspose, Flatten, Dense

from tensorflow.python.keras.layers import Lambda
from keras.layers import GaussianNoise
from keras import regularizers
from keras.constraints import NonNeg, UnitNorm
#from mylayer import MyLayer
from PIL import Image
import collections as cl
import numpy as np


class ElapsedTimer(object):
	def __init__(self):
		self.start_time = time.time()
	def elapsed(self,sec):
		if sec < 60:
			return str(sec) + " sec"
		elif sec < (60 * 60):
			return str(sec / 60) + " min"
		else:
			return str(sec / (60 * 60)) + " hr"
	def elapsed_time(self):
		print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class HOLONET():
	def __init__(self, img_rows = 128, img_cols = 128, img_ranks = 128):

		self.img_rows = img_rows
		self.img_cols = img_cols
		self.img_ranks = img_ranks
	
	
	def residual_block(self,layer_input, filters):
		"""Residual block described in paper"""
		#d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
		
		d = layer_input

		d_real = tf.keras.layers.Lambda(lambda x: tf.math.real(x))(d)
		d_imag = tf.keras.layers.Lambda(lambda x: tf.math.imag(x))(d)

		d_real = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32))(d_real)
		d_imag = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32))(d_imag)	
		
		d_real = tf.keras.layers.Lambda(lambda x: tf.tile(x,[1,1,3,3]))(d_real)
		d_imag = tf.keras.layers.Lambda(lambda x: tf.tile(x,[1,1,3,3]))(d_imag)

		d_real = tf.keras.layers.Lambda(lambda x: x[:,:,118:266,118:266])(d_real)
		d_imag = tf.keras.layers.Lambda(lambda x: x[:,:,118:266,118:266])(d_imag)
				
		ConvReal = Conv2D(filters, kernel_size=5, strides=1, padding='valid', data_format='channels_first',kernel_initializer='zeros')
		ConvImag = Conv2D(filters, kernel_size=5, strides=1, padding='valid', data_format='channels_first',kernel_initializer='zeros')
		
		d_real1 = tf.keras.layers.Lambda(lambda x: tf.math.subtract(x[0],x[1]))([ConvReal(d_real),ConvImag(d_imag)])
		d_imag1 = tf.keras.layers.Lambda(lambda x: tf.math.add(x[0],x[1]))([ConvReal(d_imag),ConvImag(d_real)])
		d_real = d_real1
		d_imag = d_imag1

		for i in range(4):
			d_real = Activation('relu')(d_real)
			d_imag = Activation('relu')(d_imag)
			
			ConvReal = Conv2D(filters, kernel_size=5, strides=1, padding='valid', data_format='channels_first',kernel_initializer='zeros')
			ConvImag = Conv2D(filters, kernel_size=5, strides=1, padding='valid', data_format='channels_first',kernel_initializer='zeros')
		
			d_real1 = tf.keras.layers.Lambda(lambda x: tf.math.subtract(x[0],x[1]))([ConvReal(d_real),ConvImag(d_imag)])
			d_imag1 = tf.keras.layers.Lambda(lambda x: tf.math.add(x[0],x[1]))([ConvReal(d_imag),ConvImag(d_real)])
			
			d_real = d_real1
			d_imag = d_imag1
		
		d = tf.keras.layers.Lambda(lambda x: tf.dtypes.complex(x[0],x[1]))([d_real,d_imag])
		d = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.complex64))(d)
		d = Add()([d, layer_input])
		return d

		
	def get_holonet(self, train_bn=True): 
		
		# INPUTS
		inputsdata = tf.keras.layers.Input(shape=(1, 128, 128), dtype=tf.complex64)

		inputskernel = tf.keras.layers.Input(shape=(128, 128, 128), dtype=tf.complex64) 
		inputsfftkernel = tf.keras.layers.Input(shape=(128, 128, 128), dtype=tf.complex64) 
		
		def SquareFFTfunc(input_img, kernel):
			fft2data = tf.keras.layers.Lambda(lambda x:tf.fft2d(x))(input_img)
			temp_mul = tf.keras.layers.Lambda(lambda x:tf.math.multiply(x[0],x[1]))([fft2data,kernel])
			temp_mul = tf.keras.layers.Lambda(lambda x:tf.ifft2d(x))(temp_mul)
			sumval = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=-3))(temp_mul)
			conj_sumval = tf.keras.layers.Lambda(lambda x: tf.math.conj(x))(sumval)
			return tf.keras.layers.Lambda(lambda x:tf.math.multiply(x[0],x[1]))([sumval, conj_sumval])
			
		def FFTfunc(input_img, kernel):
			fft2data = tf.keras.layers.Lambda(lambda x:tf.fft2d(x))(input_img)
			temp_mul = tf.keras.layers.Lambda(lambda x:tf.math.multiply(x[0],x[1]))([fft2data,kernel])
			temp_mul = tf.keras.layers.Lambda(lambda x:tf.ifft2d(x))(temp_mul)
			return tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=-3))(temp_mul)
			
			
		def scalelayer(inputimg):
			img_real = tf.keras.layers.Lambda(lambda x: tf.math.real(x))(inputimg)
			img_imag = tf.keras.layers.Lambda(lambda x: tf.math.imag(x))(inputimg)

			img_real = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32))(img_real)
			img_imag = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32))(img_imag)		
			scale = tf.keras.layers.Conv3D(1,1,use_bias= False,kernel_constraint=NonNeg())
			expand_inputreal = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x,-1))(img_real)
			expand_inputimag = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x,-1))(img_imag)
			temp_real = scale(expand_inputreal)
			temp_imag = scale(expand_inputimag)
			temp = tf.keras.layers.Lambda(lambda x: tf.complex(x[0], x[1]))([temp_real,temp_imag])
			temp = tf.keras.layers.Lambda(lambda x: tf.squeeze(x,-1))(temp)
			return tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.complex64))(temp)
        
		def denoiseblock(x, isnonlocal= False):
			temp = self.residual_block(x, 128)
			temp = self.residual_block(temp, 128)
			return temp #Add()([temp,x])
			
		def stage(x,v,I,kernel,ifftkernel):
			squarefftres = SquareFFTfunc(x,kernel)
			fftres = FFTfunc(x,kernel)
			diff = tf.keras.layers.Lambda(lambda x:tf.math.subtract(x[0],x[1]))([squarefftres,I])
			diff = scalelayer(diff)
			diff = tf.keras.layers.Lambda(lambda x:tf.math.multiply(x[0],x[1]))([diff,fftres])
			diff = tf.keras.layers.Lambda(lambda x:tf.tile(x, (1,128,1,1)))(diff)
			
			fft2data = tf.keras.layers.Lambda(lambda x:tf.fft2d(x))(diff)
			kernel = tf.keras.layers.Lambda(lambda x:tf.fft2d(x))(ifftkernel)
			temp_mul = tf.keras.layers.Lambda(lambda x:tf.math.multiply(x[0],x[1]))([fft2data,kernel])
			temp_mul = tf.keras.layers.Lambda(lambda x:tf.ifft2d(x))(temp_mul)

			diff_xv = tf.keras.layers.Lambda(lambda x:tf.math.subtract(x[0],x[1]))([v,x])
			diff_xv = scalelayer(diff_xv)
			
			x_next = tf.keras.layers.Lambda(lambda x:tf.math.add(x[0],x[1]))([x,diff_xv])
			x_next = tf.keras.layers.Lambda(lambda x:tf.math.add(x[0],x[1]))([x_next,temp_mul])
			v_next = denoiseblock(x_next)

			return x_next, v_next

		
		tempx = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(inputskernel)

		tempv = tempx
		
		for i in range(6):
			tempx, tempv = stage(tempx,tempv, inputsdata, inputskernel, inputsfftkernel)


        
		tempv = tf.keras.layers.Lambda(lambda x: tf.cast(x,'float32'))(tempv)
		tempv = Activation('sigmoid')(tempv)

#		print(tempv.dtype)	
		
		def mse_error(y_true,y_pred): 

			temp = tf.math.square(tf.math.abs(y_true-y_pred))
			return tf.math.reduce_sum(tf.cast(temp,tf.float32))

		def PSNR(y_true, y_pred):
			return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
		
		def SSIM(y_true, y_pred):
#			y_true = tf.transpose(y_true, [0, 2, 3, 1])
#			y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
			return tf.image.ssim(y_true, y_pred, 1)
		# Setup the model inputs / outputs
		model = Model(inputs=[inputsdata, inputskernel, inputsfftkernel], outputs=tempv)

		#model.add_loss(loss_total(inputs_mask,inputs,outputs))

		model.compile(
			optimizer = Adam(lr=0.0001),
			loss= mse_error
#			metrics=[PSNR,SSIM]
		)
		model.summary()

		
		return model

   
	def load_data(self,testtime):

		if testtime == True:
			trainX = []
			testX = np.load("H:\holography\GaborHolography-3d\GaborHolography-3d\code\data\sim_train_data\\"+ str(i+1)+".mat")
		else:
			testX = []
			trainX = np.empty((1000,128,128,128))
			for i in range(1000):
				mat = h5py.File("H:\holography\GaborHolography-3d\GaborHolography-3d\code\data\sim_train_data\\"+ str(i+1)+".mat")
				temp_trainX = np.transpose(mat['label']).astype(np.float32)
				trainX[i,:,:,:] = temp_trainX
				trainX = np.transpose(trainX,(0,3,1,2))
#				trainX = np.concatenate((trainX,np.swapaxes(trainX,1,2)))
#				trainX = trainX[:,:160,:244,:]
		return trainX,  testX



	def load_kernel(self):
		mat = hdf5storage.loadmat("H:\holography\otf3d.mat")
		otf3d = mat['otf3d']
		otf3d = np.transpose(otf3d,(2,0,1))
		otf3d = np.expand_dims(otf3d,axis=0)
		
		mat = hdf5storage.loadmat("H:\holography\GaborHolography-3d\GaborHolography-3d\code\DL\ifftkernel_flip.mat")
		ifftkernel = mat['ifftkernel_flip']
		ifftkernel = np.transpose(ifftkernel,(2,0,1))
		ifftkernel = np.expand_dims(ifftkernel,axis=0)
		return otf3d, ifftkernel
	
	def train(self):

		print("loading data")
		testtime = False
		imgs_train,  imgs_test = self.load_data(testtime)
		print("loading data done")
#		print(imgs_train.shape[0])
#		num_sample = imgs_train.shape[0]

#		model.load_weights('gradient_fastdeeplower32.hdf5')
		if testtime:
			model = self.get_holonet()
		else:
			model = self.get_holonet()
			model.load_weights('gradient_fastdeeplower32.hdf5')
			model_checkpoint = ModelCheckpoint('gradient_fastdeeplower32.hdf5', monitor='loss',verbose=1, save_best_only=True)
#			mat = h5py.File("H:\holography\otf3d.mat")
			otf3d,  ifftkernel = self.load_kernel()

			responseimg = np.empty((1000, 1, 128,128),dtype = 'float32')
			for i in range(1000):
				fft_img = np.fft.fft2(imgs_train[i,:,:,:])
				temp_multiply = np.multiply(fft_img,otf3d) 
				resimg = np.fft.ifft2(temp_multiply)
				sumresimg  = np.sum(resimg,axis=-3)
				responseimg[i,0,:,:] = np.multiply(sumresimg, np.conj(sumresimg)).astype('float32')

	
#			print(responseimg.dtype)
			history = model.fit([responseimg,np.tile(otf3d,(1000,1,1,1)),np.tile(ifftkernel,(1000,1,1,1))], imgs_train, batch_size=1, epochs= 20 ,verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

			img_predict = model.predict([responseimg.astype('float32'),np.tile(otf3d,(1000,1,1,1)),np.tile(ifftkernel,(1000,1,1,1))],batch_size=1, verbose=1)

#			scipy.io.savemat('temp.mat',{'fft_img':fft_img,'resimg':resimg,'temp_multiply':temp_multiply,'imgs_train':imgs_train,'otf3d':otf3d,'sumresimg':sumresimg})
			
			scipy.io.savemat('predict.mat',{'predict':img_predict})
			scipy.io.savemat('gt.mat',{'gt':imgs_train})
		

if __name__ == '__main__':
	m_holonet = HOLONET()
	timer = ElapsedTimer()
	m_holonet.train()
	timer.elapsed_time()




