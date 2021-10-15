#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:50:54 2019

@author: ababdoll
"""

import os
import warnings

#os.chdir('/data/ababdoll/data2')
#
#train_images = 'Train/img'
#train_labels = 'Train/y'
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation,Reshape, Layer, Permute, GlobalAveragePooling2D, add
from keras.layers import Conv2D, MaxPool2D, Deconvolution2D,MaxPooling2D,Cropping2D, Convolution2D,merge, UpSampling2D, ZeroPadding2D, Conv2DTranspose
from keras.layers import Input, Lambda
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers.merge import concatenate
import keras.models as models
from keras.initializers import RandomNormal
from keras.layers import LeakyReLU
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.layers.advanced_activations import ELU
from keras.utils.vis_utils import plot_model
import scipy as scipy
#from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


#os.chdir('/data/ababdoll/Abi/Building-extraction/Building')
#
#train_images = 'croped/img'
#train_labels = 'croped/ylabels(png)'
#val_images = 'croped/img1'
#val_labels = 'croped/y1'
#test_images = 'croped/img2'
#test_labels = 'croped/y2'
#
#
#
################################# data generator
#
#def generate_samples(images, labels):
#    image_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
#    mask_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
#
#    image_generator = image_datagen.flow_from_directory(
#    images,
#    class_mode = None,
#    #color_mode = 'rgb',
#    target_size=(768, 768), interpolation = 'nearest',
#    batch_size =1,
#    seed = 1)
#
#    mask_generator = mask_datagen.flow_from_directory(
#    labels,
#    class_mode = None,
#    target_size=(768, 768),interpolation = 'nearest',
#    batch_size = 1,
#    seed = 1)
#    
#    generator = zip(image_generator, mask_generator)
#    return generator
#
#def generate_val(images1, labels1):
#    image_datagen1 = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
#    mask_datagen1 = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
#
#    image_generator1 = image_datagen1.flow_from_directory(
#    images1,
#    class_mode = None,
#    #color_mode = 'rgb',
#    target_size=(768, 768),interpolation = 'nearest',
#    batch_size =1,
#    seed = 1)
#
#    mask_generator1 = mask_datagen1.flow_from_directory(
#    labels1,
#    class_mode = None,
#    target_size=(768, 768),interpolation = 'nearest',
#    batch_size = 1,
#    seed = 1)#, color_mode='grayscale')
#    
#    generator1 = zip(image_generator1, mask_generator1)
#    return generator1
#
#def generate_test_images(images2):
#    image_datagen2 = ImageDataGenerator(rescale=1./255)
#
#    image_generator2 = image_datagen2.flow_from_directory(
#    images2,
#    class_mode = None,
#    #color_mode = 'rgb',
#    target_size=(768, 768),interpolation = 'nearest',
#    batch_size =1,
#    seed = 1)
#    return image_generator2
#
#def generate_test_labels(labels2):
#    image_datagen2 = ImageDataGenerator(rescale=1./255)
#
#    image_generator2 = image_datagen2.flow_from_directory(
#    labels2,
#    class_mode = None,
#    #color_mode = 'rgb',
#    target_size=(768, 768),interpolation = 'nearest',
#    batch_size =1,
#    seed = 1)
#    return image_generator2
##
#generate = generate_samples(train_images, train_labels)
#generate1 = generate_val(val_images, val_labels)
#X_test = generate_test_images(test_images)
#Y_test = generate_test_labels(test_labels)
#next(generate)[0].shape
#count =0
#for img, y in generate:
#    print ('image', img.shape)
#    print('label', y.shape)
#    #print(np.unique(y))
#    count+=1
#    if count>1:
#        break
#    
#for img, y in generate:
#    plt.imshow(img[0])
#    plt.imshow(y[0])
#    break


image_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
mask_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')

x_img = image_datagen.flow_from_directory(
directory = '/data/ababdoll/Abi/Building-extraction/paper3/edge/', classes = ['img'],
class_mode= None,
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1)

mask = mask_datagen.flow_from_directory(
directory = '/data/ababdoll/Abi/Building-extraction/paper3/edge/', classes = ['y'],
class_mode= None,
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1) #color_mode='grayscale')
##
############### val data
image_val_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
mask_val_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')

val_img = image_val_datagen.flow_from_directory(
directory = '/data/ababdoll/Abi/Building-extraction/paper3/edge/', classes = ['img1'],
class_mode= None,
#color_mode = 'rgb',
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1)

val_mask = mask_val_datagen.flow_from_directory(
directory = '/data/ababdoll/Abi/Building-extraction/paper3/edge/', classes = ['y1'],
class_mode= None,
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1) #color_mode='grayscale')

####### test data
image_test_datagen = ImageDataGenerator(rescale=1./255)
mask_test_datagen = ImageDataGenerator(rescale=1./255)

test_img = image_test_datagen.flow_from_directory(
directory = '/data/ababdoll/Abi/Building-extraction/paper3/edge/', classes = ['img2'],
class_mode= None,
#color_mode = 'rgb',
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1, shuffle=False)

test_mask = mask_test_datagen.flow_from_directory(
directory = '/data/ababdoll/Abi/Building-extraction/paper3/edge/', classes = ['y2'],
class_mode= None,
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1, shuffle=False) #color_mode='grayscale')
#
generate = zip(x_img, mask)
generate1 = zip(val_img, val_mask)


#
#
#i =0
#next(generate)[0].shape
#for img,y in generate1:
#    #print('image:', img.shape)
#    #print('label:', y.shape)
#    plt.imshow(np.squeeze(y))
#    i +=1
#    if i>=1:
#        break



####### creat mini batches
#IMG_WIDTH = 10000
#IMG_HEIGHT = 10000
#IMG_CHANNELS = 3
#img_rows = 2048
#img_cols = 2048 
#
#def flip_axis(x, axis):
#    x = np.asarray(x).swapaxes(axis, 0)
#    x = x[::-1, ...]
#    x = x.swapaxes(0, axis)
#    return x
##
#def form_batch(X, Y, batch_size):
#    X_batch = np.zeros((batch_size, img_rows, img_cols, IMG_CHANNELS))
#    Y_batch = np.zeros((batch_size, img_rows, img_cols, 3))
#    
#
#    for i in range(batch_size):
#        #Every batch consists of images from multiple random image
#        #Other way of doing it is to make eatch batch consists of patch from same image
#        random_image_idx = np.random.randint(len(X))
#        x = X[random_image_idx]
#        y = Y[random_image_idx]
#        X_height = x.shape[0]
#        X_width = x.shape[1]
#
#        random_width = random.randint(0, X_width - img_cols - 1)
#        random_height = random.randint(0, X_height - img_rows - 1)
#        
#        Y_batch[i] = np.expand_dims(y[random_height: random_height + img_rows, random_width: random_width + img_cols, :])
#        X_batch[i] = np.array(x[random_height: random_height + img_rows, random_width: random_width + img_cols, : ])
#    return X_batch, Y_batch

#def batch_generator(X, Y, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True, rotation=True):
#    #count = 0
#    while True:
#        #count += 1
#        X_batch, Y_batch = form_batch(X, Y, batch_size)
#
#        for i in range(X_batch.shape[0]):
#            xb = X_batch[i]
#            yb = Y_batch[i]
#
#            if horizontal_flip:
#                if np.random.random() < 0.5:
#                    xb = flip_axis(xb, 1)
#                    yb = flip_axis(yb, 1)
#
#            if vertical_flip:
#                if np.random.random() < 0.5:
#                    xb = flip_axis(xb, 2)
#                    yb = flip_axis(yb, 2)
#
#            if swap_axis:
#                if np.random.random() < 0.5:
#                    xb = xb.swapaxes(0, 1)
#                    yb = yb.swapaxes(0, 1)
#                    
#            if rotation:
#                if np.random.random() < 0.5:
#                    # Random rotation in steps of 45°
#                    rotations = [90, 180, 270]
#                    # We select a rotation degree randomly
#                    rotation_choice = np.random.choice(len(rotations))
#                  # Rotate it using the random value (uses the scipy library)
#                    xb = scipy.ndimage.rotate(xb, rotations[rotation_choice], order=1,
#                                                                 reshape=False, mode='reflect')  
#                    yb = scipy.ndimage.rotate(yb, rotations[rotation_choice], order=1,
#                                                                 reshape=False, mode='reflect')    
#
#            X_batch[i] = xb
#            Y_batch[i] = yb
#        #change this to yield when using as generator
#        return X_batch, Y_batch

########## mini batch for validation
#def flip_axis1(x, axis):
#    x = np.asarray(x).swapaxes(axis, 0)
#    x = x[::-1, ...]
#    x = x.swapaxes(0, axis)
#    return x
##
#def form_batch1(X_val, Y_val, batch_size):
#    Xval_batch = np.zeros((batch_size, img_rows, img_cols, IMG_CHANNELS))
#    Yval_batch = np.zeros((batch_size, img_rows, img_cols, 3))
#    
#
#    for i in range(batch_size):
#        #Every batch consists of images from multiple random image
#        #Other way of doing it is to make eatch batch consists of patch from same image
#        random_image_idx = np.random.randint(len(X_val))
#        xval = X_val[random_image_idx]
#        yval = Y_val[random_image_idx]
#        Xval_height = xval.shape[0]
#        Xval_width = xval.shape[1]
#
#        random_width = random.randint(0, Xval_width - img_cols - 1)
#        random_height = random.randint(0, Xval_height - img_rows - 1)
#        
#        Yval_batch[i] = np.expand_dims(yval[random_height: random_height + img_rows, random_width: random_width + img_cols, :])
#        Xval_batch[i] = np.array(xval[random_height: random_height + img_rows, random_width: random_width + img_cols, : ])
#    return Xval_batch, Yval_batch

#def batch_generator1(X_val, Y_val, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True, rotation=True):
#    #count = 0
#    while True:
#        #count += 1
#        Xval_batch, Yval_batch = form_batch1(X_val, Y_val, batch_size)
#
#        for i in range(Xval_batch.shape[0]):
#            xb = Xval_batch[i]
#            yb = Yval_batch[i]
#
#            if horizontal_flip:
#                if np.random.random() < 0.5:
#                    xb = flip_axis1(xb, 1)
#                    yb = flip_axis1(yb, 1)
#
#            if vertical_flip:
#                if np.random.random() < 0.5:
#                    xb = flip_axis1(xb, 2)
#                    yb = flip_axis1(yb, 2)
#
#            if swap_axis:
#                if np.random.random() < 0.5:
#                    xb = xb.swapaxes(0, 1)
#                    yb = yb.swapaxes(0, 1)
#                    
#            if rotation:
#                if np.random.random() < 0.5:
#                    # Random rotation in steps of 45°
#                    rotations = [90, 180, 270]
#                    # We select a rotation degree randomly
#                    rotation_choice = np.random.choice(len(rotations))
#                  # Rotate it using the random value (uses the scipy library)
#                    xb = scipy.ndimage.rotate(xb, rotations[rotation_choice], order=1,
#                                                                 reshape=False, mode='reflect')  
#                    yb = scipy.ndimage.rotate(yb, rotations[rotation_choice], order=1,
#                                                                 reshape=False, mode='reflect')    
#
#            Xval_batch[i] = xb
#            Yval_batch[i] = yb
#        #change this to yield when using as generator
#        return Xval_batch, Yval_batch
            
########### mini bathc for test
#def form_batch2(X_test, Y_test, batch_size):
#    Xt_batch = np.zeros((batch_size, img_rows, img_cols, IMG_CHANNELS))
#    Yt_batch = np.zeros((batch_size, img_rows, img_cols, 3))
#    
#
#    for i in range(batch_size):
#        #Every batch consists of images from multiple random image
#        #Other way of doing it is to make eatch batch consists of patch from same image
#        random_image_idx = np.random.randint(len(X_test))
#        xt = X_test[random_image_idx]
#        yt = Y_test[random_image_idx]
#        Xt_height = xt.shape[0]
#        Xt_width = xt.shape[1]
#
#        random_width = random.randint(0, Xt_width - img_cols - 1)
#        random_height = random.randint(0, Xt_height - img_rows - 1)
#        
#        Yt_batch[i] = np.array(yt[random_height: random_height + img_rows, random_width: random_width + img_cols, :])
#        Xt_batch[i] = np.array(xt[random_height: random_height + img_rows, random_width: random_width + img_cols, : ])
#    return Xt_batch, Yt_batch
        



#def batch_generator2(X_test, Y_test, batch_size):
#    
#    for i in range(X_test.shape[0]):
#        
#        Xt_batch, Yt_batch = form_batch2(X_test, Y_test, batch_size)
#
#
#        #change this to yield when using as generator
#    return Xt_batch, Yt_batch
#    
#batch_size = len(X)
#batch_size1 = len(X_val)
#batch_size2 = len(X_test)
#
##batch_size1 = len(X_val)
#
#X_batch, Y_batch = batch_generator(X, y, batch_size)
#Xval_batch, Yval_batch = batch_generator1(X_val, y_val, batch_size1)
#Xt_batch, Yt_batch = batch_generator1(X_test, y_test, batch_size1)

#x_val, Y_val = batch_generator1(X_val, y_val, batch_size1) 


############# model 
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet(height, width, n_channels):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''


    inputs = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)

    up6 = concatenate([Conv2DTranspose(
        32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32*8, up6)

    up7 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32*4, up7)

    up8 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32*2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = conv2d_bn(mresblock9, 3, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])

    return model
   




model = MultiResUnet(1536, 1536,3)
model.summary()

model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
 



###################### train model
BS = 1
training_steps = len(x_img)//BS
val_steps = len(val_img)//BS

EPOCHS = 100
H = model.fit_generator(generate,
	validation_data=generate1, steps_per_epoch=training_steps,
    validation_steps=val_steps, epochs=EPOCHS)

#######plot histor
# ### list all data in history
# #print(H.history.keys())
# #accuracy_train=H.history['accuracy']
# ## summarize history for accuracy
# plt.figure(figsize=(7,5))
# #epochs = range(0,100)
# accu_train = H.history['accuracy']
# accu_val = H.history['val_accuracy']
# epochs = range(0,100)
# plt.plot(epochs, accu_train, 'b', label='Training')
# plt.plot(epochs, accu_val, 'r', label='Validation')
# plt.title('Model accuracy', fontsize=16)
# plt.xlabel('Epochs', fontsize=16, labelpad=10)
# plt.ylabel('Accuracy', fontsize=16, labelpad=10)
# plt.legend(fontsize=14, loc='lower right')
# plt.show()

# # summarize history for loss
# plt.figure(figsize=(7,5))
# loss_train = history.history['loss']
# loss_val = history.history['val_loss']
# epochs = range(0,100)
# plt.plot(epochs, loss_train, 'b', label='Training')
# plt.plot(epochs, loss_val, 'r', label='Validation')
# plt.title('Model loss', fontsize=16)
# plt.xlabel('Epochs', fontsize=16, labelpad=10)
# plt.ylabel('Loss', fontsize=16, labelpad=10)
# plt.legend(fontsize=14)
# plt.show()
############## predict test
BS = 1
steps = len(test_img)//BS
preds_test = model.predict_generator(test_img, steps, verbose=1 )




################# plot Y_test image and predicted image
plt.figure(figsize=(10,10))
plt.imshow(np.squeeze(test_mask[0]))
plt.axis('off')

plt.figure(figsize=(10,10))
plt.imshow(np.squeeze(preds_test[0]))
plt.axis('off')
