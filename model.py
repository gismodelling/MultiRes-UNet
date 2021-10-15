
import os
import warnings
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



################################# data generator

image_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, rotation_range=270, fill_mode='nearest')
mask_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, rotation_range=270, fill_mode='nearest')

x_img = image_datagen.flow_from_directory(
directory = '/data/.../', classes = ['img'],
class_mode= None,
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1)

mask = mask_datagen.flow_from_directory(
directory = '/data/.../', classes = ['y'],
class_mode= None,
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1) 
##
############### val data
image_val_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, rotation_range=270, fill_mode='nearest')
mask_val_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, rotation_range=270, fill_mode='nearest')

val_img = image_val_datagen.flow_from_directory(
directory = '/data/.../', classes = ['img1'],
class_mode= None,
#color_mode = 'rgb',
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1)

val_mask = mask_val_datagen.flow_from_directory(
directory = '/data/.../', classes = ['y1'],
class_mode= None,
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1)

####### test data
image_test_datagen = ImageDataGenerator(rescale=1./255)
mask_test_datagen = ImageDataGenerator(rescale=1./255)

test_img = image_test_datagen.flow_from_directory(
directory = '/data/.../', classes = ['img2'],
class_mode= None,
#color_mode = 'rgb',
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1, shuffle=False)

test_mask = mask_test_datagen.flow_from_directory(
directory = '/data/.../', classes = ['y2'],
class_mode= None,
target_size=(1536, 1536),
batch_size = 1, interpolation = 'bilinear', seed=1, shuffle=False) 
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




############# model 
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):


    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def MultiResBlock(U, inp, alpha = 1.67):
 
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


def MultiRes-UNet(height, width, n_channels):
 
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
   




model = MultiRes-UNet(1536, 1536,3)
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
