import os
import tensorflow as tf
# from tensorflow import keras
import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from math import *
import numpy as np
import albumentations as A
import random

#please check your tf2 is within the supported version of the tf2onnx
def val_process(img,label):
    img = tf.cast(img, tf.float32) #disable 255 for checking
    img = tf.squeeze(img)
    label = tf.cast(label,tf.float32)
    return img, label

transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CoarseDropout(p=0.5, max_height=3, max_width=3), #check the dropout region
        A.RandomRotate90(p=0.5), # check it
   
    ])
def aug_fn(image):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]  
    aug_img = tf.cast(aug_img, tf.float32)
    return aug_img
def train_process(image, label):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
    label = tf.cast(label,tf.float32)
    return aug_img, label

# def train_process(image, label):
#     index = tf.where(tf.equal(label,1))
#     if index != 1:
#         aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
#     else:
#         aug_img = tf.cast(image, tf.float32)
#     label = tf.cast(label,tf.float32)
#     return aug_img, label
    
AUG_BATCH =8

def CutMix(image, label, PROBABILITY = 0.3):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    DIM = 256
    imgs = []; labs = []
    for j in range(AUG_BATCH):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.int32)
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0
        WIDTH = tf.cast( DIM * tf.math.sqrt(1-b),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM,x+WIDTH//2)
        # MAKE CUTMIX IMAGE
        one = image[j,ya:yb,0:xa,:]
        two = image[k,ya:yb,xa:xb,:]
        three = image[j,ya:yb,xb:DIM,:]
        middle = tf.concat([one,two,three],axis=1)
        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)
        imgs.append(img)
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)
        lab1 = label[j,]
        lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))
    label2 = tf.stack(labs)
    return image2,label2
def MixUp(image, label, p = 0.5):
    DIM = 256
    imgs = []; labs = []
    for j in range(AUG_BATCH):
        P = tf.cast( tf.random.uniform([],0,1)<=p, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)
        a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1-a)*img1 + a*img2)
        lab1 = label[j,]
        lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
    image2 = tf.reshape(tf.stack(imgs),(len(image),DIM,DIM,3))
    label2 = tf.stack(labs)
    return image2,label2

def transform_Mix(image, label):
    r = random.uniform(0, 1)
    if r < 0.5:
        image2,label2 = CutMix(image, label)
    else:
        image2,label2 = MixUp(image, label)
    return image2,label2
    
def build_dataset(X_train, y_train,X_test, y_test, args):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)) 
    ds_train = ds_train.cache().shuffle(100).repeat()
       
    ds_train = ds_train.map(train_process, num_parallel_calls=AUTOTUNE)
    # ds_train = ds_train.batch(batch_size=args.batch_size, drop_remainder=False)
    # ds_train = ds_train.repeat()
    # ds_train = ds_train.prefetch(AUTOTUNE)  
    
    
    ds_train = ds_train.batch(batch_size=args.batch_size).map(CutMix, num_parallel_calls=AUTOTUNE) # check MixUp
    ds_train = ds_train.unbatch()
    ds_train = ds_train.shuffle(1234)
    ds_train = ds_train.batch(batch_size=args.batch_size, drop_remainder=False)
    # ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(AUTOTUNE)  
    
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    ds_test = ds_test.map(val_process, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.batch(batch_size=args.batch_size, drop_remainder=False)
    ds_test = ds_test.prefetch(AUTOTUNE)  
    return ds_train, ds_test