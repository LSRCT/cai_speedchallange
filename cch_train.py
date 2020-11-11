import sys, os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
import time
import datetime
import preprocessing as pp
import utils


def speed_model_3(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    x = tf.keras.layers.Conv2D(16, 3, strides=1, padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.AveragePooling2D((3,3))(x)
    x1 = x

    x = tf.keras.layers.Conv2D(16, 3, padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x2 = x

    x = tf.keras.layers.Add()([x1, x2])

    x = tf.keras.layers.Conv2D(16, 3, padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x3 = x

    x = tf.keras.layers.Add()([x3, x2])

    x = tf.keras.layers.AveragePooling2D((3,3))(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="linear")(x)
    return tf.keras.Model(inputs, out)


def std_scale(x):
    x =  (x - np.mean(x)) / (np.std(x))
    return x

model = speed_model_3((150, 640, 3))
opt = tf.keras.optimizers.Adam(learning_rate=4e-4)
model.compile(optimizer=opt,loss='MSE')
print(model.summary())
pathx_t = "data//dof_train.avi"
pathy_t = "data//trainY.csv"
pathx_e = "data//dof_test.avi"
pathy_e = "data//testY.csv"
mini_batch_size = 100

## normalize data
dataset = utils.DoFDataset(pathx_t, pathy_t)
dataset = dataset.batch(5, drop_remainder=True)
dat_avg = pp.calcAVG_generator(dataset, verbose=1)

dataset = utils.DoFDataset(pathx_t, pathy_t)
dataset = dataset.batch(5, drop_remainder=True)
data_std = pp.calcSTDDEV_generator(dataset, dat_avg, verbose=1)

dataset = utils.DoFDataset(pathx_t, pathy_t, preprocess_func=pp.preprocess(data_std, dat_avg))

dataset = dataset.batch(mini_batch_size, drop_remainder=True)
dataset = dataset.prefetch(3)
dataset = dataset.repeat()

dataset_test = utils.DoFDataset(pathx_e, pathy_e, preprocess_func=pp.preprocess(data_std, dat_avg))

dataset_test = dataset_test.batch(mini_batch_size, drop_remainder=True)
dataset_test = dataset_test.prefetch(3)
dataset_test = dataset_test.repeat()

print(dataset)
log_dir="logs/fit/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
no_of_frames = 14000
val_frames = 6000
history = model.fit(dataset, workers=12, use_multiprocessing=True, epochs=100, 
                    steps_per_epoch=int(no_of_frames/mini_batch_size), validation_data=(dataset_test),
                    validation_steps=int(val_frames/mini_batch_size),
                    callbacks=[tensorboard_callback])

print(history.history.keys())
savedir = os.getcwd()+"//models//resnet_100_3"
plt.plot(history.history["loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch")

model.save(savedir+"//clust_model.h5")
plt.savefig(savedir+"//lossplot.png")


