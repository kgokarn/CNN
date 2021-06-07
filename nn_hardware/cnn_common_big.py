#!/usr/bin/env python3

import keras
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import Callback
from keras.utils.generic_utils import get_custom_objects
import numpy as np
from keras import regularizers
import tensorflow as tf


def load_data():
    num_classes = 10
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train_temp = np.zeros((60000, 29, 29))
    x_test_temp = np.zeros((10000, 29, 29))
    for i in range(60000):
        x_train_temp[i, :28, :28] = x_train[i]
    for j in range(10000):
        x_test_temp[j, :28, :28] = x_test[j]
    x_train = x_train_temp
    x_test = x_test_temp

    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    return (x_train, y_train, x_test, y_test)


def pwl_activation(x):

    #x = tf.math.divide(x, 16)

    cond1 = tf.cast(tf.math.less(x, -2), tf.float32)
    cond2 = tf.cast(tf.math.logical_and(tf.math.greater_equal(x, -2), tf.math.less(x, -0.6875)), tf.float32)
    cond3 = tf.cast(tf.math.logical_and(tf.math.greater_equal(x, -0.6875), tf.math.less(x, 0.6875)), tf.float32)
    cond4 = tf.cast(tf.math.logical_and(tf.math.greater_equal(x, 0.6875), tf.math.less(x, 2)), tf.float32)
    cond5 = tf.cast(tf.math.greater_equal(x, 2), tf.float32)

    a = tf.math.multiply(cond1, -1.0)
    b = tf.math.multiply(cond2, tf.math.multiply((x - 2), 0.25))
    c = tf.math.multiply(cond3, x)
    d = tf.math.multiply(cond4, tf.math.multiply((x + 2), 0.25))
    e = tf.math.multiply(cond5, 1.0)

    f = a + b + c + d + e
    return f


get_custom_objects().update({'pwl_activation': Activation(pwl_activation)})


def first_layer_activation(x):
    return K.tanh(0.75 * x)

get_custom_objects().update({'first_layer_activation': Activation(first_layer_activation)})


def make_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(20, kernel_size=(4, 4),
                     input_shape=input_shape,
                     use_bias=False,
                     #kernel_regularizer=regularizers.l1(0.00007)
                     ))

    #model.add(Activation('tanh'))
    #model.add(Activation('first_layer_activation'))
    model.add(Activation('pwl_activation'))
    #model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(60, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(60, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(120, kernel_size=(3, 3), activation='relu'))
    #model.add(Conv2D(120, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten(data_format='channels_last'))
    model.add(Dropout(rate=0.5))
    #model.add(Dense(800))
    #model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adadelta(lr=1),
                  metrics=['accuracy'])

    return model


class WeightScale(Callback):

    def on_batch_end(self, batch, logs=None):
        weights_conv1 = self.model.get_layer(index=0).get_weights()
        #weights_conv2 = self.model.get_layer(index=3).get_weights()
        #weights_fc1 = self.model.get_layer(index=7).get_weights()
        #weights_fc2 = self.model.get_layer(index=8).get_weights()
        maximum = 0
        for w in weights_conv1:
            maximum = max(np.max(np.absolute(w)), maximum)
        #for w in weights_conv2:
        #    maximum = max(np.max(np.absolute(w)), maximum)
        #for w in weights_fc1:
        #    maximum = max(np.max(np.absolute(w)), maximum)
        #for w in weights_fc2:
        #    maximum = max(np.max(np.absolute(w)), maximum)
        self.model.get_layer(index=0).set_weights([w/maximum for w in weights_conv1])
        #self.model.get_layer(index=3).set_weights([w/maximum for w in weights_conv2])
        #self.model.get_layer(index=7).set_weights([w / maximum for w in weights_fc1])
        #self.model.get_layer(index=8).set_weights([w / maximum for w in weights_fc2])

    #def on_epoch_end(self, epoch, logs=None):
        #weights_conv1 = self.model.get_layer(index=0).get_weights()
        #weights_conv2 = self.model.get_layer(index=3).get_weights()

        #self.model.get_layer(index=0).set_weights([w + np.random.normal(0, 1/64, size=w.shape) for w in weights_conv1])
        #self.model.get_layer(index=3).set_weights([w + np.random.normal(0, 1 / 32, size=w.shape) for w in weights_conv2])

