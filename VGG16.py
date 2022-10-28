import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses

def getModel():
    (x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
    x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
    x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255
    x_train = tf.expand_dims(x_train, axis=3, name=None)
    x_test = tf.expand_dims(x_test, axis=3, name=None)
    x_train = tf.repeat(x_train, 3, axis=3)
    x_test = tf.repeat(x_test, 3, axis=3)
    x_val = x_train[-2000:,:,:,:]
    y_val = y_train[-2000:]
    x_train = x_train[:-2000,:,:,:]
    y_train = y_train[:-2000]

    model = models.Sequential()

    model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:]))

    model.add(layers.Conv2D(64, 3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(64, 3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, strides=2))

    model.add(layers.Conv2D(128, 3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(128, 3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, strides=2))

    model.add(layers.Conv2D(256, 3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, 3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, 1, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, strides=2))

    model.add(layers.Conv2D(512, 3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(512, 3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(512, 1, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, strides=2))

    model.add(layers.Conv2D(512, 3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(512, 3, strides=1, padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(512, 1, strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10, activation='softmax'))
    model.summary()

getModel()