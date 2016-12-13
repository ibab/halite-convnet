import os
import sys
from itertools import cycle

import numpy as np

from keras.models import Model, load_model
from keras.layers import Convolution2D, Deconvolution2D, Input, Reshape, Flatten, Activation, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint

REPLAY_FOLDER = 'replays'
WIDTH = 40
HEIGHT = 40

np.random.seed(0) # for reproducibility

def create_model():
    input_batch = Input(shape=(WIDTH, HEIGHT, 4))
    preprocessed = Convolution2D(8, 1, 1)(input_batch)
    conv1 = Convolution2D(8, 3, 3, border_mode='same', activation='relu')(preprocessed)
    conv2 = Convolution2D(8, 3, 3, border_mode='same', activation='relu')(conv1)
    conv3 = Convolution2D(8, 3, 3, border_mode='same')(conv2)

    skipped = merge([preprocessed, conv3], mode='sum')

    conv4 = Convolution2D(5, 1, 1)(skipped)

    output = Reshape((WIDTH, HEIGHT, 5))(Activation('softmax')(Reshape((WIDTH * HEIGHT, 5))(conv4)))
    model = Model(input=input_batch, output=output)
    model.compile('nadam', 'categorical_crossentropy', metrics=['accuracy'])

    return model

def load_data():
    for sample in cycle(os.listdir(REPLAY_FOLDER)):
        if sample[-4:] != '.npz':
            continue
        data = np.load(REPLAY_FOLDER + '/' + sample)
        X = data['X']
        y = data['y']
        yield X, y

if __name__ == '__main__':
    model = create_model()

    model.fit_generator(
        load_data(),
        callbacks=[ModelCheckpoint('model.h5', verbose=1, save_best_only=False)],
        samples_per_epoch=20000,
        nb_epoch=10,
        nb_worker=5,
    )

