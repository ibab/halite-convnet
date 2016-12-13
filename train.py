import os
import sys
from itertools import cycle

import numpy as np

from keras.models import Model, load_model
from keras.layers import Convolution2D, Deconvolution2D, Input, Reshape, Flatten, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint

REPLAY_FOLDER = 'replays'
WIDTH = 50
HEIGHT = 50

np.random.seed(0) # for reproducibility

def create_model():
    input_batch = Input(shape=(WIDTH, HEIGHT, 4))
    conv1 = Convolution2D(8, 3, 3, border_mode='same', activation='relu')(input_batch)
    conv2 = Convolution2D(8, 3, 3, border_mode='same', activation='relu')(conv1)
    conv3 = Convolution2D(5, 3, 3, border_mode='same')(conv2)
    output = Activation('softmax')(Reshape((WIDTH * HEIGHT, 5))(conv3))
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

