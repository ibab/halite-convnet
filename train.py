import os
import sys
from itertools import cycle
import h5py

import numpy as np

from keras.models import Model, load_model
from keras.layers import Convolution2D, Deconvolution2D, Input, Reshape, Flatten, Activation, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Total width and height of the wrapped area used
# as input to the convolutional network.
WIDTH = 50
HEIGHT = 50
# How many frames to take into account in each batch.
BATCH_SIZE = 256
# Fraction of data sample used for validation.
VALIDATION_SPLIT = 0.3
# How many previous frames to use as input.
LOOKBACK = 0

# For reproducibility.
np.random.seed(0)


def gated_unit(x):
    '''A single layer of the convolutional network
    using a gated activation unit.'''
    c = Convolution2D(8, 3, 3, border_mode='same')(x)
    s = Activation('sigmoid')(Convolution2D(8, 1, 1)(c))
    t = Activation('tanh')(Convolution2D(8, 1, 1)(c))
    m = merge([s, t], mode='mul')
    residual = Convolution2D(8, 1, 1, activation='relu')(m)
    skip = Convolution2D(8, 1, 1, activation='relu')(m)

    return residual, skip


def create_model():
    '''Returns the complete Keras model.'''
    input_batch = Input(shape=(WIDTH, HEIGHT, 4 + 3 * LOOKBACK))
    x = Convolution2D(8, 1, 1, activation='relu')(input_batch)

    skipped = []

    for i in range(8):
        x, skip = gated_unit(x)
        skipped.append(skip)

    out1 = merge(skipped, mode='sum')
    out2 = Convolution2D(8, 1, 1)(out1)
    out3 = Convolution2D(5, 1, 1)(out2)
    output = Reshape((WIDTH, HEIGHT, 5))(Activation('softmax')(Reshape((WIDTH * HEIGHT, 5))(out3)))
    model = Model(input=input_batch, output=output)
    model.compile('nadam', 'categorical_crossentropy', metrics=['accuracy'])

    return model


def prepare_data(group):
    '''Preprocess replay data so that it can be used
    as input and target of the network.'''

    # Copy data from file and transform
    player = group['player'][:]
    strength = group['strength'][:] / 255
    production = group['production'][:] / 20
    moves = group['moves'][:]

    n_frames = len(player)

    # Find the winner (the player with most territory at the end)
    players, counts = np.unique(player[-1], return_counts=True)
    winner_id = players[counts.argmax()]
    if winner_id == 0:
        return None

    # Broadcast production array to each time frame
    production = np.repeat(production[np.newaxis], n_frames, axis=0)
    production = production[:,:,:,np.newaxis]

    is_winner = player == winner_id
    is_loser = (player != winner_id) & (player != 0)

    batch = np.array([is_winner, is_loser, strength])
    batch = np.transpose(batch, (1, 2, 3, 0))

    lookback = []
    for i in range(1, LOOKBACK + 1):
        back = np.pad(batch[:-i], ((i, 0), (0, 0), (0, 0), (0, 0)), mode='edge')
        lookback.append(back)

    batch = np.concatenate([batch] + lookback + [production], axis=3)

    # One-hot encode the moves
    moves = np.eye(5)[np.array(moves)]

    nb, nx, ny, nc = np.shape(batch)
    if nx > WIDTH or ny > HEIGHT:
        # We don't want to work with maps larger than this
        return None

    pad_x = int((WIDTH - nx) / 2)
    extra_x = int(WIDTH - nx - 2 * pad_x)
    pad_y = int((HEIGHT - ny) / 2)
    extra_y = int(HEIGHT - ny - 2 * pad_y)

    batch = np.pad(batch, ((0, 0), (pad_x, pad_x + extra_x), (pad_y, pad_y + extra_y), (0, 0)), 'wrap')
    moves = np.pad(moves, ((0, 0), (pad_x, pad_x + extra_x), (pad_y, pad_y + extra_y), (0, 0)), 'wrap')

    # Only moves for the winning player have to be predicted.
    # If all entries are zero, this pixel won't contribute to
    # the loss.
    moves[batch[:,:,:,0] == 0] = 0

    return batch, moves


def load_data(games):
    '''Generator that loads batches of BATCH_SIZE
    frames from the specified games.'''
    xs = []
    ys = []
    size = 0
    for g in cycle(games):
        out = prepare_data(f[g])
        if out is None:
            continue

        X, y = out
        size += len(X)
        xs.append(X)
        ys.append(y)

        if size >= BATCH_SIZE:
            x_ = np.concatenate(xs, axis=0)
            y_ = np.concatenate(ys, axis=0)
            xs = [x_[BATCH_SIZE:]]
            ys = [y_[BATCH_SIZE:]]
            size = len(x_[BATCH_SIZE:])
            yield x_[:BATCH_SIZE], y_[:BATCH_SIZE]


if __name__ == '__main__':
    f = h5py.File('games.h5', 'r')
    games = np.random.permutation(list(f.keys()))
    split = int(VALIDATION_SPLIT * len(games))
    train_games = games[split:]
    val_games = games[:split]

    model = create_model()

    model.fit_generator(
        load_data(train_games),
        validation_data=load_data(val_games),
        nb_val_samples=2000,
        callbacks=[
            ModelCheckpoint(
                'model.h5',
                monitor='val_loss',
                verbose=0,
                save_best_only=True),
            #ReduceLROnPlateau(
            #    monitor='val_loss',
            #    factor=0.1,
            #    patience=10,
            #    verbose=0,
            #    mode='auto',
            #    epsilon=0.0001,
            #    cooldown=0,
            #    min_lr=0),
        ],
        samples_per_epoch=50 * BATCH_SIZE,
        nb_epoch=500,
    )
