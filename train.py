import os
import sys

import numpy as np
import json

from keras.models import Model, load_model
from keras.layers import Convolution2D, Deconvolution2D, Input, Reshape, Flatten, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint

from itertools import cycle

REPLAY_FOLDER = 'replays'
WIDTH = 100
HEIGHT = 100

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

def prepare_data(replay):
    '''Create training data from a Halite replay.'''
    frames = np.array(replay['frames'])
    player_id = frames[:,:,:,0]

    # Find the winner (the player with most territory at the end)
    players, counts = np.unique(player_id[-1], return_counts=True)
    winner_id = players[counts.argmax()]
    if winner_id == 0:
        return None

    # Broadcast production array to each time frame
    init_prod = np.array(replay['productions'])[np.newaxis]
    production = np.repeat(init_prod, replay['num_frames'], axis=0)

    strength = frames[:,:,:,1]
    is_winner = player_id == winner_id
    is_loser = (player_id != winner_id) & (player_id != 0)

    batch = np.array([is_winner, is_loser, production / 20, strength / 255])
    batch = np.transpose(batch, (1, 2, 3, 0))

    # One-hot encode the moves
    moves = np.eye(5)[np.array(replay['moves'])]

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

    # We remove the final frame, as it doesn't have any associated decisions
    return batch[:-1], moves.reshape(-1, WIDTH * HEIGHT, 5)

def data_generator():
    xs = []
    ys = []
    size = 0
    for replay_name in cycle(os.listdir(REPLAY_FOLDER)):
        if replay_name[-4:] != '.hlt':
            continue
        #print('Loading {}'.format(replay_name))
        replay = json.load(open('{}/{}'.format(REPLAY_FOLDER,replay_name)))
        out = prepare_data(replay)
        if out is not None:
            xs.append(out[0])
            ys.append(out[1])
            size += len(out[0])
        if size >= 256:
            yield np.concatenate(xs), np.concatenate(ys)
            xs = []
            ys = []
            size = 0

if __name__ == '__main__':
    model = create_model()

    model.fit_generator(
        data_generator(),
        callbacks=[ModelCheckpoint('model.h5', verbose=1, save_best_only=True)],
        samples_per_epoch=20000,
        nb_epoch=1,
    )

