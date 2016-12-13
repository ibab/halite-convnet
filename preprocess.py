import os
from itertools import cycle
import numpy as np
import json

from train import REPLAY_FOLDER, WIDTH, HEIGHT

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

xs = []
ys = []
size = 0

for i, replay_name in enumerate(os.listdir(REPLAY_FOLDER)):
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
        X = np.concatenate(xs)
        y = np.concatenate(ys)
        name = REPLAY_FOLDER + '/sample_{:04d}'.format(i)
        print('Saving sample {}'.format(name))
        np.savez(name, X=X, y=y)
        xs = []
        ys = []
        size = 0

