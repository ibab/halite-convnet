import sys
import os
from itertools import cycle
import json
import urllib
import tempfile
from time import sleep
import gzip

import h5py
import requests
import numpy as np

# How many of the most recent games to use
MOST_RECENT = 10

def get_user_ids():
    print('Downloading user IDs...')
    url = 'https://halite.io/api/web/user?fields%5B%5D=isRunning&values%5B%5D=1&orderBy=rank&limit=100&page=0'
    r = requests.get(url)
    users = r.json()['users']
    user_ids = [user['userID'] for user in users]
    return user_ids


def generate_user_games(user_id, existing, tmpdir):
    '''Downloads games from one user.'''

    print('Downloading games list for player {}'.format(user_id))

    request = "https://halite.io/api/web/game?userID={}&limit={}".format(user_id, MOST_RECENT)
    r = requests.get(request)
    games = r.json()

    game_ids = [game['replayName'] for game in games]

    testfile = urllib.request.URLopener()

    print('Downloading games from player {}'.format(user_id))

    for game_id in game_ids:
        game_id = game_id.replace('.hlt', '')

        if game_id in existing:
            continue

        request = "https://s3.amazonaws.com/halitereplaybucket/{}".format(game_id + '.hlt')
        fname = tmpdir + '/{}.gzip'.format(game_id)
        testfile.retrieve(request, fname)
        with gzip.open(fname, 'rt') as f:
            data = json.load(f)
        os.remove(fname)

        existing.add(game_id)

        print('Downloaded {}'.format(game_id))
        yield game_id, data

        sleep(1 + np.random.rand())


def prepare_data(replay):
    '''Create training data from a Halite replay.'''
    frames = np.array(replay['frames'])

    # Drop the last frame, as it doesn't have any moves associated with it
    frames = frames[:-1]

    player_id = frames[:,:,:,0]
    production = replay['productions']
    strength = frames[:,:,:,1]
    moves = np.array(replay['moves'])
    return player_id, strength, production, moves


if __name__ == '__main__':
    f = h5py.File('games.h5', 'a')
    tmpdir = tempfile.mkdtemp()

    existing = set(f.keys())

    ids = get_user_ids()
    for user_id in ids:
        for game_id, data in generate_user_games(user_id, existing, tmpdir):
            player, strength, production, moves = prepare_data(data)
            grp = f.create_group(game_id)
            opts = {'compression': 'gzip'}
            grp.create_dataset('player', data=player, **opts)
            grp.create_dataset('strength', data=strength, **opts)
            grp.create_dataset('production', data=production, **opts)
            grp.create_dataset('moves', data=moves, **opts)
            f.flush()
            print('Processed {}'.format(game_id))
