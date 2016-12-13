from networking import *
import os
import sys
import numpy as np

WIDTH = 50
HEIGHT = 50
myID, gameMap = getInit()

# Make sure not to produce stderr when loading the model
backup = sys.stderr
with open('err.log', 'w') as sys.stderr:
    from keras.models import load_model
    model = load_model('model.h5')
    model.predict(np.random.normal(size=(1, 50, 50, 4))).shape # make sure model is compiled during init
sys.stderr = backup

def frame_to_input(frame):
    game_map = np.array([[(x.owner, x.production, x.strength) for x in row] for row in frame.contents])
    data =  np.array([(game_map[:, :, 0] == myID),  # 0 : owner is me
                      ((game_map[:, :, 0] != 0) & (game_map[:, :, 0] != myID)),  # 1 : owner is enemy
                      game_map[:, :, 1] / 20,   # 2 : production
                      game_map[:, :, 2] / 255,  # 3 : strength
                      ]).astype(np.float32)
    data = np.transpose(data, (1, 2, 0))

    nx = data.shape[0]
    ny = data.shape[1]

    pad_x = int((WIDTH - nx) / 2)
    extra_x = int(WIDTH - nx - 2 * pad_x)
    pad_y = int((HEIGHT - ny) / 2)
    extra_y = int(HEIGHT - ny - 2 * pad_y)

    data = np.pad(data, ((pad_x, pad_x + extra_x), (pad_y, pad_y + extra_y), (0, 0)), 'wrap')

    return data, pad_x, extra_x, pad_y, extra_y

sendInit('ibab')

with open('status.log', 'w') as sys.stderr:
    while True:
        frame = getFrame()
        state, px, pxx, py, pyy = frame_to_input(frame)
        output = model.predict(state[np.newaxis])
        output = output.reshape(1, WIDTH, HEIGHT, 5)
        output = output[0, px:-(px + pxx), py:-(py + pyy), :]

        moves = []
        for y in range(gameMap.height):
            for x in range(gameMap.width):
                location = Location(x, y)
                if gameMap.getSite(location).owner == myID:
                    p = output[x, y, :]
                    decision = np.random.choice(np.arange(5), p=p)
                    #decision = np.argmax(p)
                    print('Decide to go {} at ({}, {}), p={}'.format(decision, x, y, p), file=sys.stderr)
                    moves.append(Move(location, decision))

        sendFrame(moves)
