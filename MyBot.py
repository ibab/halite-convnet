from networking import *
import os
import sys
import numpy as np

myID, gameMap = getInit()

# Make sure not to produce stderr when loading the model
with open(os.devnull, 'w') as sys.stderr:
    from keras.models import load_model
    model = load_model('model.h5')

model.predict(np.random.randn(1, input_dim)).shape # make sure model is compiled during init

def frame_to_input(frame):
    game_map = np.array([[(x.owner, x.production, x.strength) for x in row] for row in frame.contents])
    return np.array([(game_map[:, :, 0] == myID),  # 0 : owner is me
                      ((game_map[:, :, 0] != 0) & (game_map[:, :, 0] != myID)),  # 1 : owner is enemy
                      game_map[:, :, 1] / 20,   # 2 : production
                      game_map[:, :, 2] / 255,  # 3 : strength
                      ]).astype(np.float32)

sendInit('ibab')
while True:
    state = frame_to_input(getFrame())
    output = model.predict(state)
    actions = []
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if not state[i, j, 0]:
                continue
            m = Move(Location(i, j), np.random.choice(np.arange(5), p=output[i, j, :]))
            actions.append(m)

    sendFrame(actions)
