"""Environment constants copied from the original notebook."""

from __future__ import annotations

import numpy as np

SIGMA2 = 10**-14
W = 20 * (10**6)
T_S = 2
TAU_OVERHEAD = 0
P_MAX = 0.1
NUM_SENSORS = 5
DATA_REQ = (0.4 * 10**9) * np.ones(NUM_SENSORS)
DISTANCE_THRESHOLD = 10
INITIAL_ENERGY = 10_000_000_000
MAX_UAV_VELOCITY = 5
UAV_ALTITUDE = 10
DT = 2

GRID_SIZE = 11
WORLD_X_MAX = 100
WORLD_Y_MAX = 100
START_POSITION = np.asarray([10.0, 10.0])
TERMINAL_POSITION = np.asarray([10.0, 10.0])
CHARGER_POSITION = np.asarray([20.0, 80.0])
CHARGER_ENERGY = 100_000_000
MAX_EPISODE_STEPS = 5000

OBSTACLE_X = [0, 1, 1, 1, 1, 2, 4, 4, 4, 4, 5, 5, 5, 6, 8, 8, 8, 9, 9, 9, 9, 10, 10]
OBSTACLE_Y = [2, 6, 5, 2, 1, 1, 8, 7, 4, 3, 7, 6, 3, 6, 10, 9, 3, 10, 9, 4, 3, 10, 9]

SENSORS_XY = {
    0: [50.0, 10.0],
    1: [20.0, 60.0],
    2: [80.0, 40.0],
    3: [60.0, 60.0],
    4: [50.0, 90.0],
}

SENSOR_STATE_CELLS = {
    0: (9, 5),
    1: (4, 2),
    2: (6, 8),
    3: (4, 6),
    4: (1, 5),
}

ACTION_SPACE = {
    0: [0, 0],
    1: [MAX_UAV_VELOCITY, 0],
    2: [0, MAX_UAV_VELOCITY],
    3: [-MAX_UAV_VELOCITY, 0],
    4: [0, -MAX_UAV_VELOCITY],
}

