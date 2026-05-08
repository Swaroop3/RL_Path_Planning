"""IoT path-planning environment from the SAC notebook."""

from __future__ import annotations

import numpy as np

from .constants import (
    ACTION_SPACE,
    DATA_REQ,
    DT,
    GRID_SIZE,
    INITIAL_ENERGY,
    MAX_EPISODE_STEPS,
    NUM_SENSORS,
    OBSTACLE_X,
    OBSTACLE_Y,
    SENSOR_STATE_CELLS,
    SENSORS_XY,
    START_POSITION,
    TERMINAL_POSITION,
    T_S,
    WORLD_X_MAX,
    WORLD_Y_MAX,
)
from .rewards import channel as reward_channel
from .rewards import collect_data, compute_reward


class IoTEnv:
    """Grid-image environment with the same dynamics as the original notebook."""

    def __init__(
        self,
        num_sensors: int = NUM_SENSORS,
        initial_position: list[float] | tuple[float, float] | np.ndarray | None = None,
        x_max: int = WORLD_X_MAX,
        y_max: int = WORLD_Y_MAX,
        num_actions: int = 5,
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ):
        self.num_sensors = num_sensors
        self.state0 = np.zeros((3, GRID_SIZE, GRID_SIZE))
        self.state0[0][9][1] = 1

        self.Obstacle_x = list(OBSTACLE_X)
        self.Obstacle_y = list(OBSTACLE_Y)
        for obstacle_x, obstacle_y in zip(self.Obstacle_x, self.Obstacle_y):
            self.state0[1, obstacle_y, obstacle_x] = 1

        self.state0[1][1][9] = 1
        self.state0[1][2][2] = 0.5

        for cell_y, cell_x in SENSOR_STATE_CELLS.values():
            self.state0[2][cell_y][cell_x] = 1

        self.sensors_XY = {key: list(value) for key, value in SENSORS_XY.items()}
        self.X_max = x_max
        self.Y_max = y_max
        self.vector_state0 = np.asarray(initial_position, dtype=float) if initial_position is not None else START_POSITION.copy()
        self.Terminal = TERMINAL_POSITION.copy()
        self.max_episode_steps = max_episode_steps
        self.num_actions = num_actions
        self.actionspace = {key: list(value) for key, value in ACTION_SPACE.items()}
        self.obstacles = set(zip(self.Obstacle_x, self.Obstacle_y))
        self.reset()

    def reset(self) -> np.ndarray:
        self.agentState = np.copy(self.state0)
        self.vector_agentState = np.copy(self.vector_state0)
        self.energy_level = INITIAL_ENERGY
        self.Collected_Data = np.zeros(self.num_sensors)
        self.doneType = 0
        self.steps_counter = 0
        self.Is_Terminal = False
        self.visited_charger = 0
        self.hit_boundary = False
        self.hit_obstacle = False
        self.previous_vector_agentState = np.copy(self.vector_agentState)
        return self.agentState

    def step(self, action: int) -> tuple[np.ndarray, float, bool, None]:
        velocity = self.actionspace[int(action)]
        self.previous_vector_agentState = np.copy(self.vector_agentState)
        proposed_x = self.vector_agentState[0] + velocity[0] * DT
        proposed_y = self.vector_agentState[1] + velocity[1] * DT
        clipped_x = np.clip(proposed_x, 0, self.X_max)
        clipped_y = np.clip(proposed_y, 0, self.Y_max)
        self.hit_boundary = proposed_x != clipped_x or proposed_y != clipped_y

        proposed_i_x = int(clipped_x / 10)
        proposed_i_y = int(10 - clipped_y / 10)
        self.hit_obstacle = (proposed_i_x, proposed_i_y) in self.obstacles
        if not self.hit_obstacle:
            self.vector_agentState[0] = clipped_x
            self.vector_agentState[1] = clipped_y

        i_x = np.copy(self.vector_agentState[0]) / 10
        i_y = 10 - np.copy(self.vector_agentState[1]) / 10
        self.agentState = np.copy(self.state0)
        self.agentState[0][9][1] = 0
        self.agentState[0, int(i_y), int(i_x)] = 1
        self.energy_level -= self.propulsion_power(velocity) * T_S
        self.steps_counter += 1
        self.Is_Terminal = self.isTerminal()
        reward = self.get_reward()
        self._update_sensor_state_channel()
        return self.agentState, reward, self.Is_Terminal, None

    def propulsion_power(self, velocity: list[float]) -> float:
        velocity_magnitude = (np.linalg.norm(velocity)) ** 0.5
        return (
            79.85
            + 0.01663 * velocity_magnitude**2
            + 0.009242 * velocity_magnitude**3
            + 89.44
            * (
                (1 + velocity_magnitude**4 / 4.03**4) ** 0.5
                - velocity_magnitude**2 / (2 * 4.03**2)
            )
            ** 0.5
        )

    def isTerminal(self) -> bool:
        distance = np.linalg.norm(self.vector_agentState - self.Terminal)
        if np.all(self.Collected_Data > DATA_REQ) and distance**0.5 == 0 and self.energy_level > 0:
            self.doneType = 1
            return True
        return False

    def get_reward(self) -> float:
        return compute_reward(self)

    def channel(self) -> tuple[np.ndarray, np.ndarray]:
        return reward_channel(self.vector_agentState, self.sensors_XY)

    def resourceAllocation(self) -> None:
        self.Collected_Data = collect_data(self.vector_agentState, self.sensors_XY, self.Collected_Data)

    def _update_sensor_state_channel(self) -> None:
        data_norm = np.copy(self.Collected_Data) / DATA_REQ[0]
        pct_remaining = np.clip(1 - data_norm, 0, None)
        bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for sensor_id, (cell_y, cell_x) in SENSOR_STATE_CELLS.items():
            self.agentState[2][cell_y][cell_x] = np.digitize(pct_remaining[sensor_id], bins, right=True) / 10


# Backward-compatible alias for checkpoints or notebooks that reference the old name.
IoT_env = IoTEnv
