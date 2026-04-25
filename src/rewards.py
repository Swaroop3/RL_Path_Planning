"""Reward and data-collection logic for the IoT path-planning environment."""

from __future__ import annotations

import numpy as np

from .constants import (
    CHARGER_ENERGY,
    CHARGER_POSITION,
    DATA_REQ,
    DISTANCE_THRESHOLD,
    NUM_SENSORS,
    P_MAX,
    SIGMA2,
    T_S,
    TAU_OVERHEAD,
    UAV_ALTITUDE,
    W,
)


def channel(vector_agent_state: np.ndarray, sensors_xy: dict[int, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    distances_squared = np.array(
        [
            UAV_ALTITUDE**2
            + (vector_agent_state[0] - sensors_xy[i][0]) ** 2
            + (vector_agent_state[1] - sensors_xy[i][1]) ** 2
            for i in range(NUM_SENSORS)
        ]
    )
    channel_gain = 1e-3 / distances_squared
    return channel_gain, distances_squared**0.5


def apply_charger(vector_agent_state: np.ndarray, visited_charger: int, energy_level: float) -> tuple[int, float]:
    if (
        vector_agent_state[0] == CHARGER_POSITION[0]
        and vector_agent_state[1] == CHARGER_POSITION[1]
        and visited_charger == 0
    ):
        return 1, CHARGER_ENERGY
    return visited_charger, energy_level


def collect_data(
    vector_agent_state: np.ndarray,
    sensors_xy: dict[int, list[float]],
    collected_data: np.ndarray,
) -> np.ndarray:
    channel_gain, distances = channel(vector_agent_state, sensors_xy)
    best_sensor = int(np.argmax(channel_gain))
    if distances[best_sensor] <= DISTANCE_THRESHOLD:
        collected_data[best_sensor] += (
            np.log2(1 + P_MAX * channel_gain[best_sensor] / SIGMA2) * W * T_S * (1 - TAU_OVERHEAD)
        )
    return collected_data


def compute_reward(env: object) -> float:
    """Compute the notebook reward while mutating charger and data-collection state."""

    reward = 0.0
    indicator = np.zeros(NUM_SENSORS)

    env.visited_charger, env.energy_level = apply_charger(
        env.vector_agentState,
        env.visited_charger,
        env.energy_level,
    )

    i_x = int(env.vector_agentState[0] / 10)
    i_y = int(10 - env.vector_agentState[1] / 10)
    for obstacle_x, obstacle_y in zip(env.Obstacle_x, env.Obstacle_y):
        if obstacle_x == i_x and obstacle_y == i_y:
            reward = -20.0

    if not env.Is_Terminal:
        env.Collected_Data = collect_data(env.vector_agentState, env.sensors_XY, env.Collected_Data)
        for i in range(NUM_SENSORS):
            indicator[i] = 0 if env.Collected_Data[i] > DATA_REQ[i] else 1
        reward += (-5 * np.sum(indicator)) if np.mean(indicator) != 0 else -1
        if env.energy_level < 0:
            reward += -100
    elif env.doneType == 1:
        reward = 10.0

    return float(reward)

