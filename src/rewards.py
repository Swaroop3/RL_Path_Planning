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
    TERMINAL_POSITION,
    UAV_ALTITUDE,
    W,
)

STEP_PENALTY = -1.0
BOUNDARY_PENALTY = -50.0
OBSTACLE_PENALTY = -75.0
ENERGY_DEPLETED_PENALTY = -250.0
SUCCESS_REWARD = 1000.0
ALL_DATA_COLLECTED_REWARD = 250.0
SENSOR_COMPLETED_REWARD = 75.0
DATA_PROGRESS_SCALE = 100.0
TARGET_PROGRESS_SCALE = 0.4
INCOMPLETE_SENSOR_PENALTY = -0.2


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


def _current_target(vector_agent_state: np.ndarray, sensors_xy: dict[int, list[float]], collected_data: np.ndarray) -> np.ndarray:
    incomplete = np.where(collected_data <= DATA_REQ)[0]
    if len(incomplete) == 0:
        return TERMINAL_POSITION
    distances = [
        np.linalg.norm(vector_agent_state - np.asarray(sensors_xy[int(sensor_id)]))
        for sensor_id in incomplete
    ]
    nearest_sensor = int(incomplete[int(np.argmin(distances))])
    return np.asarray(sensors_xy[nearest_sensor])


def compute_reward(env: object) -> float:
    """Bounded shaped reward for learning the full data-collection mission."""

    if env.Is_Terminal and env.doneType == 1:
        return SUCCESS_REWARD

    env.visited_charger, env.energy_level = apply_charger(
        env.vector_agentState,
        env.visited_charger,
        env.energy_level,
    )

    reward = STEP_PENALTY
    if getattr(env, "hit_boundary", False):
        reward += BOUNDARY_PENALTY
    if getattr(env, "hit_obstacle", False):
        reward += OBSTACLE_PENALTY

    old_data = np.copy(env.Collected_Data)
    old_complete = old_data > DATA_REQ
    old_all_complete = bool(np.all(old_complete))
    target = _current_target(env.previous_vector_agentState, env.sensors_XY, old_data)

    env.Collected_Data = collect_data(env.vector_agentState, env.sensors_XY, env.Collected_Data)

    capped_old = np.minimum(old_data, DATA_REQ)
    capped_new = np.minimum(env.Collected_Data, DATA_REQ)
    data_fraction_delta = float(np.sum((capped_new - capped_old) / DATA_REQ))
    reward += DATA_PROGRESS_SCALE * data_fraction_delta

    new_complete = env.Collected_Data > DATA_REQ
    newly_completed = int(np.sum(new_complete & ~old_complete))
    reward += SENSOR_COMPLETED_REWARD * newly_completed
    if bool(np.all(new_complete)) and not old_all_complete:
        reward += ALL_DATA_COLLECTED_REWARD

    previous_distance = np.linalg.norm(env.previous_vector_agentState - target)
    current_distance = np.linalg.norm(env.vector_agentState - target)
    reward += TARGET_PROGRESS_SCALE * (previous_distance - current_distance)
    reward += INCOMPLETE_SENSOR_PENALTY * int(np.sum(~new_complete))

    if env.energy_level < 0:
        reward += ENERGY_DEPLETED_PENALTY
    return float(reward)
