"""Replay buffer for discrete SAC."""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Any

import torch

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done) -> None:
        self.buffer.append(
            Transition(
                torch.from_numpy(state).float().cpu(),
                torch.tensor([action], dtype=torch.long),
                torch.from_numpy(next_state).float().cpu(),
                torch.tensor([reward], dtype=torch.float32),
                torch.tensor([done], dtype=torch.float32),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "buffer": list(self.buffer),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self.buffer = deque(state["buffer"], maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

