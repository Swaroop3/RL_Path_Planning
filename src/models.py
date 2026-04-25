"""Network definitions for discrete Soft Actor-Critic."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CNNBackbone(nn.Module):
    """CNN architecture preserved from the original notebook."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, 1)
        self.conv2 = nn.Conv2d(16, 32, 4, 1)
        self.fc = nn.Linear(5 * 5 * 32, 64)
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))


class Actor(nn.Module):
    """Probability distribution over discrete actions."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = nn.Linear(self.backbone.out_dim, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.head(self.backbone(state)), dim=-1)

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = self.forward(state)
        dist = Categorical(probs)
        action = probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs


class Critic(nn.Module):
    """Q-values for all actions at once."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = nn.Linear(self.backbone.out_dim, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(state))

