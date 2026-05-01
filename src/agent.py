"""Discrete SAC agent."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .models import Actor, Critic
from .replay_buffer import ReplayBuffer, Transition


class SACAgent:
    def __init__(
        self,
        n_actions: int,
        replay_buffer_size: int = 100_000,
        batch_size: int = 128,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        target_entropy: float | None = None,
        device: str | None = None,
        compile_models: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(n_actions).to(self.device)
        self.critic1 = Critic(n_actions).to(self.device)
        self.critic2 = Critic(n_actions).to(self.device)
        self.target_critic1 = Critic(n_actions).to(self.device)
        self.target_critic2 = Critic(n_actions).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic1.eval()
        self.target_critic2.eval()

        if compile_models and hasattr(torch, "compile"):
            self.actor = torch.compile(self.actor)
            self.critic1 = torch.compile(self.critic1)
            self.critic2 = torch.compile(self.critic2)
            self.target_critic1 = torch.compile(self.target_critic1)
            self.target_critic2 = torch.compile(self.target_critic2)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)

        self.target_entropy = target_entropy if target_entropy is not None else -math.log(1.0 / n_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().item()

        self.memory = ReplayBuffer(replay_buffer_size)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        actions = self.select_actions(np.expand_dims(state, axis=0), deterministic=deterministic)
        return int(actions[0])

    def select_actions(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor.get_action(state_tensor, deterministic)
        return action.detach().cpu().numpy()

    def store_transition(self, state, action, next_state, reward, done) -> None:
        self.memory.push(state, action, next_state, reward, done)

    def update(self) -> dict[str, float] | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = Transition(*zip(*self.memory.sample(self.batch_size)))
        states = torch.stack([state.detach().cpu() for state in batch.state]).to(self.device)
        actions = torch.cat([action.detach().cpu() for action in batch.action]).to(self.device)
        rewards = torch.cat([reward.detach().cpu() for reward in batch.reward]).to(self.device)
        next_states = torch.stack([state.detach().cpu() for state in batch.next_state]).to(self.device)
        dones = torch.cat([done.detach().cpu() for done in batch.done]).to(self.device)

        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_log_p = torch.log(next_probs + 1e-8)
            nq1 = self.target_critic1(next_states)
            nq2 = self.target_critic2(next_states)
            soft_v = (next_probs * (torch.min(nq1, nq2) - self.alpha * next_log_p)).sum(-1)
            target_q = rewards + self.gamma * (1 - dones) * soft_v

        q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        probs = self.actor(states)
        log_p = torch.log(probs + 1e-8)
        with torch.no_grad():
            q_min = torch.min(self.critic1(states), self.critic2(states))
        actor_loss = (probs * (self.alpha * log_p - q_min)).sum(-1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        entropy = -(probs.detach() * log_p.detach()).sum(-1).mean()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy)
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float((critic1_loss.item() + critic2_loss.item()) / 2),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha),
            "entropy": float(entropy.item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q2_mean": float(q2.detach().mean().item()),
        }

    def state_dict(self, include_replay_buffer: bool = True) -> dict[str, Any]:
        state = {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_opt_state_dict": self.actor_opt.state_dict(),
            "critic1_opt_state_dict": self.critic1_opt.state_dict(),
            "critic2_opt_state_dict": self.critic2_opt.state_dict(),
            "alpha_opt_state_dict": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "n_actions": self.n_actions,
        }
        if include_replay_buffer:
            state["replay_buffer"] = self.memory.state_dict()
        return state

    def load_state_dict(self, checkpoint: dict[str, Any], load_replay_buffer: bool = True) -> None:
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2_state_dict"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt_state_dict"])
        self.critic1_opt.load_state_dict(checkpoint["critic1_opt_state_dict"])
        self.critic2_opt.load_state_dict(checkpoint["critic2_opt_state_dict"])
        self.alpha_opt.load_state_dict(checkpoint["alpha_opt_state_dict"])
        self.log_alpha.data = checkpoint["log_alpha"].to(self.device)
        self.alpha = float(self.log_alpha.exp().item())
        if load_replay_buffer and checkpoint.get("replay_buffer"):
            self.memory.load_state_dict(checkpoint["replay_buffer"])
