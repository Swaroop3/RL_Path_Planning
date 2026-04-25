"""Plotting utilities for training and evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import DATA_REQ, NUM_SENSORS


def save_training_curves(metrics: dict, output_path: str | Path, rolling_window: int = 50) -> None:
    rewards = metrics.get("cumulative_rewards", [])
    steps = metrics.get("num_steps_list", [])
    actor_losses = metrics.get("actor_losses", [])
    alphas = metrics.get("alphas", [])
    n = len(rewards)
    if n == 0:
        return

    eps = range(n)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(eps, rewards, color="royalblue", linewidth=0.7, alpha=0.5, label="per episode")
    if n >= rolling_window:
        smoothed = np.convolve(rewards, np.ones(rolling_window) / rolling_window, mode="valid")
        axes[0].plot(range(rolling_window - 1, n), smoothed, color="navy", linewidth=2, label="rolling avg")
        axes[0].legend()
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title("Training Reward")
    axes[0].grid(alpha=0.3)

    axes[1].plot(eps, steps, color="seagreen", linewidth=0.7, alpha=0.5)
    if n >= rolling_window:
        smoothed_steps = np.convolve(steps, np.ones(rolling_window) / rolling_window, mode="valid")
        axes[1].plot(range(rolling_window - 1, n), smoothed_steps, color="darkgreen", linewidth=2, label="rolling avg")
        axes[1].legend()
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Steps per Episode")
    axes[1].grid(alpha=0.3)

    axes[2].plot(eps, actor_losses, color="tomato", linewidth=0.8, alpha=0.6, label="Actor loss")
    if alphas:
        ax3b = axes[2].twinx()
        ax3b.plot(eps, alphas, color="purple", linewidth=1, linestyle="--", alpha=0.7, label="alpha")
        ax3b.set_ylabel("alpha", color="purple")
        ax3b.tick_params(axis="y", labelcolor="purple")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Actor Loss")
    axes[2].set_title("Actor Loss and Entropy Coefficient")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    fig.suptitle("SAC Training Curves", fontsize=14)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_plots(
    env,
    trajectory_x: list[float],
    trajectory_y: list[float],
    collected_data_trace: dict[int, list[float]],
    reward: float,
    steps: int,
    done: bool,
    episode: int,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for obstacle_x, obstacle_y in zip(env.Obstacle_x, env.Obstacle_y):
        rect = plt.Rectangle(
            (10 * (obstacle_x - 0.5), 10 * (10 - obstacle_y - 0.5)),
            10,
            10,
            fc="steelblue",
            ec="steelblue",
            alpha=0.75,
        )
        ax.add_patch(rect)

    x = np.array(trajectory_x)
    y = np.array(trajectory_y)
    if len(x) > 1:
        ax.quiver(
            x[:-1],
            y[:-1],
            x[1:] - x[:-1],
            y[1:] - y[:-1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color="black",
            alpha=0.55,
            width=0.003,
        )

    sensor_x = [env.sensors_XY[i][0] for i in range(NUM_SENSORS)]
    sensor_y = [env.sensors_XY[i][1] for i in range(NUM_SENSORS)]
    ax.scatter(sensor_x, sensor_y, c="black", marker="o", s=80, zorder=5, label="IoT Sensor")
    ax.scatter(10, 10, c="red", marker="s", s=100, zorder=6, label="Start/Terminal", ec="k")
    ax.scatter(20, 80, c="lime", marker="s", s=120, zorder=6, label="Charging Station", ec="k")
    ax.scatter(trajectory_x[-1], trajectory_y[-1], c="orange", marker="*", s=200, zorder=7, label="Final pos")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    goal_str = "GOAL REACHED" if done else "Max steps hit"
    ax.set_title(f"Ep {episode} | Reward: {reward:.1f} | Steps: {steps} | {goal_str}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"eval_trajectory_ep{episode}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if steps <= 0:
        return

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_SENSORS))
    for sensor_id in range(NUM_SENSORS):
        label = f"Sensor @ ({env.sensors_XY[sensor_id][0]:.0f},{env.sensors_XY[sensor_id][1]:.0f})"
        ax2.plot(range(steps), collected_data_trace[sensor_id], color=colors[sensor_id], linewidth=1.8, label=label)
        ax2.fill_between(range(steps), collected_data_trace[sensor_id], 0, alpha=0.1, color=colors[sensor_id])
    ax2.axhline(DATA_REQ[0], color="red", linestyle="--", linewidth=1.2, label=f"Requirement ({DATA_REQ[0]:.1e} b)")
    ax2.set_xlabel("Steps", fontsize=12)
    ax2.set_ylabel("Collected Data (bits)", fontsize=12)
    ax2.set_title(f"Data Collection - Episode {episode}", fontsize=12)
    ax2.set_xlim(0, max(1, steps - 1))
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(output_dir / f"eval_data_ep{episode}.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

