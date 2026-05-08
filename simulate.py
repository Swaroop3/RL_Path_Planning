from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.agent import SACAgent
from src.checkpoints import latest_checkpoint, load_checkpoint
from src.constants import MAX_EPISODE_STEPS, NUM_SENSORS
from src.environment import IoTEnv
from src.logging_utils import configure_logging, seed_everything
from src.plotting import save_evaluation_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate an existing SAC checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts_reward_v2/checkpoints/sac_best.pt"),
        help="Checkpoint to evaluate. Defaults to newest .pt in checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("artifacts_reward_v2/checkpoints")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_reward_v2/eval"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of taking argmax.",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip PNG trajectory/data plots."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir, "simulate")
    seed_everything(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {args.checkpoint_dir}")

    agent = SACAgent(n_actions=5, device=device)
    checkpoint = load_checkpoint(
        checkpoint_path, agent, device, load_replay_buffer=False
    )
    agent.actor.eval()
    logger.info(
        "Loaded checkpoint %s%s",
        checkpoint_path,
        " (legacy notebook format)" if checkpoint.get("legacy") else "",
    )

    summary_path = args.output_dir / "simulation_summary.csv"
    rows = []
    all_rewards = []
    all_steps = []
    goal_count = 0

    for episode in range(1, args.episodes + 1):
        env = IoTEnv(max_episode_steps=args.max_episode_steps)
        state = env.reset()
        done = False
        reward_sum = 0.0
        steps = 0
        trajectory_x = [float(env.vector_agentState[0])]
        trajectory_y = [float(env.vector_agentState[1])]
        collected_data_trace = {sensor_id: [] for sensor_id in range(NUM_SENSORS)}
        boundary_hits = 0
        obstacle_hits = 0

        while not done and steps < env.max_episode_steps:
            action = agent.select_action(state, deterministic=not args.stochastic)
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            steps += 1
            boundary_hits += int(env.hit_boundary)
            obstacle_hits += int(env.hit_obstacle)
            trajectory_x.append(float(env.vector_agentState[0]))
            trajectory_y.append(float(env.vector_agentState[1]))
            for sensor_id in range(NUM_SENSORS):
                collected_data_trace[sensor_id].append(
                    float(env.Collected_Data[sensor_id])
                )
            state = next_state

        if done:
            goal_count += 1
        all_rewards.append(reward_sum)
        all_steps.append(steps)

        row = {
            "checkpoint": str(checkpoint_path),
            "episode": episode,
            "reward": float(reward_sum),
            "steps": int(steps),
            "goal": bool(done),
            "energy_level": float(env.energy_level),
            "min_collected_data": float(np.min(env.Collected_Data)),
            "collected_data_sum": float(np.sum(env.Collected_Data)),
            "final_x": float(env.vector_agentState[0]),
            "final_y": float(env.vector_agentState[1]),
            "boundary_hits": int(boundary_hits),
            "obstacle_hits": int(obstacle_hits),
        }
        rows.append(row)
        logger.info(
            "Ep %s | reward %.1f | steps %s | %s",
            episode,
            reward_sum,
            steps,
            "GOAL" if done else "timeout",
        )

        if not args.no_plots:
            save_evaluation_plots(
                env,
                trajectory_x,
                trajectory_y,
                collected_data_trace,
                reward_sum,
                steps,
                done,
                episode,
                args.output_dir,
            )

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Goal reached: %s/%s (%.0f%%)",
        goal_count,
        args.episodes,
        100 * goal_count / args.episodes,
    )
    logger.info(
        "Avg reward: %.2f +/- %.2f",
        float(np.mean(all_rewards)),
        float(np.std(all_rewards)),
    )
    logger.info(
        "Avg steps: %.1f +/- %.1f", float(np.mean(all_steps)), float(np.std(all_steps))
    )
    logger.info("Simulation outputs saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
