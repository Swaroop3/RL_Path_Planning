from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from src.agent import SACAgent
from src.checkpoints import latest_checkpoint, load_checkpoint, save_checkpoint
from src.constants import MAX_EPISODE_STEPS, NUM_SENSORS
from src.environment import IoTEnv
from src.logging_utils import (
    CsvMetricLogger,
    JsonlLogger,
    configure_logging,
    configure_torch,
    seed_everything,
    write_json,
)
from src.plotting import save_training_curves


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train discrete SAC for the IoT path-planning environment."
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints")
    )
    parser.add_argument("--log-dir", type=Path, default=Path("artifacts/logs"))
    parser.add_argument(
        "--resume", type=Path, default=None, help="Checkpoint path to resume from."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda", help="cuda, cpu, or omitted for auto."
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from the newest checkpoint in checkpoint-dir.",
    )
    parser.add_argument(
        "--pause-file", type=Path, default=Path("artifacts/control/PAUSE")
    )
    parser.add_argument("--torch-threads", type=int, default=None)


    
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of new episodes to run in this invocation.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Stop after this many new environment steps.",
    )
    parser.add_argument(
        "--time-limit-minutes",
        type=float,
        default=None,
        help="Stop after this wall-clock time.",
    )
    parser.add_argument(
        "--progress-every-seconds",
        type=float,
        default=10.0,
        help="Print live training progress every N seconds. Set to 0 to disable.",
    )



    
    
    parser.add_argument("--save-every-episodes", type=int, default=100)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--save-every-seconds", type=float, default=0)
    
    
    
    
    parser.add_argument(
        "--no-save-replay-buffer",
        action="store_true",
        help="Save smaller checkpoints without replay memory.",
    )

    parser.add_argument(
        "--vector-envs",
        type=int,
        default=8,
        help="Number of environments stepped per policy batch.",
    ) #CPU

    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)#
    parser.add_argument("--batch-size", type=int, default=1024) #GPU mem
    parser.add_argument("--replay-buffer-size", type=int, default=100_000) #RAM
    parser.add_argument("--learning-starts", type=int, default=128)
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="SAC updates after each vector environment step.",
    ) #GPU comp
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--best-window", type=int, default=100)
    return parser.parse_args()


def blank_metrics() -> dict[str, list[float] | float]:
    return {
        "cumulative_rewards": [],
        "num_steps_list": [],
        "actor_losses": [],
        "critic_losses": [],
        "alphas": [],
        "entropies": [],
        "best_avg_reward": float("-inf"),
    }


def as_config(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def should_start_episode(next_episode: int, episode_stop: int | None) -> bool:
    return episode_stop is None or next_episode < episode_stop


def new_slot(episode_index: int, max_episode_steps: int) -> dict[str, Any]:
    env = IoTEnv(max_episode_steps=max_episode_steps)
    return {
        "env": env,
        "state": env.reset(),
        "episode": episode_index,
        "reward": 0.0,
        "steps": 0,
        "actor_losses": [],
        "critic_losses": [],
        "entropies": [],
        "active": True,
    }


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.pause_file.parent.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(args.log_dir, "train")
    seed_everything(args.seed)
    configure_torch(args.torch_threads)

    agent = SACAgent(
        n_actions=5,
        replay_buffer_size=args.replay_buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        tau=args.tau,
        target_entropy=args.target_entropy,
        device=args.device,
    )

    metrics = blank_metrics()
    best_avg_reward = float("-inf")
    global_step = 0
    next_episode = 0

    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume_path = latest_checkpoint(args.checkpoint_dir)
    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
            agent,
            agent.device,
            load_replay_buffer=not args.no_save_replay_buffer,
        )
        training_state = checkpoint.get("training_state", {})
        metrics = (
            training_state.get("metrics")
            or checkpoint.get("agent", {}).get("metrics")
            or blank_metrics()
        )
        for key, value in blank_metrics().items():
            metrics.setdefault(key, value)
        best_avg_reward = float(
            training_state.get(
                "best_avg_reward", metrics.get("best_avg_reward", float("-inf"))
            )
        )
        global_step = int(training_state.get("global_step", 0))
        next_episode = int(
            training_state.get("next_episode", training_state.get("episode", -1) + 1)
        )
        logger.info(
            "Loaded checkpoint %s; next episode=%s, global_step=%s, replay=%s",
            resume_path,
            next_episode,
            global_step,
            len(agent.memory),
        )

    episode_stop = next_episode + args.episodes if args.episodes is not None else None
    timestep_stop = global_step + args.timesteps if args.timesteps is not None else None
    wall_stop = (
        time.monotonic() + args.time_limit_minutes * 60
        if args.time_limit_minutes is not None
        else None
    )

    csv_logger = CsvMetricLogger(
        args.log_dir / "episode_metrics.csv",
        [
            "episode",
            "global_step",
            "reward",
            "steps",
            "goal",
            "actor_loss",
            "critic_loss",
            "entropy",
            "alpha",
            "replay_size",
            "energy_level",
            "min_collected_data",
            "collected_data_sum",
        ],
    )
    jsonl_logger = JsonlLogger(args.log_dir / "episode_metrics.jsonl")
    last_step_checkpoint = global_step
    last_time_checkpoint = time.monotonic()
    last_completed_episode = next_episode - 1
    run_start_time = time.monotonic()
    run_start_step = global_step
    run_start_completed_episodes = len(metrics["cumulative_rewards"])
    last_progress_time = run_start_time
    last_progress_step = global_step
    last_update_summary: dict[str, float] | None = None

    include_replay = not args.no_save_replay_buffer

    def training_state() -> dict[str, Any]:
        return {
            "episode": last_completed_episode,
            "next_episode": next_episode,
            "global_step": global_step,
            "metrics": metrics,
            "best_avg_reward": best_avg_reward,
            "saved_at_unix": time.time(),
        }

    def save_named(filename: str, reason: str) -> None:
        path = save_checkpoint(
            args.checkpoint_dir / filename,
            agent,
            training_state(),
            as_config(args),
            include_replay,
        )
        write_json(args.checkpoint_dir / "metrics.json", metrics)
        save_training_curves(
            metrics,
            args.checkpoint_dir / "training_curves.png",
            rolling_window=args.best_window,
        )
        logger.info("Saved %s checkpoint to %s", reason, path)

    slots: list[dict[str, Any] | None] = []
    for _ in range(max(1, args.vector_envs)):
        if should_start_episode(next_episode, episode_stop):
            slots.append(new_slot(next_episode, args.max_episode_steps))
            next_episode += 1
        else:
            slots.append(None)

    logger.info(
        "Training started on device=%s, vector_envs=%s, batch_size=%s, gradient_steps=%s",
        agent.device,
        args.vector_envs,
        args.batch_size,
        args.gradient_steps,
    )
    logger.info(
        "Pause by creating %s; resume by deleting it. Ctrl+C saves sac_interrupt.pt.",
        args.pause_file,
    )

    try:
        while any(slot is not None and slot["active"] for slot in slots):
            if wall_stop is not None and time.monotonic() >= wall_stop:
                logger.info("Wall-clock limit reached.")
                break
            if timestep_stop is not None and global_step >= timestep_stop:
                logger.info("Timestep limit reached.")
                break

            if args.pause_file.exists():
                save_named("sac_paused.pt", "pause")
                logger.info(
                    "Pause file exists; waiting until it is removed: %s",
                    args.pause_file,
                )
                while args.pause_file.exists():
                    time.sleep(5)
                logger.info("Pause file removed; continuing.")

            active_indices = [
                i for i, slot in enumerate(slots) if slot is not None and slot["active"]
            ]
            states = np.stack([slots[i]["state"] for i in active_indices])
            actions = agent.select_actions(states)
            update_stats: list[dict[str, float]] = []

            for slot_index, action in zip(active_indices, actions):
                slot = slots[slot_index]
                env = slot["env"]
                next_state, reward, done, _ = env.step(int(action))
                timeout = slot["steps"] + 1 >= env.max_episode_steps
                agent.store_transition(
                    slot["state"], int(action), next_state, reward, done or timeout
                )
                slot["state"] = next_state
                slot["reward"] += reward
                slot["steps"] += 1
                global_step += 1

                if done or timeout:
                    avg_actor = (
                        mean(slot["actor_losses"]) if slot["actor_losses"] else 0.0
                    )
                    avg_critic = (
                        mean(slot["critic_losses"]) if slot["critic_losses"] else 0.0
                    )
                    avg_entropy = mean(slot["entropies"]) if slot["entropies"] else 0.0
                    metrics["cumulative_rewards"].append(float(slot["reward"]))
                    metrics["num_steps_list"].append(int(slot["steps"]))
                    metrics["actor_losses"].append(float(avg_actor))
                    metrics["critic_losses"].append(float(avg_critic))
                    metrics["alphas"].append(float(agent.alpha))
                    metrics["entropies"].append(float(avg_entropy))

                    last_completed_episode = int(slot["episode"])
                    if len(metrics["cumulative_rewards"]) >= args.best_window:
                        rolling_avg = float(
                            np.mean(metrics["cumulative_rewards"][-args.best_window :])
                        )
                        if rolling_avg > best_avg_reward:
                            best_avg_reward = rolling_avg
                            metrics["best_avg_reward"] = best_avg_reward
                            save_named("sac_best.pt", "best")

                    row = {
                        "episode": slot["episode"],
                        "global_step": global_step,
                        "reward": float(slot["reward"]),
                        "steps": int(slot["steps"]),
                        "goal": bool(done),
                        "actor_loss": float(avg_actor),
                        "critic_loss": float(avg_critic),
                        "entropy": float(avg_entropy),
                        "alpha": float(agent.alpha),
                        "replay_size": len(agent.memory),
                        "energy_level": float(env.energy_level),
                        "min_collected_data": float(np.min(env.Collected_Data)),
                        "collected_data_sum": float(np.sum(env.Collected_Data)),
                    }
                    csv_logger.write(row)
                    jsonl_logger.write(row)
                    logger.info(
                        "Ep %s | reward %.1f | steps %s | alpha %.4f | replay %s | %s",
                        slot["episode"],
                        slot["reward"],
                        slot["steps"],
                        agent.alpha,
                        len(agent.memory),
                        "GOAL" if done else "timeout",
                    )

                    if (
                        args.save_every_episodes > 0
                        and (slot["episode"] + 1) % args.save_every_episodes == 0
                    ):
                        save_named(f"sac_ep{slot['episode'] + 1}.pt", "episode")

                    if should_start_episode(next_episode, episode_stop):
                        slots[slot_index] = new_slot(
                            next_episode, args.max_episode_steps
                        )
                        next_episode += 1
                    else:
                        slots[slot_index] = None

            if global_step >= args.learning_starts:
                for _ in range(max(0, args.gradient_steps)):
                    result = agent.update()
                    if result is not None:
                        update_stats.append(result)

            if update_stats:
                avg_update = {
                    key: float(np.mean([stat[key] for stat in update_stats]))
                    for key in update_stats[0]
                    if isinstance(update_stats[0][key], (int, float))
                }
                last_update_summary = avg_update
                for slot_index in active_indices:
                    slot = slots[slot_index]
                    if slot is not None:
                        slot["actor_losses"].append(avg_update["actor_loss"])
                        slot["critic_losses"].append(avg_update["critic_loss"])
                        slot["entropies"].append(avg_update["entropy"])

            now = time.monotonic()
            if args.progress_every_seconds > 0 and now - last_progress_time >= args.progress_every_seconds:
                elapsed = now - run_start_time
                total_step_delta = global_step - run_start_step
                step_rate = total_step_delta / elapsed if elapsed > 0 else 0.0
                interval_step_delta = global_step - last_progress_step
                interval_elapsed = now - last_progress_time
                interval_step_rate = interval_step_delta / interval_elapsed if interval_elapsed > 0 else 0.0
                completed_episodes = len(metrics["cumulative_rewards"])
                completed_this_run = completed_episodes - run_start_completed_episodes
                active_slots = sum(slot is not None and slot["active"] for slot in slots)
                recent_reward = (
                    float(np.mean(metrics["cumulative_rewards"][-10:]))
                    if metrics["cumulative_rewards"]
                    else float("nan")
                )
                status_parts = [
                    f"elapsed {format_duration(elapsed)}",
                    f"episodes {completed_this_run}/{args.episodes}" if args.episodes is not None else f"episodes {completed_this_run}",
                    f"steps {total_step_delta}/{args.timesteps}" if args.timesteps is not None else f"steps {total_step_delta}",
                    f"active_envs {active_slots}",
                    f"replay {len(agent.memory)}",
                    f"step/s {step_rate:.1f}",
                    f"step/s(last) {interval_step_rate:.1f}",
                ]
                if wall_stop is not None:
                    status_parts.append(f"time_left {format_duration(wall_stop - now)}")
                if metrics["cumulative_rewards"]:
                    status_parts.append(f"reward10 {recent_reward:.1f}")
                if last_update_summary is not None:
                    status_parts.append(f"actor {last_update_summary['actor_loss']:.4f}")
                    status_parts.append(f"critic {last_update_summary['critic_loss']:.4f}")
                    status_parts.append(f"alpha {last_update_summary['alpha']:.4f}")
                logger.info("Progress | %s", " | ".join(status_parts))
                last_progress_time = now
                last_progress_step = global_step

            if (
                args.save_every_steps > 0
                and global_step - last_step_checkpoint >= args.save_every_steps
            ):
                save_named(f"sac_step{global_step}.pt", "step")
                last_step_checkpoint = global_step

            if (
                args.save_every_seconds > 0
                and time.monotonic() - last_time_checkpoint >= args.save_every_seconds
            ):
                save_named("sac_latest.pt", "time")
                last_time_checkpoint = time.monotonic()

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        save_named("sac_interrupt.pt", "interrupt")
    finally:
        save_named("sac_final.pt", "final")
        csv_logger.close()
        jsonl_logger.close()
        logger.info(
            "Training stopped. Final checkpoint: %s",
            args.checkpoint_dir / "sac_final.pt",
        )


if __name__ == "__main__":
    main()
