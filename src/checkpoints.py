"""Checkpoint helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .agent import SACAgent


def save_checkpoint(
    path: str | Path,
    agent: SACAgent,
    training_state: dict[str, Any],
    config: dict[str, Any],
    include_replay_buffer: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "agent": agent.state_dict(include_replay_buffer=include_replay_buffer),
            "training_state": training_state,
            "config": config,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)
    return path


def load_checkpoint(
    path: str | Path,
    agent: SACAgent,
    device: str,
    load_replay_buffer: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if "agent" in checkpoint:
        agent.load_state_dict(checkpoint["agent"], load_replay_buffer=load_replay_buffer)
        return checkpoint

    # Legacy notebook checkpoint format: model and optimizer state live at top level.
    agent.load_state_dict(checkpoint, load_replay_buffer=False)
    return {
        "agent": checkpoint,
        "training_state": {
            "episode": int(checkpoint.get("episode", -1)),
            "global_step": 0,
            "metrics": checkpoint.get("metrics", {}),
            "best_avg_reward": checkpoint.get("metrics", {}).get("best_avg_reward", float("-inf")),
        },
        "config": {},
        "legacy": True,
    }


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None
