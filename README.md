# RL Path Planning

Discrete Soft Actor-Critic training for a UAV/IoT path-planning task. The UAV moves on an 11x11 grid, collects data from 5 sensors, avoids obstacles, optionally visits a charger, and must return to the terminal/start cell after all sensor data is collected.

## Entry Points

Train:

```bash
python train.py
```

Simulate/evaluate:

```bash
python simulate.py
```

Both scripts are configured through argparse defaults in the files. CLI flags can still override any default.

## Code Map

- `train.py`: training loop, checkpointing, resume, progress logging, CSV/JSONL metrics.
- `simulate.py`: loads a checkpoint, runs deterministic or stochastic evaluation, writes plots and summary CSV.
- `src/environment.py`: grid environment, movement, obstacle/wall handling, state image.
- `src/rewards.py`: reward shaping, data collection reward, invalid move penalties, terminal reward.
- `src/agent.py`: discrete SAC agent, actor/critics, entropy tuning, gradient clipping.
- `src/models.py`: CNN actor and critic networks.
- `src/replay_buffer.py`: replay memory.
- `src/checkpoints.py`: checkpoint save/load helpers.
- `src/plotting.py`: training and evaluation plots.
- `src/constants.py`: environment constants and grid layout.

## Training Controls

- `--episodes`: number of new episodes to run.
- `--timesteps`: number of new environment steps to run.
- `--max-episode-steps`: timeout per episode. Default is 500.
- `--vector-envs`: number of active environments stepped per rollout loop.
- `--batch-size`: replay samples per SAC update.
- `--gradient-steps`: SAC updates per rollout loop.
- `--learning-starts`: replay size before updates begin.
- `--target-entropy`: desired policy entropy. Default is 0.5.
- `--progress-every-seconds`: live training status interval.

## Checkpoints

Checkpoints are written to `artifacts_reward_v3/checkpoints/`.

Previous runs are intentionally kept under `artifacts/` and `artifacts_reward_v2/`. The current defaults write to `artifacts_reward_v3/` because the v2 best checkpoint learned to collect one sensor and then idle, so this version changes reward and best-checkpoint selection.

- `sac_ep<N>.pt`: numbered checkpoint every `--save-every-episodes`.
- `sac_best.pt`: best mission-progress checkpoint, ranked by goal, capped minimum sensor data, capped total sensor data, fewer steps, then reward.
- `sac_latest.pt`: periodic wall-clock checkpoint.
- `sac_final.pt`: final checkpoint when training exits.

Resume explicitly:

```bash
python train.py --resume artifacts_reward_v3/checkpoints/sac_ep2100.pt
```

Auto-resume newest checkpoint:

```bash
python train.py --auto-resume
```

Numbered checkpoint names continue from the stored episode index. If training resumes after episode 2100 and `--save-every-episodes 100`, the next numbered save is `sac_ep2200.pt`.

## Stop/Resume

Use Ctrl+C to stop training. The `finally` block writes `sac_final.pt`, so resume from that file or use `--auto-resume`.

## Outputs

- `artifacts_reward_v3/logs/train.log`: live training log.
- `artifacts_reward_v3/logs/episode_metrics.csv`: per-episode metrics.
- `artifacts_reward_v3/logs/episode_metrics.jsonl`: same data as JSONL.
- `artifacts_reward_v3/checkpoints/metrics.json`: checkpointed metric history.
- `artifacts_reward_v3/checkpoints/training_curves.png`: reward/steps/loss plot.
- `artifacts_reward_v3/eval/simulate.log`: evaluation log.
- `artifacts_reward_v3/eval/<checkpoint>_simulation_summary.csv`: evaluation summary, including checkpoint path.
- `artifacts_reward_v3/eval/<checkpoint>_trajectory_ep*.csv`: raw step-by-step position/action/reward trace.
- `artifacts_reward_v3/eval/<checkpoint>_data_ep*.csv`: raw per-step sensor data trace.
- `artifacts_reward_v3/eval/<checkpoint>_trajectory_ep*.png`: trajectory plots.
- `artifacts_reward_v3/eval/<checkpoint>_data_ep*.png`: sensor data collection plots.

If an eval for the same checkpoint already exists, the script adds `run02`, `run03`, etc. to avoid overwriting older outputs.

## Diagnosis

Watch these first:

- `goal`: should become true more often over time.
- `steps`: should drop as the policy improves.
- `boundary_hits` and `obstacle_hits`: should trend down.
- `critic_loss`, `actor_loss`, `alpha`: should stay finite.
- eval final position: repeated boundary positions usually mean a collapsed deterministic policy.

If losses become `inf` or `alpha` explodes, discard that checkpoint and start a fresh run. Continuing a numerically exploded replay buffer usually wastes time.

## Change History

Original notebook-compatible behavior:

- Obstacles were represented in state but did not block movement.
- Boundary moves were silently clipped to the map edge.
- Default episode timeout was 5000 steps.
- Reward was mostly a per-step incomplete-sensor penalty, with small obstacle and terminal rewards.
- SAC default entropy target encouraged near-maximum entropy for the 5-action policy.

Reward/environment revision:

- Obstacle moves are blocked and penalized.
- Boundary attempts are penalized.
- Default episode timeout is 500 steps.
- Reward now includes bounded step cost, data-progress reward, first sensor-completion reward, all-data-complete bonus, distance progress toward the current target, terminal success reward, and energy failure penalty.
- Training metrics now include `boundary_hits` and `obstacle_hits`.

Reward v3 revision:

- Action `0` hover/stay is penalized when it makes no data progress, while useful data-collection hovering remains allowed.
- `sac_best.pt` is selected by mission progress rather than rolling reward, so a partial one-sensor policy should not outrank a policy that covers more sensors.
- New default artifacts are written under `artifacts_reward_v3/` so old v2 run data remains untouched.

SAC stability revision:

- Default `target_entropy` is 0.5.
- Alpha is clamped to a finite range after updates.
- Actor and critic gradients are clipped.
- Critic loss uses Huber loss instead of raw MSE.
