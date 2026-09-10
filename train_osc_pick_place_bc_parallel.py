# Behavior-cloning-warm-started training for the pick-and-place task —
# same strategy as 5-arm_project_osc/train_osc_grasp_bc_parallel.py, which
# is what actually solved Phase 2 after 585k/1M steps of pure RL never once
# found a successful grasp. Applied here from the start (not as a fallback
# after trying pure RL first) since pick-and-place is a LONGER multi-stage
# sequence than grasp-only — if anything a harder exploration problem, not
# an easier one. See IMP_NOTES.md.
#
# Run collect_demonstrations.py first to produce demonstrations.npz.
#
# IMPORTANT (Windows-specific): multiprocessing on Windows uses "spawn",
# which re-imports this whole file in each worker process. Everything is
# wrapped in `if __name__ == "__main__":` so workers don't recursively
# spawn more workers.

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))

import numpy as np
import torch
import torch.nn.functional as F

from franka_osc_pick_place_env import FrankaOSCPickPlaceEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

N_ENVS = 4
DEMO_PATH = os.path.join(os.path.dirname(__file__), "demonstrations.npz")
BC_EPOCHS = 50
BC_BATCH_SIZE = 256
BC_LEARNING_RATE = 1e-3


def make_env():
    return Monitor(FrankaOSCPickPlaceEnv(use_camera=False))


def seed_replay_buffer(model, demo):
    """See 5-arm_project_osc/train_osc_grasp_bc_parallel.py's version of
    this function for the full reasoning (verified against SB3 source, not
    assumed): the buffer stores NORMALIZED actions, internal array shapes
    are (buffer_size, n_envs, dim) with buffer_size already divided by
    n_envs, and a vectorized bulk write is used instead of ~thousands of
    individual ReplayBuffer.add() calls (which was slow enough with the
    grasp-only task's ~49k transitions to look hung; this task's
    demonstrations will likely have even more transitions per episode,
    given the longer pick-and-place sequence, so the bulk-write approach
    matters even more here)."""
    buffer = model.replay_buffer
    n = len(demo["obs"])
    assert n <= buffer.buffer_size, (
        f"{n} demonstration transitions exceed the per-env replay buffer capacity "
        f"({buffer.buffer_size}) -- increase buffer_size or collect fewer demos."
    )

    low, high = model.policy.action_space.low, model.policy.action_space.high
    scaled_actions = 2.0 * (demo["actions"] - low) / (high - low) - 1.0

    n_envs = buffer.n_envs
    buffer.observations[0:n] = np.repeat(demo["obs"][:, None, :], n_envs, axis=1)
    buffer.next_observations[0:n] = np.repeat(demo["next_obs"][:, None, :], n_envs, axis=1)
    buffer.actions[0:n] = np.repeat(scaled_actions[:, None, :], n_envs, axis=1)
    buffer.rewards[0:n] = np.repeat(demo["rewards"][:, None], n_envs, axis=1)
    buffer.dones[0:n] = np.repeat(demo["dones"][:, None].astype(np.float32), n_envs, axis=1)
    buffer.timeouts[0:n] = 0.0

    buffer.pos = n
    buffer.full = False

    print(f"Seeded replay buffer with {n} demonstration transitions "
          f"(x{n_envs} tiling across parallel envs internally).")


def pretrain_actor(model, demo):
    """Supervised regression: the actor's tanh-squashed mean output should
    match the demonstrated (normalized) action for the demonstrated
    observation. Critic is NOT touched here — it learns from the seeded
    replay buffer during normal SAC training instead."""
    obs = torch.as_tensor(demo["obs"], dtype=torch.float32)
    low, high = model.policy.action_space.low, model.policy.action_space.high
    scaled_actions = 2.0 * (demo["actions"] - low) / (high - low) - 1.0
    actions = torch.as_tensor(scaled_actions, dtype=torch.float32)

    actor = model.policy.actor
    optimizer = torch.optim.Adam(actor.parameters(), lr=BC_LEARNING_RATE)

    n = obs.shape[0]
    print(f"Pretraining actor on {n} demonstration transitions for {BC_EPOCHS} epochs...")
    for epoch in range(BC_EPOCHS):
        epoch_start = time.time()
        permutation = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, BC_BATCH_SIZE):
            idx = permutation[start:start + BC_BATCH_SIZE]
            obs_batch = obs[idx]
            action_batch = actions[idx]

            mean_actions, _, _ = actor.get_action_dist_params(obs_batch)
            predicted = torch.tanh(mean_actions)
            loss = F.mse_loss(predicted, action_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        print(f"  epoch {epoch}: mse_loss={epoch_loss / n_batches:.5f} "
              f"({time.time() - epoch_start:.1f}s)")


if __name__ == "__main__":
    if not os.path.exists(DEMO_PATH):
        raise FileNotFoundError(
            f"{DEMO_PATH} not found -- run collect_demonstrations.py first."
        )
    demo = np.load(DEMO_PATH)
    print(f"Loaded {len(demo['obs'])} demonstration transitions from {DEMO_PATH}")

    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])

    # same hyperparameters as 5-arm_project_osc's proven config
    # (target_entropy=-1.0 for sustained exploration, device="cpu" since a
    # GPU doesn't help small MLPs on CPU-bound MuJoCo physics)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        target_entropy=-1.0,
        learning_starts=2_000,
        verbose=1,
        device="cpu",
    )

    seed_replay_buffer(model, demo)
    pretrain_actor(model, demo)

    print(f"Training started (OSC pick-and-place, BC-warm-started, {N_ENVS} parallel environments)...")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="sac_franka_osc_pick_place_bc_parallel",
    )

    # NOTE: this is a longer, more complex task than 5-arm_project_osc's
    # grasp-only one, which needed the full 1M steps (with BC warm-start)
    # to reach ep_len_mean~112 with reliable success. This may need more
    # than 1M steps to fully converge -- watch ep_len_mean/ep_rew_mean and
    # extend if it's still clearly improving near the end rather than
    # assuming 1M is automatically enough.
    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

    model.save("sac_franka_osc_pick_place_bc_parallel")
    print("Model saved to sac_franka_osc_pick_place_bc_parallel.zip")

    env.close()
