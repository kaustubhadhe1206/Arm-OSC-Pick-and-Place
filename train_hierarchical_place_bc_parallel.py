# Trains ONLY the carry-and-place sub-task, using HierarchicalPickPlaceEnv
# to auto-pilot the pick portion via a frozen, already-100%-reliable model
# (5-arm_project_osc's grasp-only checkpoint). See
# envs/hierarchical_pick_place_env.py's module docstring and IMP_NOTES.md
# incidents #3-#5 for the full reasoning: training one policy end-to-end
# for the full pick-and-place sequence kept forgetting/diluting the pick
# skill (even with ongoing BC regularization) because the rare, decisive
# pick-commit transitions were a small minority of a mostly-easy
# demonstration dataset. This sidesteps that entirely -- pick is not
# learned here at all, only run.
#
# Run collect_place_demonstrations.py first to produce
# place_demonstrations.npz.
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

from hierarchical_pick_place_env import HierarchicalPickPlaceEnv
from bc_regularized_sac import BCRegularizedSAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

N_ENVS = 4
DEMO_PATH = os.path.join(os.path.dirname(__file__), "place_demonstrations.npz")
PICK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pretrained", "sac_franka_grasp_frozen.zip")
BC_EPOCHS = 50
BC_BATCH_SIZE = 256
BC_LEARNING_RATE = 1e-3


def make_env():
    return Monitor(HierarchicalPickPlaceEnv(PICK_MODEL_PATH))


def seed_replay_buffer(model, demo):
    """Same logic as train_osc_pick_place_bc_parallel.py's version --
    duplicated here rather than imported, per this project's standing
    convention of keeping every training-variant file fully self-contained
    (no cross-file dependency that could let editing one script silently
    change another's behavior). See that file's version of this function
    for the full reasoning (verified against SB3 source, not assumed):
    buffer stores NORMALIZED actions, internal array shapes are
    (buffer_size, n_envs, dim) with buffer_size already divided by n_envs,
    vectorized bulk write instead of thousands of individual .add() calls."""
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
    """Same logic as train_osc_pick_place_bc_parallel.py's version --
    duplicated here for the same self-containment reason as
    seed_replay_buffer above."""
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
            f"{DEMO_PATH} not found -- run collect_place_demonstrations.py first."
        )
    demo = np.load(DEMO_PATH)
    print(f"Loaded {len(demo['obs'])} place-only demonstration transitions from {DEMO_PATH}")

    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])

    # same hyperparameters as 5-arm_project_osc's proven config
    model = BCRegularizedSAC(
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
    model.set_bc_demo_data(demo["obs"], demo["actions"])

    print(f"Training started (hierarchical place-only, {N_ENVS} parallel environments)...")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="sac_franka_hierarchical_place_bc_parallel",
    )

    # This sub-task is much shorter/simpler than the full sequence (no
    # pick to learn at all), so it may well converge well before 1M steps
    # -- watch ep_len_mean/ep_rew_mean and stop early if it plateaus at a
    # good success rate rather than assuming the full budget is needed.
    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

    model.save("sac_franka_hierarchical_place_bc_parallel")
    print("Model saved to sac_franka_hierarchical_place_bc_parallel.zip")

    env.close()
