# Collects successful CARRY-AND-PLACE-ONLY demonstrations for
# HierarchicalPickPlaceEnv. The pick portion is handled entirely by the
# frozen model inside the wrapper (reset() already returns the FIRST
# observation of the place phase, with the object already reliably
# grasped) -- this script only needs to script the carry/descend/
# release/settle sequence, not the whole pick-and-place task.
#
# Run: venv/Scripts/python.exe collect_place_demonstrations.py [n_successes]

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from hierarchical_pick_place_env import HierarchicalPickPlaceEnv

PICK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pretrained", "sac_franka_grasp_frozen.zip")


def move_to(env, target_xyz, target_gripper, max_steps, transitions, tol=0.01):
    inner = env.inner_env
    for _ in range(max_steps):
        obs_before = inner._get_obs()
        current_pos = inner.target_pos
        current_gripper = inner.data.qpos[inner.gripper_qpos_addr]
        delta_pos = np.clip(target_xyz - current_pos, -inner.max_delta_pos, inner.max_delta_pos)
        delta_gripper = np.clip(target_gripper - current_gripper, -inner.max_delta_gripper, inner.max_delta_gripper)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, delta_gripper], dtype=np.float32)
        obs_after, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs_before, action, reward, obs_after, terminated or truncated))
        if terminated or truncated:
            return terminated, truncated
        tip_pos = inner.data.site_xpos[inner.tip_id]
        if np.linalg.norm(target_xyz - tip_pos) < tol:
            break
    return False, False


def move_gripper(env, hold_pos, target_gripper, max_steps, transitions):
    inner = env.inner_env
    prev_gripper_pos = None
    stall_count = 0
    for _ in range(max_steps):
        obs_before = inner._get_obs()
        current_pos = inner.target_pos
        current_gripper = inner.data.qpos[inner.gripper_qpos_addr]
        delta_pos = np.clip(hold_pos - current_pos, -inner.max_delta_pos, inner.max_delta_pos)
        delta_gripper = np.clip(target_gripper - current_gripper, -inner.max_delta_gripper, inner.max_delta_gripper)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, delta_gripper], dtype=np.float32)
        obs_after, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs_before, action, reward, obs_after, terminated or truncated))
        if terminated or truncated:
            return terminated, truncated
        new_gripper_pos = inner.data.qpos[inner.gripper_qpos_addr]
        if prev_gripper_pos is not None and abs(new_gripper_pos - prev_gripper_pos) < 1e-4:
            stall_count += 1
            if stall_count >= 5:
                break
        else:
            stall_count = 0
        prev_gripper_pos = new_gripper_pos
    return False, False


def run_one_episode(env):
    """Returns (succeeded, transitions) for one scripted CARRY-AND-PLACE
    attempt, starting from env.reset() -- which has already auto-piloted
    a successful pick via the frozen model."""
    obs, info = env.reset()
    inner = env.inner_env
    transitions = []

    object_pos = inner.data.xpos[inner.object_id].copy()
    place_target = inner.place_target.copy()
    LIFT_HEIGHT = 0.20

    def done(terminated, truncated):
        return terminated or truncated or inner.current_step >= inner.max_episode_steps

    above_place = np.array([place_target[0], place_target[1], LIFT_HEIGHT])
    terminated, truncated = move_to(env, above_place, 0.0, 150, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    place_release_height = inner.object_floor_z + 0.02
    descend_pos = np.array([place_target[0], place_target[1], place_release_height])
    terminated, truncated = move_to(env, descend_pos, 0.0, 60, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    terminated, truncated = move_gripper(env, descend_pos, inner.gripper_open_max, 40, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    retreat_pos = np.array([place_target[0], place_target[1], place_release_height + 0.1])
    terminated, truncated = move_to(env, retreat_pos, inner.gripper_open_max, 40, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    remaining = inner.max_episode_steps - inner.current_step
    hold_steps = min(remaining, inner.place_hold_required + 20)
    for _ in range(hold_steps):
        obs_before = inner._get_obs()
        action = np.zeros(5, dtype=np.float32)
        obs_after, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs_before, action, reward, obs_after, terminated or truncated))
        if terminated:
            return True, transitions
        if truncated:
            break

    return False, transitions


def main():
    n_successes_target = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    env = HierarchicalPickPlaceEnv(PICK_MODEL_PATH)

    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    n_attempts = 0
    n_successes = 0

    while n_successes < n_successes_target:
        n_attempts += 1
        succeeded, transitions = run_one_episode(env)
        if succeeded:
            n_successes += 1
            for obs, action, reward, next_obs, dn in transitions:
                all_obs.append(obs)
                all_actions.append(action)
                all_rewards.append(reward)
                all_next_obs.append(next_obs)
                all_dones.append(dn)
        if n_attempts % 20 == 0:
            print(f"  attempts={n_attempts} successes={n_successes}/{n_successes_target} "
                  f"(success rate so far: {n_successes/n_attempts*100:.0f}%)")

    env.close()

    out_path = os.path.join(os.path.dirname(__file__), "place_demonstrations.npz")
    np.savez(
        out_path,
        obs=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        rewards=np.array(all_rewards, dtype=np.float32),
        next_obs=np.array(all_next_obs, dtype=np.float32),
        dones=np.array(all_dones, dtype=bool),
    )
    print(f"\nSaved {len(all_obs)} transitions from {n_successes} successful episodes "
          f"({n_attempts} attempts, {n_successes/n_attempts*100:.0f}% success rate) to {out_path}")


if __name__ == "__main__":
    main()
