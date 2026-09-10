# Collects successful pick-and-place demonstrations by running the SAME
# kind of hand-scripted (no RL) routine as scripted_pick_place_check.py
# across many episodes with randomized object/place positions, keeping
# only the FULL transition sequences (obs, action, reward, next_obs, done)
# from episodes that actually succeeded (terminated=True).
#
# Why: 5-arm_project_osc's pure RL ran 585k/1M steps without once finding a
# successful grasp-lift-hold, even though the task was provably achievable
# — a hard-exploration problem, not a reward-shaping one. Pick-and-place is
# a LONGER multi-stage sequence, so it's expected to be at least as hard to
# discover via pure exploration. This project defaults to the
# demonstration-driven behavior-cloning warm start from the start, rather
# than trying pure RL first (see IMP_NOTES.md).
#
# Run: venv/Scripts/python.exe collect_demonstrations.py [n_successes]

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from franka_osc_pick_place_env import FrankaOSCPickPlaceEnv


def move_to(env, target_xyz, target_gripper, max_steps, transitions, tol=0.01):
    for _ in range(max_steps):
        obs_before = env._get_obs()
        current_pos = env.target_pos
        current_gripper = env.data.qpos[env.gripper_qpos_addr]
        delta_pos = np.clip(target_xyz - current_pos, -env.max_delta_pos, env.max_delta_pos)
        delta_gripper = np.clip(target_gripper - current_gripper, -env.max_delta_gripper, env.max_delta_gripper)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, delta_gripper], dtype=np.float32)
        obs_after, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs_before, action, reward, obs_after, terminated or truncated))
        if terminated or truncated:
            return terminated, truncated
        tip_pos = env.data.site_xpos[env.tip_id]
        if np.linalg.norm(target_xyz - tip_pos) < tol:
            break
    return False, False


def move_gripper(env, hold_pos, target_gripper, max_steps, transitions):
    """See scripted_pick_place_check.py's move_gripper — tracks the
    gripper's OWN convergence (steady state, whether closed against the
    object or fully open) rather than reusing move_to()'s position-based
    early exit, which starves the gripper of steps once position is
    already converged."""
    prev_gripper_pos = None
    stall_count = 0
    for _ in range(max_steps):
        obs_before = env._get_obs()
        current_pos = env.target_pos
        current_gripper = env.data.qpos[env.gripper_qpos_addr]
        delta_pos = np.clip(hold_pos - current_pos, -env.max_delta_pos, env.max_delta_pos)
        delta_gripper = np.clip(target_gripper - current_gripper, -env.max_delta_gripper, env.max_delta_gripper)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, delta_gripper], dtype=np.float32)
        obs_after, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs_before, action, reward, obs_after, terminated or truncated))
        if terminated or truncated:
            return terminated, truncated
        new_gripper_pos = env.data.qpos[env.gripper_qpos_addr]
        if prev_gripper_pos is not None and abs(new_gripper_pos - prev_gripper_pos) < 1e-4:
            stall_count += 1
            if stall_count >= 5:
                break
        else:
            stall_count = 0
        prev_gripper_pos = new_gripper_pos
    return False, False


def run_one_episode(env):
    """Returns (succeeded, transitions) for one scripted pick-and-place
    attempt. See IMP_NOTES.md incident #1 for why the pick phase never
    switches to a literal static hold — that was found to destabilize an
    otherwise-successful grasp; every phase here keeps issuing real
    (non-frozen) gripper/position targets instead."""
    obs, info = env.reset()
    transitions = []

    object_pos = env.data.xpos[env.object_id].copy()
    place_target = env.place_target.copy()
    HOVER_HEIGHT = 0.15
    GRASP_HEIGHT = object_pos[2] + 0.01
    LIFT_HEIGHT = 0.20

    def done(terminated, truncated):
        return terminated or truncated or len(transitions) >= env.max_episode_steps

    above_object = np.array([object_pos[0], object_pos[1], object_pos[2] + HOVER_HEIGHT])
    terminated, truncated = move_to(env, above_object, env.gripper_open_max, 100, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    grasp_pos = np.array([object_pos[0], object_pos[1], GRASP_HEIGHT])
    terminated, truncated = move_to(env, grasp_pos, env.gripper_open_max, 40, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    terminated, truncated = move_gripper(env, grasp_pos, 0.0, 40, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    lift_pos = np.array([object_pos[0], object_pos[1], LIFT_HEIGHT])
    terminated, truncated = move_to(env, lift_pos, 0.0, 80, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    if not env.has_grasped_stably:
        higher_pos = lift_pos + np.array([0.0, 0.0, 0.03])
        terminated, truncated = move_to(env, higher_pos, 0.0, 30, transitions)
        if done(terminated, truncated):
            return terminated, transitions

    if not env.has_grasped_stably:
        return False, transitions

    above_place = np.array([place_target[0], place_target[1], LIFT_HEIGHT])
    terminated, truncated = move_to(env, above_place, 0.0, 150, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    place_release_height = env.object_floor_z + 0.02
    descend_pos = np.array([place_target[0], place_target[1], place_release_height])
    terminated, truncated = move_to(env, descend_pos, 0.0, 60, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    terminated, truncated = move_gripper(env, descend_pos, env.gripper_open_max, 40, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    retreat_pos = np.array([place_target[0], place_target[1], place_release_height + 0.1])
    terminated, truncated = move_to(env, retreat_pos, env.gripper_open_max, 40, transitions)
    if done(terminated, truncated):
        return terminated, transitions

    remaining = env.max_episode_steps - len(transitions)
    hold_steps = min(remaining, env.place_hold_required + 20)
    for _ in range(hold_steps):
        obs_before = env._get_obs()
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

    env = FrankaOSCPickPlaceEnv(use_camera=False)

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

    out_path = os.path.join(os.path.dirname(__file__), "demonstrations.npz")
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
