# Hand-scripted pick-and-place routine — NO RL, no policy. Extends
# 5-arm_project_osc's scripted_grasp_check.py (move above -> descend ->
# close -> lift -> hold) with a new second half (carry -> descend -> open
# -> settle) to reach the place zone. Same purpose as that script: decisively
# confirm the environment/reward/success-condition are achievable BEFORE
# trusting any training run's failures to mean "the task is impossible" vs
# "RL hasn't found it yet".
#
# Run: venv/Scripts/python.exe scripted_pick_place_check.py

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from franka_osc_pick_place_env import FrankaOSCPickPlaceEnv


def move_to(env, target_xyz, target_gripper, max_steps, tol=0.01):
    """See 5-arm_project_osc/scripted_grasp_check.py for the full
    reasoning — early-exit MUST check the actual tip position, not
    env.target_pos (a raw accumulator that reaches the waypoint almost
    instantly, well before the real arm catches up)."""
    result = None
    for _ in range(max_steps):
        current_pos = env.target_pos
        current_gripper = env.data.qpos[env.gripper_qpos_addr]
        delta_pos = np.clip(target_xyz - current_pos, -env.max_delta_pos, env.max_delta_pos)
        delta_gripper = np.clip(target_gripper - current_gripper, -env.max_delta_gripper, env.max_delta_gripper)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, delta_gripper], dtype=np.float32)
        result = env.step(action)
        _, _, terminated, truncated, _ = result
        if terminated or truncated:
            return result
        tip_pos = env.data.site_xpos[env.tip_id]
        if np.linalg.norm(target_xyz - tip_pos) < tol:
            break
    return result


def move_gripper(env, hold_pos, target_gripper, max_steps):
    """Moves the gripper toward target_gripper while holding position at
    hold_pos, for up to max_steps or until the gripper's OWN convergence
    stalls — deliberately does not reuse move_to()'s position-based early
    exit, which (per scripted_grasp_check.py's incident) fires almost
    immediately once position is already converged, starving the gripper
    of steps. Used for both closing (target_gripper=0.0) and opening
    (target_gripper=env.gripper_open_max)."""
    result = None
    prev_gripper_pos = None
    stall_count = 0
    for _ in range(max_steps):
        current_pos = env.target_pos
        current_gripper = env.data.qpos[env.gripper_qpos_addr]
        delta_pos = np.clip(hold_pos - current_pos, -env.max_delta_pos, env.max_delta_pos)
        delta_gripper = np.clip(target_gripper - current_gripper, -env.max_delta_gripper, env.max_delta_gripper)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, delta_gripper], dtype=np.float32)
        result = env.step(action)
        _, _, terminated, truncated, _ = result
        if terminated or truncated:
            return result
        new_gripper_pos = env.data.qpos[env.gripper_qpos_addr]
        if prev_gripper_pos is not None and abs(new_gripper_pos - prev_gripper_pos) < 1e-4:
            stall_count += 1
            if stall_count >= 5:
                break
        else:
            stall_count = 0
        prev_gripper_pos = new_gripper_pos
    return result


def main():
    env = FrankaOSCPickPlaceEnv(use_camera=False)
    obs, info = env.reset()

    object_pos = env.data.xpos[env.object_id].copy()
    place_target = env.place_target.copy()
    print(f"Object spawned at: {np.round(object_pos, 4)}")
    print(f"Place target at:   {np.round(place_target, 4)}")

    HOVER_HEIGHT = 0.15
    GRASP_HEIGHT = object_pos[2] + 0.01  # see scripted_grasp_check.py for why not exact center
    LIFT_HEIGHT = 0.20

    print("Phase 1: move above object, gripper open")
    above_object = np.array([object_pos[0], object_pos[1], object_pos[2] + HOVER_HEIGHT])
    move_to(env, above_object, env.gripper_open_max, max_steps=100)

    print("Phase 2: descend to grasp height")
    grasp_pos = np.array([object_pos[0], object_pos[1], GRASP_HEIGHT])
    move_to(env, grasp_pos, env.gripper_open_max, max_steps=40)

    print("Phase 3: close gripper")
    move_gripper(env, grasp_pos, 0.0, max_steps=40)
    gripper_pos_now = env.data.qpos[env.gripper_qpos_addr]
    print(f"  gripper pos: {gripper_pos_now:.4f} (0=closed, {env.gripper_open_max}=open)")

    print("Phase 4: lift")
    lift_pos = np.array([object_pos[0], object_pos[1], LIFT_HEIGHT])
    result = move_to(env, lift_pos, 0.0, max_steps=80)
    _, _, terminated, truncated, info = result

    if not info["has_grasped_stably"] and not (terminated or truncated):
        # A LITERAL STATIC HOLD (zero action) here was found to destabilize
        # the grip within ~8-40 steps even with a firm grasp, regardless of
        # grasp height or gripper squeeze strategy — but continuous ACTIVE
        # movement (as used throughout this whole routine) does not show
        # this instability at all (verified directly: the object stayed
        # gripped through an entire ~80-step, 0.65m lateral carry). This
        # was never exposed in 5-arm_project_osc's pick-only task because
        # that task TERMINATES the instant grasp_hold reaches the
        # threshold (usually mid-lift, under active correction) and never
        # needed to survive a subsequent static phase. So: if the
        # threshold isn't reached yet, keep moving (a small extra ascent)
        # rather than switching to a static hold to "wait it out".
        higher_pos = lift_pos + np.array([0.0, 0.0, 0.03])
        result = move_to(env, higher_pos, 0.0, max_steps=30)
        _, _, terminated, truncated, info = result

    height = env.data.xpos[env.object_id][2] - env.object_floor_z
    print(f"  object height above floor: {height:.4f}m, grasp_hold={info['grasp_hold']}, "
          f"has_grasped_stably={info['has_grasped_stably']}, "
          f"terminated={terminated}, truncated={truncated}")

    if terminated or truncated or not info["has_grasped_stably"]:
        print("RESULT: FAILED (pick phase never stabilized)")
        env.close()
        return

    print("Phase 5: carry — move laterally above the place target while still lifted")
    above_place = np.array([place_target[0], place_target[1], LIFT_HEIGHT])
    result = move_to(env, above_place, 0.0, max_steps=150)
    _, _, terminated, truncated, info = result
    print(f"  tip pos: {np.round(env.data.site_xpos[env.tip_id], 4)}, "
          f"object pos: {np.round(env.data.xpos[env.object_id], 4)}")
    if terminated or truncated:
        print("RESULT:", "SUCCESS (unexpected early termination during carry)" if terminated else "FAILED (truncated during carry)")
        env.close()
        return

    print("Phase 6: descend to place height")
    # release a little above the exact floor level so the object drops the
    # last few mm on its own rather than the gripper trying to descend
    # exactly to floor level (avoids fighting floor contact directly)
    place_release_height = env.object_floor_z + 0.02
    descend_pos = np.array([place_target[0], place_target[1], place_release_height])
    result = move_to(env, descend_pos, 0.0, max_steps=60)
    _, _, terminated, truncated, info = result
    if terminated or truncated:
        print("RESULT:", "SUCCESS (unexpected early termination during descend)" if terminated else "FAILED (truncated during descend)")
        env.close()
        return

    print("Phase 7: open gripper (release)")
    result = move_gripper(env, descend_pos, env.gripper_open_max, max_steps=40)
    _, _, terminated, truncated, info = result
    print(f"  gripper pos: {env.data.qpos[env.gripper_qpos_addr]:.4f}")
    if terminated or truncated:
        print("RESULT:", "SUCCESS (unexpected early termination during release)" if terminated else "FAILED (truncated during release)")
        env.close()
        return

    print("Phase 8: retreat upward a bit and hold, to let the object settle undisturbed")
    retreat_pos = np.array([place_target[0], place_target[1], place_release_height + 0.1])
    result = move_to(env, retreat_pos, env.gripper_open_max, max_steps=40)
    _, _, terminated, truncated, info = result

    if not (terminated or truncated):
        remaining = env.max_episode_steps - env.current_step
        hold_steps = min(remaining, env.place_hold_required + 20)
        for i in range(hold_steps):
            action = np.zeros(5, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"  SUCCESS at settle step {i}! reward={reward:.3f}")
                break
            if truncated:
                break
        else:
            print(f"  did not terminate after {hold_steps} settle steps: "
                  f"place_hold={info['place_hold']}/{env.place_hold_required}")

    final_object_pos = env.data.xpos[env.object_id].copy()
    final_dist_to_target = np.linalg.norm(final_object_pos[:2] - place_target[:2])
    print(f"\nFinal object position: {np.round(final_object_pos, 4)}")
    print(f"Final distance to place target (xy): {final_dist_to_target:.4f}m "
          f"(need < {env.place_radius}m)")
    print("RESULT:", "SUCCESS" if terminated else "FAILED")

    env.close()


if __name__ == "__main__":
    main()
