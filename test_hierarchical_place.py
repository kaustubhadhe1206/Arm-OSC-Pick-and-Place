import sys
import os
import mujoco

# mujoco.viewer provides the interactive 3D window to visualize the simulation
import mujoco.viewer

import numpy as np
import time

# add envs/ folder to path so we can import FrankaOSCPickPlaceEnv
sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))

from franka_osc_pick_place_env import FrankaOSCPickPlaceEnv
from stable_baselines3 import SAC

# Note: this drives the INNER FrankaOSCPickPlaceEnv directly and manually
# switches between the frozen pick model and the trained place model based
# on env.has_grasped_stably, rather than using HierarchicalPickPlaceEnv's
# reset() (which auto-pilots the ENTIRE pick phase internally before ever
# returning, so a viewer watching that would jump straight from "at home"
# to "already holding the object" with nothing shown in between — fine for
# training, useless for watching what's actually happening).

PICK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pretrained", "sac_franka_grasp_frozen.zip")

# pick which PLACE model to load: pass a path as a command-line argument,
# e.g. "python test_hierarchical_place.py checkpoints/sac_franka_hierarchical_place_bc_parallel_150000_steps"
# defaults to the final saved model if no argument is given. The pick
# model is always the frozen 5-arm_project_osc checkpoint -- it isn't
# being trained, so there's nothing to select a checkpoint of.
place_model_path = sys.argv[1] if len(sys.argv) > 1 else "sac_franka_hierarchical_place_bc_parallel"
print(f"Loading frozen pick model from: {PICK_MODEL_PATH}")
print(f"Loading place model from: {place_model_path}")
print("Press SPACE (with the viewer window focused) to pause/resume mid-episode.")

env = FrankaOSCPickPlaceEnv()

paused = {"value": False}
GLFW_KEY_SPACE = 32


def key_callback(keycode):
    if keycode == GLFW_KEY_SPACE:
        paused["value"] = not paused["value"]
        print("Paused" if paused["value"] else "Resumed")


def draw_zones(viewer, env):
    """See test_osc_pick_place.py's version — identical, duplicated here
    for the same self-containment reason as the training scripts."""
    geoms = []

    x_min, x_max = env.object_spawn_x_range
    y_min, y_max = env.object_spawn_y_range
    pick_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, 0.001])
    pick_half_size = np.array([(x_max - x_min) / 2, (y_max - y_min) / 2, 0.0005])
    geoms.append((mujoco.mjtGeom.mjGEOM_BOX, pick_half_size, pick_center, [0.2, 1.0, 0.2, 0.35]))

    px_min, px_max = env.place_target_x_range
    py_min, py_max = env.place_target_y_range
    place_center = np.array([(px_min + px_max) / 2, (py_min + py_max) / 2, 0.001])
    place_half_size = np.array([(px_max - px_min) / 2, (py_max - py_min) / 2, 0.0005])
    geoms.append((mujoco.mjtGeom.mjGEOM_BOX, place_half_size, place_center, [0.2, 0.4, 1.0, 0.35]))

    target_marker_pos = env.place_target.copy()
    target_marker_pos[2] = 0.002
    geoms.append((mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([env.place_radius, 0.001, 0.0]),
                  target_marker_pos, [1.0, 1.0, 0.0, 0.6]))

    viewer.user_scn.ngeom = len(geoms)
    for i, (geom_type, size, pos, rgba) in enumerate(geoms):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=geom_type,
            size=size,
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=np.array(rgba, dtype=np.float32),
        )


pick_model = SAC.load(PICK_MODEL_PATH, device="cpu")
place_model = SAC.load(place_model_path, env=env)

with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:

    obs, info = env.reset()
    draw_zones(viewer, env)

    while viewer.is_running():

        if paused["value"]:
            viewer.sync()
            time.sleep(0.01)
            continue

        if not env.has_grasped_stably:
            action, _ = pick_model.predict(obs[:16], deterministic=True)
        else:
            action, _ = place_model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        viewer.sync()
        time.sleep(0.01)

        if terminated or truncated:
            outcome = "SUCCESS" if terminated else "TIMEOUT"
            print(f"Episode ended: {outcome} (distance_to_object={info['distance_to_object']:.3f}m, "
                  f"has_grasped_stably={info['has_grasped_stably']}, "
                  f"grasp_hold={info['grasp_hold']}, place_hold={info['place_hold']})")
            obs, info = env.reset()
            draw_zones(viewer, env)

env.close()
