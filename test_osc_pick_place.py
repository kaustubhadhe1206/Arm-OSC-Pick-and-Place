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

# pick which model to load: pass a path as a command-line argument to test a
# specific checkpoint, e.g.
# "python test_osc_pick_place.py checkpoints/sac_franka_osc_pick_place_bc_parallel_150000_steps"
# defaults to the final saved model if no argument is given
model_path = sys.argv[1] if len(sys.argv) > 1 else "sac_franka_osc_pick_place_bc_parallel"
print(f"Loading model from: {model_path}")
print("Press SPACE (with the viewer window focused) to pause/resume mid-episode.")

env = FrankaOSCPickPlaceEnv()

# the viewer's own "Pause" button in the GUI panel does NOT actually pause
# anything here — this script drives physics stepping itself, not the
# viewer, so that button has nothing to hook into. key_callback wires up a
# real pause instead.
paused = {"value": False}
GLFW_KEY_SPACE = 32


def key_callback(keycode):
    if keycode == GLFW_KEY_SPACE:
        paused["value"] = not paused["value"]
        print("Paused" if paused["value"] else "Resumed")


def draw_zones(viewer, env):
    """Overlays flat, semi-transparent markers for the pick zone (green),
    the place zone (blue), and the exact randomized place target within it
    (a small yellow marker) — all via MuJoCo's "user scene" decoration
    layer (no physical geoms, no collision), drawn directly from the same
    numbers the env actually samples from so they can't drift out of sync
    with a later change to those ranges."""
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


model = SAC.load(model_path, env=env)

with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:

    obs, info = env.reset()
    draw_zones(viewer, env)

    while viewer.is_running():

        if paused["value"]:
            viewer.sync()
            time.sleep(0.01)
            continue

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        viewer.sync()

        # env.step() already advances physics by n_substeps (20 * 2ms = 40ms),
        # so no extra sleep is needed to keep the animation at a watchable pace
        # — sleeping the full 40ms here on top would make it needlessly slow.
        time.sleep(0.01)

        if terminated or truncated:
            outcome = "SUCCESS" if terminated else "TIMEOUT"
            print(f"Episode ended: {outcome} (distance_to_object={info['distance_to_object']:.3f}m, "
                  f"has_grasped_stably={info['has_grasped_stably']}, "
                  f"grasp_hold={info['grasp_hold']}, place_hold={info['place_hold']})")
            obs, info = env.reset()
            draw_zones(viewer, env)

env.close()
