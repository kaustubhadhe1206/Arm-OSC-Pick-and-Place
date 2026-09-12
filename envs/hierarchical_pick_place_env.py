# Wraps FrankaOSCPickPlaceEnv so only the CARRY-AND-PLACE portion of each
# episode is exposed to an outer RL trainer. The PICK portion (reach ->
# grasp -> lift -> hold) is auto-piloted internally by a FROZEN,
# already-proven policy -- 5-arm_project_osc's grasp-only model, which
# achieves reliable (user-confirmed 100% in testing) success on exactly
# this pick zone.
#
# Why this exists (see IMP_NOTES.md incidents #3-#4): training ONE policy
# end-to-end for the full pick-and-place sequence kept forgetting/diluting
# the pick skill, even with ongoing BC regularization -- the rare,
# decisive pick-commit transitions are a small minority of a mostly-easy
# demonstration dataset, so average BC loss looked great while the actual
# deployed behavior still failed to commit to a grasp. This sidesteps the
# problem entirely: pick is not being learned here at all, only RUN. The
# outer trainer only ever sees the genuinely new sub-task (carry, release,
# settle), starting from a state where the object is already reliably
# held.
#
# This works because pick-and-place's observation was deliberately
# designed as the grasp-only 16-dim observation with 6 new dims appended
# (place_target + object_to_place_target) -- verified directly (not
# assumed) to be an exact match, feature-for-feature and in the same
# order, before wiring this up. So the frozen model can be fed
# obs[:16] with no adaptation at all.

import gymnasium
import numpy as np

from franka_osc_pick_place_env import FrankaOSCPickPlaceEnv
from stable_baselines3 import SAC


class HierarchicalPickPlaceEnv(gymnasium.Env):
    def __init__(self, pick_model_path, use_camera=False, max_pick_attempts=5):
        super().__init__()
        self.inner_env = FrankaOSCPickPlaceEnv(use_camera=use_camera)
        # loaded once per env instance; used only for .predict(), never
        # trained further -- genuinely frozen. device="cpu" explicit (not
        # "auto") since this loads inside each SubprocVecEnv worker
        # process during parallel training -- 4-arm_project incident #27
        # documented a CUBLAS crash from multiple spawned processes each
        # defaulting to a GPU device.
        self.pick_model = SAC.load(pick_model_path, device="cpu")
        self.max_pick_attempts = max_pick_attempts

        self.observation_space = self.inner_env.observation_space
        self.action_space = self.inner_env.action_space

    def _run_pick_phase(self, seed=None):
        """Resets the inner env and auto-pilots it via the frozen pick
        model until has_grasped_stably is True, retrying fresh resets up
        to max_pick_attempts times if the frozen model's own (already very
        high, not literally 100%) success rate happens to miss on a
        particular spawn. Pick FAILURES stay invisible to the outer
        trainer this way -- it never needs to reason about pick failing,
        only about what to do once the object is already reliably held."""
        obs, info = None, None
        for attempt in range(self.max_pick_attempts):
            reset_seed = seed if attempt == 0 else None
            obs, info = self.inner_env.reset(seed=reset_seed)
            while not self.inner_env.has_grasped_stably:
                pick_obs = obs[:16]
                action, _ = self.pick_model.predict(pick_obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.inner_env.step(action)
                if terminated or truncated:
                    break
            if self.inner_env.has_grasped_stably:
                return obs, info
        # extremely unlikely given the frozen model's reliability, but
        # fall back to whatever the last attempt left rather than crash --
        # the outer trainer just sees one unusually short, low-reward
        # episode if this ever happens
        return obs, info

    def reset(self, seed=None, options=None):
        obs, info = self._run_pick_phase(seed=seed)
        return obs, info

    def step(self, action):
        return self.inner_env.step(action)

    def render(self):
        pass

    def close(self):
        self.inner_env.close()
