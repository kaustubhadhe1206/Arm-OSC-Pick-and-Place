# gymnasium is the standard library for creating RL environments
import gymnasium

# numpy is used for numerical operations like arrays and math
import numpy as np

# mujoco is the physics engine we use to simulate the robot
import mujoco

# os is used to build file paths that work on any operating system
import os

from osc_controller import DiffIKController, down_facing_quat


class FrankaOSCPickPlaceEnv(gymnasium.Env):
    """Phase 3: Pick and Place, extending 5-arm_project_osc's grasp-only
    task (reach -> grasp -> lift -> hold) with a second stage (carry ->
    release -> settle) that moves the object to a second fixed-box zone on
    the opposite side of the robot.

    Everything through "stable grasp achieved" reuses 5-arm_project_osc's
    proven design UNCHANGED: the same OSC controller (osc_controller.py,
    byte-for-byte copy), the same action space, the same reward-shaping
    lessons (IMP_NOTES.md there, incidents #9-#14 — gate dense bonuses
    behind the real threshold not any nonzero progress, penalize breaking
    an in-progress hold, penalize premature gripper actions with a
    distance-tapered term, weight the action-smoothness penalty more
    heavily on the gripper dimension). The carry/place stage is NEW and
    mirrors that same set of anti-exploit patterns rather than inventing
    different ones, since those specific patterns are what took incidents
    #9-#14 to discover the hard way the first time.
    """

    def __init__(self, use_camera=False, camera_size=128):
        super().__init__()

        xml_path = os.path.join(os.path.dirname(__file__), "panda", "panda_grasp.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.n_arm_joints = 7

        # Cartesian action design: policy commands a DELTA to a persistent
        # target pose (position + yaw) — unchanged from 5-arm_project_osc.
        self.max_delta_pos = 0.03    # meters per env step
        self.max_delta_yaw = 0.2     # radians per env step
        self.max_delta_gripper = 0.01  # meters per env step (range is 0-0.04)
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-self.max_delta_pos] * 3 + [-self.max_delta_yaw, -self.max_delta_gripper], dtype=np.float32),
            high=np.array([self.max_delta_pos] * 3 + [self.max_delta_yaw, self.max_delta_gripper], dtype=np.float32),
        )

        # workspace box the target position is clamped to — widened
        # slightly on the y axis vs 5-arm_project_osc since the place zone
        # now sits on the OPPOSITE side of the robot from the pick zone
        # (negative y instead of positive y — see object_spawn_y_range vs
        # place_target_y_range below).
        self.workspace_low = np.array([-0.6, -0.6, 0.02], dtype=np.float32)
        self.workspace_high = np.array([0.6, 0.6, 0.8], dtype=np.float32)

        self.gripper_open_max = 0.04
        self.gripper_to_ctrl_scale = 255.0 / self.gripper_open_max

        # observation: everything from 5-arm_project_osc's grasp env (tip
        # position, tip-to-object vector, object position, gripper
        # opening/velocity, current target pose) PLUS the place target
        # position and object-to-place-target vector, so the policy can
        # see where it needs to carry the object to. No explicit
        # "grasped stably yet" flag — the existing gripper/height/position
        # signals already make phase implicit, matching the observation
        # design philosophy from the grasp env (minimal, no redundant
        # engineered features).
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )

        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        object_jnt_id = self.model.body_jntadr[self.object_id]
        self.object_qpos_addr = self.model.jnt_qposadr[object_jnt_id]

        finger_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        self.gripper_qpos_addr = self.model.jnt_qposadr[finger_jnt_id]
        self.gripper_qvel_addr = self.model.jnt_dofadr[finger_jnt_id]

        self.home_qpos_arm = self.model.key_qpos[0][:9].copy()

        self.controller = DiffIKController(
            self.model, self.data, self.tip_id, home_qpos=self.home_qpos_arm[:self.n_arm_joints]
        )

        self.n_substeps = 20

        # PICK zone: identical bounds to 5-arm_project_osc's (proven safe
        # clearance from the base and within reach).
        self.object_spawn_x_range = (0.225, 0.525)
        self.object_spawn_y_range = (0.075, 0.375)
        self.object_floor_z = 0.015  # half the box's side length, resting flush on the floor

        # PLACE zone: same x range and size as the pick zone, on the
        # opposite side of the robot (negative y). Pushed further out than
        # a pure mirror (which would leave only a 0.15m gap between the two
        # zones' nearest edges) so the two zones are clearly, visibly
        # separated rather than nearly touching across the robot's
        # centerline — the nearest-edge gap is now 0.225m (0.075 to
        # -0.15), up from 0.15m. Worst-case pick-to-place distance (opposite
        # far corners) is now ~0.88m, still comfortably under the arm's
        # ~0.9m true max reach (checked directly, not assumed, since this
        # was the tightest constraint on how far the zones could be pushed
        # apart).
        self.place_target_x_range = (0.225, 0.525)
        self.place_target_y_range = (-0.45, -0.15)

        # radius within which the object counts as "at" the place target —
        # matches close_scale below, i.e. the same precision the proximity
        # shaping already aims for.
        self.place_radius = 0.05

        self.close_scale = 0.05

        # PICK-phase success verification: unchanged from 5-arm_project_osc
        # (incidents #24/#25 in 4-arm_project/IMP_NOTES.md established why
        # contact+closure alone isn't enough; lift-height is the
        # shape-agnostic force-closure proxy).
        self.lift_height_required = 0.05  # meters above the floor
        self.grasp_hold_required = 20
        self.grasp_hold_counter = 0

        # PLACE-phase success verification: mirrors the SAME
        # sustained-hold pattern, now for "settled in the place zone,
        # released" instead of "lifted off the floor" — a moving target
        # dropped and immediately re-grabbed shouldn't count any more than
        # a grasp that's immediately released should have.
        self.place_hold_required = 20
        self.place_hold_counter = 0

        # Latches True once the pick phase's hold requirement is met once,
        # for the rest of the episode — the reward function switches from
        # pick-shaping to place-shaping at that point. Does NOT revert to
        # False if the object is later dropped during carry (see step()'s
        # comment on the carry-phase drop penalty) — recovering from a
        # drop by re-grasping is allowed and still directed at the place
        # target, not treated as needing to restart the pick phase.
        self.has_grasped_stably = False

        # At n_substeps=20 (40ms/step), 700 steps = 28 simulated seconds.
        # Raised from 600 (24s) alongside widening the pick/place
        # separation above — the worst-case carry distance grew modestly
        # (~0.81m -> ~0.88m), so this adds proportional margin rather than
        # a specific "N seconds" target picked without reference to the
        # actual distances involved.
        self.max_episode_steps = 700
        self.current_step = 0

        self.use_camera = use_camera
        self.camera_size = camera_size
        self.camera_renderer = None
        if self.use_camera:
            self.camera_renderer = mujoco.Renderer(
                self.model, height=self.camera_size, width=self.camera_size
            )

    def get_camera_image(self):
        if self.camera_renderer is None:
            raise RuntimeError("Camera not enabled — construct FrankaOSCPickPlaceEnv(use_camera=True)")
        self.camera_renderer.update_scene(self.data, camera="scene_cam")
        return self.camera_renderer.render()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.grasp_hold_counter = 0
        self.place_hold_counter = 0
        self.has_grasped_stably = False

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = self.home_qpos_arm

        object_x = self.np_random.uniform(*self.object_spawn_x_range)
        object_y = self.np_random.uniform(*self.object_spawn_y_range)

        yaw = self.np_random.uniform(0.0, 2 * np.pi)
        quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

        self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3] = [object_x, object_y, self.object_floor_z]
        self.data.qpos[self.object_qpos_addr + 3:self.object_qpos_addr + 7] = quat

        mujoco.mj_forward(self.model, self.data)

        self.controller.reset(q_ref=self.home_qpos_arm[:self.n_arm_joints].copy())

        self.target_pos = self.data.site_xpos[self.tip_id].copy()
        self.target_yaw = 0.0

        self.prev_action_normalized = np.zeros(5, dtype=np.float32)

        # randomize the place target independently of the object's pick
        # position — same annulus-avoided-a-narrower-box philosophy as the
        # pick zone, just mirrored to the other side. z is the resting
        # height (object_floor_z), since "placed" means resting on the
        # floor there, not held at some height.
        place_x = self.np_random.uniform(*self.place_target_x_range)
        place_y = self.np_random.uniform(*self.place_target_y_range)
        self.place_target = np.array([place_x, place_y, self.object_floor_z], dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz, dyaw, dgripper = action

        # action-smoothness penalty — unchanged from 5-arm_project_osc
        # (incidents #12/#14: uniform weight was enough for the arm
        # dimensions, but the gripper needed 4x that to stop chattering
        # right at the moment of deciding to grasp; the same risk applies
        # symmetrically at the moment of deciding to release, so the same
        # weighting is used here for the whole episode, not just the pick
        # phase).
        smoothness_weights = np.array([1.0, 1.0, 1.0, 1.0, 4.0])
        normalized_action = action / self.action_space.high
        weighted_diff = smoothness_weights * (normalized_action - self.prev_action_normalized)
        action_change = np.linalg.norm(weighted_diff)
        action_smoothness_penalty = 0.05 * action_change
        self.prev_action_normalized = normalized_action

        self.target_pos = np.clip(
            self.target_pos + np.array([dx, dy, dz], dtype=np.float32),
            self.workspace_low, self.workspace_high,
        )
        self.target_yaw = (self.target_yaw + dyaw + np.pi) % (2 * np.pi) - np.pi
        desired_quat = down_facing_quat(self.target_yaw)

        current_gripper_pos = self.data.qpos[self.gripper_qpos_addr]
        target_gripper_pos = np.clip(current_gripper_pos + dgripper, 0.0, self.gripper_open_max)
        gripper_ctrl = target_gripper_pos * self.gripper_to_ctrl_scale

        for _ in range(self.n_substeps):
            mujoco.mj_forward(self.model, self.data)
            self.data.ctrl[:self.n_arm_joints] = self.controller.solve(self.target_pos, desired_quat)
            self.data.ctrl[self.n_arm_joints] = gripper_ctrl
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        tip_pos = self.data.site_xpos[self.tip_id]
        object_pos = self.data.xpos[self.object_id]
        distance_to_object = np.linalg.norm(tip_pos - object_pos)

        reward = -action_smoothness_penalty

        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            bodies = {self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2]}
            if self.object_id in bodies and self.left_finger_id in bodies:
                left_contact = True
            if self.object_id in bodies and self.right_finger_id in bodies:
                right_contact = True
        both_fingers_touching = left_contact and right_contact

        gripper_pos_now = self.data.qpos[self.gripper_qpos_addr]
        gripper_closed_enough = gripper_pos_now < (self.gripper_open_max / 2)

        object_height_above_floor = object_pos[2] - self.object_floor_z
        lifted_enough = object_height_above_floor > self.lift_height_required
        is_grasping = both_fingers_touching and gripper_closed_enough and lifted_enough

        terminated = False

        if not self.has_grasped_stably:
            # ---- PICK PHASE: identical reward structure to
            # 5-arm_project_osc's franka_osc_grasp_env.py (incidents
            # #9-#11, #14) ----
            reward += -distance_to_object
            reward += 0.1 * np.exp(-distance_to_object / self.close_scale)

            gripper_closedness = 1.0 - (gripper_pos_now / self.gripper_open_max)
            distance_factor = min(distance_to_object / self.close_scale, 1.0)
            reward -= 0.05 * gripper_closedness * distance_factor

            if is_grasping:
                reward += 0.5 * object_height_above_floor
            elif both_fingers_touching and gripper_closed_enough:
                reward += 0.05 * max(0.0, object_height_above_floor)

            if is_grasping:
                reward += 0.5
                self.grasp_hold_counter += 1
                if self.grasp_hold_counter >= self.grasp_hold_required:
                    self.has_grasped_stably = True
                    reward += 5.0
            else:
                if self.grasp_hold_counter > 0:
                    reward -= 0.3 * (self.grasp_hold_counter / self.grasp_hold_required)
                self.grasp_hold_counter = 0
        else:
            # ---- PLACE PHASE (NEW): carry the object to self.place_target,
            # release it there, and let it settle — mirrors the pick
            # phase's anti-exploit patterns rather than inventing new ones.
            object_to_target = np.linalg.norm(object_pos[:2] - self.place_target[:2])

            reward += -object_to_target
            reward += 0.1 * np.exp(-object_to_target / self.close_scale)

            is_still_holding = both_fingers_touching and gripper_closed_enough

            # Anti-premature-release: mirrors the pick phase's
            # anti-premature-CLOSE penalty (5-arm_project_osc incident
            # #10), applied to OPENING instead. Discourages releasing
            # while still far from the place target; tapers to 0 once
            # within close_scale of it, so it never penalizes the
            # necessary release-once-arrived action. Only applies while
            # actually holding the object — releasing something you were
            # never holding isn't a "premature release."
            if is_still_holding:
                gripper_openness = gripper_pos_now / self.gripper_open_max
                target_distance_factor = min(object_to_target / self.close_scale, 1.0)
                reward -= 0.05 * gripper_openness * target_distance_factor

            object_settled = (
                object_to_target < self.place_radius
                and object_height_above_floor < 0.01  # resting on the floor, not held
                and not both_fingers_touching  # genuinely released, not just hovering with contact
            )

            if object_settled:
                reward += 0.5
                self.place_hold_counter += 1
            else:
                # Penalize losing the object during carry (dropped
                # somewhere other than the place target) the same way the
                # pick phase penalizes breaking an in-progress hold — scaled
                # by how much settling progress was lost, so a drop right
                # before completion hurts more than dropping immediately.
                # Also covers a genuine drop mid-carry (is_still_holding
                # becomes False far from the target): that shows up here
                # too, since object_settled requires both proximity AND
                # release, not just release.
                if self.place_hold_counter > 0:
                    reward -= 0.3 * (self.place_hold_counter / self.place_hold_required)
                self.place_hold_counter = 0

            terminated = bool(self.place_hold_counter >= self.place_hold_required)
            if terminated:
                reward += 5.0

        # defensive clip — same bounds as 5-arm_project_osc
        # (4-arm_project/IMP_NOTES.md incident #26)
        reward = float(np.clip(reward, -2.0, 6.0))

        truncated = self.current_step >= self.max_episode_steps

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {
            "distance_to_object": float(distance_to_object),
            "grasp_hold": self.grasp_hold_counter,
            "has_grasped_stably": self.has_grasped_stably,
            "place_hold": self.place_hold_counter,
        }

    def _get_obs(self):
        tip_pos = self.data.site_xpos[self.tip_id].copy()    # shape (3,)
        object_pos = self.data.xpos[self.object_id].copy()   # shape (3,)
        tip_to_object = object_pos - tip_pos                 # shape (3,)

        gripper_pos = self.data.qpos[self.gripper_qpos_addr:self.gripper_qpos_addr + 1].copy()  # shape (1,)
        gripper_vel = self.data.qvel[self.gripper_qvel_addr:self.gripper_qvel_addr + 1].copy()  # shape (1,)

        target_yaw_trig = np.array([np.cos(self.target_yaw), np.sin(self.target_yaw)], dtype=np.float32)

        object_to_place_target = self.place_target - object_pos  # shape (3,)

        # result shape = (3)+(3)+(3)+(1)+(1)+(3)+(2)+(3)+(3) = 22
        return np.concatenate([
            tip_pos, tip_to_object, object_pos, gripper_pos, gripper_vel,
            self.target_pos.astype(np.float32), target_yaw_trig,
            self.place_target, object_to_place_target,
        ]).astype(np.float32)

    def render(self):
        pass

    def close(self):
        if self.camera_renderer is not None:
            self.camera_renderer.close()
