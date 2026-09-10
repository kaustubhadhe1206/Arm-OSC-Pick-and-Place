# Differential inverse-kinematics ("resolved-rate") controller.
#
# This is NOT full dynamic Operational Space Control (no mass-matrix
# decoupling, no torque-level control). It is the standard Jacobian
# damped-least-squares technique: each call computes the joint-angle delta
# that best reduces the Cartesian pose error between the current TCP pose
# and a desired TCP pose. That delta is integrated onto a PERSISTENT
# internal joint-angle reference (not simply added to the arm's current
# measured qpos — see solve()'s docstring for why that naive version
# doesn't work), which is then fed into the Panda's existing
# position-controlled actuators (the same "general" actuators with
# gainprm/biasprm already in panda.xml) — no new actuator logic needed. A
# nullspace secondary objective and a cap on how far the reference may lead
# the actual arm round out the design — see __init__'s comments for why
# each piece is there; all three were needed simultaneously to get the
# standalone convergence test (test_osc_controller_standalone.py) passing.
#
# This lets an RL policy command "move the gripper this far in x/y/z and
# rotate its yaw by this much" instead of raw per-joint deltas, offloading
# joint coordination onto this controller instead of onto the policy.

import numpy as np
import mujoco


class DiffIKController:
    def __init__(self, model, data, site_id, n_arm_joints=7,
                 position_gain=1.0, orientation_gain=1.0,
                 damping=1e-4, max_joint_delta=0.01,
                 home_qpos=None, nullspace_gain=0.5, max_ref_lag=0.06):
        self.model = model
        self.data = data
        self.site_id = site_id
        self.n_arm_joints = n_arm_joints
        self.position_gain = position_gain
        self.orientation_gain = orientation_gain
        # Secondary "nullspace" objective: a bare Jacobian pseudo-inverse
        # picks the MINIMUM-NORM joint velocity satisfying the 6D task each
        # step, but that's a purely local/instantaneous criterion with no
        # regard for where it leaves the redundant 7th DOF over time. This
        # was found to matter in practice, not just in theory: the
        # standalone controller test converged the task-space pose exactly
        # but settled on a posture where joint7 needed to sustain more
        # gravity-holding torque than its actuator's forcerange (+-12 N*m)
        # allows, causing a persistent ~0.11 rad/s saturated-PD hunting
        # oscillation confined to that one joint (confirmed via qvel — every
        # other joint, including both fingers, sat at ~0). Pulling the
        # redundant DOF toward a known-comfortable home posture (projected
        # through the nullspace so it never fights the primary 6D task)
        # avoids this. This is standard practice for redundant-manipulator
        # IK/OSC, not a workaround specific to this bug.
        self.home_qpos = home_qpos if home_qpos is not None else np.zeros(n_arm_joints)
        self.nullspace_gain = nullspace_gain
        self.damping = damping
        # This is a per-PHYSICS-STEP delta, not a per-control-step one. With
        # panda.xml's timestep=0.002s, max_joint_delta=0.01 rad/step is a
        # commanded rate of ~5 rad/s — already above the real Panda's actual
        # joint speed limits (~2.1-2.6 rad/s), kept slightly generous since
        # the position actuators' own PD dynamics add further smoothing.
        # A larger value (0.2, i.e. ~100 rad/s) was tried first and caused
        # the standalone test to converge partway then diverge: at that
        # rate the per-step motion is too large for the linearized Jacobian
        # to stay valid, so it overshoots, relinearizes on a bad point, and
        # spirals away instead of settling.
        self.max_joint_delta = max_joint_delta
        # Cap on how far the internal reference (see below) is allowed to
        # run ahead of the arm's ACTUAL measured qpos. Needed because a
        # persistent, freely-integrating reference (tried first, without
        # this cap) could run away UNBOUNDED once the actual arm's response
        # bandwidth fell behind the commanded advance rate — confirmed
        # empirically: q_ref drifted up to ~1.6 rad away from actual qpos,
        # and task-space error spiked into the hundreds of mm because the
        # Jacobian/task-error was (correctly) computed from where the arm
        # REALLY is, while corrections kept compounding onto a reference
        # that actual had no hope of catching up to. Clamping the lag keeps
        # enough lead to build real PD restoring torque (needed — see
        # solve()'s docstring for why zero lag has the opposite problem)
        # while guaranteeing the reference can never run away from reality.
        #
        # Raised from the original 0.025 to 0.06 after a hand-scripted
        # grasp sanity check (scripted_grasp_check.py) revealed the ARM
        # barely moved within a full training episode's step budget: at
        # 0.025, settling a ~0.5m move (home to the object) took ~6.3
        # real seconds — most of a 300-step (12s) episode consumed by
        # approach alone, before grasping could even begin. Every prior
        # training run may have been budget-starved on approach, not just
        # struggling with reward shaping. Swept 0.025 to 0.15 directly: no
        # oscillation/instability reappeared anywhere in that range (unlike
        # the earlier per-element-clipping and frame bugs, which caused
        # genuine unbounded or oscillating divergence) — residual tracking
        # error grows gracefully with lag (2.7mm at 0.025 to ~14mm at
        # 0.15), so this is a smooth speed/precision trade, not a
        # stability cliff. 0.06 settles the same ~0.5m move in ~2.6s
        # (2.4x faster) while keeping steady-state error under ~6.5mm —
        # comfortably precise enough for a 3cm object and 5cm lift
        # threshold.
        self.max_ref_lag = max_ref_lag

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        # Persistent position REFERENCE, integrated across calls — see
        # reset()'s comment for why this can't just be "current actual qpos
        # + delta" recomputed fresh each call (that scheme was tried first
        # and failed a real convergence test: task-space error grew
        # unboundedly instead of converging, confirmed to be caused by
        # exactly this).
        self.q_ref = None

    def reset(self, q_ref=None):
        """Must be called once before the first solve() (e.g. at episode
        reset), and again whenever the arm's qpos is teleported/reset
        outside the controller's control (since q_ref would otherwise be
        stale relative to the actual arm)."""
        self.q_ref = np.array(q_ref) if q_ref is not None else self.data.qpos[:self.n_arm_joints].copy()

    def solve(self, desired_pos, desired_quat):
        """Returns a new joint-angle REFERENCE (length n_arm_joints) to send
        to the position-controlled actuators, one step closer to achieving
        (desired_pos, desired_quat). Assumes the site's data (site_xpos/
        site_xmat) is already up to date for the current ACTUAL qpos (i.e.
        mj_forward or mj_step has already been called) — used to compute
        the Jacobian and task error, so the correction direction reflects
        where the arm really is right now.

        Critically, the correction is NOT added to the actual qpos to form
        the new reference (i.e. NOT `ctrl = actual_qpos + delta`) — it's
        added to self.q_ref, an internal reference that persists and
        integrates across calls independently of the actual (possibly
        lagging/disturbed) qpos. This distinction was the difference between
        the controller working and not: with "ctrl = actual_qpos + delta",
        if a disturbance (gravity) drags the actual joint away from where it
        should be, the next delta is computed as a small, local
        task-reducing correction and added to that SAME dragged position —
        so the reference always stays close to wherever the arm actually is,
        and the resulting PD tracking error (ctrl - actual) can never grow
        large enough to generate the restoring torque needed to fight the
        disturbance. Confirmed directly: in that scheme, one joint's
        REAL per-step motion stayed a fixed-sign drift for thousands of
        steps while the COMMANDED delta for that same joint flipped sign
        repeatedly — the actual motion was governed by the disturbance, not
        by the command. Advancing an independent reference lets the
        reference keep moving toward the goal regardless of how far actual
        has lagged, so PD error (and thus corrective torque) grows exactly
        as much as needed. This is the standard architecture for
        resolved-rate/differential-IK controllers — the reference is
        integrated state, not a function of instantaneous measurement."""
        if self.q_ref is None:
            self.reset()
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        Jp = self._jacp[:, :self.n_arm_joints]
        Jr = self._jacr[:, :self.n_arm_joints]
        J = np.vstack([Jp, Jr])

        current_pos = self.data.site_xpos[self.site_id].copy()
        current_mat = self.data.site_xmat[self.site_id].copy()
        current_quat = np.zeros(4)
        mujoco.mju_mat2Quat(current_quat, current_mat)

        pos_error = desired_pos - current_pos

        # Quaternion double-cover fix: q and -q represent the identical
        # rotation, but mju_subQuat's internal angle = 2*acos(qdif[0]) lands
        # in [0, 2*pi] rather than [-pi, pi] — if desired_quat and
        # current_quat happen to be in opposite hemispheres (dot < 0) it
        # computes the ~360-degree-minus-true-angle "long way around"
        # instead of the shortest rotation. Flipping desired_quat's sign
        # (a no-op on the rotation it represents) brings both into the same
        # hemisphere and fixes this. Empirically confirmed: without this,
        # standalone verification showed ~180 degree orientation error on
        # every target despite correct target quaternions.
        if np.dot(desired_quat, current_quat) < 0:
            desired_quat = -desired_quat

        # mju_subQuat(qerr, qa, qb) gives the rotation from qb to qa, but
        # EXPRESSED IN QB'S LOCAL FRAME (confirmed empirically: Exp(qerr) ==
        # R(qb)^-1 @ R(qa), not R(qa) @ R(qb)^-1). mj_jacSite's rotational
        # Jacobian (Jr), however, maps joint velocities to angular velocity
        # in the WORLD frame. Feeding subQuat's body-frame vector straight
        # into a world-frame Jacobian equation is a frame mismatch — for
        # small errors near identity orientation the two frames nearly
        # coincide so it looks fine, but at large orientation offsets (like
        # this arm's ~180 degree home orientation) it sends corrections in
        # the wrong direction. Confirmed by the standalone controller test:
        # this exact bug caused orientation error to grow monotonically
        # from 0 to 180 degrees instead of converging. Fix: rotate the
        # body-frame error into world frame via current_mat.
        ori_error_local = np.zeros(3)
        mujoco.mju_subQuat(ori_error_local, desired_quat, current_quat)
        ori_error = current_mat.reshape(3, 3) @ ori_error_local

        task_error = np.concatenate([
            self.position_gain * pos_error,
            self.orientation_gain * ori_error,
        ])

        JJt = J @ J.T
        damped_inv = np.linalg.inv(JJt + (self.damping ** 2) * np.eye(6))
        J_pinv = J.T @ damped_inv
        joint_delta = J_pinv @ task_error

        # Rate-limit the PRIMARY task term by scaling the whole vector, NOT
        # by clipping each joint independently. Per-element clipping
        # distorts the direction of the least-squares correction whenever
        # more than one joint saturates at once — confirmed empirically:
        # with per-element clipping, the standalone test's traced norm sat
        # pinned at sqrt(7)*max_joint_delta (i.e. every joint saturated) for
        # the entire run, and orientation error grew monotonically instead
        # of converging, because the resulting direction wasn't the true
        # correction direction anymore.
        norm = np.linalg.norm(joint_delta)
        if norm > self.max_joint_delta:
            joint_delta = joint_delta * (self.max_joint_delta / norm)

        # Nullspace secondary objective — see __init__'s comment. Project a
        # pull toward home_qpos through (I - J_pinv @ J) so it only acts in
        # directions that don't affect the primary 6D task, and add it AFTER
        # the primary term's rate limit rather than before: the primary
        # term's raw (pre-clip) magnitude is usually much larger than
        # max_joint_delta while far from the target, so lumping the two
        # together before a single rate-limit scale-down left the nullspace
        # term's contribution negligible (confirmed empirically — adding it
        # before clipping made no measurable difference to the standalone
        # test's results).
        null_projector = np.eye(self.n_arm_joints) - J_pinv @ J
        secondary = self.nullspace_gain * (self.home_qpos - self.q_ref)
        secondary = null_projector @ secondary

        # The nullspace term is only guaranteed task-neutral INSTANTANEOUSLY,
        # at the current configuration's exact linearization — as soon as
        # any finite step is taken, the configuration shifts and that same
        # direction is no longer exactly null, so real task-space drift
        # accumulates from it over many steps. This was confirmed
        # empirically: an unclipped secondary term (which can be much larger
        # in magnitude than the rate-limited primary term once the primary
        # task is nearly satisfied) caused position error to climb back up
        # after initially converging, even though it "should" have been
        # task-neutral at each individual instant. Capping its own norm to a
        # fraction of max_joint_delta keeps it a gentle bias, not a
        # dominant, drift-inducing motion.
        secondary_norm = np.linalg.norm(secondary)
        max_secondary = 0.5 * self.max_joint_delta
        if secondary_norm > max_secondary:
            secondary = secondary * (max_secondary / secondary_norm)

        joint_delta = joint_delta + secondary
        new_q_ref = self.q_ref + joint_delta

        # Clamp the reference's lag ahead of actual qpos — see __init__'s
        # comment on max_ref_lag.
        current_qpos = self.data.qpos[:self.n_arm_joints]
        lag = new_q_ref - current_qpos
        lag_norm = np.linalg.norm(lag)
        if lag_norm > self.max_ref_lag:
            lag = lag * (self.max_ref_lag / lag_norm)
        new_q_ref = current_qpos + lag

        self.q_ref = new_q_ref
        return self.q_ref


def down_facing_quat(yaw):
    """Quaternion for the gripper pointing straight down (-Z world) with a
    variable yaw rotation about the world Z axis on top. Yaw is composed on
    top of a fixed base "down" orientation so the policy has one free
    rotational degree of freedom for approach angle.

    The base orientation was NOT derived by guessing "180 degrees about
    world X" (that guess was verified wrong via the standalone controller
    test — it produced a ~180 degree orientation error on every target).
    Measuring the tcp site's actual world orientation at the panda.xml
    "home" keyframe showed the hand's local +Z axis (approach direction)
    already at world (0,0,-1), achieved via a 180-degree rotation about the
    world axis (1,1,0)/sqrt(2) — NOT the X axis — a consequence of the
    Panda's specific chain of per-link quat offsets (including the hand's
    -45 degree offset from link7). For a 180-degree rotation about axis
    n=(nx,ny,nz), the quaternion is [0, nx, ny, nz], giving [0, 1/sqrt2,
    1/sqrt2, 0] here — confirmed analytically (this rotation maps local
    X->world Y, local Y->world X, local Z->world -Z, matching the measured
    site_xmat exactly)."""
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    down_quat = np.array([0.0, inv_sqrt2, inv_sqrt2, 0.0])

    half_yaw = yaw / 2.0
    yaw_quat = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])

    result = np.zeros(4)
    mujoco.mju_mulQuat(result, yaw_quat, down_quat)
    return result
