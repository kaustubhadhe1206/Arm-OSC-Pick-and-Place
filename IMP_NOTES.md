# Implementation Notes — 6-arm_pick_and_place

Phase 3 of the project (Reach -> Grasp -> **Pick and Place** -> Perceive ->
Sort). Builds directly on `5-arm_project_osc`'s completed Phase 2 (Grasp),
which succeeded after a long debugging journey — every episode now
completes a successful grasp-lift-hold there. This file starts with the
carried-forward configuration/lessons, then follows the same
numbered-incident format as `5-arm_project_osc/IMP_NOTES.md` for anything
new discovered in this phase.

## Carried forward unchanged from 5-arm_project_osc

**Decision: train from scratch, not weight transfer.** Explicitly
considered and rejected loading the trained grasp model's weights
directly. Two reasons: (1) the observation space differs (pick-and-place
needs place-target information the grasp-only observation never had,
which would require the same kind of fragile column-surgery that caused
real problems in `4-arm_project`'s Phase 1->2 transfer), and (2)
`4-arm_project/IMP_NOTES.md` documents a warm-started model choking
because SAC's single GLOBAL entropy temperature treated the transferred,
already-confident weights as "exploration satisfied" and starved a
genuinely new dimension of noise — the same risk applies here (a
converged grasp policy is low-entropy everywhere, which would suppress
exploring the new carry-and-release behavior). Using the SAME successful
strategy instead — scripted demonstration -> behavior-cloning warm start —
on a fresh network, since that's what actually worked and carries neither
risk.

**`envs/osc_controller.py`** — copied byte-for-byte, no changes. This is
the differential-IK (damped-least-squares Jacobian) controller, fully
verified in `5-arm_project_osc` after finding and fixing 7 distinct bugs
(see that project's IMP_NOTES.md incidents #1-#7). Key parameters:
`max_ref_lag=0.06` (raised from an initially-too-conservative 0.025 —
incident #13 there found the original value made the controller too slow
to even reach objects within a training episode's budget), nullspace_gain
0.5 toward the home posture, `max_joint_delta=0.01`. The "down" orientation
quaternion `[0, 1/sqrt2, 1/sqrt2, 0]` — NOT the naively-guessed `[0,1,0,0]`
— was derived analytically from this specific Panda model's kinematic
chain; don't rederive it by guessing if this ever needs touching.

**`envs/panda/`** — copied unchanged (Panda model + tcp site + panda_grasp
scene with the object body).

**Action space**: 5-dim `[dx, dy, dz, dyaw, dgripper]`, deltas to a
persistent Cartesian target pose, `n_substeps=20` (40ms/step, 25Hz).

**Reward-shaping lessons applied from the start** (each was a real,
hard-won bug in 5-arm_project_osc — reapplying them here as the DEFAULT
design rather than rediscovering them):
- Gate dense "progress" bonuses (lift height there, place-distance here)
  behind the ACTUAL success threshold, not any nonzero movement in the
  right direction — a looser version is exploitable via brief
  grab-jostle-release-style cycling (incident #9).
- But ALSO provide a much SMALLER dense gradient below that threshold, or
  there's a hard reward cliff with zero signal for even attempting the
  behavior (incident #11). Both lessons together, not either alone.
- Penalize BREAKING an in-progress hold/settle, scaled by how much
  progress was lost — silently resetting a counter isn't a strong enough
  signal against grab-then-release given SAC's reward discounting
  (incident #9).
- Penalize premature gripper actions (closing before near the object,
  opening before near the place target) with a term that tapers to zero
  once actually close — reusing `close_scale` as the taper distance
  (incident #10, mirrored for release in this phase).
- Weight the action-smoothness penalty MORE heavily on the gripper
  dimension specifically (4x the arm dimensions) — chattering was
  confirmed (under deterministic evaluation, so a real learned oscillation,
  not exploration noise) to concentrate there specifically, right at the
  moment of deciding to grasp; the same risk applies at the moment of
  deciding to release (incidents #12, #14).
- Defensive reward clipping (`[-2.0, 6.0]`, from `4-arm_project` incident
  #26).
- `self.np_random` (gymnasium's seeded RNG) for all episode randomization,
  never the global `np.random` — required for `check_env()`'s
  step-determinism test to pass.

**Training strategy — the actual thing that solved Phase 2, more so than
any single reward tweak**: pure RL (SAC, `target_entropy=-1.0`) ran
585k/1M steps without once finding a successful grasp, despite the task
being provably achievable via a scripted routine. The fix was NOT more
reward shaping — it was collecting real successful demonstrations via a
hand-scripted routine (using the verified OSC controller directly) and
using them to (1) seed the replay buffer via a vectorized bulk write
(confirmed against SB3 source: buffer stores NORMALIZED actions, internal
array shapes are `(buffer_size, n_envs, dim)` with `buffer_size` already
divided by `n_envs`) and (2) pretrain the actor via supervised MSE
regression on the demonstrated (obs, action) pairs, BEFORE RL fine-tuning.
`learning_starts` lowered from 10,000 to 2,000 since SB3's warmup phase
uses pure random actions regardless of buffer/actor state (confirmed via
source, not assumed). This project should default to the SAME
demo-collection + BC-warm-start approach for the pick-and-place task from
the start, rather than trying pure RL first and waiting for it to plateau
— pick-and-place is a LONGER multi-stage sequence than grasp-only, so if
anything it's a harder exploration problem, not an easier one.

**Methodology carried forward**: write and verify a hand-scripted routine
achieving the FULL task BEFORE trusting any reward/environment design or
spending RL training budget on it. This is what decisively separated
"environment is broken" from "RL hasn't found it yet" in Phase 2
(incident #13) and is being applied here from the start via
`scripted_pick_place_check.py`, extending `scripted_grasp_check.py`'s
approach-grasp-lift-hold routine with new carry-descend-release-settle
phases.

**Operational lessons**: dedicated, separate files per phase/training
variant, never repurpose an existing file (standing project rule). Colab +
Google Drive symlinked checkpoints for training (moved off the local
laptop due to overheating during sustained parallel training —
`5-arm_project_osc/IMP_NOTES.md`). `device="cpu"` for SAC (a GPU doesn't
help small MLP networks on this CPU-bound-physics workload, and caused a
CUBLAS crash with multiple SubprocVecEnv workers in the original
4-arm_project). `N_ENVS=4` for parallel training.

## New design for this phase

**Place zone**: a second fixed box, mirrored across the y-axis from the
existing pick zone (`object_spawn_x_range=(0.225,0.525)`,
`object_spawn_y_range=(0.075,0.375)`) — same x range, negated y range
(`place_target_x_range=(0.225,0.525)`, `place_target_y_range=(-0.375,-0.075)`).
Same size and the same verified safe-clearance-from-base / within-reach
properties as the pick zone (mirroring a symmetric geometry preserves
both). A specific random point within this box (not just the zone bounds)
is sampled each episode as `self.place_target`, giving the policy (and the
reward's distance shaping) a concrete goal rather than a region.

**Task structure**: reach -> grasp -> lift -> hold (unchanged from Phase
2, reusing that exact reward logic) -> once the pick hold requirement is
met, `self.has_grasped_stably` latches True for the REST of the episode
(does not revert if the object is later dropped during carry — a
recovery/re-grasp attempt is allowed and still scored against the place
target, not treated as needing to restart the pick phase) -> reward
shifts to object-to-place-target distance shaping -> settle (object
within `place_radius` of the target, resting on the floor, gripper open
and not touching) sustained for `place_hold_required` consecutive steps ->
terminate with a completion bonus.

**Episode length**: `max_episode_steps=600` (up from Phase 2's 300).
Phase 2's trained policy completes the full grasp-lift-hold in
`ep_len_mean≈112` steps; this phase adds a full lateral traverse to the
opposite side of the robot plus a new settle-hold, so budgeted roughly
2x with margin. Not over-tuned for "more episode resets" the way Phase
2's episode length was during its pure-RL struggle (incident #13's
comment) — since this phase defaults to demonstration-driven BC
warm-start from the start, the concern about wasted resets during blind
exploration is much less relevant here.

**Observation space**: 22-dim — Phase 2's 16-dim observation (tip
position, tip-to-object vector, object position, gripper opening/velocity,
current target pose) plus `place_target` (3) and `object_to_place_target`
(3). No explicit "phase" flag — the existing gripper/height/position
signals already make the phase implicit, matching Phase 2's minimal,
no-redundant-engineered-features observation philosophy.

## Incident #1 — A literal static hold destabilizes an already-successful grasp; never exposed before because Phase 2 never needed to survive one

**Symptom:** the first scripted check attempt achieved a genuine lift
(`grasp_hold=20`, object well above the lift threshold) but then, during an
explicit "hold in place with zero actions" verification phase (mirroring
the pick-only task's design), the object consistently fell back to the
floor within 8-40 steps — reproducible across multiple fixed seeds, not a
fluke.

**Diagnosis:** this was NEVER exposed in `5-arm_project_osc`'s pick-only
task, because that task's episode TERMINATES the instant
`grasp_hold_counter` reaches the threshold — which happens mid-lift, under
continuous active correction, not during a subsequent static phase. Every
"successful" pick-only episode ended before this scenario could ever
occur. Pick-and-place is the first time anything needed to survive
BEYOND that moment.

Isolated via direct experiment (not guessed): a literal zero gripper
action recomputes `target_gripper_pos = current_actual + 0`, which
instantly matches current position — collapsing the PD actuator's
squeezing force to zero, since force is proportional to
`(target - actual)`. A small persistent negative gripper delta (keep
actively trying to close further) measurably extended hold duration (~40
steps vs ~8), confirming this mechanism is real — but even with maximum
continuous squeeze, a STATIC hold still failed by ~30-40 steps regardless
of grasp height or object friction. Friction sweeps were revealing in the
opposite direction expected: HIGHER friction made it fail FASTER (instant
drop), not slower — a signature of contact-solver stiffness sensitivity,
not a genuine insufficient-grip-force problem. Tried `noslip_iterations`
(MuJoCo's dedicated solver pass for sustained frictional contact) with no
improvement either.

The actually decisive experiment: tested CONTINUOUS MOVEMENT (an ~80-step,
0.65m lateral carry) instead of a static hold, using the same grasp. It
held the object stably the entire way with no sign of the instability.
Static holding specifically is what destabilizes this grip — moving
(even slowly) does not.

**Fix:** removed the scripted routine's explicit static "hold and verify"
phase entirely. Since `has_grasped_stably` already latches during active
lifting in most cases (confirmed: seeds where the grasp succeeded reached
`grasp_hold=20` before the lift's own `move_to()` call even finished), the
fix is simply to keep the arm ACTIVELY MOVING (a small extra ascent) if
the threshold isn't reached yet, rather than switching to a static wait.
Every phase from lift through carry passes a real (non-frozen) gripper
target to `move_to()`'s continuous delta computation, so the gripper
target is never literally static during any phase where the object needs
to stay held.

**Result:** with this fix, the full scripted pick-and-place routine
(approach -> grasp -> lift -> carry -> descend -> release -> settle)
succeeded in 6/8 runs (75%) across random object/place positions —
comparable to or better than the original pick-only task's baseline
success rate. This decisively confirms the pick-and-place environment and
reward design are achievable, the same way `scripted_grasp_check.py` did
for Phase 2. The 2 failures were the pre-existing pick-phase positioning
issue already known from Phase 2 (a fraction of random spawn positions
just don't grasp cleanly), not a new pick-and-place-specific problem.

**Open question for later, not blocking:** why does static holding
specifically destabilize this grip while movement doesn't? Plausible
candidates not yet confirmed: the controller's own small steady-state
tracking oscillation (~mm-scale, present even at rest, from
`max_ref_lag`) compounding into a slow walk-out over many identical
cycles at a fixed configuration, versus movement continuously
re-randomizing the micro-dynamics enough to prevent this from
accumulating in one direction. Not investigated further since it doesn't
block anything — the real task never requires a purely static hold — but
worth knowing about if the RL-trained policy is ever seen "fidgeting"
slightly while holding rather than sitting still, which would probably be
the trained policy independently discovering the same workaround.

## Project setup complete, not yet trained

Full pipeline built and locally verified before any training was
attempted, mirroring 5-arm_project_osc's discipline:
- `envs/franka_osc_pick_place_env.py` — passes `check_env()` and a
  random-action smoke test.
- `scripted_pick_place_check.py` — 6/8 (75%) end-to-end success across
  random seeds, confirming achievability (incident #1 above).
- `collect_demonstrations.py` — adapted from the scripted check, records
  full transitions from successful episodes only; tested at small scale
  (5 successes / 6 attempts, 83%, 1379 transitions).
- `train_osc_pick_place_bc_parallel.py` — same proven architecture as
  5-arm_project_osc's version (vectorized replay-buffer seeding, actor
  BC-pretraining, SAC fine-tuning with `target_entropy=-1.0`,
  `learning_starts=2000`). Full pipeline (seed -> pretrain -> short
  `model.learn()`) smoke-tested end-to-end without crashing before
  committing.
- `test_osc_pick_place.py` — viewer script with both zones AND the exact
  randomized place target visualized (green=pick zone, blue=place zone,
  yellow=place target point), all drawn from the same live env attributes
  the reward function uses, so they can't drift out of sync.
- Repo pushed to `https://github.com/kaustubhadhe1206/Arm-OSC-Pick-and-Place.git`;
  `colab_train.ipynb` set up mirroring 5-arm_project_osc's Drive-symlinked-
  checkpoints structure.

**Not yet done**: no training has been run. `train_osc_pick_place_bc_parallel.py`'s
`total_timesteps=1_000_000` is a starting default carried over from the
grasp-only task — this is a longer, harder task, so treat that number as
a first checkpoint to evaluate at, not an assumed sufficient budget.

## Incident #2 — Widened pick/place separation and episode budget

**Change:** the place zone (`place_target_y_range`) moved from a pure
mirror of the pick zone (`(-0.375, -0.075)`, only a 0.15m gap between the
two zones' nearest edges) to `(-0.45, -0.15)` — a 0.225m gap — so the two
zones are visibly, clearly separated rather than nearly touching across
the robot's centerline. Checked the worst-case pick-to-place distance
(opposite far corners) directly before committing to this: ~0.88m, still
under the arm's ~0.9m true max reach. `max_episode_steps` raised from 600
to 700 (28s) proportionally to the modest increase in worst-case carry
distance (~0.81m -> ~0.88m) — not picked to match a specific "N seconds"
figure requested without reference to the actual distances involved (24s
was already MORE generous than a literal "make it 8 seconds" would give,
so that request was interpreted as "give it more room for the added
distance," which is what this does).

**Verification:** re-ran `scripted_pick_place_check.py` (13 trials across
two batches): 9/13 (~69%) succeeded end-to-end, consistent with the
original (pre-widening) 75% baseline — the lower rate in the first batch
of 8 (5/8, 62.5%) was small-sample variance, not a systematic regression;
the second batch of 5 alone was 4/5 (80%). The one new failure type seen
("FAILED" with no qualifier, meaning it got all the way through release
and retreat but never settled within the hold budget) did not reproduce
in a full detailed-output pass — likely also variance, not a new
geometry-specific problem, but worth a second look if it becomes frequent
once real training data is available.

**Status:** any Colab session with demonstrations already collected or
training already started needs to redo both — the zone geometry changed,
so old `demonstrations.npz` reflects the previous, closer zone layout and
is no longer representative of the current task.
