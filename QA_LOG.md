# Q&A Log — 6-arm_pick_and_place

Substantial technical Q&A exchanges only (real explanations, not simple
yes/no procedural calls) — mirrors `5-arm_project_osc/QA_LOG.md`'s
convention.

---

**Q: proceed with creating the new project, remember what led to success
in 5-arm_project_osc, and tell me whether to use transfer learning or
train from scratch?**

A: Train from scratch, not weight transfer. Two reasons: (1) the
observation space differs (pick-and-place needs place-target information
the grasp-only observation never had, which would require the same kind
of fragile column-surgery that caused real problems in `4-arm_project`'s
Phase 1->2 transfer), and (2) `4-arm_project` directly documents a
warm-started model choking because SAC's single GLOBAL entropy
temperature treated transferred, already-confident weights as
"exploration satisfied" and starved a genuinely new dimension of noise —
the same risk applies here (a converged grasp policy is low-entropy
everywhere, suppressing exploration of the new carry-and-release
behavior). Using the actual strategy that solved Phase 2 instead — a
scripted demonstration -> behavior-cloning warm start on a FRESH network —
carries neither risk, and is proven rather than theoretical.
