# SAC subclass that keeps a behavior-cloning regularization term active in
# the actor's loss THROUGHOUT training, not just as a one-time pretraining
# step before RL takes over.
#
# Why this exists: train_osc_pick_place_bc_parallel.py's first version
# (one-time BC pretraining + replay-buffer seeding, then plain SAC) ran the
# full 1,000,000 steps and NEVER once completed a successful pick-and-place
# (ep_len_mean pinned at the episode cap the entire run, 1664 episodes).
# ep_rew_mean actually improved a lot (-204 -> -35.8) while this happened,
# and ent_coef collapsed toward 0 along the way — the policy got good at
# the DENSE shaping reward (gripper vibrating near the object, apparently
# cheap on that reward) while drifting away from the actually-successful
# behavior the BC pretraining started it with. Two things plausibly compound
# this: (1) a one-time pretraining step puts NO ongoing pressure on the
# actor to stay near the demonstrated behavior once RL gradients start
# pulling it elsewhere, and (2) the demo transitions seeded into the replay
# buffer become a shrinking fraction of it as more on-policy experience
# accumulates (buffer_size=1,000,000 vs ~100-150k demo transitions), so
# random batch sampling increasingly reflects on-policy (and, in this
# failure mode, increasingly wrong) experience over time.
#
# Fix: hold a SEPARATE, FIXED copy of the demonstration (obs, action) pairs
# outside the replay buffer, and every gradient step, add an ongoing
# MSE-to-demonstrated-action term to the actor's loss (in addition to,
# not instead of, the normal SAC actor loss) — so the actor can never fully
# drift away from the demonstrated behavior for demonstrated states,
# regardless of how the replay buffer's composition changes over the
# course of training.
#
# This is NOT full ARC/AWAC/CQL-style offline-RL regularization -- it's a
# minimal, direct fix for the specific failure mode observed: reusing SB3's
# own SAC.train() logic verbatim (copied from its source, not
# reimplemented from a guess) with one term added to the actor's loss.

import torch as th
import torch.nn.functional as F
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class BCRegularizedSAC(SAC):
    def set_bc_demo_data(self, demo_obs, demo_actions, bc_weight=100.0, bc_batch_size=256):
        """Must be called once after construction, before .learn(). Actions
        must be RAW (physical-units), matching train_osc_pick_place_bc_parallel.py's
        demonstrations.npz format — normalized internally via
        self.policy.scale_action(), the same conversion SB3 itself uses
        for the replay buffer (verified against SB3 source, not assumed —
        see seed_replay_buffer's comment in the training script).

        bc_weight is a first-guess starting point, not an empirically
        tuned value: actor_loss in this task's early logs ranged roughly
        -95 to +5 (Q-value-driven, so scales with reward/horizon), while a
        raw BC MSE loss is typically ~0.01-0.1 — at weight=100 the BC term
        contributes ~1-10, meaningful but not dominant relative to the
        Q-based term. Watch the new train/bc_loss log field; if the actor
        still drifts, raise this before assuming the whole approach is
        wrong."""
        low, high = self.policy.action_space.low, self.policy.action_space.high
        scaled_actions = 2.0 * (demo_actions - low) / (high - low) - 1.0
        self._bc_demo_obs = th.as_tensor(demo_obs, dtype=th.float32, device=self.device)
        self._bc_demo_actions = th.as_tensor(scaled_actions, dtype=th.float32, device=self.device)
        self._bc_weight = bc_weight
        self._bc_batch_size = bc_batch_size

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        assert hasattr(self, "_bc_demo_obs"), "call set_bc_demo_data() before learn()"

        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, bc_losses = [], [], []

        n_demo = self._bc_demo_obs.shape[0]

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # --- BC regularization term (the actual change vs plain SAC) ---
            demo_idx = th.randint(0, n_demo, (self._bc_batch_size,))
            demo_obs_batch = self._bc_demo_obs[demo_idx]
            demo_action_batch = self._bc_demo_actions[demo_idx]
            mean_actions, _, _ = self.actor.get_action_dist_params(demo_obs_batch)
            predicted = th.tanh(mean_actions)
            bc_loss = F.mse_loss(predicted, demo_action_batch)
            bc_losses.append(bc_loss.item())

            actor_loss = actor_loss + self._bc_weight * bc_loss
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
