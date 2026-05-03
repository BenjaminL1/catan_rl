"""
Custom PPO implementation for 1v1 Catan.

This is a synchronous, single-process PPO loop optimized for M1 Mac:
  1. Collect n_steps of experience by playing games
  2. Compute GAE advantages
  3. Run n_epochs of minibatch updates on the collected data
  4. Repeat

The key PPO idea: we have an "old" policy (used to collect experience)
and a "new" policy (being updated). The ratio new/old is clipped to
prevent the policy from changing too much in one update (catastrophic
forgetting). See Schulman et al. 2017.
"""

import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from catan_rl.algorithms.common.gae import explained_variance, safe_mean
from catan_rl.algorithms.common.rollout_buffer import CompositeRolloutBuffer
from catan_rl.eval.evaluation_manager import EvaluationManager
from catan_rl.models.build_agent_model import build_agent_model
from catan_rl.models.utils import ValueFunctionNormalizer
from catan_rl.selfplay.game_manager import GameManager
from catan_rl.selfplay.league import League
from catan_rl.selfplay.ratings import RatingTable

# Sentinel ID for the *current* learning policy in the rating table. Stays
# the same across the run; opponents have monotonically increasing IDs that
# never collide with this.
RATINGS_MAIN_ID = -99


class CatanPPO:
    """Proximal Policy Optimization trainer for Catan.

    Typical usage:
        ppo = CatanPPO(config)
        ppo.train()
    """

    def __init__(self, config: dict):
        """
        Args:
            config: merged dict from ppo/arguments.py get_config().
                    Contains both model architecture and training hyperparams.
        """
        self.config = config

        # CPU is fastest for this model at batch_size=1 rollouts (MPS kernel
        # launch overhead makes it 9x slower). CUDA is only useful if available
        # and explicitly requested. Override with config["device"] if needed.
        if config.get("device"):
            self.device = config["device"]
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        # Build policy network (config keys match build_agent_model.DEFAULT_MODEL_CONFIG)
        model_kwargs = {
            k: config[k]
            for k in (
                "obs_output_dim",
                "value_hidden_dims",
                "action_head_hidden_dim",
                "tile_in_dim",
                "tile_model_dim",
                "curr_player_main_in_dim",
                "other_player_main_in_dim",
                "dev_card_embed_dim",
                "dev_card_model_dim",
                "tile_model_num_heads",
                "proj_dev_card_dim",
                "dev_card_model_num_heads",
                "tile_encoder_num_layers",
                "proj_tile_dim",
                "dropout",
                # Phase 1.4 dev-card encoding selection
                "use_devcard_mha",
                "max_dev_seq",
                "dev_card_vocab_excl_pad",
                # Phase 2.1 axial positional embedding
                "use_axial_pos_emb",
                "axial_pos_dim",
                # Phase 2.2 transformer recipe
                "transformer_dropout",
                "transformer_activation",
                # Phase 2.4 AdaLN action heads
                "action_head_film",
                # Phase 2.5 value tower mode
                "value_head_mode",
                # Phase 3.6 opponent identity embedding
                "use_opponent_id_emb",
                "opp_id_emb_dim",
                "n_opp_kinds",
                "league_maxlen",
            )
            if k in config
        }
        self.policy = build_agent_model(device=self.device, **model_kwargs)

        # torch.compile: fuses kernels, reduces overhead. Opt-in since dev-card
        # lists trigger dynamic-shape recompiles. Enable after confirming stable.
        if config.get("torch_compile", False):
            self.policy = torch.compile(self.policy, mode="reduce-overhead")
            print("torch.compile() enabled for policy (mode=reduce-overhead).")

        # Optimizer: AdamW with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # Hyperparameters
        self.n_steps = config["n_steps"]
        self.batch_size = config["batch_size"]
        self.n_epochs = config["n_epochs"]
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_range = config["clip_range"]
        # Entropy: use start value if annealing, else fixed
        self._entropy_coef_start = config.get("entropy_coef_start", config["entropy_coef"])
        self._entropy_coef_final = config.get("entropy_coef_final", config["entropy_coef"])
        self._entropy_anneal_start = config.get("entropy_coef_anneal_start", 0)
        self._entropy_anneal_end = config.get("entropy_coef_anneal_end", 0)
        self.use_entropy_annealing = self._entropy_anneal_end > self._entropy_anneal_start
        self.entropy_coef = self._entropy_coef_start
        self.value_coef = config["value_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.total_timesteps = config["total_timesteps"]
        self.checkpoint_freq = config["checkpoint_freq"]
        # Linear LR decay (Charlesworth-style)
        self.use_linear_lr_decay = config.get("use_linear_lr_decay", False)
        self.lr_final = config.get("lr_final", config["learning_rate"] * 0.1)

        # Value normalization
        self.value_normalizer = ValueFunctionNormalizer()
        self.normalize_values = config.get("normalize_values", True)

        # Multi-env: in-process parallel games (no subprocess overhead on M1)
        self.n_envs = config.get("n_envs", 1)

        # Rollout buffer (Charlesworth-style dict observations).
        # Phase 1.3: align buffer storage dims to the configured player encoding
        # so compact-mode obs (54/61) don't get broadcast into legacy 166/173 slots.
        self.buffer = CompositeRolloutBuffer(
            n_steps=self.n_steps,
            device=self.device,
            n_envs=self.n_envs,
            curr_player_dim=int(config.get("curr_player_main_in_dim", 166)),
            next_player_dim=int(config.get("other_player_main_in_dim", 173)),
            store_opponent_id=bool(config.get("use_opponent_id_emb", False)),
        )

        # KL early stopping: break out of update epochs if policy drifts too far
        self.target_kl = config.get("target_kl")

        # Phase 1.1: PPO2-style value clipping. Reduces variance of value
        # updates by symmetrically clipping how far the new value estimate
        # can move from the buffer's stored old value within one minibatch.
        # Loss = max(MSE_unclipped, MSE_clipped) — pessimistic w.r.t. the clip.
        self.use_value_clipping = bool(config.get("use_value_clipping", True))
        self.clip_range_vf = float(config.get("clip_range_vf", 0.2))

        # Phase 1.2: per-rollout advantage normalization mode.
        # 'rollout' = mean/std once over the whole buffer (low-variance grad).
        # 'batch'   = mean/std per-minibatch (legacy default).
        # 'none'    = pass advantages through raw.
        self.advantage_norm = str(config.get("advantage_norm", "rollout"))
        if self.advantage_norm not in ("rollout", "batch", "none"):
            raise ValueError(
                f"advantage_norm must be 'rollout'|'batch'|'none', got {self.advantage_norm!r}"
            )

        # Phase 1.5: D6 dihedral-symmetry augmentation per minibatch.
        # 0.0 = off (default for back-compat); 0.5 recommended in phase1_full.
        self.symmetry_aug_prob = float(config.get("symmetry_aug_prob", 0.0))
        if not 0.0 <= self.symmetry_aug_prob <= 1.0:
            raise ValueError(f"symmetry_aug_prob must be in [0, 1], got {self.symmetry_aug_prob!r}")

        # Entropy floor: prevent policy collapse by boosting entropy_coef if needed
        self.entropy_floor = config.get("entropy_floor", 0.003)
        self.entropy_floor_coef = config.get("entropy_floor_coef", 0.01)
        self._last_entropy = float("inf")  # updated after each update() call

        # Stagnation detection
        self.stagnation_window = config.get("stagnation_window", 10)
        self.stagnation_threshold = config.get("stagnation_threshold", 0.03)
        self._eval_win_rate_history: list = []

        # League (Charlesworth-style): past policies for self-play
        self.league = League(
            maxlen=config.get("league_maxlen", 100),
            add_every=config.get("add_policy_every", 4),
            random_weight=config.get("league_random_weight", 0.0),
            heuristic_weight=config.get("heuristic_opponent_weight", 0.0),
            # Phase 3.1 PFSP-hard / PFSP-var sampling.
            pfsp_mode=config.get("pfsp_mode", "linear"),
            pfsp_p=config.get("pfsp_p", 2.0),
            pfsp_window=config.get("pfsp_window", 32),
            # Phase 3.2 latest-policy regularization.
            latest_policy_weight=config.get("latest_policy_weight", 0.0),
        )

        # Add initial policy so league always has at least one (required for policy opponents)
        self.league.add(copy.deepcopy(self._raw_policy_state_dict()))

        # ── Phase 3.4: TrueSkill / Glicko-2 league ratings ─────────────────
        # ``use_trueskill`` defaults off so legacy configs stay byte-for-byte
        # identical. ``trueskill_decay`` (default 1.001 per update) inflates
        # σ slowly to handle non-stationary policies — without it, the
        # main's σ collapses to ~1.0 within a few hundred updates and PFSP
        # win predictions become brittle.
        self.use_trueskill = bool(config.get("use_trueskill", False))
        self.trueskill_decay = float(config.get("trueskill_decay", 1.001))
        self.rating_table: RatingTable | None = RatingTable() if self.use_trueskill else None

        # ── Phase 3.5: Nash-weighted checkpoint pruning ────────────────────
        # ``prune_strategy='nash'`` periodically replaces the FIFO eviction
        # rule with a replicator-dynamics evaluation: the policy with the
        # lowest Nash mixture mass is dropped, keeping the league biased
        # toward strategically-distinct opponents.
        self.prune_strategy = config.get("prune_strategy", "fifo")
        if self.prune_strategy not in ("fifo", "nash"):
            raise ValueError(
                f"prune_strategy must be 'fifo' or 'nash', got {self.prune_strategy!r}"
            )
        self.nash_prune_every = int(config.get("prune_every", 20))
        self.nash_top_k = int(config.get("nash_top_k", 32))
        self.nash_prune_games = int(config.get("nash_prune_round_games", 50))
        self._adds_since_prune = 0

        # ── Phase 3.3: Duo-exploiter cycle ─────────────────────────────────
        # ``duo`` mode periodically pauses main training to spin up an
        # exploiter — a fresh policy initialized from main that trains
        # exclusively against a frozen main snapshot for ``exploiter_n_updates``
        # PPO updates. The exploiter's final state is then injected back
        # into main's league with an amplified PFSP priority for its first
        # ``exploiter_priority_games`` matches.
        self.exploiter_mode = config.get("exploiter_mode", "off")
        if self.exploiter_mode not in ("off", "duo"):
            raise ValueError(f"exploiter_mode must be 'off' or 'duo', got {self.exploiter_mode!r}")
        self.exploiter_cycle_steps = int(config.get("exploiter_cycle_steps", 1_000_000))
        self.exploiter_n_updates = int(config.get("exploiter_n_updates", 32))
        self.exploiter_priority_multiplier = float(config.get("exploiter_priority_multiplier", 1.5))
        self.exploiter_priority_games = int(config.get("exploiter_priority_games", 64))
        self._exploiter_cycle_count = 0
        self._last_exploiter_at = 0  # main step at which the prior cycle ended

        # Build policy factory for league/GameManager (creates policy instances for inference)
        def build_policy_fn(device: str = "cpu"):
            return build_agent_model(device=device, **model_kwargs)

        # Environment manager: use league for policy opponents when league has policies.
        # Phase 1.3: thermometer-vs-compact obs encoding flag plumbed into env.
        # Phase 3.2: ``current_policy_state_fn`` lets the league's
        # ``current_self`` opponent draw a live snapshot of self.policy.
        # Phase 3.4: ``record_match_fn`` updates the TrueSkill rating table
        # from every concluded league match.
        self.game_manager = GameManager(
            n_envs=self.n_envs,
            opponent_type="random",
            max_turns=config.get("max_turns", 500),
            league=self.league,
            build_policy_fn=build_policy_fn,
            device=self.device,
            use_thermometer_encoding=bool(config.get("use_thermometer_encoding", True)),
            current_policy_state_fn=self._raw_policy_state_dict,
            record_match_fn=self._record_rating_match if self.use_trueskill else None,
            use_opponent_id_emb=bool(config.get("use_opponent_id_emb", False)),
            opp_id_mask_prob=float(config.get("opp_id_mask_prob", 0.40)),
            league_maxlen=int(config.get("league_maxlen", 100)),
        )

        # Evaluation opponent: starts as configured, auto-upgrades to heuristic
        # once the model crosses eval_upgrade_threshold win rate vs the current opponent.
        self._eval_opponent = config.get("opponent_type", "random")
        self._eval_upgrade_threshold = config.get("eval_upgrade_threshold", 0.95)
        self._use_thermometer_encoding = bool(config.get("use_thermometer_encoding", True))
        self.eval_manager = EvaluationManager(
            n_games=config.get("eval_games", 40),
            opponent_type=self._eval_opponent,
            max_turns=config.get("max_turns", 500),
            use_thermometer_encoding=self._use_thermometer_encoding,
        )

        # Logging
        self.log_dir = config.get("log_dir", "runs/train")
        self.writer = SummaryWriter(self.log_dir)

        # Checkpoint directory
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints/train")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training state
        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        # Phase 0: per-head entropy collapse tracking (one counter per head).
        # Imported lazily to avoid circular imports at module load.
        from catan_rl.models.action_heads_module import MultiActionHeads as _MAH

        self._head_collapse_counter: dict[str, int] = {n: 0 for n in _MAH.HEAD_NAMES}
        self._collapsed_heads_last: str = ""

    # ── Phase 3.3: exploiter cycle snapshot/restore ────────────────────────
    def _snapshot_for_exploiter(self) -> dict:
        """Save enough state to swap in an exploiter trainer and roll back.

        Captures:
          - policy state_dict (the *main* weights we want to preserve)
          - optimizer state_dict (Adam moments — would otherwise be reset)
          - value_normalizer running mean/var
          - global_step / episode counters / entropy_coef
          - league reference (object identity, not a copy — restored verbatim)
          - GameManager's league + record_match callback wiring
          - rollout buffer position (we abandon any in-progress rollout
            and the cycle starts with a fresh buffer.reset())

        Notes:
          The policy state_dict is enough to fully reconstruct the model
          since architecture is fixed by ``model_kwargs``. We don't copy
          the optimizer's *param-group* references because they live on
          the params themselves; we copy ``state_dict()`` and re-`load` it.
        """
        return {
            "policy": copy.deepcopy(self._raw_policy_state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "value_normalizer": copy.deepcopy(self.value_normalizer.state_dict()),
            "global_step": self.global_step,
            "entropy_coef": self.entropy_coef,
            "episode_rewards": list(self.episode_rewards),
            "episode_lengths": list(self.episode_lengths),
            "episode_wins": list(self.episode_wins),
            "league": self.league,
            "game_manager_league": self.game_manager.league,
            "game_manager_record_match_fn": self.game_manager._record_match_fn,
            "game_manager_current_policy_state_fn": (self.game_manager._current_policy_state_fn),
        }

    def _restore_from_snapshot(self, snap: dict) -> None:
        """Roll the trainer back to the state captured by ``_snapshot_for_exploiter``.

        Re-loads policy + optimizer. Resets the rollout buffer (we discard
        the exploiter's last partial rollout). Reinstates the league
        reference on both ``self`` and ``self.game_manager``.
        """
        self.policy.load_state_dict(snap["policy"])
        self.optimizer.load_state_dict(snap["optimizer"])
        self.value_normalizer.load_state_dict(snap["value_normalizer"])
        self.global_step = snap["global_step"]
        self.entropy_coef = snap["entropy_coef"]
        self.episode_rewards = snap["episode_rewards"]
        self.episode_lengths = snap["episode_lengths"]
        self.episode_wins = snap["episode_wins"]
        self.league = snap["league"]
        self.game_manager.league = snap["game_manager_league"]
        self.game_manager._record_match_fn = snap["game_manager_record_match_fn"]
        self.game_manager._current_policy_state_fn = snap["game_manager_current_policy_state_fn"]
        self.buffer.reset()

    def _raw_policy_state_dict(self) -> dict:
        """Return policy state dict without torch.compile's _orig_mod. prefix.

        torch.compile wraps the policy in OptimizedModule, which prefixes all
        keys with '_orig_mod.'. League policies and checkpoints are always stored
        in bare format so they can be loaded into uncompiled CatanPolicy instances.
        """
        sd = self.policy.state_dict()
        if next(iter(sd)).startswith("_orig_mod."):
            sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}
        return sd

    # ── Phase 3.4: TrueSkill rating helpers ───────────────────────────────
    def _record_rating_match(self, opponent_policy_id: int, win: int) -> None:
        """Update the rating table after a single match.

        Skips matches where the opponent has no stable policy ID (random,
        heuristic, current_self) — those would corrupt ratings by injecting
        many "match" rows for non-policy entities. Random/heuristic remain
        useful as PFSP signal but not as Elo signal.
        """
        if self.rating_table is None or opponent_policy_id < 0:
            return
        self.rating_table.record_match(RATINGS_MAIN_ID, opponent_policy_id, a_won=bool(win))

    def _decay_ratings_sigma(self) -> None:
        """Inflate every rating's σ by ``trueskill_decay``.

        Without this, σ collapses toward 1.0 within a few hundred matches
        and the rating system becomes overconfident in stale numbers. The
        backend's own update already shrinks σ; this re-injects uncertainty
        each PPO update so non-stationarity is reflected.
        """
        if self.rating_table is None or self.trueskill_decay == 1.0:
            return
        from catan_rl.selfplay.ratings import Rating

        for k, r in list(self.rating_table.ratings.items()):
            self.rating_table.ratings[k] = Rating(mu=r.mu, sigma=r.sigma * self.trueskill_decay)

    # ── Phase 3.3: Duo-exploiter cycle ─────────────────────────────────────
    def _maybe_run_exploiter_cycle(self) -> None:
        """Run an exploiter cycle if one is due.

        Cycle structure:
          1. Snapshot main (policy + optimizer + value normalizer + league refs).
          2. Replace ``self.policy`` with a fresh build initialized from main.
          3. Build a frozen-main league as the only opponent.
          4. Train ``exploiter_n_updates`` PPO updates against frozen main.
          5. Restore main from snapshot.
          6. Push exploiter snapshot into the (restored) main league with
             priority boost (``exploiter_priority_multiplier``) for its
             first ``exploiter_priority_games`` matches.

        Scalars during the cycle are namespaced under ``exploiter/`` so the
        main-training plot lines stay clean.
        """
        if self.exploiter_mode != "duo":
            return
        # Fire every ``exploiter_cycle_steps`` of *main* progress. ``_last_exploiter_at``
        # tracks the step count at which the previous cycle ended, so the next
        # cycle waits ``exploiter_cycle_steps`` further main steps.
        next_due = self._last_exploiter_at + self.exploiter_cycle_steps
        if self.global_step < next_due:
            return

        cycle_idx = self._exploiter_cycle_count + 1
        print(
            f"[exploiter] starting cycle {cycle_idx} at main step "
            f"{self.global_step:,} (target {self.exploiter_n_updates} updates)"
        )

        snap = self._snapshot_for_exploiter()
        try:
            # Step 1+2: rebuild policy from main snapshot, fresh optimizer.
            self.policy.load_state_dict(snap["policy"])
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config.get("weight_decay", 1e-4),
            )

            # Step 3: frozen-main league, sampled exclusively.
            frozen_league = League.frozen_main_for_exploiter(snap["policy"])
            self.league = frozen_league
            self.game_manager.league = frozen_league
            # Disable rating updates during the cycle — exploiter ratings are
            # not commensurable with the main rating curve.
            self.game_manager._record_match_fn = None
            # ``current_self`` would self-evaluate the *exploiter*; null it out.
            self.game_manager._current_policy_state_fn = None

            # Step 4: short PPO loop. We reuse ``collect_rollouts`` and
            # ``update`` so the exploiter trains with the same recipe.
            self._train_exploiter_inner(cycle_idx)

            exploiter_state = copy.deepcopy(self._raw_policy_state_dict())
        finally:
            # Step 5: ALWAYS restore main, even if the cycle errors out.
            self._restore_from_snapshot(snap)

        # Step 6: inject exploiter into main league with priority boost.
        new_id = self.league.add_with_boost(
            exploiter_state,
            multiplier=self.exploiter_priority_multiplier,
            boost_games=self.exploiter_priority_games,
        )
        if self.writer is not None:
            self.writer.add_scalar("train/exploiter_cycles_completed", cycle_idx, self.global_step)
            self.writer.add_scalar("train/exploiter_added_policy_id", new_id, self.global_step)
        print(f"[exploiter] cycle {cycle_idx} done; injected as league policy id {new_id}")

        self._exploiter_cycle_count = cycle_idx
        self._last_exploiter_at = self.global_step

    def _train_exploiter_inner(self, cycle_idx: int) -> None:
        """Inner PPO loop run *as the exploiter* against frozen main.

        Logs scalars under ``exploiter/`` to keep the main learning curve
        clean. Doesn't increment ``self.global_step`` — main's step counter
        is preserved across the cycle so eval/checkpoint scheduling still
        fires from main's perspective.
        """
        saved_global_step = self.global_step
        for update_idx in range(1, self.exploiter_n_updates + 1):
            self.collect_rollouts()
            update_stats = self.update()
            # Roll back the global_step bump that ``collect_rollouts`` did,
            # so main's step counter doesn't drift across cycles.
            self.global_step = saved_global_step

            if self.writer is None:
                continue
            tag_step = (cycle_idx - 1) * self.exploiter_n_updates + update_idx
            self.writer.add_scalar("exploiter/update", update_idx, tag_step)
            for key in ("entropy", "approx_kl", "value_loss", "policy_loss"):
                if key in update_stats:
                    self.writer.add_scalar(f"exploiter/{key}", update_stats[key], tag_step)

    # ── Phase 3.5: Nash-weighted checkpoint pruning ────────────────────────
    def _maybe_run_nash_pruning(self) -> None:
        """Periodically replace FIFO eviction with Nash-weighted pruning.

        Triggered every ``prune_every`` league adds **and** only when the
        league is at capacity (otherwise FIFO/Nash agree: nothing to evict).

        Cost: ``nash_top_k * (nash_top_k - 1) / 2 * nash_prune_round_games``
        h2h games. With defaults (32, 50) that's ~24,800 games — substantial,
        which is why this runs at most every 20 league adds.
        """
        if self.prune_strategy != "nash":
            return
        if len(self.league) < self.league.policies.maxlen:
            return  # No eviction needed yet.
        if self._adds_since_prune < self.nash_prune_every:
            return
        self._adds_since_prune = 0

        # Pick the most recent K entries — proxy for "currently relevant"
        # without a full league-wide round-robin (would cost ``maxlen^2/2``).
        ids = list(self.league._policy_ids)[-self.nash_top_k :]
        if len(ids) < 2:
            return
        # Map each ID back to its state_dict (deque-aligned with _policy_ids).
        state_dicts: dict[int, dict] = {
            pid: sd
            for pid, sd in zip(self.league._policy_ids, self.league.policies, strict=True)
            if pid in ids
        }
        payoff = self._build_payoff_matrix(ids, state_dicts)
        evicted = self.league.prune_nash(payoff, ids)
        if self.rating_table is not None:
            self.rating_table.ratings.pop(evicted, None)

    def _build_payoff_matrix(self, ids: list[int], state_dicts: dict[int, dict]) -> "np.ndarray":
        """Round-robin h2h, returning the (k, k) win-rate matrix.

        Uses the existing eval manager's deterministic ``evaluate_h2h`` so
        first-mover bias cancels across both orderings of each seed pair.
        """
        from catan_rl.eval.evaluation_manager import EvaluationManager, standard_eval_seeds

        em = EvaluationManager(opponent_type="policy", max_turns=500)
        seeds = standard_eval_seeds(0, max(2, self.nash_prune_games // 2))
        k = len(ids)
        payoff = np.full((k, k), 0.5, dtype=np.float64)  # diagonal stays 0.5

        # Use a single shared inference policy for memory; load weights per-call.
        from catan_rl.models.build_agent_model import build_agent_model

        pol_a = build_agent_model(device=self.device)
        pol_b = build_agent_model(device=self.device)

        for i in range(k):
            pol_a.load_state_dict(state_dicts[ids[i]])
            pol_a.eval()
            for j in range(i + 1, k):
                pol_b.load_state_dict(state_dicts[ids[j]])
                pol_b.eval()
                h2h = em.evaluate_h2h(pol_a, pol_b, seeds, device=self.device)
                wr = float(h2h["win_rate_a"])
                payoff[i, j] = wr
                payoff[j, i] = 1.0 - wr
        return payoff

    def _log_rating_scalars(self, writer: "SummaryWriter | None", step: int) -> None:
        """Push TrueSkill scalars to TensorBoard.

        Logs the main agent's μ/σ plus the top-K opponents' conservative
        ratings. Names use the ``eval/trueskill_*`` prefix so they live next
        to the eval-harness scalars rather than the per-update train scalars.
        """
        if self.rating_table is None or writer is None:
            return
        main = self.rating_table.get(RATINGS_MAIN_ID)
        writer.add_scalar("eval/trueskill_main_mu", main.mu, step)
        writer.add_scalar("eval/trueskill_main_sigma", main.sigma, step)
        writer.add_scalar("eval/trueskill_main_conservative", main.conservative, step)
        for rank, (key, rating) in enumerate(self.rating_table.top_k(5), start=1):
            if key == RATINGS_MAIN_ID:
                continue
            writer.add_scalar(f"eval/trueskill_top{rank}_conservative", rating.conservative, step)

    def _obs_to_tensor_dict(self, obs: dict) -> dict[str, torch.Tensor]:
        """Convert a single env obs dict (numpy/ints) to a batched tensor dict (B=1)."""
        return self._batch_obs_to_tensor_dict([obs])

    def _batch_obs_to_tensor_dict(self, obs_list: list[dict]) -> dict[str, torch.Tensor]:
        """Convert a list of B env obs dicts to a single batched tensor dict.

        Uses np.stack + from_numpy for a single memcopy per array (no intermediate
        per-element tensors). Dev cards are (B, 16) int64; player_modules.py tensor
        branch handles this directly without pad_sequence.
        """
        device = self.device
        tile = np.stack([o["tile_representations"] for o in obs_list])  # (B,19,79)
        curr = np.stack([o["current_player_main"] for o in obs_list])  # (B,166)
        nxt = np.stack([o["next_player_main"] for o in obs_list])  # (B,173)
        hid = np.stack([o["current_player_hidden_dev"] for o in obs_list])  # (B,16) int32
        cpd = np.stack([o["current_player_played_dev"] for o in obs_list])  # (B,16) int32
        npd = np.stack([o["next_player_played_dev"] for o in obs_list])  # (B,16) int32
        out = {
            "tile_representations": torch.from_numpy(tile).to(device),
            "current_player_main": torch.from_numpy(curr).to(device),
            "next_player_main": torch.from_numpy(nxt).to(device),
            "current_player_hidden_dev": torch.from_numpy(hid).to(device).long(),
            "current_player_played_dev": torch.from_numpy(cpd).to(device).long(),
            "next_player_played_dev": torch.from_numpy(npd).to(device).long(),
        }
        # Phase 3.6: opponent identity passes through when present in obs.
        if "opponent_kind" in obs_list[0]:
            kind = np.stack([int(o["opponent_kind"]) for o in obs_list])
            pid = np.stack([int(o["opponent_policy_id"]) for o in obs_list])
            out["opponent_kind"] = torch.from_numpy(kind).to(device).long()
            out["opponent_policy_id"] = torch.from_numpy(pid).to(device).long()
        return out

    def _batch_masks_to_tensor(self, masks_list: list[dict]) -> dict[str, torch.Tensor]:
        """Stack a list of B per-env mask dicts into batched bool tensors."""
        device = self.device
        return {
            k: torch.from_numpy(np.stack([m[k] for m in masks_list])).to(device)
            for k in masks_list[0]
        }

    def _run_batched_opponent_turns(
        self, pending: list[int], partial_rewards: list[float]
    ) -> dict[int, tuple]:
        """Run batched opponent NN inference until all pending envs finish their turn.

        Args:
            pending: list of env indices with _opp_turn_in_progress == True
            partial_rewards: reward accumulated before opponent acts, aligned with pending

        Returns:
            dict mapping env_idx → (final_obs, total_reward, terminated, truncated, info)
        """
        gm = self.game_manager
        opp_policy = gm.rollout_opp_policy
        results = {}
        # Use dict so slot alignment survives as pending shrinks each loop
        accum = {env_i: partial_rewards[slot] for slot, env_i in enumerate(pending)}

        while pending:
            # Batch opponent obs/masks across all still-pending envs
            opp_pairs = [gm.get_opponent_obs_masks(i) for i in pending]
            opp_obs_t = self._batch_obs_to_tensor_dict([p[0] for p in opp_pairs])
            opp_mask_t = self._batch_masks_to_tensor([p[1] for p in opp_pairs])

            with torch.inference_mode():
                opp_actions, _, _ = opp_policy.act(opp_obs_t, opp_mask_t, deterministic=True)
            opp_actions_np = opp_actions.cpu().numpy()  # (len(pending), 6)

            still_pending = []
            for slot, env_i in enumerate(pending):
                turn_done, obs, rew_delta, terminated, truncated, info = gm.apply_opponent_action(
                    env_i, opp_actions_np[slot]
                )
                if turn_done:
                    results[env_i] = (obs, accum[env_i] + rew_delta, terminated, truncated, info)
                else:
                    still_pending.append(env_i)
            pending = still_pending

        return results

    def collect_rollouts(self) -> dict[str, float]:
        """Play n_steps of games and store transitions in the buffer.

        Main agent inference is batched (batch=n_envs). Opponent NN inference
        is also batched via _run_batched_opponent_turns — all envs whose
        opponent turn was triggered in the same outer loop share one forward
        pass through the shared rollout opponent policy.

        Returns:
            Dict with rollout statistics (mean reward, mean length, etc.).
        """
        self.policy.eval()
        self.buffer.reset()

        observations, _ = self.game_manager.reset_all()
        ep_rewards = [0.0] * self.n_envs
        ep_lengths = [0] * self.n_envs
        steps_collected = 0

        while steps_collected < self.n_steps:
            # Clamp batch to remaining steps (handles n_steps % n_envs != 0)
            batch_n = min(self.n_envs, self.n_steps - steps_collected)

            obs_batch = [observations[i] for i in range(batch_n)]
            masks_list = self.game_manager.get_masks()[:batch_n]

            obs_t = self._batch_obs_to_tensor_dict(obs_batch)
            masks_t = self._batch_masks_to_tensor(masks_list)

            with torch.inference_mode():
                actions, values, log_probs = self.policy.act(obs_t, masks_t)

            actions_np = actions.cpu().numpy()  # (batch_n, 6)
            values_squeezed = values.squeeze(-1)
            if self.normalize_values:
                values_np = self.value_normalizer.denormalize(values_squeezed).cpu().numpy()
            else:
                values_np = values_squeezed.cpu().numpy()  # (batch_n,)
            log_probs_np = log_probs.cpu().numpy()  # (batch_n,)

            # Phase 1: step all main agents; collect envs that need opponent NN turns
            pending_opp = []  # env indices whose opponent turn is deferred
            pending_prewd = []  # partial reward accumulated before opponent acts
            immediate = []  # env indices with complete transitions

            for i in range(batch_n):
                new_obs, reward, terminated, truncated, info = self.game_manager.step_one(
                    i, actions_np[i]
                )
                if info.get("opp_turn_pending"):
                    pending_opp.append(i)
                    pending_prewd.append(reward)
                else:
                    observations[i] = new_obs
                    immediate.append((i, new_obs, reward, terminated, truncated, info))

            # Phase 2: batch opponent inference for deferred turns
            if pending_opp and self.game_manager.rollout_opp_policy is not None:
                opp_results = self._run_batched_opponent_turns(pending_opp, pending_prewd)
                for env_i, (fin_obs, fin_rew, fin_term, fin_trunc, fin_info) in opp_results.items():
                    observations[env_i] = fin_obs
                    immediate.append((env_i, fin_obs, fin_rew, fin_term, fin_trunc, fin_info))

            # Phase 3: store all completed transitions to buffer
            for env_i, _new_obs, reward, terminated, truncated, info in immediate:
                self.buffer.add(
                    obs_batch[env_i],
                    actions_np[env_i],
                    reward,
                    bool(terminated),
                    bool(truncated),
                    float(values_np[env_i]),
                    float(log_probs_np[env_i]),
                    masks_list[env_i],
                )
                ep_rewards[env_i] += reward
                ep_lengths[env_i] += 1
                self.global_step += 1
                steps_collected += 1

                done = terminated or truncated
                if done:
                    self.episode_rewards.append(ep_rewards[env_i])
                    self.episode_lengths.append(ep_lengths[env_i])
                    # Truncations are not wins regardless of who had higher VP at cutoff;
                    # only genuine terminations with `is_success` count.
                    won = bool(terminated) and bool(info.get("is_success"))
                    self.episode_wins.append(1.0 if won else 0.0)
                    ep_rewards[env_i] = 0.0
                    ep_lengths[env_i] = 0

        # Bootstrapped last values — one batched forward pass for all envs
        with torch.inference_mode():
            obs_t = self._batch_obs_to_tensor_dict(observations)
            raw_last = self.policy.get_value(obs_t).squeeze(-1)
            if self.normalize_values:
                last_values = self.value_normalizer.denormalize(raw_last).cpu().numpy()
            else:
                last_values = raw_last.cpu().numpy()  # (n_envs,)

        self.buffer.compute_returns_and_advantages(
            last_values, self.gamma, self.gae_lambda, advantage_norm=self.advantage_norm
        )
        self.buffer.finalize(self.device)

        if self.normalize_values:
            self.value_normalizer.update(
                torch.tensor(self.buffer.returns[: self.n_steps], dtype=torch.float32)
            )

        stats = {}
        if self.episode_rewards:
            stats["mean_reward"] = safe_mean(self.episode_rewards[-100:])
            stats["mean_length"] = safe_mean(self.episode_lengths[-100:])
            stats["mean_win_rate"] = safe_mean(self.episode_wins[-100:])
        return stats

    def update(self) -> dict[str, float]:
        """Run up to n_epochs of PPO updates with KL early stopping.

        Epochs are terminated early if the mean KL divergence between the
        old and new policy exceeds target_kl. This prevents over-updating
        stale data while allowing the full n_epochs when updates are safe.

        Phase 0 also logs per-head entropy for collapse detection. The joint
        ``entropy`` scalar is preserved (back-compat); new keys
        ``entropy_head_<name>`` and ``entropy_head_<name>_cond`` give the
        unconditional and relevance-weighted means respectively.

        Returns:
            Dict with training loss statistics including approx_kl,
            epochs_completed, and per-head entropy diagnostics.
        """
        from catan_rl.models.action_heads_module import MultiActionHeads

        self.policy.train()

        total_pg_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_kl = 0.0
        n_updates = 0
        epochs_completed = 0

        # Per-head entropy accumulators. We track the unconditional mean
        # (sum / N) AND the conditional mean over relevant samples
        # (sum_weighted / sum_weight). Drop in either signals collapse.
        head_names = MultiActionHeads.HEAD_NAMES
        head_ent_sum: dict[str, float] = {h: 0.0 for h in head_names}
        head_weighted_ent_sum: dict[str, float] = {h: 0.0 for h in head_names}
        head_weight_sum: dict[str, float] = {h: 0.0 for h in head_names}
        head_sample_count: dict[str, int] = {h: 0 for h in head_names}

        for _epoch in range(self.n_epochs):
            epoch_kl = 0.0
            epoch_batches = 0

            for batch in self.buffer.get_batches(
                self.batch_size, symmetry_aug_prob=self.symmetry_aug_prob
            ):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_prob = batch["old_log_prob"]
                old_values = batch["old_values"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                masks = batch["masks"]

                # Phase 1.2: advantage normalization. With 'rollout' mode the
                # buffer has already normalized globally, so the per-batch
                # standardization here is a no-op (and is skipped). 'batch'
                # mode runs the legacy per-minibatch standardization.
                if self.advantage_norm == "batch":
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # 'rollout' and 'none' both bypass per-batch normalization.

                # Forward pass: re-evaluate old actions under current policy.
                # Per-head dict is requested so we can log collapse diagnostics
                # without a second forward pass.
                values, new_log_prob, entropy, per_head = self.policy.evaluate_actions(
                    obs, masks, actions, return_per_head=True
                )
                values = values.squeeze(-1)  # (B,)

                # ── Policy loss (clipped surrogate objective) ─────────
                ratio = torch.exp(new_log_prob - old_log_prob)
                pg_loss_1 = -advantages * ratio
                pg_loss_2 = -advantages * torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                )
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                # ── Value loss (Phase 1.1: optionally clipped) ────────
                if self.normalize_values:
                    norm_returns = self.value_normalizer.normalize(returns)
                    # The values stored in the buffer were denormalized for
                    # GAE. Renormalize so the clip range applies on the same
                    # scale as the network output.
                    norm_old_values = self.value_normalizer.normalize(old_values)
                else:
                    norm_returns = returns
                    norm_old_values = old_values

                if self.use_value_clipping:
                    # PPO2: clip the new value within ±clip_range_vf of the old
                    # value, then take the elementwise-max of the two squared
                    # errors. This is pessimistic — same direction as the
                    # policy clip — and keeps the value head from chasing
                    # individual outliers.
                    v_clipped = norm_old_values + (values - norm_old_values).clamp(
                        -self.clip_range_vf, self.clip_range_vf
                    )
                    v_loss_unclipped = (values - norm_returns).pow(2)
                    v_loss_clipped = (v_clipped - norm_returns).pow(2)
                    value_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = nn.functional.mse_loss(values, norm_returns)

                # ── Entropy bonus ─────────────────────────────────────
                entropy_loss = -entropy

                # ── Total loss ────────────────────────────────────────
                loss = pg_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    batch_kl = (old_log_prob - new_log_prob).mean().item()

                    # Per-head entropy bookkeeping (no extra forward).
                    batch_size = entropy.detach().shape if entropy.detach().dim() else None
                    for name, parts in per_head.items():
                        ent_ps = parts["entropy_per_sample"].detach()
                        weight = parts["weight"].detach()
                        head_ent_sum[name] += float(ent_ps.sum().item())
                        head_weighted_ent_sum[name] += float((ent_ps * weight).sum().item())
                        head_weight_sum[name] += float(weight.sum().item())
                        head_sample_count[name] += int(ent_ps.numel())
                    del batch_size  # silence unused warning

                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_fraction += clip_fraction
                total_kl += batch_kl
                epoch_kl += batch_kl
                n_updates += 1
                epoch_batches += 1

            epochs_completed += 1

            # KL early stopping: if mean KL this epoch exceeds target, the
            # data is too stale to update safely — stop before damaging the policy.
            if self.target_kl is not None and epoch_batches > 0:
                if (epoch_kl / epoch_batches) > self.target_kl:
                    break

        n_updates = max(n_updates, 1)
        ev = explained_variance(
            self.buffer.values[: self.n_steps], self.buffer.returns[: self.n_steps]
        )

        mean_entropy = total_entropy / n_updates
        self._last_entropy = mean_entropy  # used by entropy floor in _update_annealing

        result: dict[str, float] = {
            "policy_loss": total_pg_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": mean_entropy,
            "clip_fraction": total_clip_fraction / n_updates,
            "explained_variance": ev,
            "approx_kl": total_kl / n_updates,
            "epochs_completed": float(epochs_completed),
        }
        # Per-head entropy diagnostics (Phase 0).
        for name in head_names:
            count = max(head_sample_count[name], 1)
            uncond = head_ent_sum[name] / count
            weight_sum = head_weight_sum[name]
            cond = head_weighted_ent_sum[name] / weight_sum if weight_sum > 1e-9 else 0.0
            result[f"entropy_head_{name}"] = uncond
            result[f"entropy_head_{name}_cond"] = cond

        # Collapse flag: any head whose unconditional entropy dropped below
        # threshold for `entropy_collapse_consecutive_updates` consecutive updates.
        threshold = float(self.config.get("entropy_collapse_threshold", 0.0005))
        consecutive_required = int(self.config.get("entropy_collapse_consecutive_updates", 3))
        collapsed_heads: list[str] = []
        for name in head_names:
            if result[f"entropy_head_{name}"] < threshold:
                self._head_collapse_counter[name] += 1
                if self._head_collapse_counter[name] >= consecutive_required:
                    collapsed_heads.append(name)
            else:
                self._head_collapse_counter[name] = 0
        result["entropy_collapse_flag"] = 1.0 if collapsed_heads else 0.0
        if collapsed_heads:
            self._collapsed_heads_last = ",".join(collapsed_heads)
        return result

    def _update_annealing(self, update_num: int) -> None:
        """Apply entropy annealing and linear LR decay (Charlesworth-style)."""
        # Entropy annealing
        if self.use_entropy_annealing:
            start_u = self._entropy_anneal_start
            end_u = self._entropy_anneal_end
            if start_u <= update_num <= end_u:
                frac = (update_num - start_u) / max(end_u - start_u, 1)
                self.entropy_coef = self._entropy_coef_start + frac * (
                    self._entropy_coef_final - self._entropy_coef_start
                )
            elif update_num > end_u:
                self.entropy_coef = self._entropy_coef_final

        # Entropy floor: if the policy has become nearly deterministic, temporarily
        # raise entropy_coef to restore exploration. This fires after annealing has
        # set entropy_coef, so it only overrides when truly needed.
        if self._last_entropy < self.entropy_floor:
            self.entropy_coef = max(self.entropy_coef, self.entropy_floor_coef)

        # Linear LR decay
        if self.use_linear_lr_decay:
            num_updates = self.total_timesteps // self.n_steps
            if update_num < num_updates:
                frac = 1.0 - update_num / num_updates
                lr = self.lr_final + frac * (self.config["learning_rate"] - self.lr_final)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

    def train(self, verbose: bool = True) -> None:
        """Main training loop.

        Alternates between:
          1. Collecting experience (playing games)
          2. Updating the policy (learning from experience)

        Charlesworth-style: entropy annealing, linear LR decay, update-based eval.
        """
        print(
            f"Starting training for {self.total_timesteps:,} timesteps "
            f"({self.n_envs} envs, {self.n_steps} steps/update)..."
        )
        start_time = time.time()
        last_eval_step = 0
        last_checkpoint_step = 0
        eval_interval = self.config.get("eval_freq", self.checkpoint_freq)
        # Update count for annealing (resume-safe: derived from global_step)
        update_num = self.global_step // self.n_steps

        try:
            while self.global_step < self.total_timesteps:
                # ── Annealing (before collect, so update uses correct LR/entropy) ─
                self._update_annealing(update_num)

                # ── Collect experience ────────────────────────────────────
                rollout_stats = self.collect_rollouts()

                # ── Update policy ─────────────────────────────────────────
                update_stats = self.update()
                update_num += 1

                # ── League: add policy every N updates (Charlesworth-style) ─
                added = self.league.maybe_add(update_num, self._raw_policy_state_dict())
                if added:
                    self._adds_since_prune += 1
                    # Phase 3.5: Nash pruning fires after the add so the
                    # league's just-grown size is reflected.
                    self._maybe_run_nash_pruning()

                # Phase 3.4: ratings decay + scalar logging.
                self._decay_ratings_sigma()
                self._log_rating_scalars(self.writer, self.global_step)

                # Phase 3.3: duo exploiter cycle scheduling (stubbed).
                self._maybe_run_exploiter_cycle()

                # ── Logging ───────────────────────────────────────────────
                elapsed = time.time() - start_time
                fps = self.global_step / max(elapsed, 1e-8)

                if verbose:
                    wr = rollout_stats.get("mean_win_rate", 0)
                    mr = rollout_stats.get("mean_reward", 0)
                    ev = update_stats.get("explained_variance", 0)
                    ent = update_stats.get("entropy", 0)
                    kl = update_stats.get("approx_kl", 0)
                    ep = int(update_stats.get("epochs_completed", self.n_epochs))
                    print(
                        f"Step {self.global_step:>8,} | "
                        f"WR {wr:.2f} | "
                        f"R {mr:+.3f} | "
                        f"EV {ev:.2f} | "
                        f"Ent {ent:.3f} | "
                        f"KL {kl:.4f} | "
                        f"Ep {ep}/{self.n_epochs} | "
                        f"FPS {fps:.0f}"
                    )

                # TensorBoard logging
                for key, val in rollout_stats.items():
                    self.writer.add_scalar(f"rollout/{key}", val, self.global_step)
                for key, val in update_stats.items():
                    self.writer.add_scalar(f"train/{key}", val, self.global_step)
                self.writer.add_scalar("train/fps", fps, self.global_step)
                self.writer.add_scalar("train/entropy_coef", self.entropy_coef, self.global_step)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)

                # ── Periodic evaluation ───────────────────────────────────
                if self.global_step - last_eval_step >= eval_interval:
                    self.policy.eval()
                    eval_stats = self.eval_manager.evaluate(self.policy, self.device)
                    self.policy.train()

                    wr_now = eval_stats["win_rate"]
                    print(
                        f"  [EVAL vs {self._eval_opponent}] Win rate: {wr_now:.2f}, "
                        f"Avg VP: {eval_stats['avg_vp']:.1f}, "
                        f"Avg Length: {eval_stats['avg_game_length']:.0f}"
                    )

                    for key, val in eval_stats.items():
                        self.writer.add_scalar(f"eval/{key}", val, self.global_step)
                    self.writer.add_scalar(
                        "eval/opponent_is_heuristic",
                        1.0 if self._eval_opponent == "heuristic" else 0.0,
                        self.global_step,
                    )

                    # Upgrade eval opponent from random → heuristic once threshold is crossed
                    if self._eval_opponent == "random" and wr_now >= self._eval_upgrade_threshold:
                        self._eval_opponent = "heuristic"
                        self.eval_manager = EvaluationManager(
                            n_games=self.config.get("eval_games", 40),
                            opponent_type="heuristic",
                            max_turns=self.config.get("max_turns", 500),
                            use_thermometer_encoding=self._use_thermometer_encoding,
                        )
                        # Reset win-rate history — old random-opponent values are not comparable
                        self._eval_win_rate_history = []
                        print(
                            f"  [EVAL] Upgraded eval opponent to HEURISTIC "
                            f"(WR {wr_now:.2f} >= {self._eval_upgrade_threshold:.2f})"
                        )
                        self.writer.add_scalar("eval/opponent_upgraded", 1.0, self.global_step)

                    # Stagnation detection: warn if win rate hasn't improved
                    self._eval_win_rate_history.append(wr_now)
                    window = self._eval_win_rate_history[-self.stagnation_window :]
                    if len(window) >= self.stagnation_window:
                        improvement = max(window) - min(window)
                        if improvement < self.stagnation_threshold:
                            print(
                                f"  [WARN] STAGNATION detected: win rate range "
                                f"{min(window):.2f}–{max(window):.2f} over last "
                                f"{self.stagnation_window} evals (< {self.stagnation_threshold} "
                                f"improvement). Check EV and entropy in TensorBoard."
                            )
                            self.writer.add_scalar("train/stagnation_flag", 1.0, self.global_step)

                    last_eval_step = self.global_step

                    # Early stopping check
                    target = self.config.get("win_rate_target")
                    if target and wr_now >= target:
                        print(f"  Win rate target {target} reached! Saving final model.")
                        self.save(os.path.join(self.checkpoint_dir, "best_model.pt"))
                        break

                # ── Periodic checkpoint ───────────────────────────────────
                if self.global_step - last_checkpoint_step >= self.checkpoint_freq:
                    self.save(
                        os.path.join(self.checkpoint_dir, f"checkpoint_{self.global_step:08d}.pt")
                    )
                    last_checkpoint_step = self.global_step
        except KeyboardInterrupt:
            # Graceful interrupt: always save a checkpoint you can resume from.
            interrupt_path = os.path.join(
                self.checkpoint_dir,
                f"interrupt_{self.global_step:08d}.pt",
            )
            print("\nKeyboardInterrupt received. Saving interrupt checkpoint...")
            self.save(interrupt_path)
            self.writer.close()
            print(
                f"Training interrupted at step {self.global_step:,}. "
                f"Checkpoint saved to {interrupt_path}."
            )
            return

        # Final save
        self.save(os.path.join(self.checkpoint_dir, "final_model.pt"))
        self.writer.close()
        print(f"Training complete. Total steps: {self.global_step:,}")

    def save(self, path: str) -> None:
        """Save everything needed to resume training."""
        torch.save(
            {
                "policy_state_dict": self._raw_policy_state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "value_normalizer": self.value_normalizer.state_dict(),
                "config": self.config,
                "eval_win_rate_history": self._eval_win_rate_history,
                "eval_opponent": self._eval_opponent,
            },
            path,
        )
        print(f"  Saved checkpoint: {path}")

    @classmethod
    def load(cls, path: str, config_override: dict | None = None) -> "CatanPPO":
        """Load a saved trainer, optionally overriding config values."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if config_override:
            config.update(config_override)

        trainer = cls(config)
        # torch.compile wraps policy in OptimizedModule (_orig_mod prefix in state_dict keys).
        # Load into the underlying model to handle checkpoints saved before compile was enabled.
        policy_target = (
            trainer.policy._orig_mod if hasattr(trainer.policy, "_orig_mod") else trainer.policy
        )
        policy_target.load_state_dict(checkpoint["policy_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Force LR from config — overrides the stale LR stored in optimizer state.
        # This is needed when resuming with a deliberately reset learning rate.
        for g in trainer.optimizer.param_groups:
            g["lr"] = config["learning_rate"]
        trainer.global_step = checkpoint["global_step"]
        if "value_normalizer" in checkpoint:
            trainer.value_normalizer.load_state_dict(checkpoint["value_normalizer"])
        if "eval_win_rate_history" in checkpoint:
            trainer._eval_win_rate_history = checkpoint["eval_win_rate_history"]
        if "eval_opponent" in checkpoint:
            trainer._eval_opponent = checkpoint["eval_opponent"]
            trainer.eval_manager = EvaluationManager(
                n_games=config.get("eval_games", 40),
                opponent_type=trainer._eval_opponent,
                max_turns=config.get("max_turns", 500),
                use_thermometer_encoding=trainer._use_thermometer_encoding,
            )

        print(
            f"Loaded checkpoint from {path} (step {trainer.global_step:,}, "
            f"eval opponent: {trainer._eval_opponent})"
        )
        return trainer
