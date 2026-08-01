"""Ruleset epochs for the 1v1 Colonist.io env (spec ``preroll-dev-cards-r1``).

Two epochs exist, and eval numbers are only comparable **within** one:

* ``R0`` — the shipped rules up to 2026-07. At a ``roll_pending`` node the
  action mask offers exactly ``ROLL_DICE``, so **no policy seat** — neither the
  learner nor a snapshot opponent — may play a dev card before rolling.
* ``R1`` — Colonist-faithful. Both policy seats may play **one** dev card from
  :data:`PRE_ROLL_DEV_TYPES` before the dice.

**The epoch scopes the MASK, not the scripted heuristic.** The heuristic
opponent's pre-roll Knight (``agents/heuristic.py:heuristic_pre_roll``, driven
unconditionally by ``CatanEnv._opponent_pre_roll``) is shipped R0 behaviour and
deliberately runs under BOTH epochs. It never goes through
``compute_action_masks``, and every banked R0 number was measured with it on —
so gating it on R1 would leave ``ruleset="R0"`` unable to reproduce the epoch
it names, the opposite of this module's purpose. The asymmetry R1 removes is
the one between a *learner* and its *policy* opponents.

**Defaults.** ``compute_action_masks``, :class:`catan_rl.env.CatanEnv` and
``EvalHarness`` all default to ``R0`` — no caller changes the rules a
checkpoint is judged under merely by upgrading. Training opts into R1
explicitly through ``RolloutConfig.ruleset``, which lands in the saved
checkpoint config and is what eval reads back to detect a cross-epoch h2h.

This lives in its own module (rather than in ``masks.py`` or ``catan_env.py``)
because ``env/masks.py``, ``env/catan_env.py``, ``eval/harness.py``,
``eval/cross_arch.py``, ``conformance/recorder.py`` and
``scripts/play_vs_model.py`` all need the constants, and ``masks.py`` must not
import ``catan_env``.
"""

from __future__ import annotations

from catan_rl.policy.obs_schema import ActionType

#: The pre-2026-07 epoch: no pre-roll dev-card window on either seat.
RULESET_R0 = "R0"
#: The Colonist-faithful epoch: one pre-roll dev card per turn, both seats.
RULESET_R1 = "R1"

VALID_RULESETS: frozenset[str] = frozenset({RULESET_R0, RULESET_R1})

#: Dev-card action types legal at a ``roll_pending`` node under R1.
#:
#: ``PLAY_ROAD_BUILDER`` is deliberately ABSENT (spec D4). It writes
#: ``road_building_roads_left = 2`` while ``roll_pending`` is still true, and
#: the road-builder block sits BELOW the roll block in both ``masks.py`` and
#: ``catan_env.step`` — so the two free roads would defer across the roll, a
#: possible 7, the discard and the robber. Hoisting the road-builder block
#: above the roll gate is a worse trap: its no-legal-road fallback sets
#: ``type_mask[END_TURN]``, making a turn that never rolls reachable. Roads
#: cannot interact with a dice roll, so excluding it deletes the whole failure
#: class at ~zero tactical cost.
PRE_ROLL_DEV_TYPES: frozenset[int] = frozenset(
    {
        int(ActionType.PLAY_KNIGHT),
        int(ActionType.PLAY_YOP),
        int(ActionType.PLAY_MONOPOLY),
    }
)

#: Engine ``devCards`` keys corresponding to :data:`PRE_ROLL_DEV_TYPES`.
#:
#: Lets a caller answer "could this player play anything pre-roll?" from the
#: hand alone, without building an obs + mask + network sample. Keep in sync
#: with :data:`PRE_ROLL_DEV_TYPES` — ``tests/unit/env/test_preroll.py`` pins
#: the two as the same set.
PRE_ROLL_DEV_CARD_NAMES: frozenset[str] = frozenset({"KNIGHT", "YEAROFPLENTY", "MONOPOLY"})


def validate_ruleset(ruleset: str) -> str:
    """Return ``ruleset`` if it names a known epoch, else raise ``ValueError``."""
    if ruleset not in VALID_RULESETS:
        raise ValueError(f"ruleset={ruleset!r}; supported: {sorted(VALID_RULESETS)}")
    return ruleset
