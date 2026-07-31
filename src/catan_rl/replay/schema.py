"""Replay-system schema for 1v1 Catan game records.

A replay is a JSON file describing every step of one Catan game. The
recorder (``catan_rl.replay.recorder``) produces it; the viewer
(``catan_rl.replay.viewer``) consumes it. The two scripts are
otherwise fully decoupled — the JSON schema is the only contract.

Schema layout (top level)::

    {
      "schema_version": 2,
      "metadata": { ... },
      "board_static": { ... },
      "steps": [ {...}, {...}, ... ]
    }

Per-step (``ReplayStep``)::

    {
      "step_idx": 0,
      "kind": "setup" | "main" | "terminal",
      "actor": "player_a" | "player_b",
      "dice_roll": [int, int] | null,
      "actions": [ {"kind": "...", "args": {...}}, ... ],
      "events":  [ {"kind": "...", ...}, ... ],
      "log_lines": ["...", "..."],
      "state_after": { ... },
      "policy_internals": [ {...}, ... ]      # v2, may be []
    }

The discriminator key on every event variant is **``kind``**, not
``type`` (a Python builtin). All resource keys follow Charlesworth
ordering (``WOOD, BRICK, WHEAT, ORE, SHEEP``) — see
:mod:`catan_rl.policy.obs_schema` for the source of truth.

Board geometry is stored in **axial coordinates** only — no pixel
coords — so a future viewer at any resolution can render without
re-recording. :mod:`catan_rl.replay.hex_math` converts (q, r) →
pixel at render time.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

#: Current on-disk schema version. Bumped whenever any of the keys
#: below change shape; ship a v(N-1) -> v(N) migration in
#: :mod:`catan_rl.replay.migrations` at the same time. Named
#: ``REPLAY_SCHEMA_VERSION`` (not ``SCHEMA_VERSION``) so it doesn't
#: collide with :data:`catan_rl.checkpoint.SCHEMA_VERSION`.
#:
#: v1 → v2 (2026-07-27): added the optional ``ReplayStep.policy_internals``
#: list and the ``Metadata`` provenance flags (``mode`` / ``sims`` /
#: ``clairvoyant`` / ``reveal_bot``). Both default to empty/False, so the
#: v1 → v2 migration in :mod:`catan_rl.replay.migrations` is a no-op bump.
REPLAY_SCHEMA_VERSION = 2


class ReplaySchemaError(RuntimeError):
    """Raised on a malformed payload during :func:`event_from_dict` or
    :func:`Replay.read_json` (corrupt JSON, missing required keys,
    schema version newer than this codebase supports, etc.)."""


# ---------------------------------------------------------------------------
# Resource / dev-card ordering
# ---------------------------------------------------------------------------

#: Resource ordering used in JSON keys + dataclass field order.
#: Matches :data:`catan_rl.policy.obs_schema.RESOURCES_CW` — the
#: project's canonical Charlesworth order. Re-declared here so the
#: replay package has zero runtime dependency on the policy package.
STATE_RESOURCE_ORDER: tuple[str, ...] = ("WOOD", "BRICK", "WHEAT", "ORE", "SHEEP")

#: Dev-card key ordering for the omniscient hand + the played-card
#: count. Matches the engine's ``DevCardOrder`` plus the conventional
#: VP-card slot.
STATE_DEV_CARD_ORDER: tuple[str, ...] = (
    "KNIGHT",
    "VP",
    "ROAD_BUILDER",
    "YEAR_OF_PLENTY",
    "MONOPOLY",
)


# ---------------------------------------------------------------------------
# Player + board geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerSpec:
    """Identity of one of the two players in this game."""

    kind: Literal["random", "heuristic", "policy", "human"]
    """``"human"`` marks a seat played by a person through the GUI
    harness (``scripts/play_vs_model.py``). It is NOT constructible as
    an actor — :func:`catan_rl.replay.player_factory.build_actor`
    rejects it — but it must be a legal *record* value so a human game
    is never mislabelled as ``"heuristic"``."""

    ckpt_path: str | None
    """Filesystem path to the checkpoint file when ``kind="policy"``;
    ``None`` for random/heuristic players."""

    color: str
    """Engine-side player color (e.g., ``"red"``, ``"blue"``). The
    viewer renders settlements and roads in this color."""

    seat_index: int
    """0 or 1 — the slot this player occupies in the engine's
    ``playerQueue`` at game start. ``player_a`` always refers to
    seat 0 by the JSON convention, regardless of which kind sits
    there. See :class:`Metadata.winner_seat` for who actually won."""


@dataclass(frozen=True, slots=True)
class HexStatic:
    """One hex tile's immutable identity. Pixel coords are NOT
    stored — the viewer computes them via :func:`hex_math.axial_to_pixel`
    against whatever window resolution it likes."""

    hex_idx: int
    q: int
    r: int
    resource: Literal["WOOD", "BRICK", "WHEAT", "ORE", "SHEEP", "DESERT"]
    number_token: int | None
    """``2-12`` excluding 7 for resource hexes; ``None`` for the desert."""

    has_robber_initial: bool


@dataclass(frozen=True, slots=True)
class VertexStatic:
    """One vertex's adjacency list. Variable-length: 1-3 hexes
    (interior vertices touch 3, coastal touch 1-2)."""

    vertex_idx: int
    adjacent_hex_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EdgeStatic:
    """One edge between two vertices."""

    edge_idx: int
    v1_idx: int
    v2_idx: int


@dataclass(frozen=True, slots=True)
class PortStatic:
    """One port's location + trade ratio."""

    port_idx: int
    vertex_idx_pair: tuple[int, int]
    ratio: Literal["2:1", "3:1"]
    resource: str | None
    """One of ``WOOD/BRICK/WHEAT/ORE/SHEEP`` for 2:1 specific ports;
    ``None`` for 3:1 generic ports."""


@dataclass(frozen=True, slots=True)
class BoardStatic:
    """Full board geometry — captured once at game start, immutable
    through the game.

    All coords are axial (q, r). Pixel rendering is the viewer's job.
    """

    hexes: tuple[HexStatic, ...]
    vertices: tuple[VertexStatic, ...]
    edges: tuple[EdgeStatic, ...]
    ports: tuple[PortStatic, ...]


# ---------------------------------------------------------------------------
# Sub-actions (what happened during a step)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubAction:
    """One atomic engine-level action taken during a step.

    The ``kind`` discriminator is one of:

    * ``"BuildSettlement"`` — ``args = {"vertex_idx": int}``
    * ``"BuildCity"`` — ``args = {"vertex_idx": int}``
    * ``"BuildRoad"`` — ``args = {"edge_idx": int}``
    * ``"RollDice"`` — ``args = {"d1": int, "d2": int}``
    * ``"BankTrade"`` — ``args = {"give_resource", "receive_resource", "give_amount"}``
    * ``"BuyDevCard"`` — ``args = {}``
    * ``"PlayKnight"`` — ``args = {}``
    * ``"PlayYearOfPlenty"`` — ``args = {"res_a": str, "res_b": str}``
    * ``"PlayMonopoly"`` — ``args = {"resource": str}``
    * ``"PlayRoadBuilder"`` — ``args = {}``
    * ``"MoveRobber"`` — ``args = {"hex_idx": int, "victim": str | None}``
    * ``"Discard"`` — ``args = {"discarded": {"WOOD": int, ...}}``
    * ``"EndTurn"`` — ``args = {}``

    The viewer renders these in the event-log panel but does not
    re-execute them — ``state_after`` is the source of truth.
    """

    kind: str
    args: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step events (markers on the step bar)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepEvent:
    """Base class for tagged-union step events. Subclasses set the
    ``kind`` class-var to their discriminator string."""

    kind: ClassVar[str] = "StepEvent"


@dataclass(frozen=True, slots=True)
class LongestRoadChange(StepEvent):
    """Holder of the Longest Road bonus changed this step."""

    kind: ClassVar[str] = "LongestRoadChange"
    prev_owner: str | None
    new_owner: str | None
    length: int


@dataclass(frozen=True, slots=True)
class LargestArmyChange(StepEvent):
    """Holder of the Largest Army bonus changed this step."""

    kind: ClassVar[str] = "LargestArmyChange"
    prev_owner: str | None
    new_owner: str | None
    knights: int


@dataclass(frozen=True, slots=True)
class Monopoly(StepEvent):
    """A Monopoly dev card was played; ``count`` resources of
    ``resource`` were transferred to ``player`` from the other
    player."""

    kind: ClassVar[str] = "Monopoly"
    player: str
    resource: Literal["WOOD", "BRICK", "WHEAT", "ORE", "SHEEP"]
    count: int


@dataclass(frozen=True, slots=True)
class Robber(StepEvent):
    """The robber was moved to ``hex_idx`` by ``player``."""

    kind: ClassVar[str] = "Robber"
    player: str
    hex_idx: int


@dataclass(frozen=True, slots=True)
class Steal(StepEvent):
    """``robber`` stole a card from ``victim``. ``resource`` is the
    actual resource if the recorder knows (omniscient context;
    typically the engine reveals it post-hoc); ``"UNKNOWN"`` if
    inference failed."""

    kind: ClassVar[str] = "Steal"
    robber: str
    victim: str
    resource: str


@dataclass(frozen=True, slots=True)
class GameEnd(StepEvent):
    """The game terminated this step. ``winner`` reached the VP cap;
    ``vp_breakdown`` maps each player_name → final VP count."""

    kind: ClassVar[str] = "GameEnd"
    winner: str
    vp_breakdown: dict[str, int]


@dataclass(frozen=True, slots=True)
class UnknownEvent(StepEvent):
    """Forward-compat catch-all. Produced by :func:`event_from_dict`
    when ``strict=False`` and the payload's ``kind`` isn't in
    :data:`EVENT_REGISTRY`. The viewer renders these as a neutral
    grey marker on the step bar so a v1 viewer can still scrub
    through a v2 replay that adds new event types."""

    kind: ClassVar[str] = "UnknownEvent"
    original_kind: str
    payload: dict[str, Any]


#: Dispatch table: ``kind`` → dataclass. Populated at module-import
#: time. Public so tests can iterate the known set; ``event_from_dict``
#: looks up here.
EVENT_REGISTRY: dict[str, type[StepEvent]] = {
    LongestRoadChange.kind: LongestRoadChange,
    LargestArmyChange.kind: LargestArmyChange,
    Monopoly.kind: Monopoly,
    Robber.kind: Robber,
    Steal.kind: Steal,
    GameEnd.kind: GameEnd,
}

_LOG = logging.getLogger("catan_rl.replay")


def event_to_dict(event: StepEvent) -> dict[str, Any]:
    """Serialise a :class:`StepEvent` instance into a JSON-safe dict.

    The output always carries ``kind`` as the discriminator. For
    :class:`UnknownEvent` we splat the saved payload back out so a
    round-trip leaves the original v2 event bytes intact (and a v1
    viewer hand-off remains lossless). The payload is deep-copied
    so the returned dict is fully independent of the stored
    ``UnknownEvent`` instance."""
    if isinstance(event, UnknownEvent):
        unknown_out = copy.deepcopy(event.payload)
        unknown_out["kind"] = event.original_kind
        return unknown_out
    # Standard variant: serialise dataclass fields keyed by name.
    # Defensively skip ``kind`` even though it's annotated as
    # ``ClassVar`` — depending on the Python / dataclass version,
    # inherited ClassVar fields can occasionally still appear in
    # ``__dataclass_fields__`` and we don't want a duplicate write
    # to clobber the discriminator with the instance attribute.
    out: dict[str, Any] = {"kind": event.kind}
    for fld in event.__dataclass_fields__.values():
        if fld.name == "kind":
            continue
        out[fld.name] = getattr(event, fld.name)
    return out


def event_from_dict(payload: dict[str, Any], *, strict: bool = False) -> StepEvent:
    """Parse one event dict into the right :class:`StepEvent` variant.

    Args:
        payload: A dict with a string ``"kind"`` key + variant-specific
            fields.
        strict: When ``True``, an unknown ``kind`` raises
            :class:`ReplaySchemaError`. When ``False`` (default),
            unknown ``kind`` logs a warning and returns
            :class:`UnknownEvent` carrying the original payload —
            forward-compat for a v1 viewer reading a v2 replay.

    Raises:
        ReplaySchemaError: if ``payload["kind"]`` is missing, or if
            ``strict=True`` and the kind isn't in
            :data:`EVENT_REGISTRY`, or if a known variant is missing
            a required field.
    """
    if "kind" not in payload:
        raise ReplaySchemaError(f"event payload missing 'kind' discriminator: {payload!r}")
    kind = payload["kind"]
    if not isinstance(kind, str):
        raise ReplaySchemaError(f"event 'kind' must be str, got {type(kind).__name__}")
    cls = EVENT_REGISTRY.get(kind)
    if cls is None:
        if strict:
            raise ReplaySchemaError(
                f"unknown event kind {kind!r}; known kinds: {sorted(EVENT_REGISTRY)}"
            )
        _LOG.warning("replay: unknown event kind %r; rendering as UnknownEvent", kind)
        # Deep-copy the payload so subsequent mutation of the caller's
        # source dict cannot corrupt the stored ``UnknownEvent``.
        return UnknownEvent(original_kind=kind, payload=copy.deepcopy(payload))
    # Build kwargs by inspecting the dataclass fields. Explicitly
    # skip the ``kind`` discriminator (it's a ClassVar on each
    # variant; certain Python / dataclass version combinations leak
    # it into ``__dataclass_fields__`` and the constructor refuses it
    # as a kwarg).
    kwargs: dict[str, Any] = {}
    for fld in cls.__dataclass_fields__.values():
        if fld.name == "kind":
            continue
        if fld.name not in payload:
            raise ReplaySchemaError(
                f"event {kind!r} missing required field {fld.name!r}: {payload!r}"
            )
        kwargs[fld.name] = payload[fld.name]
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Player + step state snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerStateSnapshot:
    """One player's state at end of a step. All counts are absolute
    (not deltas). Built by deep-copy at capture time so subsequent
    engine mutations cannot retroactively alter the snapshot."""

    name: str
    """Engine-side player name (``"Agent"`` or ``"Opponent"``); the
    JSON-level alias is ``actor`` = ``player_a`` / ``player_b`` based
    on seat."""

    vp: int
    resources: dict[str, int]
    """Charlesworth-ordered keys; values are non-negative ints."""

    dev_cards_hand: dict[str, int]
    """Omniscient hand — the engine merges
    ``player.devCards + player.newDevCards`` at capture time so the
    viewer sees what the player actually holds at end-of-step (not
    the engine's internal two-bucket split)."""

    dev_cards_played: dict[str, int]
    """Cumulative count of each dev-card type the player has played
    so far. ``VP`` here is the count of VP cards the player owns
    (treated as "played" since they're permanent points)."""


@dataclass(frozen=True, slots=True)
class StepStateSnapshot:
    """Complete game state at end of one step. Captured by deep-copy."""

    settlements: dict[str, tuple[int, ...]]
    """``"player_a" -> tuple[vertex_idx, ...]``."""

    cities: dict[str, tuple[int, ...]]
    roads: dict[str, tuple[int, ...]]
    """``"player_a" -> tuple[edge_idx, ...]``."""

    robber_hex: int
    players: dict[str, PlayerStateSnapshot]
    """``"player_a" -> snapshot, "player_b" -> snapshot``."""

    longest_road_holder: str | None
    largest_army_holder: str | None

    last_seven_roller: str | None = None
    """The most recent player to have rolled a 7 (engine
    ``dice.py:last_7_roller_obj``). Persists across turns until ANOTHER
    7 rolls — this is the Karma-buff invariant. The viewer uses it to
    render a "Karma armed" indicator on the OTHER player's panel.
    Defaults to ``None`` for forward compat with v1 replays written
    before this field was added (the writer always populates it; the
    reader treats it as optional for tolerance)."""


# ---------------------------------------------------------------------------
# Steps + replay top-level
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyInternals:
    """What the policy was weighing at ONE decision (v2).

    Recorded only for steps a neural policy chose; human / heuristic /
    random steps carry none. Every value is a plain Python ``int`` /
    ``float`` / ``bool`` — tensors are converted at the capture
    boundary so the viewer's no-torch import contract holds.

    **Interpretation warning.** ``type_probs`` is a 13-way softmax over
    action TYPES. At most steps it reads roughly ``0.97 END_TURN``; it
    is a hypothesis generator for contested decisions (build-vs-trade,
    robber placement), never a verdict, and the policy grading its own
    homework is a known failure mode."""

    type_mask: tuple[bool, ...]
    """Legality of each of the 13 action types at this decision."""

    type_probs: tuple[float, ...]
    """DENSE masked-softmax probabilities over the 13 action types."""

    chosen_action: tuple[int, ...]
    """The sampled 6-tuple ``[type, corner, edge, tile, res1, res2]``."""

    value: float
    """The value head's scalar estimate for the pre-action state."""

    corner_top: tuple[tuple[int, float], ...] = ()
    """``(index, prob)`` pairs for the TOP-8 corner-head entries only.
    Dense 54/72/19 distributions roughly double the file for no human-
    review value, so the pointer heads are truncated.

    EMPTY when the chosen action type does not consult this head (per
    ``MultiActionHeads.head_relevance``): an ``END_TURN`` never looked
    at a corner, and its all-False mask would otherwise be recorded as
    ``masked_log_softmax``'s uniform safe fallback — pure noise wearing
    the shape of an opinion. Empty means "not consulted", NOT
    "undecided"."""

    edge_top: tuple[tuple[int, float], ...] = ()
    tile_top: tuple[tuple[int, float], ...] = ()

    res1_probs: tuple[float, ...] = ()
    """DENSE 5-wide masked-softmax over the FIRST resource argument
    (Charlesworth order) — the card given / taken / monopolised /
    discarded. Only 5 floats, so no truncation. Empty unless the type
    consults it (YoP, Monopoly, BankTrade, Discard)."""

    res2_probs: tuple[float, ...] = ()
    """DENSE 5-wide masked-softmax over the SECOND resource argument
    (YoP's 2nd card, BankTrade's receive). Empty for every other type."""

    belief_logits: tuple[float, ...] | None = None
    """Raw aux belief-head logits over the opponent's hidden dev-card
    types, or ``None`` when the policy has no belief head."""


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """One step of the replay — corresponds to one player turn under
    the agreed step semantics (see module docstring for details)."""

    step_idx: int
    kind: Literal["setup", "main", "terminal"]
    actor: Literal["player_a", "player_b"]
    dice_roll: tuple[int, int] | None
    actions: tuple[SubAction, ...]
    events: tuple[StepEvent, ...]
    log_lines: tuple[str, ...]
    state_after: StepStateSnapshot

    policy_internals: tuple[PolicyInternals, ...] = ()
    """v2, optional. Empty for every step not chosen by a neural policy
    (human, heuristic, random, and any policy step recorded before the
    capture landed). Appended LAST so v1 positional construction of
    :class:`ReplayStep` is unchanged."""


@dataclass(frozen=True, slots=True)
class Metadata:
    """Top-level metadata for one replay."""

    player_a: PlayerSpec
    player_b: PlayerSpec
    seed: int
    max_turns: int
    intended_hex_size: tuple[int, int]
    """The canvas size (W, H) the recorder rendered against. Used by
    the viewer for proportional layout if it wants to fit-to-window
    using the recorder's aspect ratio. Pixel coords are NOT
    derived from this — see module docstring for the axial-only
    contract."""

    recorded_at_utc: str
    """ISO-8601 timestamp; for human reference only."""

    winner: str | None
    winner_seat: int | None
    """0 or 1 (matches a ``PlayerSpec.seat_index``), or ``None`` if
    the game truncated without a winner."""

    final_vp: tuple[int, int]
    """``(vp_seat_0, vp_seat_1)``."""

    total_steps: int
    partial: bool
    """``True`` if the game did NOT play out to a terminal state — the
    driver's loop was cut short (window close, ``KeyboardInterrupt``,
    crash) and the record was salvaged anyway. Set by
    ``scripts/play_vs_model.py``; a ``partial`` record is a valid trace
    of the steps it contains but is NOT a valid outcome/strength read.
    The batch recorder writes only at end-of-game, so its records are
    always ``False``."""

    mode: str = "raw_policy"
    """v2. ``"raw_policy"`` or ``"search"`` — how the policy seat chose
    its moves."""

    sims: int | None = None
    """v2. MCTS sims/move when ``mode == "search"``; ``None`` otherwise."""

    clairvoyant: bool = False
    """v2. ``True`` when the mover had access to information a real
    opponent would not. The deployed determinized search deep-copies the
    env, so its simulated worlds share the opponent's true hidden dev
    cards AND the true future dice — a game recorded with
    ``clairvoyant=True`` is NOT a valid strength read."""

    reveal_bot: bool = False
    """v2. ``True`` when the human could see the policy's full hand
    (``play_vs_model.py --reveal-bot``). Also not a valid strength read;
    recorded so a revealed analysis game can never be misfiled as one."""

    git_sha: str | None = None
    """v2. ``git rev-parse HEAD`` of the tree the game was played on, or
    ``None`` when it could not be read. Without it a recorded game names no
    code, and the ruleset, the obs schema and the harness all move.

    MAY CARRY A ``-dirty`` SUFFIX, making the value 45 chars rather than 40:
    the writer appends it when the game-affecting trees (``src scripts crates
    configs``) had uncommitted edits, because a bare SHA would then name code
    the game was NOT played on. Consumers must not assume a 40-char string —
    compare on ``value.removesuffix("-dirty")`` and read the suffix as a flag."""

    ckpt_sha256: str | None = None
    """v2. SHA-256 of the checkpoint FILE the policy seat was loaded from, or
    ``None`` when it could not be read. ``PlayerSpec.ckpt_path`` alone is not
    attribution: it points into a gitignored ``runs/`` directory under a
    ``keep_last_n`` rotation, so tomorrow the same path may hold different
    bytes, or nothing at all."""


@dataclass(frozen=True, slots=True)
class Replay:
    """One complete game replay. Top-level JSON object."""

    schema_version: int
    metadata: Metadata
    board_static: BoardStatic
    steps: tuple[ReplayStep, ...]
