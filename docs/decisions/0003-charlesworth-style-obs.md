# ADR 0003: Charlesworth-Style Dict Observation

**Status:** Accepted
**Date:** 2026-04-30

## Context

Catan state has heterogeneous structure: 19 hex tiles, 54 vertices, 72 edges, plus per-player scalars (resources, VP, dev cards, ports). A flat-vector observation conflates all of this and makes architectural priors hard to encode.

Henry Charlesworth (2018) showed that a **dict observation** — per-tile features encoded by a transformer, dev-card sequences encoded by attention, player scalars by MLP, all fused — performed substantially better than flat-vector PPO baselines on 4-player Catan.

## Decision

Adopt a Charlesworth-style dict observation:

| Component | Shape | Encoder |
|---|---|---|
| `tile_representations` | `(19, 79)` | TransformerEncoder (2 layers, 4 heads) → projection |
| `current_player_main` | `(166,)` | MLP |
| `next_player_main` | `(173,)` | MLP |
| `*_dev_*` (3 sequences) | `(15,)` int | Embedding + multi-head attention |

Final fusion: concat → Linear → LayerNorm → ReLU → 512-dim state vector.

## Consequences

- Network has 1.54M params — small enough for CPU-only training on M1 Pro.
- Tile transformer has no positional encoding (Phase 2.1 will add 2D axial pos-emb).
- Per-tile self-attention is over 19 tokens — O(n²) is irrelevant; the value is in the global readout.
- Dev-card MHA over a 15-padded sequence is wasteful (sequence is a multiset). Phase 1.4 of the roadmap replaces it with count encoding.
- Phase 2.3 will add a bipartite hex/vertex/edge GNN that exploits Catan's graph structure more directly than the transformer can.

## Alternatives Considered

- **Flat-vector PPO.** Rejected based on Charlesworth's empirical results.
- **Pure GNN from day 1.** More inductive bias but harder to debug; deferred to Phase 2.3 once the trainer is otherwise stable.

## Related

- `src/catan_rl/models/observation/`
- `docs/obs_schema.md`
