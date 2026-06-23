# ThePhantom video-parsing — BLOCKER resolution (empirical)

Video: https://www.youtube.com/watch?v=9Sm86ml04aI  (1080p, dur 4739.5s ≈ 79 min)
POV / "You" = **ThePhantom** (bottom-right self-seat in HUD). Opponent = rayman147 (GREEN).
The video contains **MANY back-to-back games**, not one. (Critical: the prior
artifacts mixed frames from different games.)

================================================================================
## BLOCKER 1 — WINNER / TERMINAL  →  VERDICT: GO (with caveats)
================================================================================

### (1) The terminal IS readable — exact victory log line (eyeball-confirmed)

The reliable terminal signal is the Colonist LOG line:

    🏆  <player> won the game!  🏆

Captured literally for the long game ending at **t≈4294.6s** (OCR + visual):

    "rayman147 won the game!"     (preceded by "rayman147 built a Settlement (+1 VP)")
    then  "rayman147: gg" / "ThePhantom: gg"

  → Long game winner = **rayman147**.  ThePhantom (POV) LOST this game.
  Artifact: overlays/B1_victory_log_4295.png  (clean, trophy icons both sides)

New-game reset marker (board cleared): the log shows the empty-log placeholder

    "Happy settling! Learn how to play in the rulebook / List of commands: /help"

then setup lines ("<player> placed a Settlement / placed a Road / received
starting resources"). Reset for the next game seen at t≈4302s.

A post-game end-screen also appears for several seconds: a side panel
("Map / Replay / Share / Resource Stats") + a center modal banner with
"Time: MM:SS - Turns: N" and per-category stat tabs. This is a SECOND, longer-
lived terminal marker than the single log line.

### (2) The top-left "X - Y" counter is NOT the game score

Tracked across one whole game (engine-OCR'd):
  t=100 0-0 | t=1200 1-1 | t=1800 1-1 | t=2400 1-2 | t=3000 2-2 | t=3600 3-2 |
  t=4200 3-2 | t=4400 (NEXT game) 3-3.
It increments by 1 occasionally, roughly symmetric, **does not reset between
games** (3-2 at end → 3-3 next game), and is ~3 at the end of a game the winner
took with 15 VP. So it is NOT VP, NOT a per-game win/series counter (a series
counter would read 0-1/1-0 after one game, not 3-3). It sits beside the
settings/spectator/rulebook UI icons; most consistent with a cumulative
session tally (e.g. 7s/robber/knights). **Do not use it for anything.**

### (3) Verdict + edge cases
GO: the per-GAME winner is reliably read from the LOG line "X won the game!".
Recommended production rule:
  • Primary winner = the player name on the "won the game!" log line.
  • Confirm with the end-screen panel (Replay/Share/Resource Stats) as a
    persistence-robust terminal detector, then read backwards for the win line.
  • Segment games by the "Happy settling / List of commands" reset marker.

CAVEATS / residual risk (honest):
  • The win LOG line is on-screen only briefly before the log auto-clears / the
    end-screen modal covers it. For game A (ended ~t=597) the win line had
    already scrolled past the crop — only the end-screen ("Well Played!",
    Time 10:40, Turns 71) was visible. So you MUST sample finely (≤1-2s) around
    the end-screen, or OCR the end-screen + scroll-back, to catch the line.
  • The center banner text "Victory!!!" vs "Well Played!" is NOT a trustworthy
    per-POV winner signal: the long game showed "Victory!!!" yet the log said
    rayman147 (not the ThePhantom POV) won. Use the LOG line, not the banner.
  • resign/forfeit & video-cut-off: if no "won the game!" line and no end-screen
    is ever found for a game segment (e.g. the video cuts off mid-game — which
    the LAST game in THIS video does, it is still in play at t=4739), set
    winner = null for that segment. A "<player> left/forfeited" line (if present)
    would also imply the other player won; not observed in sampled frames.

================================================================================
## BLOCKER 2 — BOARD ORIENTATION LOCK  →  VERDICT: GO (resolved)
================================================================================

### Root cause (proven, diag_residuals.py)
The 19 hex centers form a regular hex lattice with a D6 (12-element) symmetry.
All 12 candidate affines fit the detected tokens with **identical** geometric
residual — best-vs-2nd gap = **0.00 px** on every frame. So "pick lowest
residual" is decided by FP/argmin noise and silently flips rotation/reflection
between frames while keeping residual <2px. THAT is what relabelled every
hex/vertex/edge id across board_120 vs board_500.

### The fix (orient_lock2.py): screen-anchored + content-verified, redundant
PART A (geometric, content-free, deterministic): choose the unique D6 orientation
  by a SCREEN-SPACE rule — engine hex 8 must land top-center, engine hex 11 must
  land rightmost. This ordered, non-collinear landmark pair fixes rotation AND
  reflection. Penalty separation is huge: best 2.9 vs 2nd 423.9 (146×), NO tie.
PART B (independent cross-check): ≥2 ground-truth anchors, each an INDEPENDENT
  read of a NUMBER token AND its hex RESOURCE colour, at fixed engine ids
  (H8 = SHEEP/2, H16 = ORE/10, eyeball-confirmed). Exactly **1 of 12**
  orientations passes the content check; the other 11 are rejected. If PART A
  and PART B disagree, the frame is REJECTED (no silently-flipped board emitted).

### Frame-stability proof (prove_stability.py) — the proof the prior lock lacked
Ran the locked reader on game-A frames. On the 5 fully-rendered frames
(t=240,350,450,500,540) the engine-indexed board map is **19/19 hexes BYTE-
IDENTICAL**, desert = hex 11 on every frame:

  see board/AGREEMENT_TABLE.txt   (all rows OK; ALL 19 IDENTICAL = True)

  (Setup frames t=120/180 differ on H6 number and H11 desert-vs-wheat ONLY
   because the board isn't fully rendered yet — robber not placed / token not
   shown — a content-OCR artifact, NOT an orientation flip; the multiset/None
   checks catch it. t=300 is an animation frame with only 12 tokens → REJECTED,
   correctly.)

### Consistency check rejects a deliberate mis-orientation (catches a flip in prod)
Forcing any of the 11 wrong orientations and running PART B → all FAIL the
anchor check → rejected. Verified across all 12 (exactly 1 accepted).

### Verdict + residual risk
GO: the new lock makes the board map frame-stable across the same game. The
desert anchor was never used (per the blocker's instruction); orientation is
fixed by screen geometry + 2 redundant OCR anchors.
Residual risk:
  • Anchors (H8=SHEEP/2, H16=ORE/10) are board-specific; for a NEW game's board
    you must re-establish the anchors once (run PART A screen-rule, eyeball 2
    hexes) — or better, drop hard-coded content anchors and rely on PART A
    (screen rule) alone, which already uniquely picks the orientation (146× gap),
    using PART B only as a self-consistency reject.
  • Token detector occasionally drops/adds a token during animations
    (t=300: 12 toks; some frames: 19 toks = a piece misread as a token) → the
    affine residual blows up (>40px) and the frame should be skipped; the lock
    already rejects these via the content check. Add an explicit residual gate
    (e.g. reject if mean residual > 5px) in production.

================================================================================
## KEY ARTIFACTS (/tmp/phantom/blockers/)
================================================================================
BLOCKER 1:
  overlays/B1_victory_log_4295.png       "rayman147 won the game!" (clean)
  overlays/B1_victory_modal_4295.png     "Victory!!!" end-screen (long game)
  overlays/B1_gameA_modal_598.png        "Well Played!" end-screen (game A)
  overlays/B1_game1_end_full_t4295.png   full end frame
  ocr/scan_*.json                        log OCR across the whole video
  scan_terminal.py / scan_fine.py        the scanners
BLOCKER 2:
  board/AGREEMENT_TABLE.txt              cross-frame 19/19 agreement proof
  board/locked_board_{240,350,450,500,540}.json   per-frame locked boards
  board/stability_result.json           machine-readable proof
  overlays/B2_locked_overlay_{240,500}.png        annotated engine-id overlays
  overlays/B2_locked_zoom_500.png        zoom (desert=H11, anchors circled)
  orient_lock.py / orient_lock2.py       the deterministic lock
  diag_residuals.py                      root-cause proof (0.00px gap)
  prove_stability.py                     the stability + rejection harness
