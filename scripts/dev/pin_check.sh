#!/bin/bash
# pin_check.sh — the identical-output pin for pipeline speed changes (council Section 4).
#
# Re-runs prepare-frames on the proven pin video (9Sm86ml04aI) under the sweep-exact
# env pins and requires, versus the archived reference:
#   a. deep-diff of the FULL meta.json for every game dir (every key; any volatile key
#      must be EXPLICITLY listed in VOLATILE_KEYS below, never silently excluded),
#   b. sha256 match on post_setup.png and empty_baseline.png for every game,
#   c. exact integer match of the '[ocr-cache] hits= misses=' counters,
#   d. '[ocr-threads]' line present (and, post thread-pin bundle, showing cv2=1).
#
# Usage:
#   scripts/dev/pin_check.sh baseline   # archive reference frames + stdout (run ONCE,
#                                       # under CURRENT committed code, sweep idle)
#   scripts/dev/pin_check.sh check      # after a change: re-run and diff vs reference
#
# DO NOT run while a sweep is active (it competes for CPU and the OCR timing-derived
# frame schedule is deterministic, but the run takes ~2h of the machine).
set -euo pipefail
cd "$(dirname "$0")/../.."

VIDEO=9Sm86ml04aI
REF=runs/human/pin_reference
FRAMES=data/human/vlm_spike/frames
VOLATILE_KEYS='["post_setup_png","empty_baseline_png"]'   # absolute paths; explicitly excluded
ENV_PINS=(OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CATAN_OCR_THREADS=1)

run_prepare() {
  env "${ENV_PINS[@]}" PYTHONPATH=src .venv/bin/python scripts/vlm_spike.py \
    prepare-frames --video "$VIDEO" 2>&1 | tee "$1"
}

case "${1:-}" in
  baseline)
    mkdir -p "$REF"
    echo "[pin] baseline run under current committed code…"
    run_prepare "$REF/baseline_stdout.txt"
    for d in "$FRAMES/${VIDEO}"__g*; do
      cp -R "$d" "$REF/$(basename "$d")"
    done
    grep -E '^\[ocr-(cache|threads)\]' "$REF/baseline_stdout.txt" || true
    echo "[pin] reference archived to $REF"
    ;;
  check)
    test -d "$REF" || { echo "[pin] FAIL: no reference — run 'baseline' first"; exit 2; }
    echo "[pin] check run under candidate code…"
    run_prepare "$REF/check_stdout.txt"
    fail=0
    # (c) ocr-cache integer canary
    base_cache=$(grep -m1 '^\[ocr-cache\]' "$REF/baseline_stdout.txt" || true)
    new_cache=$(grep -m1 '^\[ocr-cache\]' "$REF/check_stdout.txt" || true)
    if [ "$base_cache" != "$new_cache" ]; then
      echo "[pin] FAIL (c): ocr-cache counters differ:"; echo "  base: $base_cache"; echo "  new : $new_cache"; fail=1
    else echo "[pin] ok (c): $new_cache"; fi
    # (d) ocr-threads line present
    grep -m1 '^\[ocr-threads\]' "$REF/check_stdout.txt" || { echo "[pin] WARN (d): no [ocr-threads] line"; }
    # (a)+(b) per-game full meta diff + PNG hashes
    for refdir in "$REF/${VIDEO}"__g*; do
      g=$(basename "$refdir"); newdir="$FRAMES/$g"
      test -d "$newdir" || { echo "[pin] FAIL: game dir $g missing in check run"; fail=1; continue; }
      .venv/bin/python - "$refdir" "$newdir" "$VOLATILE_KEYS" <<'PY' || fail=1
import json, sys, hashlib, pathlib
ref, new, volatile = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), set(json.loads(sys.argv[3]))
a = json.loads((ref/"meta.json").read_text()); b = json.loads((new/"meta.json").read_text())
for k in volatile: a.pop(k, None); b.pop(k, None)
ok = True
keys = sorted(set(a) | set(b))
for k in keys:
    if a.get(k) != b.get(k):
        print(f"[pin] FAIL (a): {ref.name} meta key '{k}' differs"); ok = False
for png in ("post_setup.png", "empty_baseline.png"):
    pa, pb = ref/png, new/png
    if pa.exists() != pb.exists():
        print(f"[pin] FAIL (b): {ref.name}/{png} presence differs"); ok = False
    elif pa.exists():
        ha = hashlib.sha256(pa.read_bytes()).hexdigest(); hb = hashlib.sha256(pb.read_bytes()).hexdigest()
        if ha != hb: print(f"[pin] FAIL (b): {ref.name}/{png} sha256 differs"); ok = False
print(f"[pin] {'ok' if ok else 'FAIL'}: {ref.name}")
sys.exit(0 if ok else 1)
PY
    done
    if [ "$fail" -eq 0 ]; then echo "[pin] PASS — outputs identical to reference"; else echo "[pin] FAIL — do NOT ship this change"; exit 1; fi
    ;;
  *)
    echo "usage: $0 {baseline|check}"; exit 2;;
esac
