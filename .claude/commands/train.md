---
description: Start or resume a training run with sane defaults
---

Start training the Catan agent.

If `$ARGUMENTS` contains a checkpoint path, resume from it; otherwise start fresh.

Steps:
1. Confirm the user wants to start training (compute-expensive, days-long).
2. If resuming, verify the checkpoint exists and print its `global_step`.
3. Launch `python scripts/train.py --verbose` (with `--resume <ckpt>` if applicable) in the background.
4. Print the TensorBoard command to monitor: `tensorboard --logdir runs/train/`.
5. Report the PID and how to gracefully interrupt (Ctrl-C → auto-saves `interrupt_*.pt`).

Do NOT auto-launch without confirmation. Do NOT modify `arguments.py` unless the user asked.
