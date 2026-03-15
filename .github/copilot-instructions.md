# Model Usage & Quota Management
- **DEFAULT MODEL:** Always assume the use of standard, unlimited models (e.g., GPT-4o or Claude 4.5 Sonnet) for routine queries.
- **RESTRICTION:** DO NOT suggest or initiate a switch to "Premium" models (Claude 4.6 Opus / GPT-5) unless I explicitly request a "High-Reasoning Deep Dive."
- **TRIGGER CRITERIA:** Only recommend a model upgrade if the current task involves:
    1. Complex Reinforcement Learning math (e.g., PPO gradient clipping or vector geometry).
    2. Deep architectural refactoring of the `env.py` or `train.py` logic.
    3. Resolving "stalled" training performance that simpler models have failed to diagnose.
- **CONFIRMATION:** If you believe a task requires Claude 4.6, preface your response with: "⚠️ This task is mathematically complex. Should I proceed using your Claude 4.6 quota?"
- **CONCISENESS:** Keep responses brief for simple Git or Python syntax questions to minimize token usage across all models.