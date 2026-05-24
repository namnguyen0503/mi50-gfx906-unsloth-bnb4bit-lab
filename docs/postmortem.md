# Postmortem

## What consumed the most time

- Rebuilding bitsandbytes multiple times before isolating GPU visibility root cause.
- Re-running large-model tests before completing minimal runtime probes.
- Environment drift between two virtualenvs with different HIP build behavior.

## Core lessons

1. In mixed-GPU systems, solve visibility/dispatch first (`ROCR_VISIBLE_DEVICES`).
2. Verify basic torch kernels before model-level debugging.
3. Keep a single golden environment and avoid silent package drift.
4. Treat isolated kernel wins as necessary but not sufficient for end-to-end success.

## Final engineering conclusions

- Stable path exists for MI50 with Gemma4-31B at moderate context.
- Long-context Gemma4 LoRA in this stack remains memory-limited.
- noflash experiment is a `NEGATIVE_RESULT` for the stated training goal.
