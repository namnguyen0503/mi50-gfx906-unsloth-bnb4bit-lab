# Gemma4 Triton-gfx906 Fused CE Patch

This directory documents the experimental patch to enable `seq8192` LoRA training on the 32GB MI50/gfx906.

## Features
- **Triton-gfx906 all-text-attention patch**: Monkeypatches all 60 text attention layers to use memory-efficient Triton kernels instead of standard SDPA. This avoids the 8 GiB dense fp32 sliding SDPA workspace allocation at `seq8192`.
- **Fused active linear CE**: Bypasses the standard full-logits Cross-Entropy calculation. Instead of materializing `[tokens × vocab]` dense logits (which takes ~8 GiB at `seq8192`), it fuses the operation, drastically reducing peak memory.

## Caveats
- Experimental text-only path.
- Not vision-compatible.
- Process-local monkeypatch.
- Requires `nlzy/triton-gfx906` via `PYTHONPATH`.
- Memory/capability optimization, not speed optimization.
- Not production packaging.
