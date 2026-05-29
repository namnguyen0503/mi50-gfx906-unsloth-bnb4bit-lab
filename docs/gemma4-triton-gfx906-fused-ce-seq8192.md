# Gemma4 Triton gfx906 Fused CE Seq8192

## Overview

This document describes a new experimental result: Gemma4 31B bnb4bit real-active seq8192 LoRA training on a single AMD MI50 32GB/gfx906. This was achieved using a Triton-gfx906 all-text-attention patch combined with a fused active linear Cross-Entropy (CE) loss implementation.

## Key Features

- **Context Length**: `seq8192` passed for LoRA `r8`, `r16`, and `r32`.
- **Active Tokens**: `active_tokens=8191`.
- **LoRA Target**: Original target includes attention + MLP.
- **Attention Patch**: All 60 text attention layers patched.
- **Loss**: Fused active linear CE.
- **Success Criteria**: Forward, backward, and optimizer step all passed successfully with finite loss and no NaN/Inf.

## VRAM Peaks

- `r8 seq8192`: 25.562 / 27.188 GiB alloc/reserved
- `r16 seq8192`: 26.002 / 27.832 GiB
- `r32 seq8192`: 26.875 / 28.719 GiB

## Root Cause Analysis

The previous full/global-only `seq8192` OOM was caused by unpatched sliding attention. An 8 GiB memory request matched the dense FP32 sliding SDPA workspace requirements:
`1 * 32 * 8192 * 8192 * 4 bytes = 8.0 GiB` where `32` is the observed Gemma4 text `num_attention_heads`, `8192 * 8192` is the dense attention score/workspace shape, and `4 bytes` is fp32..

Additionally, the standard CE loss calculation is a major bottleneck because it materializes `[tokens × vocab]` dense logits.
- `seq4096` fp32 dense logits ≈ 4 GiB
- `seq8192` fp32 dense logits ≈ 8 GiB

The fused active linear CE avoids materializing these dense logits in memory, saving significant VRAM.

## Speed Tradeoff

- Old baseline (`r128 seq2048 FP16`): `compute_sec = 44.279s`
- New optimized path (`r128 seq2048 hot mean`): `72.429s`
- **Takeaway**: The new path is slower but uses significantly less VRAM, making `seq8192` possible on the MI50.

## Caveats

- Experimental text-only path.
- Not vision-compatible.
- Process-local monkeypatch.
- Relies on `nlzy/triton-gfx906` via `PYTHONPATH`.
- Uses a fused active CE, not the stock full-logits CE.
- This is a memory/capability optimization, not a speed optimization.
- No raw dataset text provided.
- Not intended for production packaging.
