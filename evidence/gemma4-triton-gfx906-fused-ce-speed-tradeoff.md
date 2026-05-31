# Evidence: Gemma4 Triton Fused CE Speed Tradeoff

## Overview
While the Triton-gfx906 patch and fused active linear CE enable training at `seq8192` on a 32GB MI50 by drastically reducing VRAM consumption, this comes at a computational speed cost.

## Benchmark Results (r128 seq2048)
To compare speed on an apples-to-apples basis where both methods fit into memory, we tested `r128 seq2048`.

- **Old Path (Baseline FP16)**: `compute_sec = 44.279s`
- **New Optimized Path (hot mean)**: `compute_sec = 72.429s`

## Analysis
The new fused CE and Triton attention path is noticeably slower per step (~1.6x slower in this specific test) but avoids large `[tokens × vocab]` dense logit materialization (saving ~8 GiB at seq8192) and bypasses the unpatched sliding attention SDPA workspace (saving ~8 GiB). This tradeoff is necessary to fit `seq8192` into the 32GB limit of the MI50.

> Note: this speed comparison is an apples-to-apples `r128 seq2048` benchmark against the old FP16 baseline. It is not the same workload as the successful `r8/r16/r32 seq8192` memory gate.
