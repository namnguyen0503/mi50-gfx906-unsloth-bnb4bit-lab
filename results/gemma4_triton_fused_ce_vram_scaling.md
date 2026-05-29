# Gemma4 Triton-gfx906 + fused CE VRAM scaling

This table records additional VRAM scaling points for the experimental text-only
Triton-gfx906 + fused active linear CE path.

Caveats:
- Experimental text-only path.
- Process-local monkeypatch.
- Not vision-compatible.
- Memory/capability optimization, not speed optimization.
- Raw dataset text is not logged.

| Config | Rank | Seq len | Active tokens | Peak alloc GiB | Peak reserved GiB | Status | Notes |
|---|---:|---:|---:|---:|---:|---|---|
| Triton all-attn + fused CE | r8 | 1024 | TODO | 18.441 | 18.775 | OBSERVED_PARTIAL | Live speed run was still `RUNNING`; do not treat as final verified row yet |
| Triton all-attn + fused CE | r128 | 2048 | 2047 | 27.978 | 28.758 | VERIFIED_OK | Hot mean ~72.9s; old FP16 baseline was 44.279s |
| Triton all-attn + fused CE | r8 | 4096 | 4095 | 22.747 | 24.590 | VERIFIED_OK | Original attention+MLP LoRA target |
| Triton all-attn + fused CE | r16 | 4096 | 4095 | 22.975 | 24.910 | VERIFIED_OK | Original attention+MLP LoRA target |
| Triton all-attn + fused CE | r32 | 4096 | 4095 | 23.431 | 25.830 | VERIFIED_OK | Original attention+MLP LoRA target |
| Triton all-attn + fused CE | r8 | 8192 | 8191 | 25.562 | 27.188 | VERIFIED_OK | Main seq8192 gate |
| Triton all-attn + fused CE | r16 | 8192 | 8191 | 26.002 | 27.832 | VERIFIED_OK | Main seq8192 gate |
| Triton all-attn + fused CE | r32 | 8192 | 8191 | 26.875 | 28.719 | VERIFIED_OK | Main seq8192 gate |

## Interpretation

The optimized path is slower at short/mid context than the old FP16 baseline,
but it substantially improves memory headroom and enables real-active seq8192
training on a single MI50 32GB/gfx906.

The old path remains the faster choice for short-context training when it fits.
The Triton-gfx906 + fused CE path is the memory-safe long-context path.
