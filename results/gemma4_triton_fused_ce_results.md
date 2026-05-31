# Gemma4 Triton Fused CE Results

Summary of results using the Triton-gfx906 all-text-attention patch (60 layers) and fused active linear CE.

## Seq8192 Fits
| Rank | Seq | Peak Alloc (GiB) | Peak Reserved (GiB) | Status |
|---|---|---:|---:|---|
| r8 | 8192 | 25.562 | 27.188 | VERIFIED_OK |
| r16 | 8192 | 26.002 | 27.832 | VERIFIED_OK |
| r32 | 8192 | 26.875 | 28.719 | VERIFIED_OK |

- **active_tokens**: 8191
- **Loss**: Finite, no NaN/Inf.
- **Passes**: Forward, backward, and optimizer.step all OK.

## Speed Tradeoff (r128 seq2048)
| Path | compute_sec |
|---|---|
| Old Baseline FP16 | 44.279s |
| New Triton Fused CE (hot mean) | 72.429s |

This is a memory/capability optimization, enabling `seq8192` at the cost of slower compute.
