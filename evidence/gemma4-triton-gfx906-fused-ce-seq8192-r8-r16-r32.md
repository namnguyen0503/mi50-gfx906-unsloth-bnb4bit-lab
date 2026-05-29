# Evidence: Gemma4 Triton gfx906 Fused CE Seq8192 r8/r16/r32

## Configuration
- **Model**: Gemma4-31B bnb 4-bit
- **Hardware**: Single AMD MI50 32GB/gfx906
- **Context**: `seq8192` (active_tokens=8191)
- **Ranks Tested**: `r8`, `r16`, `r32`
- **Modifications**: Triton-gfx906 all-text-attention patch (60 layers) + fused active linear CE.
- **LoRA Target**: Attention + MLP.

## Results
| Rank | Peak Alloc (GiB) | Peak Reserved (GiB) | Status |
|---|---:|---:|---|
| r8 | 25.562 | 27.188 | VERIFIED_OK |
| r16 | 26.002 | 27.832 | VERIFIED_OK |
| r32 | 26.875 | 28.719 | VERIFIED_OK |

## Verification Details
- Forward pass: Success.
- Backward pass: Success.
- Optimizer step: Success.
- Loss: Finite loss observed, no NaN/Inf.
- Previous baseline for `r8 seq8192 fullpad` without these patches was `VERIFIED_OOM`.

## Caveats
- Experimental text-only path, not vision-compatible.
- Process-local monkeypatch.
- nlzy/triton-gfx906 via PYTHONPATH.
- Fused active CE, not stock full-logits CE.
- Memory/capability optimization, not a speed optimization.
