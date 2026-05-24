# noflash-attention Experiment

## Objective

Test whether noflash SDPA patching can reduce memory enough to make Gemma4-31B long-context LoRA training feasible.

## Isolated SDPA results

- seq4096 fp16: `VERIFIED_OK`, forward+backward, peak_reserved ~4.068 GB
- seq8192 fp16: `VERIFIED_OK`, forward+backward, peak_reserved ~10.420 GB

## Gemma4 integration results

- Import strategy A (Unsloth first, then noflash): `VERIFIED_ERROR` due to patched SDPA `__doc__` being `None` in one import path.
- Import strategy B (restore docstring before importing Unsloth): import issue bypassed.
- Gemma4 r64 seq4096: `VERIFIED_OOM`, `SDPA_CALL_COUNT=60`, forward did not complete.

Sweep results (strategy B):

- r8 seq4096: `VERIFIED_OOM` (forward)
- r16 seq4096: `VERIFIED_OOM` (forward)
- r32 seq4096: `VERIFIED_OOM` (forward)
- r8 seq8192: `VERIFIED_OOM` (forward)

## SDPA signatures seen in Gemma4

- `(1, 32, 4096, 256)` bf16
- `(1, 32, 4096, 512)` bf16
- `(1, 32, 8192, 256)` bf16
- `(1, 32, 8192, 512)` bf16

## Conclusion

`NEGATIVE_RESULT`: noflash works for isolated SDPA kernels but did not rescue Gemma4-31B long-context training in this stack.

## Reproducer script

- `scripts/bench_gemma4_noflash_sweep.py`
