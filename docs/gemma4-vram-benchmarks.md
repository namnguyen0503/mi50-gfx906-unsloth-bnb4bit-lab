# Gemma4 VRAM Benchmarks

All entries below are measured outcomes from this lab.

## Verified table

| Model / config | Forward | Backward | Optimizer step | Peak alloc | Status |
|---|---|---|---|---:|---|
| Gemma4-31B inference (generate) | VERIFIED_OK | N/A | N/A | ~17.55 GB | VERIFIED_OK |
| Gemma4-31B FastModel load only | N/A | N/A | N/A | ~17.78 GB | VERIFIED_OK |
| LoRA r8 seq128 | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | ~18.52 GB | VERIFIED_OK |
| LoRA r128 seq2048 | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | ~29.65 GB | VERIFIED_OK |
| LoRA r128 seq4096 | VERIFIED_OK | VERIFIED_OOM | N/A | ~31.9 GB | VERIFIED_OOM |
| LoRA r64 seq4096 | VERIFIED_OK | VERIFIED_OOM | N/A | ~28.06 GB | VERIFIED_OOM |
| LoRA r8 seq8192 fullpad | VERIFIED_OOM | N/A | N/A | ~28.31 GB | VERIFIED_OOM |

## Interpretation

- `seq=2048` is the highest clearly stable long-run candidate in this stack.
- `seq>=4096` for Gemma4-31B LoRA was not feasible on MI50 32 GB in measured runs.

## Reproducer script

Main reproducer for this table:

- `scripts/bench_gemma4_lora_vram.py`
