# Gemma4 VRAM Benchmarks

All entries below are measured outcomes from this lab.

## Verified table

| Model / config | Input shape | Forward | Backward | Optimizer step | Peak alloc | Peak reserved | Status |
|---|---|---|---|---|---:|---:|---|
| Gemma4-31B inference (generate) | N/A | VERIFIED_OK | N/A | N/A | ~17.55 GB | N/A | VERIFIED_OK |
| Gemma4-31B FastModel load only | N/A | N/A | N/A | N/A | ~17.78 GB | N/A | VERIFIED_OK |
| LoRA r8 seq128 | `(1, 128)` | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | ~18.52 GB | N/A | VERIFIED_OK |
| LoRA r128 seq2048 fullpad | `(1, 2048)` | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | 29.654 GB | 30.887 GB | VERIFIED_OK |
| LoRA r128 seq4096 | N/A | VERIFIED_OK | VERIFIED_OOM | N/A | ~31.9 GB | N/A | VERIFIED_OOM |
| LoRA r64 seq4096 | N/A | VERIFIED_OK | VERIFIED_OOM | N/A | ~28.06 GB | N/A | VERIFIED_OOM |
| LoRA r8 seq8192 fullpad | N/A | VERIFIED_OOM | N/A | N/A | ~28.31 GB | N/A | VERIFIED_OOM |

## Interpretation

- `fullpad seq=2048` is the highest clearly stable long-run candidate in this stack.
- Verified details for the main successful run: `input_shape=(1, 2048)`, `forward=True`, `backward=True`, `optimizer_step=True`.
- This fits, but the reserved VRAM headroom is tight (~1.1GB on a 31.984GB MI50).
- `seq>=4096` for Gemma4-31B LoRA was not feasible on MI50 32 GB in measured runs.

Evidence file:

- `evidence/gemma4-lora-r128-seq2048-fullpad-ok.md`

## Reproducer script

Main reproducer for this table:

- `scripts/bench_gemma4_lora_vram.py`
