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
- The verified `r128 seq2048 fullpad` row above used `use_gradient_checkpointing="unsloth"`.
- Unless otherwise stated, the original verified `r128 seq2048 fullpad` row above used the BF16 path.
- HF/PyTorch-style `use_gradient_checkpointing=True` also passed in the local comparison script (`torch` mode), but with much tighter headroom: `31.686GB` reserved, about `0.298GB` below the `31.984GB` MI50 limit.
- `use_gradient_checkpointing=False` (`none`) failed at forward with `VERIFIED_OOM`.
- The real verified benefit of `unsloth` here is VRAM headroom: `29.654GB` allocated / `30.887GB` reserved versus `30.848GB` allocated / `31.686GB` reserved for `torch`.
- Do not interpret `step_sec` as a clean speed comparison; compute-only forward/backward/optimizer timing was roughly similar.
- FP16 was separately verified for the same `r128 seq2048 fullpad` workload. Peak VRAM stayed identical, but FP16 compute time was much lower on this MI50/gfx906 setup.
- In the reversed-order dtype run, FP16 `compute_sec=44.279` versus BF16 `compute_sec=119.220`, about `2.69x` faster.
- Requested dtype was honored in the observed model path for both modes, while the first trainable LoRA parameter and the loss remained `torch.float32`.
- FP16 should be treated as the preferred performance dtype for this hardware in this workload, pending longer-run NaN/stability checks.
- LoRA and rsLoRA both fit the measured `r128 seq2048 fullpad` 31B budget in this environment.
- rsLoRA did not materially change VRAM or runtime class in this one-step benchmark.
- DoRA is supported by the current Unsloth/FastModel path, but it is `VERIFIED_OOM` at backward for this exact 31B config because of the extra memory pressure.
- This does not rule out DoRA for 9B or lower-rank configs; 9B headroom is much larger.
- `seq>=4096` for Gemma4-31B LoRA was not feasible on MI50 32 GB in measured runs.

Evidence file:

- `evidence/gemma4-lora-r128-seq2048-fullpad-ok.md`
- `evidence/gemma4-gradient-checkpointing-comparison.md`
- `evidence/gemma4-dtype-fp16-vs-bf16.md`
- `evidence/gemma4-peft-variant-lora-rslora-dora.md`
- `evidence/gemma4-realdata-peft-nan-speed-r8-seq1024-fp16.md`

Additional real-data micro-run:

- Private JSONL dataset (`schema=text`, `raw_text_logged=no`): LoRA / rsLoRA / DoRA all completed `3/3` FP16 steps with finite loss/grad.
- LoRA and rsLoRA stayed near `21s/step`; DoRA fit at `r8/seq1024` but used more VRAM and ran much slower.
- See `evidence/gemma4-realdata-peft-nan-speed-r8-seq1024-fp16.md`.

## Reproducer script

Main reproducer for this table:

- `scripts/bench_gemma4_lora_vram.py`
