# Benchmark Index

This page is the central map for benchmark and evidence files in this repository. It is intentionally concise and points readers to the canonical detailed notes.

## 1. Environment snapshot

- GPU: MI50 / gfx906
- torch: `2.7.0+rocm6.3`
- HIP: `6.3.42131-fa1d09cbd`
- Unsloth: `2026.5.6`
- Transformers: `5.8.0`
- `Xformers=None`, `FA2=False`
- Environment details: `docs/environment.md`

## 2. Core Gemma4 VRAM fit/OOM benchmarks

- Canonical table: `docs/gemma4-vram-benchmarks.md`
- Key points:
  - `r128 seq2048 fullpad`: `VERIFIED_OK`, `29.654GB` alloc / `30.887GB` reserved
  - `r128 seq4096`: `VERIFIED_OOM`
  - `r64 seq4096`: `VERIFIED_OOM`
  - `r8 seq8192`: `VERIFIED_OOM`

## 3. Gradient checkpointing comparison

- Evidence: `evidence/gemma4-gradient-checkpointing-comparison.md`
- Key points:
  - `none`: OOM at forward
  - `torch=True`: `VERIFIED_OK`, but `31.686GB` reserved, very tight headroom
  - `unsloth`: `VERIFIED_OK`, `30.887GB` reserved
  - Benefit verified here: VRAM headroom, not proven speedup

## 4. Precision / dtype comparison

- Evidence: `evidence/gemma4-dtype-fp16-vs-bf16.md`
- Key points:
  - FP16 and BF16 both `VERIFIED_OK`
  - Same peak VRAM
  - FP16 was about `2.69x` faster by `compute_sec` in the reversed-order run
  - Caveat: no long-run stability or quality claim

## 5. PEFT variant comparison: synthetic / fullpad

- Evidence: `evidence/gemma4-peft-variant-lora-rslora-dora.md`
- Key points:
  - LoRA `r128 seq2048`: `VERIFIED_OK`
  - rsLoRA `r128 seq2048`: `VERIFIED_OK`
  - DoRA `r128 seq2048`: OOM at backward
  - Caveat: not a quality benchmark, and does not rule out DoRA for 9B

## 6. PEFT variant real-data FP16 micro-run

- Evidence: `evidence/gemma4-realdata-peft-nan-speed-r8-seq1024-fp16.md`
- Key points:
  - Private JSONL dataset, `schema=text`, `raw_text_logged=no`
  - LoRA / rsLoRA / DoRA all `3-step VERIFIED_OK`
  - No NaN/Inf observed in this short run
  - LoRA and rsLoRA: about `21s/step`
  - DoRA: about `86.9s/step` with higher VRAM
  - Caveat: no quality claim

## 7. noflash-attention experiments

- Detailed note: `docs/noflash-attention-experiment.md`
- Key points:
  - Isolated SDPA tests: `VERIFIED_OK`
  - Full Gemma4 noflash runs remained `VERIFIED_OOM`
  - Conclusion: `NEGATIVE_RESULT` for rescuing this 31B stack

## 8. SDPA backend probe

- Probe script: `scripts/probe_sdpa_backends_gfx906.py`
- Key points:
  - Backend availability probe, not a performance benchmark
  - `MATH` path works in this environment
  - `FLASH` / `EFFICIENT` are not available / usable here

## 9. FineTome token stats / context planning

- Docs: `docs/finetome-token-stats.md`
- Results: `results/finetome_token_stats.md`
- Purpose: context-length planning and corpus characterization, not training performance

## 10. How to interpret labels

- `VERIFIED_OK`: reproduced successfully with the documented config
- `VERIFIED_OOM`: reproduced memory failure at a known phase
- `NEGATIVE_RESULT`: scoped failure for a specific attempted strategy or goal, not a universal method judgment
- `UNVERIFIED`: discussed but not reproduced to the same evidence standard
