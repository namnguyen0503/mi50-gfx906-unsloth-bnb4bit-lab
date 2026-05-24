# MI50/gfx906 ROCm + Unsloth + bitsandbytes 4-bit Lab

This repository documents one working MI50/gfx906 setup. It is intended as a reference for experienced ROCm/LLM users, not a guaranteed one-click installer.

## Scope

Engineering case study for running Gemma4/Qwen 4-bit LoRA workflows on AMD MI50 with ROCm 6.3, including:

- Environment constraints and root-cause debugging for mixed-GPU systems.
- bitsandbytes gfx906 build notes.
- Verified VRAM benchmarks (`VERIFIED_OK` / `VERIFIED_OOM`).
- `noflash-attention` negative-result experiment.
- FineTome token length distribution analysis for context-length planning.

## Credits

Human operator and final engineering judgement: Nam Nguyen. GPT-5.5 Thinking and GPT-5.3 Codex were used as AI planning/debugging/documentation assistants. All claims are based on reproduced commands, logs, or measured VRAM results.

## Golden environment (documented)

- Python: 3.13
- torch: `2.7.0+rocm6.3`
- HIP runtime string: `6.3.42131-fa1d09cbd`
- transformers: 5.8.0
- unsloth: 2026.5.6
- unsloth_zoo: 2026.5.4
- bitsandbytes: source build for `gfx906:sramecc-:xnack-`

Required runtime env vars:

```bash
ROCR_VISIBLE_DEVICES=0
HIP_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=9.0.6
TORCHDYNAMO_DISABLE=1
TORCH_INDUCTOR_DISABLE=1
TORCH_COMPILE_DISABLE=1
```

## Verified benchmark snapshot

| Workload | Result | Peak VRAM (alloc) | Notes |
|---|---|---:|---|
| Gemma4-31B inference generate | VERIFIED_OK | ~17.55 GB | load ~17.28 GB |
| Gemma4-31B FastModel load | VERIFIED_OK | ~17.78 GB | baseline before LoRA |
| LoRA r8 seq128 | VERIFIED_OK | ~18.52 GB | forward+backward+step |
| LoRA r128 seq2048 | VERIFIED_OK | ~29.65 GB | forward+backward+step |
| LoRA r128 seq4096 | VERIFIED_OOM | ~31.9 GB | OOM at backward |
| LoRA r64 seq4096 | VERIFIED_OOM | ~28.06 GB | OOM at backward |
| LoRA r8 seq8192 fullpad | VERIFIED_OOM | ~28.31 GB | OOM at forward (tried +8 GiB) |

## noflash-attention summary

- Isolated SDPA tests: `VERIFIED_OK`
  - seq4096 fp16: peak_reserved ~4.068 GB
  - seq8192 fp16: peak_reserved ~10.420 GB
- Gemma4 with noflash still `VERIFIED_OOM` for tested long-context configs:
  - r64 seq4096 (SDPA call counter > 0)
  - sweep: r8/r16/r32 seq4096 and r8 seq8192 all OOM at forward
- Conclusion: `NEGATIVE_RESULT` for rescuing Gemma4-31B long-context training in this stack.

## Reproduction scripts

- `scripts/probe_sdpa_backends_gfx906.py`
- `scripts/bench_gemma4_lora_vram.py`
- `scripts/bench_gemma4_noflash_sweep.py`

## What worked / failed / lessons

### What worked
- Stable 4-bit LoRA at moderate context (`seq=2048`, rank up to 128).
- Reproducible bnb runtime probe and SDPA instrumentation.

### What failed
- Long-context Gemma4 LoRA (`seq>=4096`) on MI50 32 GB in this software stack.
- noflash SDPA monkeypatch did not make Gemma4 long-context training feasible.

### Lessons learned
- `ROCR_VISIBLE_DEVICES=0` is critical in mixed-GPU systems when using `HSA_OVERRIDE_GFX_VERSION`.
- Do not trust only version labels; verify with runtime probes and measured VRAM.
- Isolated kernel success does not guarantee full-model training feasibility.

## Repository map

- `docs/environment.md`
- `docs/rocm-gfx906-debugging.md`
- `docs/bitsandbytes-build.md`
- `docs/gemma4-vram-benchmarks.md`
- `docs/noflash-attention-experiment.md`
- `docs/finetome-token-stats.md`
- `docs/postmortem.md`
- `docs/mi50-gfx906-full-guide.en.md` (public-facing English full guide)
- `docs/mi50-gfx906-full-guide.vi.md` (original Vietnamese guide, sanitized)
- `results/vram_table.md`
- `results/noflash_results.md`
- `results/finetome_token_stats.md`
- `scripts/`
