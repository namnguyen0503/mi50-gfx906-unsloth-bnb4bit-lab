# MI50/gfx906 ROCm + Unsloth + bitsandbytes 4-bit Lab

This repository documents one working MI50/gfx906 setup. It is intended as a reference for experienced ROCm/LLM users, not a guaranteed one-click installer.

## Quick Links

| Start here | Link | Purpose |
|---|---|---|
| Benchmark map | [Benchmark index](docs/benchmark-index.md) | Central overview of all benchmarks and evidence |
| Full guide | [Full MI50/gfx906 guide — English](docs/mi50-gfx906-full-guide.en.md) | Reproduce the working setup |
| Full guide | [Full MI50/gfx906 guide — Vietnamese](docs/mi50-gfx906-full-guide.vi.md) | Original full guide in Vietnamese |
| Evidence | [Evidence index](evidence/README.md) | Verified benchmark notes and supporting evidence |
| Results | [VRAM result table](results/vram_table.md) | Compact VRAM-oriented results table |

Core docs:

- [Benchmark index](docs/benchmark-index.md)
- [Environment](docs/environment.md)
- [Gemma4 VRAM benchmarks](docs/gemma4-vram-benchmarks.md)
- [Full MI50/gfx906 guide — English](docs/mi50-gfx906-full-guide.en.md)
- [Full MI50/gfx906 guide — Vietnamese](docs/mi50-gfx906-full-guide.vi.md)
- [noflash-attention experiment](docs/noflash-attention-experiment.md)
- [Triton Fused CE Seq8192 experiment](docs/gemma4-triton-gfx906-fused-ce-seq8192.md)
- [Evidence index](evidence/README.md)
- [VRAM result table](results/vram_table.md)
- [Golden environment package manifest](docs/golden-env-package-manifest.md)
- [Env hotfix patches](patches/README.md)

Direct evidence links:

- [Gemma4 r128 seq2048 fullpad OK](evidence/gemma4-lora-r128-seq2048-fullpad-ok.md)
- [Gradient checkpointing comparison](evidence/gemma4-gradient-checkpointing-comparison.md)
- [FP16 vs BF16 dtype benchmark](evidence/gemma4-dtype-fp16-vs-bf16.md)
- [PEFT variant: LoRA vs rsLoRA vs DoRA](evidence/gemma4-peft-variant-lora-rslora-dora.md)
- [Real-data PEFT FP16 3-step micro-run](evidence/gemma4-realdata-peft-nan-speed-r8-seq1024-fp16.md)
- [Real-data PEFT 100-sample CPT eval](evidence/gemma4-realdata-peft-100sample-cpt-eval-r8-seq1024-fp16.md)
- [Triton Fused CE Seq8192 OK](evidence/gemma4-triton-gfx906-fused-ce-seq8192-r8-r16-r32.md)
- [Triton Fused CE speed tradeoff](evidence/gemma4-triton-gfx906-fused-ce-speed-tradeoff.md)

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

- Central map of benchmark and evidence files: [`docs/benchmark-index.md`](docs/benchmark-index.md)

| Workload | Result | Peak VRAM (alloc) | Notes |
|---|---|---:|---|
| Gemma4-31B inference generate | VERIFIED_OK | ~17.55 GB | load ~17.28 GB |
| Gemma4-31B FastModel load | VERIFIED_OK | ~17.78 GB | baseline before LoRA |
| LoRA r8 seq128 | VERIFIED_OK | ~18.52 GB | forward+backward+step |
| LoRA r128 seq2048 fullpad | VERIFIED_OK | 29.654 GB | input_shape=(1,2048), forward/backward/optimizer.step OK, reserved 30.887 GB |
| LoRA r128 seq4096 | VERIFIED_OOM | ~31.9 GB | OOM at backward |
| LoRA r64 seq4096 | VERIFIED_OOM | ~28.06 GB | OOM at backward |
| LoRA r8 seq8192 fullpad | VERIFIED_OOM | ~28.31 GB | OOM at forward (tried +8 GiB) |
| LoRA r32 seq8192 triton+fused_ce | VERIFIED_OK | ~26.88 GB | input_shape=(1,8192), fused CE, reserved 28.719 GB |

## Key verified benchmark

- `Gemma4-31B bnb 4-bit + LoRA r128 + fullpad seq2048: VERIFIED_OK`
  `input_shape=(1,2048), forward/backward/optimizer.step OK, peak=29.654GB allocated / 30.887GB reserved`
  Evidence: `evidence/gemma4-lora-r128-seq2048-fullpad-ok.md`
  This fits, but the reserved VRAM headroom is tight (~1.1GB on a 31.984GB MI50).

## Gradient checkpointing comparison

- For the verified `r128 seq2048 fullpad` run, `use_gradient_checkpointing="unsloth"` was used.
- `use_gradient_checkpointing=True` (`torch` mode in the comparison script) also passed, but with much tighter reserved VRAM headroom: `31.686GB reserved`, about `0.298GB` below the `31.984GB` MI50 limit.
- `use_gradient_checkpointing=False` (`none`) failed at forward with `VERIFIED_OOM`.
- Evidence: `evidence/gemma4-gradient-checkpointing-comparison.md`
- Interpretation: the verified benefit is VRAM headroom, not a proven speedup. Compute-only forward/backward/optimizer timing was roughly similar between `torch` and `unsloth` modes.

## Precision / dtype comparison

- Unless otherwise stated, the original verified `r128 seq2048 fullpad` benchmark was recorded on the BF16 path.
- FP16 was separately tested and verified for the same one-step workload.
- FP16 and BF16 used the same peak VRAM in the verified runs: `29.654GB` allocated / `30.887GB` reserved.
- FP16 was much faster by `compute_sec` on this MI50/gfx906 setup, with the reversed-order run showing about `2.69x` speedup versus BF16.
- FP16 should be considered the preferred performance dtype for this hardware in this workflow, pending longer-run NaN/stability checks.
- Requested dtype was honored in the observed model path for both modes: FP16 kept `model_config_dtype=torch.float16` and `embedding_dtype=torch.float16`; BF16 kept `model_config_dtype=torch.bfloat16` and `embedding_dtype=torch.bfloat16`.
- The observed path did not silently upcast BF16 model/embedding tensors to full FP32, although the first trainable LoRA parameter and the loss remained `torch.float32` in both runs.
- Evidence: `evidence/gemma4-dtype-fp16-vs-bf16.md`

## PEFT variant comparison

- LoRA and rsLoRA both fit the `r128 seq2048 fullpad` 31B benchmark in this environment.
- rsLoRA did not materially change VRAM or runtime class in this measured one-step case.
- DoRA is supported by the current Unsloth/FastModel path here, but it OOMs at backward for this exact 31B config because of the added memory pressure.
- This is a runtime/VRAM/fit benchmark only, not a quality benchmark.
- This does not rule out DoRA for 9B or lower-rank configs; 9B headroom is much larger.
- Evidence: `evidence/gemma4-peft-variant-lora-rslora-dora.md`

## Real-data PEFT micro-run

- Private JSONL dataset, schema detected as `text`, `raw_text_logged=no`.
- LoRA, rsLoRA, and DoRA all completed `3/3` FP16 steps with finite loss and sampled finite gradients.
- LoRA and rsLoRA were nearly identical in this short run, at about `21s/step` and `22.377GB` alloc / `23.283GB` reserved.
- DoRA fit at `r8/seq1024`, but used more VRAM and was much slower: `23.569GB` alloc / `25.645GB` reserved, about `86.9s/step`.
- This is a micro-run only and does not make any quality or long-run stability claim.
- Evidence: [`evidence/gemma4-realdata-peft-nan-speed-r8-seq1024-fp16.md`](evidence/gemma4-realdata-peft-nan-speed-r8-seq1024-fp16.md)

## Real-data CPT held-out loss probe

- Stronger real-data adapter comparison than the 3-step micro-run: `100` train samples and `32` fixed held-out eval samples, with `train_eval_overlap=no`.
- LoRA, rsLoRA, and DoRA all completed `100/100` FP16 steps with finite loss and sampled finite gradients.
- rsLoRA had the best held-out CPT `eval_loss_after` (`1.774377`) and the best `eval_loss_delta` (`6.113859`) while staying in LoRA's speed/VRAM class.
- LoRA was slightly fastest at `21.729s/step` and had the lowest peak reserved VRAM at `23.283GB`.
- DoRA ran successfully, but was much slower (`86.668s/step`), used more VRAM (`25.562GB` reserved), and did not beat rsLoRA on held-out CPT loss in this Gemma4-31B run.
- Current practical takeaway: for this Gemma4-31B MI50 FP16 CPT probe, rsLoRA is the best practical adapter candidate. It matched LoRA's speed/VRAM class while achieving the best held-out CPT eval loss. DoRA ran successfully but was much slower and did not beat rsLoRA in this test. This is not a final persona-quality claim.
- This is a CPT held-out loss probe only, not a final SFT or persona-quality eval.
- Evidence: [`evidence/gemma4-realdata-peft-100sample-cpt-eval-r8-seq1024-fp16.md`](evidence/gemma4-realdata-peft-100sample-cpt-eval-r8-seq1024-fp16.md)

## Triton-gfx906 Fused CE Seq8192 Success

- The previous full/global-only `seq8192` `VERIFIED_OOM` was overcome using a process-local monkeypatch of all 60 text attention layers with memory-efficient Triton-gfx906 kernels and fused active linear CE.
- `seq8192` LoRA `r8/r16/r32` all `VERIFIED_OK` with finite loss and no NaN/Inf.
- This resolved two major memory bottlenecks: dense logits (`seq8192` fp32 dense logits ≈ 8 GiB) and the unpatched sliding attention SDPA workspace (1 * 32 * 8192 * 8192 * 4 bytes = 8.0 GiB).
- **Speed tradeoff**: The optimized path is slower (`r128 seq2048` took `72.429s` vs `44.279s`) but dramatically reduces memory usage, enabling `seq8192` on the 32GB MI50.
- Evidence: [`docs/gemma4-triton-gfx906-fused-ce-seq8192.md`](docs/gemma4-triton-gfx906-fused-ce-seq8192.md)

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

## Installation guides

- `docs/unsloth-install-guide-gfx906.en.md` (current public-facing install guide)
- `docs/legacy-unsloth-install-guide-gfx906.vi.md` (historical/original base guide, sanitized)

## Patch artifacts and environment manifest

- `patches/README.md`
- `patches/torch-hf-storage-shim.py`
- `patches/unsloth-vllm-import-guard.patch`
- `docs/golden-env-package-manifest.md`
- `requirements-golden.txt`

## What worked / failed / lessons

### What worked
- Stable 4-bit LoRA at moderate context (`fullpad seq=2048`, rank up to 128).
- Long-context `seq8192` LoRA via the Triton-gfx906 patch + fused active linear CE, fitting into the 32GB limit.
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
- `docs/golden-env-package-manifest.md`
- `docs/unsloth-install-guide-gfx906.en.md`
- `docs/legacy-unsloth-install-guide-gfx906.vi.md`
- `docs/rocm-gfx906-debugging.md`
- `docs/bitsandbytes-build.md`
- `docs/gemma4-vram-benchmarks.md`
- `docs/benchmark-index.md`
- `docs/noflash-attention-experiment.md`
- `docs/gemma4-triton-gfx906-fused-ce-seq8192.md`
- `docs/finetome-token-stats.md`
- `docs/postmortem.md`
- `docs/mi50-gfx906-full-guide.en.md` (public-facing English full guide)
- `docs/mi50-gfx906-full-guide.vi.md` (original Vietnamese guide, sanitized)
- `evidence/README.md`
- `evidence/gemma4-dtype-fp16-vs-bf16.md`
- `evidence/gemma4-gradient-checkpointing-comparison.md`
- `evidence/gemma4-lora-r128-seq2048-fullpad-ok.md`
- `evidence/gemma4-peft-variant-lora-rslora-dora.md`
- `evidence/gemma4-realdata-peft-nan-speed-r8-seq1024-fp16.md`
- `evidence/gemma4-realdata-peft-100sample-cpt-eval-r8-seq1024-fp16.md`
- `evidence/gemma4-triton-gfx906-fused-ce-seq8192-r8-r16-r32.md`
- `evidence/gemma4-triton-gfx906-fused-ce-speed-tradeoff.md`
- `results/vram_table.md`
- `results/noflash_results.md`
- `results/gemma4_triton_fused_ce_results.md`
- `results/finetome_token_stats.md`
- `patches/`
- `requirements-golden.txt`
- `scripts/`
