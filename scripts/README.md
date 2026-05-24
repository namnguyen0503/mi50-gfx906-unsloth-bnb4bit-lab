# Scripts

## Core benchmark scripts

- `probe_sdpa_backends_gfx906.py`: probes SDPA backend availability checks (MATH vs FLASH_ATTENTION vs EFFICIENT_ATTENTION) on gfx906, including VRAM peaks and error states. This is a compatibility probe, not a performance benchmark.
- `bench_gemma4_lora_vram.py`: main reproducer for Gemma4-31B LoRA VRAM table (`VERIFIED_OK` / `VERIFIED_OOM`) with explicit OOM phase reporting.
- `bench_gemma4_noflash_sweep.py`: sanitized noflash experiment sweep reproducer with SDPA call counting and unique q/k/v signature capture.

## Probe and helper scripts

- `probe_torch_rocm.py`: quick torch/ROCm sanity probe.
- `probe_bnb_runtime.py`: bitsandbytes 4-bit runtime probe.
- `check_finetome_gemma4_tokens.py`: token-length summary helper.

These scripts are minimal reproducibility helpers for this lab report.
