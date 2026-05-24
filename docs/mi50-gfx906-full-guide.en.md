# MI50 (gfx906) ROCm 6.3 + Unsloth + bitsandbytes 4-bit: Full Forensic Guide (EN)

This is the English, sanitized version of the original Vietnamese forensic guide.

## 0) Scope and evidence policy

This document is a reproducibility and incident-analysis report for one MI50/gfx906 setup. It is not a one-click installer. Results are labeled as:

- `VERIFIED_OK`
- `VERIFIED_OOM`
- `UNVERIFIED`
- `NEGATIVE_RESULT`

## 1) Hardware and runtime context

### 1.1 Why topology mattered

The host had mixed AMD GPUs (MI50 + RX6600). This was the dominant source of runtime mis-dispatch when `HSA_OVERRIDE_GFX_VERSION` was used without strict ROCr visibility control.

### 1.2 Golden stack (verified)

- Python: 3.13
- torch: `2.7.0+rocm6.3`
- HIP build string: `6.3.42131-fa1d09cbd`
- transformers: 5.8.0
- unsloth: 2026.5.6
- unsloth_zoo: 2026.5.4
- bitsandbytes: source build with `gfx906:sramecc-:xnack-`

### 1.3 Usable VRAM clarification

Usable VRAM in this setup was approximately `31.984 GB` from torch device properties. Some tools or labels can show confusing 16GB strings; those were not used as final capacity truth.

## 2) Required environment variables

```bash
ROCR_VISIBLE_DEVICES=0
HIP_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=9.0.6
TORCHDYNAMO_DISABLE=1
TORCH_INDUCTOR_DISABLE=1
TORCH_COMPILE_DISABLE=1
```

### 2.1 Root cause and dispatch logic

`ROCR_VISIBLE_DEVICES=0` was essential because `HSA_OVERRIDE_GFX_VERSION=9.0.6` affects all ROCr-visible GPUs. In a mixed MI50+RX6600 system, leaving RX6600 visible could route gfx906-targeted kernels to the wrong device, producing `invalid device function` errors.

## 3) Failure signatures and fixes

### 3.1 `invalid device function` in basic torch ops

Observed even in basic kernels (`torch.randn`, `zero_`) under bad visibility/runtime conditions.

Example signature:

```text
RuntimeError: HIP error: invalid device function
```

Interpretation: this is often a dispatch/runtime mismatch problem, not necessarily a bitsandbytes bug.

### 3.2 Unsloth import-path issues

- `No module named torch.distributed.checkpoint.hf_storage` on older torch layout.
- `PackageNotFoundError: vllm` with namespace edge cases.

### 3.3 Gemma4 model recognition

Older transformers failed with `model type gemma4 not recognized`; this stack required a newer transformers line (5.8.0 used here).

## 4) bitsandbytes build notes

Build target used:

```bash
AMDGPU_TARGETS="gfx906:sramecc-:xnack-" python setup.py bdist_wheel
```

Verification approach:

```bash
strings libbitsandbytes_rocm63.so | grep amdgcn
```

Expected to include `gfx906:sramecc-:xnack-`.

## 5) Verified Gemma4 VRAM benchmarks

| Configuration | Forward | Backward | Optimizer step | Peak alloc | Label |
|---|---|---|---|---:|---|
| Gemma4-31B inference generate | VERIFIED_OK | N/A | N/A | ~17.55 GB | VERIFIED_OK |
| Gemma4-31B FastModel load only | N/A | N/A | N/A | ~17.78 GB | VERIFIED_OK |
| LoRA r8 seq128 | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | ~18.52 GB | VERIFIED_OK |
| LoRA r128 seq2048 | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | ~29.65 GB | VERIFIED_OK |
| LoRA r128 seq4096 | VERIFIED_OK | VERIFIED_OOM | N/A | ~31.9 GB | VERIFIED_OOM |
| LoRA r64 seq4096 | VERIFIED_OK | VERIFIED_OOM | N/A | ~28.06 GB | VERIFIED_OOM |
| LoRA r8 seq8192 (fullpad) | VERIFIED_OOM | N/A | N/A | ~28.31 GB | VERIFIED_OOM |

Operational conclusion: `seq=2048` is the highest clearly stable Gemma4-31B LoRA setting verified in this stack.

## 6) SDPA backend reality on gfx906

- PyTorch SDPA `FLASH_ATTENTION` and `EFFICIENT_ATTENTION` were not compiled for gfx906 in this stack.
- `MATH` backend was the practical fallback.
- This strongly impacts long-context memory behavior.

## 7) xFormers / FA2 correction notes

- xFormers was not usable in the current ROCm/gfx906 environment (mismatched/unusable package state).
- Unsloth runtime detection showed: `Xformers=None, FA2=False`.
- Therefore this report does **not** claim “xFormers fallback works normally on MI50.”

## 8) noflash-attention experiment

### 8.1 Isolated SDPA tests

- seq4096 fp16: `VERIFIED_OK`, forward+backward, peak_reserved ~4.068 GB
- seq8192 fp16: `VERIFIED_OK`, forward+backward, peak_reserved ~10.420 GB

### 8.2 Gemma4 integration tests

- r64 seq4096: `VERIFIED_OOM`, `SDPA_CALL_COUNT=60`, forward did not complete
- Sweep results:
  - r8 seq4096: `VERIFIED_OOM`
  - r16 seq4096: `VERIFIED_OOM`
  - r32 seq4096: `VERIFIED_OOM`
  - r8 seq8192: `VERIFIED_OOM`

Observed Gemma4 SDPA signatures included:

- `q/k/v (1, 32, 4096, 256)` bf16
- `q/k/v (1, 32, 4096, 512)` bf16
- `q/k/v (1, 32, 8192, 256)` bf16
- `q/k/v (1, 32, 8192, 512)` bf16

Conclusion: `NEGATIVE_RESULT` for the stated goal (making Gemma4-31B long-context LoRA training feasible on this stack).

## 9) FineTome train[:3000] token statistics

- count: 3000
- mean: 568.05
- p50: 457
- p75: 662
- p90: 1003
- p95: 1320
- p99: 2195
- max: 6531
- >2048: 44 / 3000 = 1.47%
- >3072: 10 / 3000 = 0.33%
- >4096: 4 / 3000 = 0.13%
- >8192: 0

## 10) Kaggle 2xT4 comparison boundary

The referenced Kaggle setup (2xT4, balanced multi-GPU mapping, xFormers behavior, rank-8, short-sequence tendencies) is not directly equivalent to this MI50 single-GPU path.

In this repository, no claim is made that FA2 was active for that path; and no claim is made that xFormers was usable on MI50 in this stack.

## 11) Sanitization policy used here

- `/home/elysia` -> `/home/<user>`
- `/media/elysia/NVME PCIE3` -> `/path/to/NVME`
- Private project/checkpoint names -> `<private-project>`, `<private-checkpoint>`
