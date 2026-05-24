# Environment Reference

Public path placeholders:

- Home path: `/home/<user>`
- HF cache path: `/path/to/hf_cache`

## Golden stack (VERIFIED)

- Python: 3.13
- torch: `2.7.0+rocm6.3`
- HIP build string: `6.3.42131-fa1d09cbd`
- transformers: 5.8.0
- unsloth: 2026.5.6
- unsloth_zoo: 2026.5.4
- bitsandbytes: 0.50.0.dev0 source build

## Required runtime env vars

```bash
ROCR_VISIBLE_DEVICES=0
HIP_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=9.0.6
TORCHDYNAMO_DISABLE=1
TORCH_INDUCTOR_DISABLE=1
TORCH_COMPILE_DISABLE=1
```

## Why this exact combo

Mixed GPU topology (MI50 + RX6600) caused kernel dispatch errors when ROCr could still see the second GPU while `HSA_OVERRIDE_GFX_VERSION` was active. `ROCR_VISIBLE_DEVICES=0` was the key control point.
