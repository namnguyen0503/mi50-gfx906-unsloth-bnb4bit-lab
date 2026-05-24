# Golden Environment Package Manifest

This manifest lists the high-value package versions for the golden MI50/gfx906 environment used in the verified runs in this repository. It is intentionally concise and is not a full `pip freeze` dump.

## Core stack

- Python: 3.13
- torch: `2.7.0+rocm6.3`
- HIP build string: `6.3.42131-fa1d09cbd`
- transformers: `5.8.0`
- unsloth: `2026.5.6`
- unsloth_zoo: `2026.5.4`
- bitsandbytes: `0.50.0.dev0`

## Supporting packages

- accelerate: `1.13.0`
- peft: `0.18.1`
- triton: `3.3.0`

## Build note

`bitsandbytes` was source-built for:

```bash
gfx906:sramecc-:xnack-
```

The binary target mattered, but the dominant runtime issue in this lab was mixed-GPU dispatch behavior rather than bitsandbytes source content alone.

## Related artifacts

- `requirements-golden.txt`
- `patches/torch-hf-storage-shim.py`
- `patches/unsloth-vllm-import-guard.patch`
