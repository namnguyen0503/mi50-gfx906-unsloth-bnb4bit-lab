# Patch Artifacts

This directory contains small reproducibility artifacts for the golden MI50/gfx906 environment documented in this repository.

Included patches:

- `torch-hf-storage-shim.py`: exact shim content used for `torch.distributed.checkpoint.hf_storage` in the golden env.
- `unsloth-vllm-import-guard.patch`: sanitized unified diff showing the `try/except` guard added around `importlib_version("vllm")` checks in Unsloth.

These are intentionally minimal artifacts:

- no raw site-packages tree
- no wheels or built binaries
- no venv snapshot

Related files:

- `docs/golden-env-package-manifest.md`
- `requirements-golden.txt`
- `docs/environment.md`
