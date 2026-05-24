# Unsloth Install Guide for gfx906 (EN, updated)

This is the current public-facing installation guide for setting up Unsloth workflows on AMD MI50/gfx906 with ROCm 6.3.x. It reflects the final conclusions of this repository, not the earlier historical draft.

For deeper incident analysis and debugging detail, see:

- `docs/mi50-gfx906-full-guide.en.md`
- `docs/rocm-gfx906-debugging.md`
- `docs/bitsandbytes-build.md`

## 1) Scope

This guide focuses on the install path that matched the final verified lab environment:

- ROCm 6.3.x userspace
- Python 3.13
- `torch==2.7.0+rocm6.3`
- HIP build string `6.3.42131-fa1d09cbd`
- Unsloth-based 4-bit workflows on MI50/gfx906

It is still a reproducibility-oriented engineering guide, not a guaranteed one-click setup for every host.

## 2) Install ROCm 6.3.3 base packages

On Ubuntu 24.04 (Noble), the historical working path used AMD's Noble package source for ROCm 6.3.3:

```bash
sudo rm /etc/apt/sources.list.d/rocm.list
sudo apt-get update

wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb
sudo apt-get install ./amdgpu-install_6.3.60303-1_all.deb

sudo amdgpu-install --usecase=rocm
```

If Secure Boot prompts for MOK enrollment on reboot, complete that step before proceeding.

## 3) Use the final required runtime environment variables

The final verified runtime combination in this repository is:

```bash
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export TORCHDYNAMO_DISABLE=1
export TORCH_INDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
```

Why `ROCR_VISIBLE_DEVICES=0` is critical:

- `HSA_OVERRIDE_GFX_VERSION=9.0.6` affects all GPUs visible to ROCr.
- On a mixed MI50 + RX6600 system, leaving the second GPU visible can cause gfx906-targeted kernels to dispatch incorrectly.
- In this lab, that mis-dispatch produced `RuntimeError: HIP error: invalid device function`, including in basic torch ops.
- `HIP_VISIBLE_DEVICES` alone was not a sufficient final control point for this topology.

Important correction from the older draft:

- The older `HIP_VISIBLE_DEVICES=1` instruction is not recommended for the final setup.
- The verified final setup used `ROCR_VISIBLE_DEVICES=0` together with `HIP_VISIBLE_DEVICES=0`.

## 4) Create a clean Python 3.13 virtual environment

Example:

```bash
python3.13 -m venv /home/<user>/unsloth_env
source /home/<user>/unsloth_env/bin/activate
```

You may use a different path, such as `/path/to/venv`, but keep the environment isolated.

## 5) Install PyTorch ROCm first and verify compatibility

Install from the ROCm 6.3 wheel index:

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

The final golden environment documented in this repository used:

- `torch==2.7.0+rocm6.3`
- HIP build string `6.3.42131-fa1d09cbd`

Do not rely only on a visible package version label. This lab found that seemingly similar torch environments could still differ in runtime behavior on MI50.

## 6) Install Unsloth stack

The historical guide initially installed only a minimal Unsloth stack before bitsandbytes. That staging can still be reasonable. Example baseline sequence:

```bash
pip install --no-deps unsloth
pip install accelerate peft Pillow requests
pip install unsloth_zoo
```

Transformer version requirements changed over time. In the final verified repo environment, Gemma4 support required a newer transformers line and the documented golden stack used `transformers==5.8.0`.

## 7) Build bitsandbytes from source for gfx906

This repository's final target is:

```bash
AMDGPU_TARGETS="gfx906:sramecc-:xnack-"
```

Example source-build flow:

```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes

export MAX_JOBS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export MAKEFLAGS="-j$(nproc)"

export ROCM_HOME=/opt/rocm
export COMPUTE_BACKEND=hip
export PYTORCH_ROCM_ARCH="gfx906"
export AMDGPU_TARGETS="gfx906:sramecc-:xnack-"
export CMAKE_ARGS="-DCOMPUTE_BACKEND=hip -DAMDGPU_TARGETS=gfx906:sramecc-:xnack-"

pip install .
```

For verification and interpretation details, see `docs/bitsandbytes-build.md`.

## 8) Disable compiler paths on MI50

Before training or runtime probes, keep compiler paths disabled:

```bash
export TORCHDYNAMO_DISABLE=1
export TORCH_INDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
```

These settings matched the final lab configuration and avoided unsupported or unstable compiler paths for MI50 in this stack.

## 9) Minimal verification sequence

After installation, verify in this order:

1. Run a basic torch probe.
2. Run a bitsandbytes linear/runtime probe.
3. Only then run model load or LoRA scripts.

In this repository, the helper scripts are:

- `scripts/probe_torch_rocm.py`
- `scripts/probe_bnb_runtime.py`
- `scripts/probe_sdpa_backends_gfx906.py`

## 10) Recommended reading order

1. `docs/unsloth-install-guide-gfx906.en.md`
2. `docs/rocm-gfx906-debugging.md`
3. `docs/bitsandbytes-build.md`
4. `docs/mi50-gfx906-full-guide.en.md`

## 11) Historical note

The original Vietnamese base guide is preserved separately as a historical reference:

- `docs/legacy-unsloth-install-guide-gfx906.vi.md`
