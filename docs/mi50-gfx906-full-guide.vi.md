# MI50 (gfx906) ROCm 6.3 + Unsloth + bitsandbytes 4-bit: Forensic Guide (VI, sanitized)

Tài liệu này là bản tiếng Việt đã được sanitize từ guide forensic gốc.

## 0) Scope và nguyên tắc

- Mục tiêu: ghi lại stack đã chạy được trên MI50/gfx906 với Unsloth + bnb 4-bit.
- Đây là tài liệu forensic/reproducibility, không phải one-click installer.
- Kết quả được gán nhãn: `VERIFIED_OK`, `VERIFIED_OOM`, `UNVERIFIED`, `NEGATIVE_RESULT`.

## 1) Hardware và runtime context

### 1.1 Topology quan trọng

Hệ thống có 2 GPU AMD khác kiến trúc (MI50 + RX6600). Đây là điểm gây nhiều lỗi nhất khi dùng `HSA_OVERRIDE_GFX_VERSION`.

Ví dụ truy vấn:

```bash
lspci -nn | grep -i amd
```

### 1.2 ROCm/HIP stack (golden)

- Python: 3.13
- torch: `2.7.0+rocm6.3`
- HIP string: `6.3.42131-fa1d09cbd`
- transformers: 5.8.0
- unsloth: 2026.5.6
- unsloth_zoo: 2026.5.4
- bitsandbytes: source build target `gfx906:sramecc-:xnack-`

### 1.3 VRAM thực tế

- MI50 trong setup này có tổng bộ nhớ GPU quan sát được khoảng `31.984 GB`.
- Nếu một số công cụ hiện chuỗi 16GB gây nhầm lẫn, hãy lấy số từ `torch.cuda.get_device_properties(0).total_memory` trong env đã lock GPU.

## 2) Env vars bắt buộc

```bash
ROCR_VISIBLE_DEVICES=0
HIP_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=9.0.6
TORCHDYNAMO_DISABLE=1
TORCH_INDUCTOR_DISABLE=1
TORCH_COMPILE_DISABLE=1
```

### 2.1 Vì sao `ROCR_VISIBLE_DEVICES=0` là critical

- `HSA_OVERRIDE_GFX_VERSION=9.0.6` tác động trên tất cả GPU nhìn thấy bởi ROCr.
- Nếu RX6600 vẫn visible, runtime có thể dispatch kernel gfx906 sai GPU -> `invalid device function`.
- `HIP_VISIBLE_DEVICES` một mình là chưa đủ để chặn toàn bộ path dispatch ở tầng thấp.

## 3) Các lỗi đã gặp và cách quy về root cause

### 3.1 `invalid device function` ở torch op cơ bản

Signature:

```text
RuntimeError: HIP error: invalid device function
```

Có thể xuất hiện ở `torch.randn`/`zero_` ngay cả khi chưa dùng bnb.

Kết luận: ưu tiên check GPU visibility + dispatch runtime trước khi nghiệm thu bnb/model.

### 3.2 Liên quan Unsloth import

- `No module named torch.distributed.checkpoint.hf_storage` -> cần shim alias trên stack torch cũ.
- `PackageNotFoundError: vllm` với namespace package `vllm/` -> cần guard import/version check.

### 3.3 Gemma4 architecture recognition

- `model type gemma4 not recognized` trên transformers cũ.
- Bắt buộc transformers >= 5.5; stack này dùng 5.8.0.

## 4) bitsandbytes build notes

Build:

```bash
AMDGPU_TARGETS="gfx906:sramecc-:xnack-" python setup.py bdist_wheel
```

Verify binary target:

```bash
strings libbitsandbytes_rocm63.so | grep amdgcn
```

Mong đợi có chuỗi `gfx906:sramecc-:xnack-`.

## 5) Benchmarks Gemma4-31B (verified)

| Cấu hình | Input shape | Forward | Backward | Optimizer step | Peak alloc | Peak reserved | Label |
|---|---|---|---|---|---:|---:|---|
| Inference generate | N/A | VERIFIED_OK | N/A | N/A | ~17.55 GB | N/A | VERIFIED_OK |
| FastModel load | N/A | N/A | N/A | N/A | ~17.78 GB | N/A | VERIFIED_OK |
| LoRA r8 seq128 | `(1, 128)` | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | ~18.52 GB | N/A | VERIFIED_OK |
| LoRA r128 seq2048 fullpad | `(1, 2048)` | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | 29.654 GB | 30.887 GB | VERIFIED_OK |
| LoRA r128 seq4096 | N/A | VERIFIED_OK | VERIFIED_OOM | N/A | ~31.9 GB | N/A | VERIFIED_OOM |
| LoRA r64 seq4096 | N/A | VERIFIED_OK | VERIFIED_OOM | N/A | ~28.06 GB | N/A | VERIFIED_OOM |
| LoRA r8 seq8192 fullpad | N/A | VERIFIED_OOM | N/A | N/A | ~28.31 GB | N/A | VERIFIED_OOM |

Kết luận operational: `fullpad seq=2048` là mốc ổn định đã verify cho Gemma4-31B LoRA trong stack này.

Chi tiết verify cho run thành công chính:

- `input_shape=(1, 2048)`
- `forward=True`
- `backward=True`
- `optimizer_step=True`
- Fit được, nhưng headroom VRAM reserved khá sít (~1.1GB trên MI50 31.984GB).
- Evidence: `evidence/gemma4-lora-r128-seq2048-fullpad-ok.md`

## 6) SDPA backend reality trên gfx906

- PyTorch SDPA `FLASH_ATTENTION` / `EFFICIENT_ATTENTION`: không được compile cho gfx906 trong stack này.
- Fallback thực tế: `MATH`.
- Do đó chi phí bộ nhớ với context dài rất cao.
- Script SDPA trong repo là probe khả năng backend, không phải performance benchmark.

## 7) xFormers / FA2 thông tin đã chỉnh lý

- xFormers trong env hiện tại không usable trên ROCm/gfx906 (mismatch/unusable).
- Unsloth detect: `Xformers=None, FA2=False`.
- Không được diễn giải là xFormers fallback đang hoạt động bình thường trên MI50.

## 8) noflash-attention experiment

### 8.1 Isolated SDPA

- seq4096 fp16: forward+backward `VERIFIED_OK`, peak_reserved ~4.068 GB.
- seq8192 fp16: forward+backward `VERIFIED_OK`, peak_reserved ~10.420 GB.

### 8.2 Gemma4 integration

- r64 seq4096: `VERIFIED_OOM`, `SDPA_CALL_COUNT=60`, forward không hoàn tất.
- Sweep:
  - r8 seq4096: `VERIFIED_OOM`
  - r16 seq4096: `VERIFIED_OOM`
  - r32 seq4096: `VERIFIED_OOM`
  - r8 seq8192: `VERIFIED_OOM`

SDPA signature đã quan sát:

- `(1, 32, 4096, 256)` bf16
- `(1, 32, 4096, 512)` bf16
- `(1, 32, 8192, 256)` bf16
- `(1, 32, 8192, 512)` bf16

Kết luận: `NEGATIVE_RESULT` cho mục tiêu cứu long-context Gemma4-31B train trong stack hiện tại.

## 9) FineTome train[:3000] token stats

- count: 3000
- mean: 568.05
- p50: 457
- p75: 662
- p90: 1003
- p95: 1320
- p99: 2195
- max: 6531
- `>2048`: 44/3000 = 1.47%
- `>3072`: 10/3000 = 0.33%
- `>4096`: 4/3000 = 0.13%
- `>8192`: 0

## 10) So sánh với notebook Kaggle 2xT4

- Notebook 2xT4 là context khác: 2 GPU balanced map, xFormers behavior, rank8, data ngắn.
- Không được suy diễn kết quả đó trực tiếp sang MI50 1-GPU long-context path.
- Trong stack này, không có bằng chứng FA2/xFormers usable cho MI50.

## 11) Sanitize conventions

Trong bản public này:

- `/home/elysia` -> `/home/<user>`
- `/media/elysia/NVME PCIE3` -> `/path/to/NVME`
- private project/checkpoint paths -> `<private-project>` / `<private-checkpoint>`
