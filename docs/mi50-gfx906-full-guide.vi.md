# MI50 (gfx906) ROCm 6.3 + Unsloth + bitsandbytes 4-bit: Forensic Guide (VI, sanitized)

Tai lieu nay la ban tieng Viet da duoc sanitize tu guide forensic goc.

## 0) Scope va nguyen tac

- Muc tieu: ghi lai stack da chay duoc tren MI50/gfx906 voi Unsloth + bnb 4-bit.
- Day la tai lieu forensic/reproducibility, khong phai one-click installer.
- Ket qua duoc gan nhan: `VERIFIED_OK`, `VERIFIED_OOM`, `UNVERIFIED`, `NEGATIVE_RESULT`.

## 1) Hardware va runtime context

### 1.1 Topology quan trong

He thong co 2 GPU AMD khac kien truc (MI50 + RX6600). Day la diem gay nhieu loi nhat khi dung `HSA_OVERRIDE_GFX_VERSION`.

Vi du truy van:

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

### 1.3 VRAM thuc te

- MI50 trong setup nay co tong bo nho GPU quan sat duoc ~`31.984 GB`.
- Neu mot so cong cu hien chuoi 16GB gay nham lan, lay so do tu `torch.cuda.get_device_properties(0).total_memory` trong env da lock GPU.

## 2) Env vars bat buoc

```bash
ROCR_VISIBLE_DEVICES=0
HIP_VISIBLE_DEVICES=0
HSA_OVERRIDE_GFX_VERSION=9.0.6
TORCHDYNAMO_DISABLE=1
TORCH_INDUCTOR_DISABLE=1
TORCH_COMPILE_DISABLE=1
```

### 2.1 Vi sao `ROCR_VISIBLE_DEVICES=0` la critical

- `HSA_OVERRIDE_GFX_VERSION=9.0.6` tac dong tren tat ca GPU nhin thay boi ROCr.
- Neu RX6600 van visible, runtime co the dispatch kernel gfx906 sai GPU -> `invalid device function`.
- `HIP_VISIBLE_DEVICES` mot minh la chua du de chan toan bo path dispatch o tang thap.

## 3) Cac loi da gap va cach quy ve root cause

### 3.1 `invalid device function` o torch op co ban

Signature:

```text
RuntimeError: HIP error: invalid device function
```

Co the xuat hien o `torch.randn`/`zero_` ngay ca khi chua dung bnb.

Ket luan: uu tien check GPU visibility + dispatch runtime truoc khi nghiem thu bnb/model.

### 3.2 Lien quan Unsloth import

- `No module named torch.distributed.checkpoint.hf_storage` -> can shim alias tren stack torch cu.
- `PackageNotFoundError: vllm` voi namespace package `vllm/` -> can guard import/version check.

### 3.3 Gemma4 architecture recognition

- `model type gemma4 not recognized` tren transformers cu.
- Bat buoc transformers >= 5.5; stack nay dung 5.8.0.

## 4) bitsandbytes build notes

Build:

```bash
AMDGPU_TARGETS="gfx906:sramecc-:xnack-" python setup.py bdist_wheel
```

Verify binary target:

```bash
strings libbitsandbytes_rocm63.so | grep amdgcn
```

Mong doi co chuoi `gfx906:sramecc-:xnack-`.

## 5) Benchmarks Gemma4-31B (verified)

| Cau hinh | Forward | Backward | Optimizer step | Peak alloc | Label |
|---|---|---|---|---:|---|
| Inference generate | VERIFIED_OK | N/A | N/A | ~17.55 GB | VERIFIED_OK |
| FastModel load | N/A | N/A | N/A | ~17.78 GB | VERIFIED_OK |
| LoRA r8 seq128 | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | ~18.52 GB | VERIFIED_OK |
| LoRA r128 seq2048 | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | ~29.65 GB | VERIFIED_OK |
| LoRA r128 seq4096 | VERIFIED_OK | VERIFIED_OOM | N/A | ~31.9 GB | VERIFIED_OOM |
| LoRA r64 seq4096 | VERIFIED_OK | VERIFIED_OOM | N/A | ~28.06 GB | VERIFIED_OOM |
| LoRA r8 seq8192 fullpad | VERIFIED_OOM | N/A | N/A | ~28.31 GB | VERIFIED_OOM |

Ket luan operational: `seq=2048` la moc on dinh da verify cho Gemma4-31B LoRA trong stack nay.

## 6) SDPA backend reality tren gfx906

- PyTorch SDPA `FLASH_ATTENTION` / `EFFICIENT_ATTENTION`: khong duoc compile cho gfx906 trong stack nay.
- Fallback thuc te: `MATH`.
- Do do chi phi bo nho voi context dai rat cao.
- Script SDPA trong repo la probe kha nang backend, khong phai performance benchmark.

## 7) xFormers / FA2 thong tin da chinh ly

- xFormers trong env hien tai khong usable tren ROCm/gfx906 (mismatch/unusable).
- Unsloth detect: `Xformers=None, FA2=False`.
- Khong duoc dien giai la xFormers fallback dang hoat dong binh thuong tren MI50.

## 8) noflash-attention experiment

### 8.1 Isolated SDPA

- seq4096 fp16: forward+backward `VERIFIED_OK`, peak_reserved ~4.068 GB.
- seq8192 fp16: forward+backward `VERIFIED_OK`, peak_reserved ~10.420 GB.

### 8.2 Gemma4 integration

- r64 seq4096: `VERIFIED_OOM`, `SDPA_CALL_COUNT=60`, forward khong hoan tat.
- Sweep:
  - r8 seq4096: `VERIFIED_OOM`
  - r16 seq4096: `VERIFIED_OOM`
  - r32 seq4096: `VERIFIED_OOM`
  - r8 seq8192: `VERIFIED_OOM`

SDPA signature da quan sat:

- `(1, 32, 4096, 256)` bf16
- `(1, 32, 4096, 512)` bf16
- `(1, 32, 8192, 256)` bf16
- `(1, 32, 8192, 512)` bf16

Ket luan: `NEGATIVE_RESULT` cho muc tieu cuu long-context Gemma4-31B train trong stack hien tai.

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

## 10) So sanh voi notebook Kaggle 2xT4

- Notebook 2xT4 la context khac: 2 GPU balanced map, xFormers behavior, rank8, data ngan.
- Khong duoc suy dien ket qua do truc tiep sang MI50 1-GPU long-context path.
- Trong stack nay, khong co bang chung FA2/xFormers usable cho MI50.

## 11) Sanitize conventions

Trong ban public nay:

- `/home/elysia` -> `/home/<user>`
- `/media/elysia/NVME PCIE3` -> `/path/to/NVME`
- private project/checkpoint paths -> `<private-project>` / `<private-checkpoint>`
