# Legacy Unsloth Install Guide for gfx906 (VI, sanitized)

> Warning: This is a historical base guide. Some instructions were later corrected by the forensic guide, especially GPU visibility variables.

Tài liệu này là bản tiếng Việt đã sanitize từ guide cài đặt gốc. Nó được giữ lại như một mốc lịch sử để tham chiếu quá trình dựng môi trường ban đầu, không phải hướng dẫn public được khuyến nghị cuối cùng.

Tài liệu cập nhật nên đọc thêm:

- `docs/unsloth-install-guide-gfx906.en.md`
- `docs/mi50-gfx906-full-guide.en.md`
- `docs/rocm-gfx906-debugging.md`

## 1) Xóa repo cũ và cài lại lõi AMD ROCm 6.3.3

Vì Ubuntu là 24.04 (Noble), guide gốc bắt đầu bằng việc gỡ tàn dư repo sai đời phát hành rồi cài bộ cài ROCm phù hợp:

```bash
# Xóa repo cũ bị sai
sudo rm /etc/apt/sources.list.d/rocm.list
sudo apt-get update

# Tải công cụ cài đặt chính thức của AMD cho Noble
wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb
sudo apt-get install ./amdgpu-install_6.3.60303-1_all.deb

# Cài ROCm cốt lõi
sudo amdgpu-install --usecase=rocm
```

Ghi chú lịch sử:

- Guide gốc nhắc khởi động lại máy và enroll MOK nếu màn hình Secure Boot xuất hiện.
- Phần ROCm 6.3.3 này vẫn hữu ích như một ghi chú cài nền tảng.

## 2) Hướng dẫn GPU visibility cũ

Guide gốc dùng:

```bash
export HIP_VISIBLE_DEVICES=1
export HSA_OVERRIDE_GFX_VERSION=9.0.6
```

Cảnh báo:

- Đây là chỉ dẫn lịch sử, không còn là cấu hình khuyến nghị cuối cùng.
- Kết luận forensic sau đó cho thấy trên hệ mixed GPU (MI50 + RX6600), chỉ đặt `HIP_VISIBLE_DEVICES` là chưa đủ để chặn toàn bộ đường dispatch ở tầng ROCr.
- Hướng dẫn public hiện tại yêu cầu thêm `ROCR_VISIBLE_DEVICES=0` và dùng `HIP_VISIBLE_DEVICES=0` cho final setup đã verify.

## 3) Tạo môi trường ảo sạch (Python 3.13)

Ví dụ từ guide gốc:

```bash
python3.13 -m venv /home/<user>/unsloth_env
source /home/<user>/unsloth_env/bin/activate
```

Nếu bạn dùng vị trí khác, thay bằng đường dẫn phù hợp như `/path/to/venv`.

## 4) Cài Unsloth và thư viện phụ trợ cơ bản

Guide gốc tách bước cài `bitsandbytes` ra riêng:

```bash
pip install --no-deps unsloth
pip install transformers==4.55.0 accelerate peft triton==3.3.0
pip install unsloth_zoo Pillow requests
```

Ghi chú lịch sử:

- Đây là snapshot tại thời điểm guide gốc được viết.
- Stack cuối cùng được verify trong repo này đã dùng phiên bản mới hơn cho `transformers`, `unsloth`, và `unsloth_zoo`.

## 5) Tự build bitsandbytes cho gfx906

Guide gốc mô tả việc build bitsandbytes từ source do wheel public không còn hỗ trợ `gfx906` đầy đủ.

```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes
```

Guide gốc còn nhắc sửa `pyproject.toml` để tránh lỗi tương thích Python 3.13 trong một số trạng thái source tree cũ.

Biến môi trường build theo guide gốc:

```bash
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export MAKEFLAGS="-j$(nproc)"

export ROCM_HOME=/opt/rocm
export COMPUTE_BACKEND=hip
export PYTORCH_ROCM_ARCH="gfx906"
export AMDGPU_TARGETS="gfx906"
export CMAKE_ARGS="-DCOMPUTE_BACKEND=hip -DAMDGPU_TARGETS=gfx906"
```

Chạy build:

```bash
pip install .
cd ..
```

Cảnh báo:

- Hướng dẫn public cập nhật trong repo này đã chuẩn hóa target thành `gfx906:sramecc-:xnack-`.
- Xem tài liệu mới: `docs/bitsandbytes-build.md`.

## 6) Cài PyTorch ROCm

Guide gốc dùng:

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

Ghi chú:

- Repo forensic hiện tại xác nhận golden env dùng `torch==2.7.0+rocm6.3` với HIP build string `6.3.42131-fa1d09cbd`.
- Không phải mọi torch build nhìn giống nhau theo version label đều chạy tốt trên MI50.

## 7) Tắt compiler path của PyTorch

Guide gốc nhấn mạnh việc vô hiệu hóa compiler path trước khi chạy training script:

```bash
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
```

Kết luận cuối cùng của repo này đã chuẩn hóa thành:

```bash
export TORCHDYNAMO_DISABLE=1
export TORCH_INDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
```

## 8) Tóm tắt các điểm đã bị thay thế

- `HIP_VISIBLE_DEVICES=1` không còn là khuyến nghị cuối cùng.
- Thiếu `ROCR_VISIBLE_DEVICES=0` là sai sót quan trọng với topology mixed GPU.
- Build target `AMDGPU_TARGETS="gfx906"` đã được supersede bởi `gfx906:sramecc-:xnack-`.
- Version snapshot của `transformers`/`unsloth` trong guide gốc không còn đại diện cho golden env cuối cùng.

## 9) Sanitization policy

- Original home path -> `/home/<user>`
- Local/private paths -> `/path/to/venv`, `/path/to/workdir`, `<private-project>`
