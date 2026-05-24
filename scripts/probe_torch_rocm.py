import os

import torch


def main():
    print("=== TORCH ROCM PROBE ===")
    for k in [
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "TORCHDYNAMO_DISABLE",
        "TORCH_INDUCTOR_DISABLE",
        "TORCH_COMPILE_DISABLE",
    ]:
        print(f"{k}={os.environ.get(k)}")

    print(f"torch={torch.__version__}")
    print(f"torch_hip={getattr(torch.version, 'hip', None)}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu_name={props.name}")
        print(f"gpu_total_mem_gb={props.total_memory / 1024**3:.3f}")
        x = torch.randn((2, 64), device="cuda", dtype=torch.float16)
        y = x.zero_()
        print(f"randn_zero_ok shape={tuple(y.shape)} dtype={y.dtype}")


if __name__ == "__main__":
    main()
