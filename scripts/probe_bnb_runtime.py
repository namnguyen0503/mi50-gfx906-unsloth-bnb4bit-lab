import importlib.util
import os
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="bitsandbytes 4-bit runtime probe. Requires GPU.")
    p.parse_args()

    import bitsandbytes as bnb
    import bitsandbytes.cextension as ce
    import torch

    print("=== BNB RUNTIME PROBE ===")
    for k in [
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "TORCH_COMPILE_DISABLE",
    ]:
        print(f"{k}={os.environ.get(k)}")

    print(f"torch={torch.__version__}")
    print(f"torch_hip={getattr(torch.version, 'hip', None)}")
    print(f"bitsandbytes={bnb.__version__}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu_name={props.name}")
        print(f"gpu_total_mem_gb={props.total_memory / 1024**3:.3f}")

    print(f"bnb.__file__={bnb.__file__}")
    print(f"ce.__file__={ce.__file__}")
    bnb_dir = Path(bnb.__file__).resolve().parent
    for candidate in sorted(bnb_dir.glob("libbitsandbytes*.so")):
        print(f"bnb_shared_object={candidate}")

    vllm_spec = importlib.util.find_spec("vllm")
    print(f"vllm_spec={vllm_spec}")

    linear = torch.nn.Linear(64, 64, bias=True, dtype=torch.float16).to("cuda")
    x = torch.randn(2, 64, device="cuda", dtype=torch.float16)
    linear_q = bnb.nn.Linear4bit(
        64,
        64,
        bias=True,
        compute_dtype=None,
        compress_statistics=True,
        quant_type="nf4",
        device="meta",
    )
    linear_q.weight = bnb.nn.Params4bit(data=linear.weight, quant_type="nf4", requires_grad=False)
    linear_q.bias = torch.nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
    linear_q = linear_q.to("cuda")
    y = linear_q(x)
    print(f"success_shape={tuple(y.shape)}")
    print(f"success_dtype={y.dtype}")


if __name__ == "__main__":
    main()
