import contextlib

import torch
import torch.nn.functional as F


def gb(x):
    return x / (1024 ** 3)


def run_backend(name, backend):
    print(f"\n=== BACKEND {name} ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    status = "VERIFIED_ERROR"
    note = ""
    try:
        q = torch.randn((1, 32, 4096, 128), device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn((1, 32, 4096, 128), device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn((1, 32, 4096, 128), device="cuda", dtype=torch.float16, requires_grad=True)
        with torch.nn.attention.sdpa_kernel(backend):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        loss = y.float().mean()
        loss.backward()
        torch.cuda.synchronize()
        status = "VERIFIED_OK"
    except torch.OutOfMemoryError as e:
        torch.cuda.synchronize()
        status = "VERIFIED_OOM"
        note = str(e).replace("\n", " ")[:700]
    except RuntimeError as e:
        torch.cuda.synchronize()
        status = "VERIFIED_ERROR"
        note = str(e).replace("\n", " ")[:700]
    except Exception as e:  # pragma: no cover
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
        status = "VERIFIED_ERROR"
        note = f"{type(e).__name__}: {e}"

    peak_alloc = gb(torch.cuda.max_memory_allocated())
    peak_reserved = gb(torch.cuda.max_memory_reserved())
    print(f"status={status}")
    print("shape=(1,32,4096,128)")
    print("dtype=torch.float16")
    print(f"peak_alloc_gb={peak_alloc:.3f}")
    print(f"peak_reserved_gb={peak_reserved:.3f}")
    if note:
        print(f"note={note}")


def main():
    print("=== SDPA BACKEND BENCH (gfx906) ===")
    print(f"torch={torch.__version__}")
    print(f"torch_hip={getattr(torch.version, 'hip', None)}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    run_backend("MATH", torch.nn.attention.SDPBackend.MATH)
    run_backend("FLASH_ATTENTION", torch.nn.attention.SDPBackend.FLASH_ATTENTION)
    run_backend("EFFICIENT_ATTENTION", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)


if __name__ == "__main__":
    main()
