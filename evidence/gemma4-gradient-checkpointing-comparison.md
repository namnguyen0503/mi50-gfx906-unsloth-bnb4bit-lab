# Gemma4-31B r128 seq2048 gradient checkpointing comparison

Context: verified fullpad benchmark on MI50/gfx906 for `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` with rank `128`, `seq_len=2048`, `input_shape=(1, 2048)`, `trainable_params=979435520`, torch `2.7.0+rocm6.3`, HIP `6.3.42131-fa1d09cbd`, and `Xformers=None, FA2=False`.

## Mode mapping

- `none` = `use_gradient_checkpointing=False`
- `torch` = `use_gradient_checkpointing=True`
- `unsloth` = `use_gradient_checkpointing="unsloth"`

The local comparison script treats `true` as invalid input. Use `torch` for HF/PyTorch-style gradient checkpointing.

## Verified results

| mode | status | phase | peak_alloc_gb | peak_reserved_gb | reserved_headroom_gb | forward_sec | backward_sec | optimizer_sec | interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| none | VERIFIED_OOM | forward | 31.014 | 31.658 | 0.326 | N/A | N/A | N/A | No gradient checkpointing fails at forward with `HIP out of memory, tried to allocate 128 MiB`. |
| torch | VERIFIED_OK | done | 30.848 | 31.686 | 0.298 | 44.947 | 75.577 | 0.275 | HF/PyTorch-style gradient checkpointing works, but reserved VRAM headroom is extremely tight. |
| unsloth | VERIFIED_OK | done | 29.654 | 30.887 | 1.097 | 45.601 | 75.834 | 0.178 | `use_gradient_checkpointing="unsloth"` works with materially better VRAM headroom. |

## Interpretation

- `none` is `VERIFIED_OOM` at forward.
- `torch=True` works, but is extremely close to the MI50 VRAM limit in this setup.
- `use_gradient_checkpointing="unsloth"` works with materially better VRAM headroom.
- Versus `torch`, `unsloth` reduced peak allocated VRAM by `1.194GB` and peak reserved VRAM by `0.799GB`.
- MI50 max memory in this setup was `31.984GB`, so reserved headroom was about `0.298GB` for `torch` and `1.097GB` for `unsloth`.

## Timing warning

- Do not interpret `step_sec` as a clean speedup metric because it includes load-time variance and broader timing noise.
- Compute-only timing is roughly similar:
  - `torch`: `forward + backward + optimizer = 120.799s`
  - `unsloth`: `forward + backward + optimizer = 121.613s`
- The verified benefit here is VRAM headroom, not a proven end-to-end speed advantage.
