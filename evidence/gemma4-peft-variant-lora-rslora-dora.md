# Gemma4-31B r128 seq2048 PEFT variant comparison

## Context

This evidence note records a local one-step PEFT variant micro-benchmark for:

- model: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`
- rank: `128`
- seq_len: `2048`
- fullpad input shape: `(1, 2048)`
- gradient checkpointing: `use_gradient_checkpointing="unsloth"`
- optimizer: `AdamW8bit`
- workload: one forward + backward + `optimizer.step`
- process isolation: each PEFT variant ran in a separate child process

Environment:

- torch `2.7.0+rocm6.3`
- HIP `6.3.42131-fa1d09cbd`
- ROCm/gfx906 MI50
- Unsloth `2026.5.6`
- Transformers `5.8.0`
- `Xformers=None`, `FA2=False`

## Verified results

| variant | use_rslora | use_dora | status | phase | trainable_params | extra_trainable_params_vs_lora | peak_alloc_gb | peak_reserved_gb | interpretation |
|---|---|---|---|---|---:|---:|---:|---:|---|
| lora | `False` | `False` | VERIFIED_OK | done | 979435520 | 0 | 29.654 | 30.887 | Baseline LoRA path fits this 31B `r128 seq2048 fullpad` budget. |
| rslora | `True` | `False` | VERIFIED_OK | done | 979435520 | 0 | 29.654 | 30.885 | rsLoRA was runtime/VRAM-neutral in this measured one-step case. |
| dora | `False` | `True` | VERIFIED_OOM | backward | 983664640 | 4229120 | 30.578 | 31.648 | DoRA is supported in this environment, but the extra memory pressure pushes this exact 31B config over budget at backward. |

## DoRA OOM note

```text
HIP out of memory. Tried to allocate 442.00 MiB. GPU 0 has a total capacity of 31.98 GiB of which 22.00 MiB is free. Of the allocated memory 30.47 GiB is allocated by PyTorch, and 1.17 GiB is reserved by PyTorch but unallocated.
```

## Conclusion

- LoRA: baseline `VERIFIED_OK`.
- rsLoRA: `VERIFIED_OK`, with the same trainable parameter count and the same VRAM class as LoRA in this measured one-step case.
- DoRA: supported by the current Unsloth/FastModel path in this environment, adds `4,229,120` trainable parameters, but is `VERIFIED_OOM` at backward for this exact 31B `r128 seq2048 fullpad` test.
- This is a runtime/VRAM/fit benchmark only, not a quality benchmark.
- For this specific 31B budget question, DoRA is a `NEGATIVE_RESULT` only in the narrow sense that it does not fit this exact configuration.

## Caveat

- No quality conclusion is made here.
- This result should not be generalized into “DoRA is bad” or “rsLoRA is better” as methods.
- 9B headroom is much larger, so DoRA remains a valid quality candidate for 9B experiments and should not be ruled out by this 31B result.
