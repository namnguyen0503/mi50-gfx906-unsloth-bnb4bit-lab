# Gemma4-31B r128 seq2048 FP16 vs BF16 dtype comparison

## Context

This evidence note records a local one-step dtype micro-benchmark for:

- model: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`
- rank: `128`
- seq_len: `2048`
- fullpad input shape: `(1, 2048)`
- gradient checkpointing: `use_gradient_checkpointing="unsloth"`
- optimizer: `AdamW8bit`
- workload: one forward + backward + `optimizer.step`
- process isolation: each precision ran in a separate child process

Environment:

- torch `2.7.0+rocm6.3`
- HIP `6.3.42131-fa1d09cbd`
- ROCm/gfx906 MI50
- Unsloth `2026.5.6`
- Transformers `5.8.0`
- `Xformers=None`, `FA2=False`

## Original benchmark summary

| precision | status | model dtype | embedding dtype | trainable param dtype | loss dtype | peak_alloc_gb | peak_reserved_gb | forward_sec | backward_sec | optimizer_sec | compute_sec | speedup |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bf16 | VERIFIED_OK | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.float32` | 29.654 | 30.887 | 44.089 | 74.908 | 0.185 | 119.182 | baseline |
| fp16 | VERIFIED_OK | `torch.float16` | `torch.float16` | `torch.float32` | `torch.float32` | 29.654 | 30.887 | 17.620 | 26.840 | 0.181 | 44.641 | `~2.67x` vs bf16 |

## Reversed-order benchmark summary

| precision | status | model dtype | embedding dtype | trainable param dtype | loss dtype | peak_alloc_gb | peak_reserved_gb | forward_sec | backward_sec | optimizer_sec | compute_sec | speedup |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| fp16 | VERIFIED_OK | `torch.float16` | `torch.float16` | `torch.float32` | `torch.float32` | 29.654 | 30.887 | 17.701 | 26.399 | 0.179 | 44.279 | `~2.69x` vs bf16 |
| bf16 | VERIFIED_OK | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.float32` | 29.654 | 30.887 | 44.297 | 74.743 | 0.180 | 119.220 | baseline |

## Main conclusion

- FP16 and BF16 both fit with identical peak VRAM in this one-step benchmark: `29.654GB` allocated / `30.887GB` reserved.
- FP16 was honored:
  - `requested_dtype=torch.float16`
  - `model_config_dtype=torch.float16`
  - `embedding_dtype=torch.float16`
- BF16 was honored:
  - `requested_dtype=torch.bfloat16`
  - `model_config_dtype=torch.bfloat16`
  - `embedding_dtype=torch.bfloat16`
- In both cases, the first trainable LoRA parameter and the observed loss tensor remained `torch.float32`.
- The reversed-order benchmark confirms the large compute-time gap was not simply a run-order artifact.
- FP16 was about `2.69x` faster by `compute_sec` in the reversed-order run (`119.220 / 44.279`).

## Caveat

- This is a one-step runtime/VRAM/dtype benchmark only.
- No quality or long-run numerical stability conclusion is made here.
- Longer FP16 training should still monitor NaNs and loss spikes.
- `step_sec` is not the primary metric because it includes broader timing and load-time noise; `compute_sec` is the more relevant comparison.

## ROCm note

- On this MI50/gfx906 stack, BF16 tensors were exposed as BF16 in the observed model path, but arithmetic throughput was still much slower than FP16 in the one-step benchmark.
- This note does not overclaim whether that behavior is fully native or partly emulated internally; it only records the measured dtype exposure, VRAM, and runtime.
