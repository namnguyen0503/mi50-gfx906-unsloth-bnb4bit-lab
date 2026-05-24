# Gemma4-31B LoRA r128 seq2048 fullpad — VERIFIED_OK

This note records the verified fullpad reproduction of the main successful MI50 benchmark in this repository.

## Sanitized command summary

Run the Gemma4 LoRA VRAM reproducer with the documented MI50 runtime environment, a sanitized Hugging Face cache path such as `/path/to/hf_cache`, and the `fullpad` seq2048 configuration for:

- model: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`
- rank: `128`
- sequence length: `2048`

## Verified output

```text
STATUS=VERIFIED_OK
model_id=unsloth/gemma-4-31B-it-unsloth-bnb-4bit
rank=128
seq_len=2048
input_shape=(1, 2048)
trainable_params=979435520
load_vram_gb=17.783
lora_vram_gb=21.439
forward_status=True
backward_status=True
optimizer_step_status=True
peak_alloc_gb=29.654
peak_reserved_gb=30.887
oom_phase=
note=
```

## Result

- `Gemma4-31B bnb 4-bit + LoRA r128 + fullpad seq2048` is `VERIFIED_OK`.
- Peak allocated: `29.654GB`
- Peak reserved: `30.887GB`
- Fits, but close to VRAM limit.
