# Gemma4-31B r8 seq1024 real-data PEFT FP16 micro-run

## Privacy note

- Dataset source: private JSONL dataset (`<private-elysia-cpt-dataset>.jsonl`)
- Schema detected: `text`
- `raw_text_logged=no`
- No raw samples are included in this evidence note.

## Context / config

- model: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`
- rank: `8`
- seq_len: `1024`
- dtype: `torch.float16`
- gradient checkpointing: `use_gradient_checkpointing="unsloth"`
- optimizer: `AdamW8bit`
- batch size: `1`
- grad accumulation: `1`
- max steps: `3`
- process isolation: each variant ran in a separate child process
- fixed sample seed: `3407`
- same selected sample indices for all variants

Environment:

- torch `2.7.0+rocm6.3`
- HIP `6.3.42131-fa1d09cbd`
- ROCm/gfx906 MI50
- Unsloth `2026.5.6`
- Transformers `5.8.0`
- `Xformers=None`, `FA2=False`

## Verified results

| variant | status | steps | first_loss | NaN/Inf | peak_alloc_gb | peak_reserved_gb | compute_sec_per_step | interpretation |
|---|---|---:|---:|---|---:|---:|---:|---|
| lora | VERIFIED_OK | 3 | 7.265254 | no | 22.377 | 23.283 | 21.630 | Baseline LoRA path completed all 3 FP16 steps with finite loss/grad. |
| rslora | VERIFIED_OK | 3 | 7.265254 | no | 22.377 | 23.283 | 21.146 | rsLoRA completed all 3 FP16 steps and stayed in the same speed/VRAM class as LoRA. |
| dora | VERIFIED_OK | 3 | 7.265254 | no | 23.569 | 25.645 | 86.911 | DoRA fit at `r8/seq1024`, but used materially more VRAM and was much slower in this micro-run. |

## Conclusion

- LoRA, rsLoRA, and DoRA all completed `3/3` FP16 steps with finite loss and sampled finite gradients.
- LoRA and rsLoRA were nearly identical in this short real-data run.
- DoRA fit at `r8/seq1024`, but had materially higher VRAM usage and about `4x` slower compute per step than LoRA/rsLoRA.
- DoRA added `4,229,120` trainable parameters versus LoRA in this measured run.

## Caveats

- No quality conclusion is made here.
- No long-run FP16 stability conclusion is made here.
- No raw dataset content is included.
- This is a micro-run only: it demonstrates short-horizon finite-loss / finite-grad behavior, runtime class, and VRAM class on one private dataset.
