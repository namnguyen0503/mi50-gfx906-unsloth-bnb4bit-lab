# Gemma4-31B r8 seq1024 real-data PEFT 100-sample CPT held-out loss probe

## Privacy note

- Dataset source: private JSONL dataset
- Public placeholder: `<private-elysia-cpt-dataset>.jsonl`
- Schema detected: `text`
- Dataset records seen by the runner: `1700`
- `raw_text_logged=no`
- `train_eval_overlap=no`
- No raw dataset samples, prompts, or completions are included here.

## Config

- Model: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`
- Rank: `8`
- Sequence length: `1024`
- Dtype: `torch.float16`
- Gradient checkpointing: `use_gradient_checkpointing="unsloth"`
- Optimizer: `AdamW8bit`
- Batch size: `1`
- Grad accumulation: `1`
- Train steps: `100`
- Held-out eval samples: `32`
- Fixed seed: `3407`
- Same disjoint train/eval split used for all variants
- Each variant ran in a separate child process to avoid VRAM fragmentation
- `local_files_only=true`
- `device_map=sequential`

Environment summary:

- GPU: MI50 / gfx906
- torch: `2.7.0+rocm6.3`
- HIP: `6.3.42131-fa1d09cbd`
- Unsloth: `2026.5.6`
- Transformers: `5.8.0`
- `Xformers=None`, `FA2=False`

## Result table

| variant | status | train_steps | eval_loss_before | eval_loss_after | eval_delta | eval_delta_per_compute_hour | first_train_loss | last_train_loss | mean_train_loss | last10_train_loss_mean | peak_alloc_gb | peak_reserved_gb | train_sec_per_step | NaN/Inf | interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| lora | VERIFIED_OK | 100 | 7.888236 | 1.804142 | 6.084094 | 10.080124 | 7.265254 | 1.444924 | 2.231947 | 1.713362 | 22.379 | 23.283 | 21.729 | no | Completed all 100 FP16 steps with finite loss/grad; slightly fastest and lowest reserved VRAM in this run. |
| rslora | VERIFIED_OK | 100 | 7.888236 | 1.774377 | 6.113859 | 10.091610 | 7.265254 | 1.358464 | 2.135640 | 1.688593 | 22.379 | 23.285 | 21.810 | no | Best held-out CPT eval loss after training, best eval delta, and best eval delta per compute hour while staying in LoRA's speed/VRAM class. |
| dora | VERIFIED_OK | 100 | 7.888236 | 1.803650 | 6.084586 | 2.527395 | 7.265254 | 1.454543 | 2.211795 | 1.718692 | 23.568 | 25.562 | 86.668 | no | Completed successfully, but used more VRAM and about 4x more compute time per step without beating rsLoRA on held-out CPT loss. |

## Key findings

- LoRA, rsLoRA, and DoRA all completed `100/100` FP16 steps with finite train loss and sampled finite gradients.
- All three variants started from the same held-out baseline: `eval_loss_before=7.888236`.
- rsLoRA had the best held-out CPT eval loss after training: `1.774377`.
- rsLoRA had the largest held-out CPT eval loss reduction: `6.113859`.
- rsLoRA had the best `eval_loss_delta_per_compute_hour`: `10.091610`, narrowly above LoRA's `10.080124`.
- LoRA was slightly fastest at `21.729s/step` and had the lowest peak reserved VRAM at `23.283GB`.
- DoRA completed successfully on this `r8 seq1024` Gemma4-31B setup, but it was much slower (`86.668s/step`) and used more VRAM (`25.562GB` reserved) without improving over rsLoRA on held-out CPT loss.
- Current practical takeaway: for this Gemma4-31B MI50 FP16 CPT probe, rsLoRA is the best practical adapter candidate. It matched LoRA's speed/VRAM class while achieving the best held-out CPT eval loss. DoRA ran successfully but was much slower and did not beat rsLoRA in this test. This is not a final persona-quality claim.

## Caveats

- No raw dataset text is included.
- This is still a short-run probe relative to real CPT training.
- Held-out CPT loss here is a domain-modeling proxy, not a final SFT quality eval and not a persona/vibe eval.
- No final quality claim is made.
- No long-run FP16 stability claim is made beyond this 100-step probe.
- This is not a universal claim that DoRA is bad.
- These results do not rule out DoRA on smaller 9B models or in dedicated SFT/persona-quality evaluation workflows.
