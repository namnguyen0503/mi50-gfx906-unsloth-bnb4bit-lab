# VRAM Result Table

| config | method | fullpad | input_shape | forward | backward | optimizer_step | peak_alloc_gb | peak_reserved_gb | status |
|---|---|---|---|---|---|---|---:|---:|---|
| gemma4 inference generate | inference | N/A | N/A | VERIFIED_OK | N/A | N/A | ~17.55 | N/A | VERIFIED_OK |
| gemma4 FastModel load | load | N/A | N/A | N/A | N/A | N/A | ~17.78 | N/A | VERIFIED_OK |
| LoRA r8 seq128 | LoRA | no | `(1, 128)` | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | ~18.52 | N/A | VERIFIED_OK |
| LoRA r128 seq2048 fullpad | LoRA | yes | `(1, 2048)` | OK | OK | OK | 29.654 | 30.887 | VERIFIED_OK |
| LoRA r128 seq4096 | LoRA | no | N/A | VERIFIED_OK | VERIFIED_OOM | N/A | ~31.9 | N/A | VERIFIED_OOM |
| LoRA r64 seq4096 | LoRA | no | N/A | VERIFIED_OK | VERIFIED_OOM | N/A | ~28.06 | N/A | VERIFIED_OOM |
| LoRA r8 seq8192 fullpad | LoRA | yes | N/A | VERIFIED_OOM | N/A | N/A | ~28.31 | N/A | VERIFIED_OOM |
| LoRA r8 seq4096 triton+fused_ce | LoRA | yes | `(1, 4096)` | OK | OK | OK | 22.747 | 24.590 | VERIFIED_OK |
| LoRA r16 seq4096 triton+fused_ce | LoRA | yes | `(1, 4096)` | OK | OK | OK | 22.975 | 24.910 | VERIFIED_OK |
| LoRA r32 seq4096 triton+fused_ce | LoRA | yes | `(1, 4096)` | OK | OK | OK | 23.431 | 25.830 | VERIFIED_OK |
| LoRA r8 seq8192 triton+fused_ce | LoRA | yes | `(1, 8192)` | OK | OK | OK | 25.562 | 27.188 | VERIFIED_OK |
| LoRA r16 seq8192 triton+fused_ce | LoRA | yes | `(1, 8192)` | OK | OK | OK | 26.002 | 27.832 | VERIFIED_OK |
| LoRA r32 seq8192 triton+fused_ce | LoRA | yes | `(1, 8192)` | OK | OK | OK | 26.875 | 28.719 | VERIFIED_OK |

## Gradient checkpointing note

- The `LoRA r128 seq2048 fullpad` `VERIFIED_OK` row was measured with `use_gradient_checkpointing="unsloth"`.
- `use_gradient_checkpointing=True` (`torch` mode) also passed for the same config, with `30.848` allocated and `31.686` reserved.
- `use_gradient_checkpointing=False` (`none`) failed at forward with `VERIFIED_OOM`.
- See `evidence/gemma4-gradient-checkpointing-comparison.md`.

## Precision / dtype note

- Unless otherwise stated, the original `LoRA r128 seq2048 fullpad` `VERIFIED_OK` row used the BF16 path.
- FP16 was separately verified for the same one-step workload with the same peak VRAM: `29.654` allocated / `30.887` reserved.
- In the reversed-order dtype benchmark, FP16 `compute_sec=44.279` versus BF16 `compute_sec=119.220`, about `2.69x` faster.
- Requested dtype matched the observed model/embedding dtype for both modes, while the first trainable LoRA parameter and the loss stayed `torch.float32`.
- See `evidence/gemma4-dtype-fp16-vs-bf16.md`.

## PEFT variant note

- LoRA and rsLoRA both fit the measured `r128 seq2048 fullpad` 31B budget in this environment.
- rsLoRA stayed in the same VRAM/runtime class as baseline LoRA in this one-step benchmark.
- DoRA is supported by the current Unsloth/FastModel path, but it is `VERIFIED_OOM` at backward for this exact 31B config.
- DoRA added `4,229,120` trainable parameters versus baseline LoRA in the measured run.
- This does not rule out DoRA for 9B or lower-rank configs.
- See `evidence/gemma4-peft-variant-lora-rslora-dora.md`.

## Real-data PEFT micro-run note

- Private JSONL dataset (`schema=text`, `raw_text_logged=no`): LoRA / rsLoRA / DoRA all completed `3/3` FP16 steps with finite loss/grad.
- LoRA and rsLoRA both used `22.377` alloc / `23.283` reserved and stayed around `21s/step`.
- DoRA fit at `r8/seq1024`, but used `23.569` alloc / `25.645` reserved and ran at about `86.911s/step`.
- See `evidence/gemma4-realdata-peft-nan-speed-r8-seq1024-fp16.md`.

## Real-data 100-sample CPT held-out loss probe note

- Private JSONL dataset (`schema=text`, `dataset_num_records=1700`, `raw_text_logged=no`): LoRA / rsLoRA / DoRA all completed `100/100` FP16 steps with finite loss/grad on a fixed disjoint train/eval split.
- rsLoRA had the best held-out CPT `eval_loss_after=1.774377` and `eval_loss_delta=6.113859` while staying near LoRA's speed/VRAM class (`21.810s/step`, `22.379` alloc / `23.285` reserved).
- LoRA was slightly fastest (`21.729s/step`) and had the lowest peak reserved VRAM (`23.283`).
- DoRA completed successfully, but used `23.568` alloc / `25.562` reserved and ran at `86.668s/step`, so it was not cost-effective in this Gemma4-31B probe.
- This is a CPT held-out loss probe, not a final SFT/persona-quality eval.
- See `evidence/gemma4-realdata-peft-100sample-cpt-eval-r8-seq1024-fp16.md`.

## Triton-gfx906 Fused CE note
  * `fullpad=yes` for the Triton+fused CE rows means fixed-shape full sequence blocks. For example, the successful `seq8192` rows used `active_tokens=8191` and `pad_tokens_added=1`, so they are real-active fullpad runs, not mostly-padded shortcuts.

- LoRA `seq8192` passed for `r8`, `r16`, `r32` using a Triton-gfx906 all-text-attention patch and fused active linear CE.
- This resolves the dense logits and unpatched sliding attention SDPA workspace OOM issues, fitting within ~28.7GB peak reserved.
- Slower compute: a `r128 seq2048` test took `72.429s` compared to the old `44.279s` baseline, but enables `seq8192` capability.
- See `evidence/gemma4-triton-gfx906-fused-ce-seq8192-r8-r16-r32.md` and `evidence/gemma4-triton-gfx906-fused-ce-speed-tradeoff.md`.
