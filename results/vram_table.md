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
