# VRAM Result Table

| config | forward | backward | optimizer_step | peak_alloc_gb | status |
|---|---|---|---|---:|---|
| gemma4 inference generate | VERIFIED_OK | N/A | N/A | ~17.55 | VERIFIED_OK |
| gemma4 FastModel load | N/A | N/A | N/A | ~17.78 | VERIFIED_OK |
| LoRA r8 seq128 | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | ~18.52 | VERIFIED_OK |
| LoRA r128 seq2048 | VERIFIED_OK | VERIFIED_OK | VERIFIED_OK | ~29.65 | VERIFIED_OK |
| LoRA r128 seq4096 | VERIFIED_OK | VERIFIED_OOM | N/A | ~31.9 | VERIFIED_OOM |
| LoRA r64 seq4096 | VERIFIED_OK | VERIFIED_OOM | N/A | ~28.06 | VERIFIED_OOM |
| LoRA r8 seq8192 fullpad | VERIFIED_OOM | N/A | N/A | ~28.31 | VERIFIED_OOM |
