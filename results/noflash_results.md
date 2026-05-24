# noflash-attention Results

## Isolated SDPA

| config | forward | backward | peak_alloc_gb | peak_reserved_gb | status |
|---|---|---|---:|---:|---|
| B=1,H=32,S=4096,D=128 fp16 | True | True | 3.547 | 4.068 | VERIFIED_OK |
| B=1,H=32,S=8192,D=128 fp16 | True | True | 9.813 | 10.420 | VERIFIED_OK |

## Gemma4 integration

| config | forward | backward | optimizer_step | sdpa_calls | peak_alloc_gb | peak_reserved_gb | status |
|---|---|---|---|---:|---:|---:|---|
| r64 seq4096 (strategy B) | False | False | False | 60 | 28.059 | 30.461 | VERIFIED_OOM |
| r8 seq4096 | False | False | False | 60 | 30.423 | 30.625 | VERIFIED_OOM |
| r16 seq4096 | False | False | False | 60 | 30.655 | 30.832 | VERIFIED_OOM |
| r32 seq4096 | False | False | False | 60 | 31.110 | 31.348 | VERIFIED_OOM |
| r8 seq8192 | False | False | False | 60 | 26.868 | 30.660 | VERIFIED_OOM |

`NEGATIVE_RESULT`: noflash SDPA patching did not make Gemma4 long-context LoRA training feasible in this stack.
