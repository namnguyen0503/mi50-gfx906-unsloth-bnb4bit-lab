# ROCm gfx906 Debugging Notes

## Root cause (VERIFIED)

`ROCR_VISIBLE_DEVICES=0` was required. Without it, `HSA_OVERRIDE_GFX_VERSION=9.0.6` affected all visible GPUs, and a mixed MI50 + RX6600 system could route gfx906 kernels incorrectly.

## Typical failure signatures

- `RuntimeError: HIP error: invalid device function`
- Failure in basic ops (`torch.randn`, `zero_`) indicates runtime/kernel mismatch or wrong dispatch, not just bitsandbytes.

## Practical checks

1. Confirm only intended GPU is visible to ROCr and HIP.
2. Run basic torch probe before model load.
3. Run bnb linear probe before full training script.
4. Disable compiler paths on MI50:
   - `TORCHDYNAMO_DISABLE=1`
   - `TORCH_INDUCTOR_DISABLE=1`
   - `TORCH_COMPILE_DISABLE=1`

## Environment A/B finding

- Golden env (A): torch ops + bnb probe stable.
- Alternate env (B): basic torch kernels failed despite similar version labels.
- Lesson: trust runtime probes over version strings.
