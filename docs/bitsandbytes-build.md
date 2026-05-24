# bitsandbytes Build (gfx906)

## Target and rationale

bitsandbytes was built from source with gfx906 target enabled.

```bash
AMDGPU_TARGETS="gfx906:sramecc-:xnack-" python setup.py bdist_wheel
```

## Verification method

Check shared object symbols:

```bash
strings libbitsandbytes_rocm63.so | grep amdgcn
```

Expected target string includes:

- `amdgcn-amd-amdhsa--gfx906:sramecc-:xnack-`

## Important interpretation

In this lab, bitsandbytes binary content was not the main issue; environment dispatch/runtime constraints were the dominant failure source.
