# Triton on Windows with an Intel Arc GPU (PyTorch XPU)

This is the setup that worked for the unpack kernels and the trainer. Versions used: PyTorch 2.15.0+xpu, triton-xpu 3.9.0, Intel Arc A770 16 GB, Windows.

## What Triton needs on Windows

Triton compiles a small C launcher for every kernel, so it needs a C compiler and the Level Zero headers and import library.

1. **Visual Studio 2022** with the C++ build tools (provides `cl` and `vcvars64.bat`).
2. **Level Zero SDK v1.35.0**: `level-zero-win-sdk-1.35.0.zip` from the releases page of github.com/oneapi-src/level-zero. Unzip it anywhere. It supplies `lib\ze_loader.lib` and `include\level_zero\ze_api.h`.
3. A Python venv with PyTorch XPU (2.15.0+xpu) and triton-xpu (3.9.0) installed.

## Environment

Open a plain `cmd` window and run:

```
set LEVEL_ZERO_SDK=PATH\TO\level-zero-win-sdk-1.35.0
call setup\triton_env.bat
```

`triton_env.bat` does the following, which you can also do by hand:

```
call "<VS 2022>\VC\Auxiliary\Build\vcvars64.bat"
set LIB=%LEVEL_ZERO_SDK%\lib;%LIB%
set INCLUDE=%LEVEL_ZERO_SDK%\include;%INCLUDE%
set CC=cl
```

Then check the stack (this compiles a vector-add kernel and two `tl.dot` kernels and prints their error and time):

```
python setup\triton_check.py
```

Expected: `add ok, maxerr 0.0` and two `dot ... ok` lines. If compilation fails with a missing `ze_loader.lib` or `ze_api.h`, `LIB` / `INCLUDE` do not point at the SDK. If `cl` is not found, `vcvars64.bat` was not called in the same window.

## Notes from this setup

- `torch.xpu.synchronize()` does not wait for Triton kernels here. Time Triton work with a host readback (for example `float(t.flatten()[0])`).
- `torch.xpu.empty_cache()` crashed the driver. Do not call it; use `PYTORCH_ALLOC_CONF=expandable_segments:True` instead.
- Block pointers are removed in this triton-xpu build (tensor descriptors are used instead). The kernels in this repo use plain pointer arithmetic.
- `grf_mode='large'` is rejected; the valid values are 128, 256, 512, auto, default.
- On the A770 (Alchemist) `tl.dot` did not reach the XMX/DPAS matrix engines (see the README). Memory-bound kernels such as the unpack work well.
- With several GPUs installed, select one with `ZE_AFFINITY_MASK=<index>`.
