# arc-ternary-triton-pytorch-kernels

Triton unpack kernels and a PyTorch XPU trainer for ternary PQ2_0 (2-bit) mixture-of-experts weights, built to train Flint-40B on a single Intel Arc A770 16 GB under Windows.

Only the 37.0M per-expert-row scale multipliers are trained. The 2-bit expert codes and the Q4_1 dense parts stay frozen and packed on the GPU; expert matrices are unpacked to fp32 on the fly, in forward and again in backward. Base model (ternary GGUF): [TheSanchezian/Ternary-Flint-40B-Intel-Arc](https://huggingface.co/TheSanchezian/Ternary-Flint-40B-Intel-Arc).

## Hardware and software

- GPU: Intel Arc A770 16 GB (Alchemist)
- OS: Windows, Visual Studio 2022 build tools
- PyTorch 2.15.0+xpu, triton-xpu 3.9.0
- Level Zero SDK v1.35.0 (needed by Triton to compile launchers)

## Results

Training speed on 1,024-token chunks (answer tokens per second), one A770. Each row adds one change to the row above it.

| Step | Change | Measured |
|---|---|---|
| 0 | Original per-expert loop, fp32 | 79-87 s/step, ~12 tok/s |
| 1 | Bit-shift unpack instead of a 16-bit/LUT gather | 26.4 s/step (one-step test) |
| 2 | Batched experts, fixed padding, `PYTORCH_ALLOC_CONF=expandable_segments:True` | 27-30 s/step live, ~33-35 tok/s |
| 3 | Triton one-pass unpack kernel (bit-identical to the torch unpack) | 16.5-21 s/step live, 40-58 tok/s |
| 4 | Expert-sorted grouping (G=16), id-gather unpack, routing reuse in recompute, one H2D copy per layer | 9.4-9.8 s/step live, ~100-106 tok/s; peak 12.1-13.2 GiB |
| 5 | Blocked attention (query blocks under checkpoint) | examples up to 4,096 tokens (peak 12.51 GiB); 8,192 tokens synthetic (peak 13.10 GiB) |

Unpack kernel alone (all 256 experts of one matrix kind in one layer):

| | Time per matrix kind per layer | Throughput |
|---|---|---|
| torch unpack | ~28-31 ms | |
| Triton unpack | ~2.8 ms | ~400-427 GB/s |

The layer expert pipeline went from 116 ms to 34 ms with the Triton unpack. In step 4 one training step went from 27-28 s to 8.6-9.2 s, and GPU busy time from 30-45% to ~75-80%.

## What worked

- Shift-and-mask unpack. A LUT gather into fp16 crashed the driver.
- One-pass Triton unpack, bit-identical to torch (`torch.equal`). Memory-bound, runs at ~400-427 GB/s.
- Batched expert groups with fixed padding, and `expandable_segments:True`.
- Sorting experts by token count, groups of 16, with the expert ids gathered inside the unpack kernel.
- Computing routing once per layer and reusing it in the recompute pass, plus a single host-to-device copy per layer.
- Blocked attention under `torch.utils.checkpoint` for long examples.

## What did not work

- **`tl.dot` on the A770.** Triton `tl.dot` did not reach the XMX/DPAS matrix engines. A fp32 fused unpack-and-matmul kernel reached ~92 GFLOP/s and a fp16 hi/lo kernel ~43 GFLOP/s, against ~330 GFLOP/s for torch fp32 `bmm`. These experimental fused matmul kernels (v1 fp32, v2 fp16 hi/lo) are not included; the trainer uses the Triton unpack followed by torch `bmm`.
- Block pointers are removed in this triton-xpu build (tensor descriptors instead).
- `grf_mode='large'` is rejected (valid: 128, 256, 512, auto, default).
- `torch.xpu.synchronize()` does not wait for Triton kernels. Time with a host readback.
- `torch.xpu.empty_cache()` crashed the driver.
- Reduced precision: fp16 all-model training gave NaN; fp16 experts-only gave 11-16% gradient error; bf16 gave 30% gradient error.

## Layout

```
kernels/ternary_unpack.py   production Triton unpack kernels (contiguous range, expert-id gather)
tests/test_unpack.py        standalone correctness + timing test (random data, no model)
trainer/                    flint_torch.py (model), flint_train.py, flint_supervisor.py, flint_rt.py (GGUF reader), start_train.bat
setup/                      Windows Triton setup notes, triton_env.bat, triton_check.py
```

The PQ2_0 layout the kernels decode: `W[e,n,g*128+j*4+s] = (((code>>(2s))&3)-1)*scale[e,n,g]`, with `codes` uint8 `[E,N,G,32]` and `scale` fp16 `[E,N,G]`.

## Setup

See [setup/windows_triton_setup.md](setup/windows_triton_setup.md). In short: install Visual Studio 2022 build tools and the Level Zero SDK, call `setup\triton_env.bat` (set `LEVEL_ZERO_SDK` first), then run `python setup\triton_check.py`.

## Run the test

No model needed. From a shell with the Triton environment set up:

```
python tests\test_unpack.py
```

It checks the contiguous-range kernel (both variants, with and without a preallocated output) and the expert-id kernel against a plain torch reference with `torch.equal`, then times them with a host readback. It exits non-zero if any comparison fails.

## Run the trainer

You need the Flint-40B PQ2_0 GGUF, its `tokenizer.json`, and a data folder with `train.jsonl` / `val.jsonl` (the data-builder script is not part of this repo). Set the variables and start:

```
set FLINT_GGUF=PATH\TO\Ternary-Flint-40B-PQ2_0.gguf
set TOKENIZER=PATH\TO\tokenizer.json
set DATA_DIR=PATH\TO\data
set RUN_DIR=PATH\TO\run
set PYTHON=PATH\TO\venv\Scripts\python.exe
set LEVEL_ZERO_SDK=PATH\TO\level-zero-win-sdk-1.35.0
trainer\start_train.bat
```

The script sets `PYTORCH_ALLOC_CONF=expandable_segments:True`, loads the MSVC and Level Zero environment, and runs `flint_supervisor.py`, which restarts `flint_train.py` after each clean stop and after crashes, resuming from the last save. Create an empty file named `STOP` in `RUN_DIR` to end the run. Defaults in the script: chunk 1,024, learning rate 2e-4, `--moe-group 16`, examples up to 4,096 tokens.

`flint_torch.py` falls back to a plain torch unpack (and prints a warning) if the Triton kernels cannot be imported.

## Credits and license

- Flint-40B model and its authors: Apache License 2.0. This repo builds on the model; see [NOTICE](../NOTICE).
- Base model on Hugging Face: [TheSanchezian/Ternary-Flint-40B-Intel-Arc](https://huggingface.co/TheSanchezian/Ternary-Flint-40B-Intel-Arc).
- Code in this repo: Apache License 2.0, see [LICENSE](../LICENSE).
- Trained and written with help from Claude (Anthropic).
