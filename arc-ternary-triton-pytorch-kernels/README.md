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
                            flint_lora.py, flint_train_lora*.py (LoRA, fast path, packing, experimental bf16x2), export_folded.py, start_train_lora_pack.bat
kernels/fused_kernels.py    fused RMSNorm and SwiGLU Triton kernels (fast path)
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

## Two-GPU model split (longer examples)

`trainer/flint_torch_split.py` splits Flint across two Arc GPUs: layers 0..26 plus embeddings on `xpu:0`, layers 27..48 plus final norm, head and loss on `xpu:1` (the split point is chosen automatically from layer sizes). Only the hidden state (and its gradient) crosses between the cards. Each GPU holds its own multipliers and Adam state; saves use the same format as the single-GPU trainer, so a run can switch between the two.

This is a model split, not data parallel: the two cards take turns, so speed matches one GPU. The gain is memory headroom for longer training examples.

Measured on 2x Arc A770 16 GB (one forward + backward pass per example, chunk 1,024 tokens, fp32):

| Example length | tok/s | Peak GPU 0 | Peak GPU 1 |
|---:|---:|---:|---:|
| 4,096 | 101 | 7.1 GiB | 6.4 GiB |
| 6,144 | 103 | 7.1 GiB | 6.5 GiB |
| 8,192 | 101 | 7.4 GiB | 6.7 GiB |
| 10,240 | 96 | 7.6 GiB | 6.8 GiB |
| 12,288 | 91 | 7.8 GiB | 7.0 GiB |
| 16,384 | 78 | 8.4 GiB | 7.5 GiB |

For comparison, the single-GPU trainer peaks at about 12.8 GiB at 4,096 tokens.

Correctness: split vs single-device on the same 512-token example matched with loss within 1e-6 to 4e-4 relative and gradients within about 1e-5 relative, apart from repetitions where MoE routing ties flip (gradient differences of about 2e-3 to 4e-3; single-GPU runs show the same kind of routing-tie noise, at about 1e-3). A Triton unpack probe was bit-identical on both devices, and the cross-device hand-off was exact.

Run it with `trainer\start_train_split.bat` (same variables as `start_train.bat`, plus `MAXTOK`, default 16,384, and `SPLIT_AT`, default automatic).

## LoRA, fast path and sequence packing

After the multiplier phases, training moved to LoRA adapters on top of the frozen ternary weights.

- **LoRA** (`trainer/flint_lora.py`): rank 16, alpha 32, on attention, the two dense FFN layers and the shared experts, plus one adapter per projection kind shared by all 256 routed experts. 27.14M trainable fp32 parameters (103.5 MB). Learning rate 1e-5 (1e-4 diverged).
- **Fast path** (`flint_train_lora_fast.py --fast`): the trained per-row multipliers are folded into the GGUF scales once with `trainer/export_folded.py`, so the forward pass no longer multiplies them in, and RMSNorm and SwiGLU use fused Triton kernels (`kernels/fused_kernels.py`). Requires `--freeze-mult`; pass the folded GGUF as `--gguf`.
- **Sequence packing** (`flint_train_lora_pack.py --pack`): short examples are packed into one 1,024-token chunk with a block-diagonal causal mask, positions (RoPE) restarting at each example, no KV carried between examples, and a loss mask per example. Examples longer than the chunk are trained unpacked, as before.

Measured on one Arc A770 16 GB, fp32, chunk 1,024:

| Test | Result |
|---|---|
| Fast path, real-conditions test | 1.43x faster (1.6-1.7x on full chunks); loss matched the unfused path within 0.03%; peak 13.2 GiB |
| Fast path, live training | +10-25% (about 68-78 tok/s), smaller than the test because of fixed per-step cost on short examples |
| Packing, 200 real training examples, packed vs unpacked | 58.5 -> 105.5 trained tok/s (1.80x); 200 -> 97 optimizer steps; 26.6 -> 14.7 min; peak 12.48 -> 12.29 GiB |
| Packing, leakage check | changing one packed example changed another example's attention output by exactly 0.0 |
| Packing, per-example loss vs unpacked | within 1.5e-6 relative |
| Packing, LoRA gradient vs unpacked | 5.1e-3 relative L2 (expected: one update now averages several examples) |
| Live run, mixed data up to 10,240 tokens | 77-83 trained tok/s; about 16% of steps were packable because many examples exceed 1,024 tokens |

Run it with `trainer\start_train_lora_pack.bat` (variables listed at the top of the script). The script restarts the trainer after each 2-hour stop or crash and retries at 9,216 tokens after an out-of-memory crash.

Known issue: periodic evals run only when the step count lands exactly on a multiple of `--eval-every`; an example that spans several chunks can step past it, so some evals are skipped.

Also measured (micro-benchmark, one A770): `torch.mm` bf16 reaches about 71 TFLOPS (56 with fp32 output) against about 18 TFLOPS fp32, while Triton `tl.dot` stays near 0.05-0.14 TFLOPS on this build. Fast matmuls on Arc should go through oneDNN (`torch.mm`/`bmm`), not `tl.dot`.

## Experimental: bf16x2 two-product matmuls

`trainer/flint_train_lora_bf16x2.py --fast --pack --bf16x2` runs the weight matmuls on the A770's bf16 matrix (XMX) units through oneDNN (`torch.mm`/`bmm` with `out_dtype=float32`) while keeping fp32-level accuracy:

- Weight scales are rounded to bf16 once, so ternary weights (-1/0/+1 times scale) are exact in bf16.
- Each fp32 activation is split into `hi = bf16(x)` and `lo = bf16(x - hi)`; both halves are stacked into one matmul twice as tall (fused Triton split kernel), and the two output halves are added.
- Custom autograd functions save only packed weights and inputs; weights are re-unpacked in backward.

Measured on one Arc A770 16 GB:

| Test | fp32 (packing trainer) | bf16x2 |
|---|---|---|
| Single matmul error vs fp32 | - | about 2.5e-6 relative L2, forward and backward |
| Gradient match with expert routing held fixed | noise 6.5e-6 | 2.7e-5 relative L2 |
| Eval loss, 40 examples, effect of bf16 scale rounding alone | 0.708419 | 0.708258 (0.02%) |
| 200 short training examples, trained tok/s | 74.7 | 71.2 (0.95x) |
| One 10,240-token example, tok/s | 73.5 | 80.4 (1.09x) |
| Per-step training loss, 99 steps | - | mean 0.4% relative difference |
| Peak reserved VRAM | 12.65 GiB | 12.53 GiB |

Single matmuls run 1.2-3x faster than fp32, but the split, unpack and the parts left in fp32 (attention scores, routing) cancel most of it, so end to end it is about even on mixed data. It stays off by default. Note: tiny weight changes (such as the scale rounding, or a 1e-5 nudge to the router in pure fp32) flip some near-tie expert choices in this MoE, which alone moves LoRA gradients by about 20%; compare with routing held fixed.

## Credits and license

- Flint-40B model and its authors: Apache License 2.0. This repo builds on the model; see [NOTICE](../NOTICE).
- Base model on Hugging Face: [TheSanchezian/Ternary-Flint-40B-Intel-Arc](https://huggingface.co/TheSanchezian/Ternary-Flint-40B-Intel-Arc).
- Code in this repo: Apache License 2.0, see [LICENSE](../LICENSE).
- Trained and written with help from Claude (Anthropic).
