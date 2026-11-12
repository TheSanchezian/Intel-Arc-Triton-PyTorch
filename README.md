# Intel-Arc-Triton-PyTorch

Triton kernels and PyTorch (XPU) tools for running and training quantized models on Intel Arc GPUs under Windows.

| Folder | What it is |
|---|---|
| [arc-ternary-triton-pytorch-kernels](arc-ternary-triton-pytorch-kernels/) | Triton unpack kernels for ternary PQ2_0 (2-bit) MoE weights, plus a PyTorch XPU trainer: Flint-40B trained on one Arc A770 16 GB, 12 -> ~100 tokens/s |

More Intel-Arc quant formats will be added as separate folders.

License: Apache 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).