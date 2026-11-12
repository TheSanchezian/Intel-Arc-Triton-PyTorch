# Intel-Arc-Triton-PyTorch

Triton kernels and PyTorch (XPU) tools for running and training quantized models on Intel Arc GPUs under Windows.

| Folder | What it is |
|---|---|
| [arc-ternary-triton-pytorch-kernels](arc-ternary-triton-pytorch-kernels/) | Triton unpack kernels for ternary PQ2_0 (2-bit) MoE weights, plus a PyTorch XPU trainer: Flint-40B trained on one Arc A770 16 GB, 12 -> ~100 tokens/s |

More Intel-Arc quant formats will be added as separate folders.

## Contributing: Arc developers wanted

This repo is meant to grow into a shared toolbox for Intel Arc. If you work on Arc hardware (A-series Alchemist, B-series Battlemage, Lunar Lake or other Xe GPUs), contributions are very welcome:

- **New quant formats:** int4, int8, other GGUF or OpenVINO layouts, each in its own folder.
- **Faster kernels:** especially anything that reaches the XMX/DPAS matrix engines from Triton or SYCL.
- **Benchmarks on other cards and drivers:** B580, A750, Lunar Lake, Linux.
- **Bug reports and fixes** for the Windows setup.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the few rules (bit-exact tests, measured numbers only). Open an issue to discuss ideas before large changes.
License: Apache 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).