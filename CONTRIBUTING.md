# Contributing

Thanks for helping Intel Arc tooling grow. A few rules keep this repo trustworthy:

1. **One folder per format.** Name it `arc-<format>-triton-pytorch-kernels/` (for example `arc-int4-triton-pytorch-kernels/`), with its own README, kernels, and tests.
2. **Correctness first.** Every kernel ships with a standalone test (random data, no model needed) that compares against a plain PyTorch reference: `torch.equal` for exact kernels, or a stated tolerance with a reason.
3. **Measured numbers only.** Report the GPU, driver version, OS, PyTorch and Triton versions, and how you timed it. On XPU, `torch.xpu.synchronize()` does not wait for Triton kernels, so time with a host readback (for example `.item()`).
4. **Negative results are welcome.** If something did not work (no matrix-engine use, driver crashes, precision loss), document it; it saves everyone time.
5. **No personal paths or secrets** in code or configs. Use placeholders.
6. **License.** Contributions are under Apache 2.0, the same as the repo.

Open an issue first for large changes or new formats, so the work is not duplicated.