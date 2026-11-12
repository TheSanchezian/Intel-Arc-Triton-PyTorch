import torch, triton, triton.language as tl, time
@triton.jit
def add_k(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)
@triton.jit
def mm_k(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        av = tl.load(a + rm[:, None] * K + (k0 + rk)[None, :])
        bv = tl.load(b + rn[:, None] * K + (k0 + rk)[None, :])
        acc += tl.dot(av, tl.trans(bv))
    tl.store(c + rm[:, None] * N + rn[None, :], acc)
print("triton", triton.__version__, "device", torch.xpu.get_device_name(0), flush=True)
n = 1 << 20; x = torch.randn(n, device="xpu"); y = torch.randn(n, device="xpu"); o = torch.empty_like(x)
add_k[(n // 1024,)](x, y, o, n, B=1024); torch.xpu.synchronize()
print("add ok, maxerr", float((o - x - y).abs().max()), flush=True)
for dt in (torch.float32, torch.float16):
    a = torch.randn(256, 2048, device="xpu", dtype=dt); b = torch.randn(512, 2048, device="xpu", dtype=dt); c = torch.empty(256, 512, device="xpu")
    g = (256 // 64, 512 // 64)
    mm_k[g](a, b, c, 256, 512, 2048, BM=64, BN=64, BK=32); torch.xpu.synchronize()
    ref = a.float() @ b.float().t()
    t0 = time.perf_counter()
    for _ in range(50): mm_k[g](a, b, c, 256, 512, 2048, BM=64, BN=64, BK=32)
    torch.xpu.synchronize(); ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"dot {dt} ok, relerr {float((c - ref).norm() / ref.norm()):.2e}, {ms:.3f} ms", flush=True)