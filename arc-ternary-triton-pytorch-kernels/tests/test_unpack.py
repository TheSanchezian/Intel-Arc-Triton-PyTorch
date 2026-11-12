"""Standalone test for the PQ2_0 Triton unpack kernels. No model needed: random codes and scales.

Checks (torch.equal, i.e. bit-identical) the contiguous-range kernel (both variants) and the expert-id-gather kernel
against a plain torch reference, then times them. Timing uses a host readback because torch.xpu.synchronize() does
not wait for Triton kernels on this stack.

  python tests/test_unpack.py [--device xpu] [--experts 256] [--rows 512] [--groups 16]

Defaults match one Flint-40B expert matrix kind (256 experts, N=512 rows, K=2048 columns = 16 blocks of 128).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kernels import ternary_unpack as tu

SHIFTS = (0, 2, 4, 6)


def reference(codes, sc, shifts):
    """W[e,n,g*128+j*4+s] = (((codes[e,n,g,j] >> 2s) & 3) - 1) * sc[e,n,g], fp32."""
    v = ((codes.to(torch.int32).unsqueeze(-1) >> shifts) & 3).to(torch.float32) - 1   # [E,N,G,32,4]
    E, N, G = codes.shape[:3]
    return (v.reshape(E, N, G, 128) * sc.unsqueeze(-1).to(torch.float32)).reshape(E, N, G * 128)


def readback(t):
    if isinstance(t, (list, tuple)):
        t = t[-1]
    return float(t.flatten()[0])


def timeit(fn, reps=10):
    readback(fn()); readback(fn())                      # warm-up (compile) + settle
    t0 = time.perf_counter()
    for _ in range(reps):
        o = fn()
    readback(o)                                          # host readback = real completion
    return (time.perf_counter() - t0) / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--groups", type=int, default=16, help="experts per call for the range / id timing")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device(a.device)
    E, N, G = a.experts, a.rows, 16
    K = G * 128
    gen = torch.Generator().manual_seed(a.seed)
    codes = torch.randint(0, 256, (E, N, G, 32), generator=gen, dtype=torch.uint8)
    # valid ternary data uses codes 0..2; the bit math also holds for code 3, so test all four values
    sc = (torch.rand(E, N, G, generator=gen) * 0.1 + 0.001).to(torch.float16)
    codes, sc = codes.to(dev).contiguous(), sc.to(dev)
    shifts = torch.tensor(SHIFTS, dtype=torch.int32, device=dev)

    print(f"device {dev}, triton {tu.triton.__version__}, E={E} N={N} K={K}")
    ok = True

    # 1. contiguous range kernel, both variants, with and without a preallocated out= buffer
    for gsz in (1, 7, 16, 32):
        buf = torch.empty(gsz * N * K, device=dev)
        for variant in ("gather", "reshape"):
            eq_a = eq_o = True
            for e0 in range(0, E, gsz):
                e1 = min(e0 + gsz, E)
                ref = reference(codes[e0:e1], sc[e0:e1], shifts)
                eq_a &= torch.equal(tu.unpack_group_triton(codes, sc, e0, e1, variant=variant, BR=16), ref)
                eq_o &= torch.equal(tu.unpack_group_triton(codes, sc, e0, e1, out=buf, variant=variant, BR=16), ref)
            ok &= eq_a and eq_o
            print(f"  range  g={gsz:<2} {variant:<7}: equal(alloc)={eq_a} equal(out=)={eq_o}")

    # 2. expert-id gather kernel: unsorted ids, repeated-free and with repeats
    for gsz in (1, 7, 16, 32):
        eq = True
        for trial in range(4):
            ids = torch.randperm(E, generator=gen)[:gsz].to(dev)
            ref = reference(codes.index_select(0, ids), sc.index_select(0, ids), shifts)
            eq &= torch.equal(tu.unpack_ids_triton(codes, sc, ids), ref)
            eq &= torch.equal(tu.unpack_ids_triton(codes, sc, ids.to(torch.int32)), ref)
        ok &= eq
        print(f"  ids    g={gsz:<2}: equal={eq}")

    print("BITWISE:", "PASS" if ok else "FAIL")

    # 3. timing (whole layer = all experts, in groups of --groups), host readback
    g = a.groups
    nbytes = codes.numel() + sc.numel() * 2 + E * N * K * 4
    ids_all = [torch.arange(e0, min(e0 + g, E), device=dev) for e0 in range(0, E, g)]
    buf = torch.empty(g * N * K, device=dev)
    t_ref = timeit(lambda: [reference(codes[e0:e0 + g], sc[e0:e0 + g], shifts) for e0 in range(0, E, g)], a.reps)
    t_rng = timeit(lambda: [tu.unpack_group_triton(codes, sc, e0, min(e0 + g, E), out=buf, variant="reshape", BR=16)
                            for e0 in range(0, E, g)], a.reps)
    t_ids = timeit(lambda: [tu.unpack_ids_triton(codes, sc, ids) for ids in ids_all], a.reps)
    print(f"\ntiming, all {E} experts of one matrix kind, groups of {g}, host readback, {a.reps} reps")
    print(f"  torch reference        : {t_ref:8.2f} ms")
    print(f"  triton range (reshape) : {t_rng:8.2f} ms   {nbytes / (t_rng / 1000) / 1e9:6.0f} GB/s (codes+scales read, fp32 write)")
    print(f"  triton id-gather       : {t_ids:8.2f} ms   {nbytes / (t_ids / 1000) / 1e9:6.0f} GB/s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
