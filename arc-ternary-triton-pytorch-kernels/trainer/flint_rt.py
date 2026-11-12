"""Flint-40B (ternary MoE, GGUF PQ2_0 + Q4_1) on an Intel Arc A770 with a small custom OpenCL runtime.

No OpenVINO, no PyTorch: numpy + pyopencl. The weights are read from the original GGUF (not included), repacked for
the kernels (the integer codes are unchanged) and cached as .npy files for fast later loads.
Per token: fused q/k/v Q4_1 GEMV (RMSNorm computed on the fly) -> split-KV attention (learned sinks, 513-token
sliding-window ring buffer on 35 layers, global cache on 14 layers; the current token's k/v written in-kernel) ->
o GEMV into the residual -> MoE: f32 router (norm on the fly) -> top-6 (sigmoid, normalized x 1.24) + pre-FFN norm
and Hadamard-512 prep in the same launch -> 6 experts (2-bit ternary GEMV) | shared expert (Q4_1, SiLU*up fused)
-> LM head. Residual in f32. The next token is chosen on the GPU (argmax), so the host only queues work.
Usage (see README.md):
  python flint_rt.py speed   --gguf Flint.gguf --tokenizer tokenizer.json [--short]
  python flint_rt.py prompts --gguf Flint.gguf --tokenizer tokenizer.json
  python flint_rt.py check|profile --gguf ... --tokenizer ...
  python flint_rt.py ppl|long|prefill|compare --gguf ... --tokenizer ... --data wikitext-2-test.txt
License: Apache-2.0.
"""
import argparse, math, os, struct, sys, time
import numpy as np


CFG = {"tokenizer": None, "data": None, "cache": "flint_cache"}   # set from the command line
HID, NL, H, KVH, HD, WIN = 2048, 49, 32, 4, 128, 513

# ----------------------------------------------------------------------------------------------- GGUF
E, TOPK, EFF, BLK, BB = 256, 6, 512, 128, 34
_SCAL = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf(path):
    mm = np.memmap(path, np.uint8, "r")
    buf = memoryview(mm)
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, buf, pos)
        pos += struct.calcsize(fmt)
        return v[0]

    def string():
        nonlocal pos
        n = take("<Q")
        s = bytes(buf[pos:pos + n]).decode("utf-8", "replace")
        pos += n
        return s

    def value(t):
        nonlocal pos
        if t in _SCAL:
            return take(_SCAL[t])
        if t == 8:
            return string()
        if t == 9:
            at, n = take("<I"), take("<Q")
            if at == 8:
                first = [string() for _ in range(min(n, 4))]
                for _ in range(n - len(first)):
                    n_s = take("<Q")
                    pos += n_s
                return {"array_of": "string", "len": n, "first": first}
            if at in _SCAL:
                sz = struct.calcsize(_SCAL[at])
                vals = [struct.unpack_from(_SCAL[at], buf, pos + i * sz)[0] for i in range(min(n, 64))]
                pos += n * sz
                return vals if n <= 64 else {"array_of": at, "len": n, "first": vals[:4]}
            raise ValueError(f"nested array type {at}")
        raise ValueError(f"unknown GGUF value type {t}")

    if bytes(buf[0:4]) != b"GGUF":
        sys.exit("not a GGUF file")
    pos = 4
    ver, n_t, n_kv = take("<I"), take("<Q"), take("<Q")
    kv = {}
    for _ in range(n_kv):
        k = string()
        kv[k] = value(take("<I"))
    tensors = {}
    for _ in range(n_t):
        name = string()
        nd = take("<I")
        dims = [take("<Q") for _ in range(nd)]
        tensors[name] = {"dims": dims, "type": take("<I"), "off": take("<Q")}
    align = int(kv.get("general.alignment", 32))
    data0 = (pos + align - 1) // align * align
    order = sorted(tensors, key=lambda n: tensors[n]["off"])
    for i, n in enumerate(order):
        tensors[n]["off"] += data0
        end = (tensors[order[i + 1]]["off"] + data0) if i + 1 < len(order) else len(mm)
        tensors[n]["size"] = end - tensors[n]["off"]
    return mm, ver, kv, tensors


# ----------------------------------------------------------------------------------------------- weights
LUT_CLAMP = np.array([sum(min((b >> (2 * k)) & 3, 2) << (2 * k) for k in range(4)) for b in range(256)], np.uint8)
ZEROS_PER_BYTE = np.array([sum(((b >> (2 * k)) & 3) == 1 for k in range(4)) for b in range(256)], np.int64)
THREES_PER_BYTE = np.array([sum(((b >> (2 * k)) & 3) == 3 for k in range(4)) for b in range(256)], np.int64)
SHAPES = {"ffn_gate_exps": (EFF, HID), "ffn_up_exps": (EFF, HID), "ffn_down_exps": (HID, EFF)}


def raw_expert_tensor(mm, tensors, L, kind):
    N, K = SHAPES[kind]
    t = tensors[f"blk.{L}.{kind}.weight"]
    nbytes = E * N * (K // BLK) * BB
    if t["size"] < nbytes:
        sys.exit(f"blk.{L}.{kind}: {t['size']} bytes on disk < {nbytes} expected for PQ2_0 [{E},{N},{K}]")
    return np.asarray(mm[t["off"]:t["off"] + nbytes]).reshape(E, N, K // BLK, BB)


def pack_for_gpu(raw):
    """raw [E,N,G,34] -> interleaved codes [E, N/16, G, 8, 16] uint32 + scales [E, N/16, G, 16] f16."""
    En, N, G, _ = raw.shape
    d = np.ascontiguousarray(raw[..., :2]).view(np.float16).reshape(En, N, G)
    qs = LUT_CLAMP[raw[..., 2:]]                                 # [E,N,G,32] uint8, 3 -> 2
    w32 = np.ascontiguousarray(qs).view(np.uint32).reshape(En, N, G, 8)   # weight j of each uint at bits 2j
    wi = np.ascontiguousarray(w32.reshape(En, N // 16, 16, G, 8).transpose(0, 1, 3, 4, 2))
    si = np.ascontiguousarray(d.reshape(En, N // 16, 16, G).transpose(0, 1, 3, 2))
    return wi, si



# ----------------------------------------------------------------------------------------------- kernels
# expert (2-bit ternary, Hadamard-512) kernels, single-pass attention, Q4_1 GEMV
MOE_COMMON = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define SG16 __attribute__((intel_reqd_sub_group_size(16)))
#define INV512 0.04419417382415922f

/* 16 rows (lane = row) x 128 inputs from local memory xl (at offset xo) */
#define DOT128(acc, v, xl, xo)                                                  \
  {                                                                             \
    uint wv_[8] = {v.s0, v.s1, v.s2, v.s3, v.s4, v.s5, v.s6, v.s7};             \
    _Pragma("unroll") for (int u = 0; u < 8; ++u) {                             \
      float xv_ = (xl)[(xo) + u * 16 + lane];                                   \
      _Pragma("unroll") for (int j = 0; j < 16; ++j)                            \
        acc += (float)((int)((wv_[u] >> (2 * j)) & 3u) - 1)                     \
               * intel_sub_group_shuffle(xv_, j);                               \
    }                                                                           \
  }

/* in-place Walsh-Hadamard (natural order, unscaled) on NB blocks of 512 in local memory, NT threads */
#define FWHT_LOCAL(l, NB, tid, NT)                                              \
  for (uint h = 1; h < 512; h <<= 1) {                                          \
    for (uint q = (tid); q < (NB) * 256; q += (NT)) {                           \
      uint b_ = q / 256, i_ = q % 256;                                          \
      uint j_ = b_ * 512 + (i_ / h) * 2 * h + (i_ % h);                         \
      float a_ = (l)[j_], c_ = (l)[j_ + h];                                     \
      (l)[j_] = a_ + c_; (l)[j_ + h] = a_ - c_;                                 \
    }                                                                           \
    barrier(CLK_LOCAL_MEM_FENCE);                                               \
  }
"""
MOE_GATE_UP = r"""
__attribute__((reqd_work_group_size(16, KSG, 1))) SG16
__kernel void flint_gate_up(const __global INPUT0_TYPE* x, const __global INPUT1_TYPE* idx,
                            const __global uint* wg, const __global INPUT3_TYPE* sg,
                            const __global uint* wu, const __global INPUT5_TYPE* su,
                            __global OUTPUT0_TYPE* act) {
    int tot = 1;
    for (int i = 0; i < INPUT0_DIMS_SIZE; ++i) tot *= INPUT0_DIMS[i];
    const int M = tot / 2048;
    const uint G = 16;
    const uint lane = get_local_id(0), ks = get_local_id(1), gid = get_group_id(0);
    const uint tid = ks * 16 + lane, slot = gid / 32, nb = gid % 32;
    __local float xl[2048];
    __local float rg[KSG][16], ru[KSG][16];
    for (int m = 0; m < M; ++m) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = tid; i < 2048; i += 16 * KSG) xl[i] = (float)x[m * 2048 + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        FWHT_LOCAL(xl, 4, tid, 16 * KSG);
        const uint rb = (uint)idx[m * 6 + slot] * 32 + nb;
        const uint g0 = (G * ks) / KSG, g1 = (G * (ks + 1)) / KSG;
        float ag = 0.0f, au = 0.0f;
        for (uint g = g0; g < g1; ++g) {
            const size_t base = ((size_t)(rb * G + g) * 8) * 16;
            uint8 vg = intel_sub_group_block_read8(wg + base);
            uint8 vu = intel_sub_group_block_read8(wu + base);
            float tg = 0.0f, tu = 0.0f;
            DOT128(tg, vg, xl, g * 128);
            DOT128(tu, vu, xl, g * 128);
            ag += tg * (float)sg[(rb * G + g) * 16 + lane];
            au += tu * (float)su[(rb * G + g) * 16 + lane];
        }
        rg[ks][lane] = ag; ru[ks][lane] = au;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (ks == 0) {
            float gs = 0.0f, us = 0.0f;
            for (int k = 0; k < KSG; ++k) { gs += rg[k][lane]; us += ru[k][lane]; }
            gs *= INV512; us *= INV512;
            act[(m * 6 + slot) * 512 + nb * 16 + lane] = (OUTPUT0_TYPE)(gs / (1.0f + exp(-gs)) * us);
        }
    }
}
"""
MOE_DOWN = r"""
__attribute__((reqd_work_group_size(16, KSD, 1))) SG16
__kernel void flint_down(const __global INPUT0_TYPE* a, const __global INPUT1_TYPE* idx,
                         const __global INPUT2_TYPE* rw, const __global uint* wd,
                         const __global INPUT4_TYPE* sd, __global OUTPUT0_TYPE* y) {
    int tot = 1;
    for (int i = 0; i < INPUT0_DIMS_SIZE; ++i) tot *= INPUT0_DIMS[i];
    const int M = tot / 3072;
    const uint G = 4, T = 24;
    const uint lane = get_local_id(0), ks = get_local_id(1), nb = get_group_id(0);
    const uint tid = ks * 16 + lane;
    __local float al[3072];
    __local float r[KSD][16];
    for (int m = 0; m < M; ++m) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = tid; i < 3072; i += 16 * KSD) al[i] = (float)a[m * 3072 + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        FWHT_LOCAL(al, 6, tid, 16 * KSD);
        const uint t0 = (T * ks) / KSD, t1 = (T * (ks + 1)) / KSD;
        float acc = 0.0f;
        for (uint t = t0; t < t1; ++t) {
            const uint slot = t / G, g = t % G;
            const uint rb = (uint)idx[m * 6 + slot] * 128 + nb;
            const size_t base = ((size_t)(rb * G + g) * 8) * 16;
            uint8 v = intel_sub_group_block_read8(wd + base);
            float s = 0.0f;
            DOT128(s, v, al, slot * 512 + g * 128);
            acc += s * (float)sd[(rb * G + g) * 16 + lane] * (float)rw[m * 6 + slot];
        }
        r[ks][lane] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (ks == 0) {
            float t = 0.0f;
            for (int k = 0; k < KSD; ++k) t += r[k][lane];
            y[m * 2048 + nb * 16 + lane] = (OUTPUT0_TYPE)(t * INV512);
        }
    }
}
"""
ATT_COMMON = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define HD 128
"""
ATT_SINGLE = r"""
#define WG 256
/* one work-group per query head (32); 8 work-groups share a KV head (its K/V come from L3 after the first) */
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void flint_attn(const __global INPUT0_TYPE* qkv, const __global INPUT1_TYPE* posp,
                         const __global INPUT2_TYPE* pk, const __global INPUT3_TYPE* pv,
                         const __global INPUT4_TYPE* qnw, const __global INPUT5_TYPE* knw,
                         const __global INPUT6_TYPE* invf, const __global INPUT7_TYPE* sinks,
                         __global OUTPUT0_TYPE* out) {
    const int t = get_local_id(0), h = get_group_id(0), g = h / 8;
    const long pos = (long)posp[0];
    __local float qs[HD], kn[HD], vn[HD], red[WG], acc2[2 * HD];
    __local float sc[NSLOT + 1];
    float qv = 0.0f, kv = 0.0f;
    if (t < HD) {
        qs[t] = (float)qkv[h * HD + t];
        kn[t] = (float)qkv[4096 + g * HD + t];
        vn[t] = (float)qkv[4608 + g * HD + t];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    red[t] = (t < HD) ? qs[t] * qs[t] : kn[t - HD] * kn[t - HD];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 64; s > 0; s >>= 1) {
        if ((t & 127) < s) red[t] += red[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float rq = rsqrt(red[0] / 128.0f + 1e-6f), rk = rsqrt(red[HD] / 128.0f + 1e-6f);
    if (t < HD) { qv = qs[t] * rq * (float)qnw[t]; kv = kn[t] * rk * (float)knw[t]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < HD) { qs[t] = qv; kn[t] = kv; }
    barrier(CLK_LOCAL_MEM_FENCE);
#if ROPE
    if (t < HD) {
        const int p = t ^ 64;
        const float ang = (float)pos * (float)invf[t & 63], c = cos(ang), s = sin(ang);
        qv = qs[t] * c + ((t < 64) ? -qs[p] : qs[p]) * s;
        kv = kn[t] * c + ((t < 64) ? -kn[p] : kn[p]) * s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < HD) { qs[t] = qv; kn[t] = kv; }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if RING
    long start = pos - (WIN - 1);
    if (start < 0) start = 0;
#else
    const long start = 0;
#endif
    const int nv = (int)(pos - start) + 1;             /* keys start..pos, all read from the updated state */
    float lmax = -INFINITY;
    for (int s = t; s < nv; s += WG) {
        float a = 0.0f;
        if (s < nv) {
#if RING
            const int slot = (int)((start + s) % WIN);
#else
            const int slot = s;
#endif
            const __global INPUT2_TYPE* kr = pk + ((size_t)g * NSLOT + slot) * HD;
            for (int d = 0; d < HD; d += 8) {
                const float8 k8 = convert_float8(vload8(0, kr + d));
                a += k8.s0 * qs[d] + k8.s1 * qs[d + 1] + k8.s2 * qs[d + 2] + k8.s3 * qs[d + 3]
                   + k8.s4 * qs[d + 4] + k8.s5 * qs[d + 5] + k8.s6 * qs[d + 6] + k8.s7 * qs[d + 7];
            }
        } else {
            for (int d = 0; d < HD; ++d) a += qs[d] * kn[d];
        }
        a *= 0.08838834764831845f;
        sc[s] = a;
        lmax = fmax(lmax, a);
    }
    red[t] = lmax;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmax(red[t], red[t + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float sink = (float)sinks[h];
    const float m = fmax(red[0], sink);
    barrier(CLK_LOCAL_MEM_FENCE);
    float lsum = 0.0f;
    for (int s = t; s < nv; s += WG) { const float e = exp(sc[s] - m); sc[s] = e; lsum += e; }
    red[t] = lsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float L = red[0] + exp(sink - m);
    const int d = t & 127, hf = t >> 7;
    float a = 0.0f;
    for (int s = hf; s < nv; s += 2) {
        float vv;
        if (s < nv) {
#if RING
            const int slot = (int)((start + s) % WIN);
#else
            const int slot = s;
#endif
            vv = (float)pv[((size_t)g * NSLOT + slot) * HD + d];
        } else {
            vv = vn[d];
        }
        a += sc[s] * vv;
    }
    acc2[hf * HD + d] = a;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < HD) out[h * HD + t] = (OUTPUT0_TYPE)((acc2[t] + acc2[HD + t]) / L);
}
"""
GEMV_SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(16, KS, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_q41(const __global uint* restrict w, const __global half* restrict d,
                       const __global half* restrict m, const __global half* restrict x,
                       __global half* restrict y, const int G) {
    const int lane = get_local_id(0), ks = get_local_id(1), nb = get_group_id(0);
    const int g0 = (G * ks) / KS, g1 = (G * (ks + 1)) / KS;
    float acc = 0.0f;
    for (int g = g0; g < g1; ++g) {
        const size_t blk = (size_t)nb * G + g;
        const uint4 v = intel_sub_group_block_read4(w + blk * 64);
        const float x0 = (float)x[g * 32 + lane], x1 = (float)x[g * 32 + 16 + lane];
        const float sx = sub_group_reduce_add(x0 + x1);
        const uint u[4] = {v.s0, v.s1, v.s2, v.s3};
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int j = k * 4 + b;                                   /* weight j and j+16 */
                s += (float)((u[k] >> (8 * b)) & 15u) * intel_sub_group_shuffle(x0, j)
                   + (float)((u[k] >> (8 * b + 4)) & 15u) * intel_sub_group_shuffle(x1, j);
            }
        }
        acc += (float)d[blk * 16 + lane] * s + (float)m[blk * 16 + lane] * sx;
    }
    __local float red[KS][16];
    red[ks][lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float t = 0.0f;
        for (int k = 0; k < KS; ++k) t += red[k][lane];
        y[nb * 16 + lane] = (half)t;
    }
}

/* f32 router: 256 rows x K; work-group = 16 rows x KS subgroups, lane = row, coalesced over K via block reads */
__attribute__((reqd_work_group_size(16, KS, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_f32(const __global float* restrict w, const __global half* restrict x,
                       __global half* restrict y, const int K) {
    const int lane = get_local_id(0), ks = get_local_id(1), nb = get_group_id(0);
    const int c0 = (K / 16 * ks) / KS * 16, c1 = (K / 16 * (ks + 1)) / KS * 16;
    float acc = 0.0f;
    for (int r = 0; r < 16; ++r) {                                         /* 16 rows of this block */
        const __global float* wr = w + (size_t)(nb * 16 + r) * K;
        float s = 0.0f;
        for (int c = c0 + lane; c < c1; c += 16) s += wr[c] * (float)x[c];
        s = sub_group_reduce_add(s);
        if (lane == r) acc = s;
    }
    __local float red[KS][16];
    red[ks][lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float t = 0.0f;
        for (int k = 0; k < KS; ++k) t += red[k][lane];
        y[nb * 16 + lane] = (half)t;
    }
}
"""
KSG, KSD = 16, 8


SLIDING = [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
           0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]   # = RoPE pattern (convert_flint_k5.py)
PROMPTS = [("Fact", "Question: What is the capital of France?\nAnswer:"),
           ("Math", "Question: What is 25 + 37?\nAnswer:"),
           ("Code", "Write a one-line Python function to check if a number is even:\n```python\n")]


def _kernel_sources():
    return ({"CL_COMMON": MOE_COMMON, "CL_GATE_UP": MOE_GATE_UP, "CL_DOWN": MOE_DOWN, "KSG": KSG, "KSD": KSD},
            {"CL_COMMON": ATT_COMMON, "CL_ATTN": ATT_SINGLE}, {"SRC": GEMV_SRC})


DOWN_PF = r"""
/* expert down with the weight blocks + scales PREFETCHED before the Hadamard (addresses depend only on the
   expert ids), so the memory reads overlap the local-memory butterfly. One token (M = 1). */
#define MAXT ((24 + KSD - 1) / KSD)
__attribute__((reqd_work_group_size(16, KSD, 1))) SG16
__kernel void flint_down_pf(const __global float* a, const __global int* idx, const __global float* rw,
                            const __global uint* wd, const __global half* sd, __global float* y) {
    const uint G = 4, T = 24;
    const uint lane = get_local_id(0), ks = get_local_id(1), nb = get_group_id(0);
    const uint tid = ks * 16 + lane;
    __local float al[3072];
    __local float r[KSD][16];
    const uint t0 = (T * ks) / KSD, t1 = (T * (ks + 1)) / KSD;
    uint8 v[MAXT];
    float sc[MAXT];
    #pragma unroll
    for (uint j = 0; j < MAXT; ++j) {
        if (t0 + j < t1) {
            const uint t = t0 + j, slot = t / G, g = t % G;
            const uint rb = (uint)idx[slot] * 128 + nb;
            v[j] = intel_sub_group_block_read8(wd + ((size_t)(rb * G + g) * 8) * 16);
            sc[j] = (float)sd[(rb * G + g) * 16 + lane] * rw[slot];
        }
    }
    for (uint i = tid; i < 3072; i += 16 * KSD) al[i] = a[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    FWHT_LOCAL(al, 6, tid, 16 * KSD);
    float acc = 0.0f;
    #pragma unroll
    for (uint j = 0; j < MAXT; ++j) {
        if (t0 + j < t1) {
            const uint t = t0 + j, slot = t / G, g = t % G;
            float s = 0.0f;
            DOT128(s, v[j], al, slot * 512 + g * 128);
            acc += s * sc[j];
        }
    }
    r[ks][lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float t = 0.0f;
        for (int k = 0; k < KSD; ++k) t += r[k][lane];
        y[nb * 16 + lane] = t * INV512;
    }
}
"""

RT_SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void embed(const __global half* E, const __global int* tp, __global float* x) {
    const int i = get_global_id(0);
    x[i] = (float)E[(size_t)tp[0] * 2048 + i];
}
/* x (+= add if do_add) ; out = RMSNorm(x) * w  (half and float copies) */
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void rmsnorm(__global float* x, const __global float* add, const int do_add, const __global float* w,
             __global half* outh, __global float* outf) {
    __local float r[256];
    const int t = get_local_id(0);
    float s = 0.0f;
    for (int i = t; i < 2048; i += 256) {
        float v = x[i];
        if (do_add) { v += add[i]; x[i] = v; }
        s += v * v;
    }
    r[t] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 128; k > 0; k >>= 1) { if (t < k) r[t] += r[t + k]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float inv = rsqrt(r[0] / 2048.0f + 1e-6f);
    for (int i = t; i < 2048; i += 256) { const float o = x[i] * inv * w[i]; outh[i] = (half)o; outf[i] = o; }
}
/* k = RoPE(RMSNorm(k_new)) and v_new written into the KV caches at the token's slot; one subgroup per kv head */
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void kv_store(const __global half* qkv, const __global long* posp, const __global float* knw,
                       const __global float* invf, __global half* kc, __global half* vc,
                       const int nslot, const int ring, const int rope) {
    const int lane = get_sub_group_local_id(), head = get_global_id(0) / 16;
    const long pos = posp[0];
    const int slot = ring ? (int)(pos % nslot) : (int)pos;
    float v[8];
    for (int i = 0; i < 8; ++i) v[i] = (float)qkv[4096 + head * 128 + lane * 8 + i];
    float ss = 0.0f;
    for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
    ss = sub_group_reduce_add(ss);
    const float r = rsqrt(ss / 128.0f + 1e-6f);
    for (int i = 0; i < 8; ++i) v[i] = v[i] * r * knw[lane * 8 + i];
    if (rope) {
        for (int i = 0; i < 8; ++i) {
            const float partner = intel_sub_group_shuffle_xor(v[i], 8);
            const int d = lane * 8 + i;
            const float ang = (float)pos * invf[d & 63];
            v[i] = v[i] * cos(ang) + ((lane < 8) ? -partner : partner) * sin(ang);
        }
    }
    const size_t base = ((size_t)head * nslot + slot) * 128 + lane * 8;
    for (int i = 0; i < 8; ++i) {
        kc[base + i] = (half)v[i];
        vc[base + i] = qkv[4608 + head * 128 + lane * 8 + i];
    }
}
/* sigmoid -> top-6 of 256 -> normalize -> x 1.24 (one work-group) */
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void topk6(const __global float* logits, __global int* idx, __global float* rw) {
    __local float p[256], bv[256], selv[6];
    __local int bi[256], seli[6];
    const int t = get_local_id(0);
    p[t] = 1.0f / (1.0f + exp(-logits[t]));
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int r = 0; r < 6; ++r) {
        bv[t] = p[t]; bi[t] = t;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 128; k > 0; k >>= 1) {
            if (t < k) {
                const float a = bv[t], b = bv[t + k];
                if (b > a || (b == a && bi[t + k] < bi[t])) { bv[t] = b; bi[t] = bi[t + k]; }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (t == 0) { selv[r] = bv[0]; seli[r] = bi[0]; p[bi[0]] = -1.0f; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) {
        float s = 0.0f;
        for (int r = 0; r < 6; ++r) s += selv[r];
        for (int r = 0; r < 6; ++r) { idx[r] = seli[r]; rw[r] = selv[r] / (s + 1e-20f) * 1.2400000095367432f; }
    }
}

/* ---------------- split-KV attention, router, top-6 ---------------- */
/* router: one work-group (64 threads) per expert row, coalesced float4 reads, tree reduction */
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void router2(const __global float4* w, const __global float4* x, __global float* y) {
    __local float r[64];
    const int row = get_group_id(0), t = get_local_id(0);
    const __global float4* wr = w + (size_t)row * 512;
    float s = 0.0f;
    for (int i = t; i < 512; i += 64) s += dot(wr[i], x[i]);
    r[t] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 32; k > 0; k >>= 1) { if (t < k) r[t] += r[t + k]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (t == 0) y[row] = r[0];
}
/* top-6 with one subgroup: lane owns experts lane*16 .. lane*16+15; no barriers */
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void topk6b(const __global float* logits, __global int* idx, __global float* rw) {
    const int lane = get_sub_group_local_id();
    float p[16];
    for (int j = 0; j < 16; ++j) p[j] = 1.0f / (1.0f + exp(-logits[lane * 16 + j]));
    float sv[6]; int si[6]; float s = 0.0f;
    for (int r = 0; r < 6; ++r) {
        float bv = -2.0f; int bi = 0;
        for (int j = 0; j < 16; ++j) if (p[j] > bv) { bv = p[j]; bi = lane * 16 + j; }
        const float gm = sub_group_reduce_max(bv);
        const int gi = sub_group_reduce_min(bv == gm ? bi : (1 << 30));
        sv[r] = gm; si[r] = gi; s += gm;
        for (int j = 0; j < 16; ++j) if (lane * 16 + j == gi) p[j] = -1.0f;
    }
    if (lane == 0)
        for (int r = 0; r < 6; ++r) { idx[r] = si[r]; rw[r] = sv[r] / (s + 1e-20f) * 1.2400000095367432f; }
}
/* split-KV attention, part 1: work-group (64 threads) = (query head h, chunk of 64 keys); writes m, l, acc[128] */
#define CH 64
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void attn_part(const __global half* qkv, const __global long* posp, const __global half* kc,
               const __global half* vc, const __global float* qnw, const __global float* invf,
               __global float* part, const int nslot, const int win, const int rope, const int nsplit) {
    __local float qs[128], red[64], pk[64];
    const int t = get_local_id(0), h = get_group_id(0), sp = get_group_id(1), g = h / 8;
    const long pos = posp[0];
    long start = win ? pos - (win - 1) : 0;
    if (start < 0) start = 0;
    const int nv = (int)(pos - start) + 1, s0 = sp * CH;
    __global float* P = part + ((size_t)h * nsplit + sp) * 130;
    if (s0 >= nv) {                                          /* whole work-group: empty chunk */
        if (t == 0) { P[0] = -INFINITY; P[1] = 0.0f; }
        P[2 + t] = 0.0f; P[66 + t] = 0.0f;
        return;
    }
    float a = (float)qkv[h * 128 + t], b = (float)qkv[h * 128 + 64 + t];
    red[t] = a * a + b * b;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 32; k > 0; k >>= 1) { if (t < k) red[t] += red[t + k]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float rq = rsqrt(red[0] / 128.0f + 1e-6f);
    a *= rq * qnw[t]; b *= rq * qnw[64 + t];
    if (rope) {
        const float ang = (float)pos * invf[t], c = cos(ang), sn = sin(ang);
        const float a2 = a * c - b * sn, b2 = b * c + a * sn;          /* rot_half: [-x2, x1] */
        a = a2; b = b2;
    }
    qs[t] = a; qs[64 + t] = b;
    barrier(CLK_LOCAL_MEM_FENCE);
    const int s = s0 + t;
    float sc = -INFINITY;
    if (s < nv) {
        const int slot = win ? (int)((start + s) % win) : s;
        const __global half* kr = kc + ((size_t)g * nslot + slot) * 128;
        float acc = 0.0f;
        for (int d = 0; d < 128; d += 8) {
            const float8 k8 = convert_float8(vload8(0, kr + d));
            acc += k8.s0 * qs[d] + k8.s1 * qs[d + 1] + k8.s2 * qs[d + 2] + k8.s3 * qs[d + 3]
                 + k8.s4 * qs[d + 4] + k8.s5 * qs[d + 5] + k8.s6 * qs[d + 6] + k8.s7 * qs[d + 7];
        }
        sc = acc * 0.08838834764831845f;
    }
    red[t] = sc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 32; k > 0; k >>= 1) { if (t < k) red[t] = fmax(red[t], red[t + k]); barrier(CLK_LOCAL_MEM_FENCE); }
    const float m = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    const float e = (s < nv) ? exp(sc - m) : 0.0f;
    pk[t] = e; red[t] = e;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 32; k > 0; k >>= 1) { if (t < k) red[t] += red[t + k]; barrier(CLK_LOCAL_MEM_FENCE); }
    const int n = min(CH, nv - s0);
    float a0 = 0.0f, a1 = 0.0f;
    for (int kk = 0; kk < n; ++kk) {
        const int slot = win ? (int)((start + s0 + kk) % win) : s0 + kk;
        const __global half* vr = vc + ((size_t)g * nslot + slot) * 128;
        a0 += pk[kk] * (float)vr[t];
        a1 += pk[kk] * (float)vr[64 + t];
    }
    if (t == 0) { P[0] = m; P[1] = red[0]; }
    P[2 + t] = a0; P[66 + t] = a1;
}
/* split-KV attention, part 2: per query head, combine the chunks + the learned sink */
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void attn_combine(const __global float* part, const __global float* sinks, const __global long* posp,
                  __global half* out, const int win, const int nsplit) {
    const int d = get_local_id(0), h = get_group_id(0);
    const long pos = posp[0];
    long start = win ? pos - (win - 1) : 0;
    if (start < 0) start = 0;
    const int nv = (int)(pos - start) + 1, ns = (nv + CH - 1) / CH;
    const __global float* P = part + (size_t)h * nsplit * 130;
    const float sink = sinks[h];
    float M = sink;
    for (int i = 0; i < ns; ++i) M = fmax(M, P[i * 130]);
    float L = exp(sink - M), acc = 0.0f;
    for (int i = 0; i < ns; ++i) {
        const float w = exp(P[i * 130] - M);
        L += P[i * 130 + 1] * w;
        acc += P[i * 130 + 2 + d] * w;
    }
    out[h * 128 + d] = (half)(acc / L);
}

/* ---------------- GPU-side greedy ---------------- */
/* greedy next token ON the GPU: argmax over the logits -> next input token, position += 1, history[count++] */
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void argmax_next(const __global half* logits, const int V, __global int* tp, __global long* posb,
                 __global int* hist, __global int* cnt) {
    __local float bv[256];
    __local int bi[256];
    const int t = get_local_id(0);
    float best = -INFINITY; int idx = 0;
    for (int i = t; i < V; i += 256) { const float v = (float)logits[i]; if (v > best) { best = v; idx = i; } }
    bv[t] = best; bi[t] = idx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 128; k > 0; k >>= 1) {
        if (t < k) {
            const float a = bv[t], b = bv[t + k];
            if (b > a || (b == a && bi[t + k] < bi[t])) { bv[t] = b; bi[t] = bi[t + k]; }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) { tp[0] = bi[0]; posb[0] = posb[0] + 1; hist[cnt[0]] = bi[0]; cnt[0] = cnt[0] + 1; }
}

/* ---------------- "last work-group finishes" fusions ---------------- */
/* returns 1 in exactly one (the last) work-group, after all work-groups' global writes are visible; resets the
   counter. All threads of every work-group must call it (uniformly). */
#if defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ >= 200
#define FENCE_REL() atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device)
#define FENCE_ACQ() atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device)
#else
#define FENCE_REL() mem_fence(CLK_GLOBAL_MEM_FENCE)
#define FENCE_ACQ() mem_fence(CLK_GLOBAL_MEM_FENCE)
#endif
/* OpenCL 1.2 mem_fence only orders within a work-item; it does NOT make writes visible to other work-groups
   (K17 compare: logits completely wrong). Device-scope release (writers) / acquire (last group) fixes that. */
static inline int last_wg(__global int* counter, const int total, const int tid, __local int* flag) {
    FENCE_REL();
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if (tid == 0) {
        const int prev = atomic_inc(counter);
        flag[0] = (prev == total - 1);
        if (prev == total - 1) counter[0] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    FENCE_ACQ();
    return flag[0];
}
/* RMSNorm epilogue run by one work-group of NT threads: x (+= add) ; out = norm(x) * w */
static inline void norm_epilogue(__global float* x, volatile __global float* tmp, const __global float* add,
                          const int do_add, const __global float* w, __global half* outh, __global float* outf,
                          const int tid, const int NT, __local float* r) {
    /* the GEMV work-groups wrote their rows to `tmp` with plain stores (like the attention partials);
       only this last work-group touches the residual: x += tmp (+ add), then RMSNorm */
    float s = 0.0f;
    for (int i = tid; i < 2048; i += NT) {
        float v = x[i] + tmp[i];
        if (do_add) v += add[i];
        x[i] = v;
        s += v * v;
    }
    r[tid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = NT / 2; k > 0; k >>= 1) { if (tid < k) r[tid] += r[tid + k]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float inv = rsqrt(r[0] / 2048.0f + 1e-6f);
    for (int i = tid; i < 2048; i += NT) { const float o = x[i] * inv * w[i]; outh[i] = (half)o; outf[i] = o; }
}
/* router (one work-group per row) + top-6 done by the last work-group */
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void router_topk(const __global float4* w, const __global float4* x, __global float* y, __global int* counter,
                 __global int* idx, __global float* rw) {
    __local float r[64], p[256], bv[64], selv[6];
    __local int bi[64], seli[6], flag[1];
    const int row = get_group_id(0), t = get_local_id(0);
    const __global float4* wr = w + (size_t)row * 512;
    float s = 0.0f;
    for (int i = t; i < 512; i += 64) s += dot(wr[i], x[i]);
    r[t] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 32; k > 0; k >>= 1) { if (t < k) r[t] += r[t + k]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (t == 0) y[row] = r[0];
    if (!last_wg(counter, 256, t, flag)) return;
    volatile __global float* vy = y;
    for (int i = t; i < 256; i += 64) p[i] = 1.0f / (1.0f + exp(-vy[i]));
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int rr = 0; rr < 6; ++rr) {
        float b = -2.0f; int ib = 0;
        for (int i = t; i < 256; i += 64) if (p[i] > b) { b = p[i]; ib = i; }
        bv[t] = b; bi[t] = ib;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 32; k > 0; k >>= 1) {
            if (t < k) {
                const float a0 = bv[t], b0 = bv[t + k];
                if (b0 > a0 || (b0 == a0 && bi[t + k] < bi[t])) { bv[t] = b0; bi[t] = bi[t + k]; }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (t == 0) { selv[rr] = bv[0]; seli[rr] = bi[0]; p[bi[0]] = -1.0f; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) {
        float sum = 0.0f;
        for (int rr = 0; rr < 6; ++rr) sum += selv[rr];
        for (int rr = 0; rr < 6; ++rr) { idx[rr] = seli[rr]; rw[rr] = selv[rr] / (sum + 1e-20f) * 1.2400000095367432f; }
    }
}
/* split-KV attention with the current token's k/v computed in-kernel (and written to the cache by one work-group
   per kv head), and the combine (+ sink) done by the last work-group of each query head */
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void attn_fused(const __global half* qkv, const __global long* posp, __global half* kc, __global half* vc,
                const __global float* qnw, const __global float* knw, const __global float* invf,
                const __global float* sinks, __global float* part, __global int* counters, __global half* out,
                const int nslot, const int win, const int rope, const int nsplit) {
    __local float qs[128], kcur[128], vcur[128], red[64], pk[64];
    __local int flag[1];
    const int t = get_local_id(0), h = get_group_id(0), sp = get_group_id(1), g = h / 8;
    const long pos = posp[0];
    long start = win ? pos - (win - 1) : 0;
    if (start < 0) start = 0;
    const int nv = (int)(pos - start) + 1, s0 = sp * CH;
    __global float* P = part + ((size_t)h * nsplit + sp) * 130;
    if (s0 >= nv) {
        if (t == 0) { P[0] = -INFINITY; P[1] = 0.0f; }
        P[2 + t] = 0.0f; P[66 + t] = 0.0f;
    } else {
        const int is_cur = (nv - 1 < s0 + CH);                 /* this chunk holds the current token */
        float a = (float)qkv[h * 128 + t], b = (float)qkv[h * 128 + 64 + t];
        red[t] = a * a + b * b;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 32; k > 0; k >>= 1) { if (t < k) red[t] += red[t + k]; barrier(CLK_LOCAL_MEM_FENCE); }
        const float rq = rsqrt(red[0] / 128.0f + 1e-6f);
        barrier(CLK_LOCAL_MEM_FENCE);
        a *= rq * qnw[t]; b *= rq * qnw[64 + t];
        float ang = 0.0f, c = 1.0f, sn = 0.0f;
        if (rope) { ang = (float)pos * invf[t]; c = cos(ang); sn = sin(ang); }
        qs[t] = a * c - b * sn; qs[64 + t] = b * c + a * sn;
        if (is_cur) {
            float ka = (float)qkv[4096 + g * 128 + t], kb = (float)qkv[4096 + g * 128 + 64 + t];
            red[t] = ka * ka + kb * kb;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 32; k > 0; k >>= 1) { if (t < k) red[t] += red[t + k]; barrier(CLK_LOCAL_MEM_FENCE); }
            const float rk = rsqrt(red[0] / 128.0f + 1e-6f);
            ka *= rk * knw[t]; kb *= rk * knw[64 + t];
            kcur[t] = ka * c - kb * sn; kcur[64 + t] = kb * c + ka * sn;
            vcur[t] = (float)qkv[4608 + g * 128 + t]; vcur[64 + t] = (float)qkv[4608 + g * 128 + 64 + t];
            if ((h & 7) == 0) {                                 /* one work-group per kv head writes the cache */
                const int slot = win ? (int)(pos % win) : (int)pos;
                const size_t base = ((size_t)g * nslot + slot) * 128;
                kc[base + t] = (half)kcur[t]; kc[base + 64 + t] = (half)kcur[64 + t];
                vc[base + t] = (half)vcur[t]; vc[base + 64 + t] = (half)vcur[64 + t];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int s = s0 + t;
        float sc = -INFINITY;
        if (s < nv - 1) {
            const int slot = win ? (int)((start + s) % win) : s;
            const __global half* kr = kc + ((size_t)g * nslot + slot) * 128;
            float acc = 0.0f;
            for (int d = 0; d < 128; d += 8) {
                const float8 k8 = convert_float8(vload8(0, kr + d));
                acc += k8.s0 * qs[d] + k8.s1 * qs[d + 1] + k8.s2 * qs[d + 2] + k8.s3 * qs[d + 3]
                     + k8.s4 * qs[d + 4] + k8.s5 * qs[d + 5] + k8.s6 * qs[d + 6] + k8.s7 * qs[d + 7];
            }
            sc = acc * 0.08838834764831845f;
        } else if (s == nv - 1) {
            float acc = 0.0f;
            for (int d = 0; d < 128; ++d) acc += qs[d] * kcur[d];
            sc = acc * 0.08838834764831845f;
        }
        red[t] = sc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 32; k > 0; k >>= 1) { if (t < k) red[t] = fmax(red[t], red[t + k]); barrier(CLK_LOCAL_MEM_FENCE); }
        const float m = red[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        const float e = (s < nv) ? exp(sc - m) : 0.0f;
        pk[t] = e; red[t] = e;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 32; k > 0; k >>= 1) { if (t < k) red[t] += red[t + k]; barrier(CLK_LOCAL_MEM_FENCE); }
        const int n = min(CH, nv - s0);
        float a0 = 0.0f, a1 = 0.0f;
        for (int kk = 0; kk < n; ++kk) {
            const int si = s0 + kk;
            if (si == nv - 1) { a0 += pk[kk] * vcur[t]; a1 += pk[kk] * vcur[64 + t]; }
            else {
                const int slot = win ? (int)((start + si) % win) : si;
                const __global half* vr = vc + ((size_t)g * nslot + slot) * 128;
                a0 += pk[kk] * (float)vr[t];
                a1 += pk[kk] * (float)vr[64 + t];
            }
        }
        if (t == 0) { P[0] = m; P[1] = red[0]; }
        P[2 + t] = a0; P[66 + t] = a1;
    }
    if (!last_wg(counters + h, nsplit, t, flag)) return;
    volatile __global float* VP = part + (size_t)h * nsplit * 130;
    const int ns = (nv + CH - 1) / CH;
    const float sink = sinks[h];
    float M = sink;
    for (int i = 0; i < ns; ++i) M = fmax(M, VP[i * 130]);
    float L = exp(sink - M), o0 = 0.0f, o1 = 0.0f;
    for (int i = 0; i < ns; ++i) {
        const float wgt = exp(VP[i * 130] - M);
        L += VP[i * 130 + 1] * wgt;
        o0 += VP[i * 130 + 2 + t] * wgt;
        o1 += VP[i * 130 + 66 + t] * wgt;
    }
    out[h * 128 + t] = (half)(o0 / L);
    out[h * 128 + 64 + t] = (half)(o1 / L);
}

/* ---------------- fusions without any cross-work-group sync ---------------- */
/* work-group sum / max via subgroup reductions + one combine step (2 barriers instead of a 6-step tree) */
static inline float wg_sum(float v, __local float* tmp) {
    v = sub_group_reduce_add(v);
    if (get_sub_group_local_id() == 0) tmp[get_sub_group_id()] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    float t = 0.0f;
    for (uint i = 0; i < get_num_sub_groups(); ++i) t += tmp[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    return t;
}
static inline float wg_max(float v, __local float* tmp) {
    v = sub_group_reduce_max(v);
    if (get_sub_group_local_id() == 0) tmp[get_sub_group_id()] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    float t = -INFINITY;
    for (uint i = 0; i < get_num_sub_groups(); ++i) t = fmax(t, tmp[i]);
    barrier(CLK_LOCAL_MEM_FENCE);
    return t;
}

/* split-KV attention part: q RMSNorm+RoPE in every work-group; the chunk holding the current token computes k/v
   in-kernel and (one work-group per kv head) writes them to the cache slot nobody else reads; combine is separate */
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void attn_part_kv(const __global half* qkv, const __global long* posp, __global half* kc, __global half* vc,
                const __global float* qnw, const __global float* knw, const __global float* invf,
                __global float* part, const int nslot, const int win, const int rope, const int nsplit) {
    __local float qs[128], kcur[128], vcur[128], red[64], pk[64];
    const int t = get_local_id(0), h = get_group_id(0), sp = get_group_id(1), g = h / 8;
    const long pos = posp[0];
    long start = win ? pos - (win - 1) : 0;
    if (start < 0) start = 0;
    const int nv = (int)(pos - start) + 1, s0 = sp * CH;
    __global float* P = part + ((size_t)h * nsplit + sp) * 130;
    if (s0 >= nv) {
        if (t == 0) { P[0] = -INFINITY; P[1] = 0.0f; }
        P[2 + t] = 0.0f; P[66 + t] = 0.0f;
    } else {
        const int is_cur = (nv - 1 < s0 + CH);                 /* this chunk holds the current token */
        float a = (float)qkv[h * 128 + t], b = (float)qkv[h * 128 + 64 + t];
        const float rq = rsqrt(wg_sum(a * a + b * b, red) / 128.0f + 1e-6f);      /* subgroup reductions */
        a *= rq * qnw[t]; b *= rq * qnw[64 + t];
        float ang = 0.0f, c = 1.0f, sn = 0.0f;
        if (rope) { ang = (float)pos * invf[t]; c = cos(ang); sn = sin(ang); }
        qs[t] = a * c - b * sn; qs[64 + t] = b * c + a * sn;
        if (is_cur) {
            float ka = (float)qkv[4096 + g * 128 + t], kb = (float)qkv[4096 + g * 128 + 64 + t];
            const float rk = rsqrt(wg_sum(ka * ka + kb * kb, red) / 128.0f + 1e-6f);
            ka *= rk * knw[t]; kb *= rk * knw[64 + t];
            kcur[t] = ka * c - kb * sn; kcur[64 + t] = kb * c + ka * sn;
            vcur[t] = (float)qkv[4608 + g * 128 + t]; vcur[64 + t] = (float)qkv[4608 + g * 128 + 64 + t];
            if ((h & 7) == 0) {                                 /* one work-group per kv head writes the cache */
                const int slot = win ? (int)(pos % win) : (int)pos;
                const size_t base = ((size_t)g * nslot + slot) * 128;
                kc[base + t] = (half)kcur[t]; kc[base + 64 + t] = (half)kcur[64 + t];
                vc[base + t] = (half)vcur[t]; vc[base + 64 + t] = (half)vcur[64 + t];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int s = s0 + t;
        float sc = -INFINITY;
        if (s < nv - 1) {
            const int slot = win ? (int)((start + s) % win) : s;
            const __global half* kr = kc + ((size_t)g * nslot + slot) * 128;
            float acc = 0.0f;
            for (int d = 0; d < 128; d += 8) {
                const float8 k8 = convert_float8(vload8(0, kr + d));
                acc += k8.s0 * qs[d] + k8.s1 * qs[d + 1] + k8.s2 * qs[d + 2] + k8.s3 * qs[d + 3]
                     + k8.s4 * qs[d + 4] + k8.s5 * qs[d + 5] + k8.s6 * qs[d + 6] + k8.s7 * qs[d + 7];
            }
            sc = acc * 0.08838834764831845f;
        } else if (s == nv - 1) {
            float acc = 0.0f;
            for (int d = 0; d < 128; ++d) acc += qs[d] * kcur[d];
            sc = acc * 0.08838834764831845f;
        }
        const float m = wg_max(sc, red);
        const float e = (s < nv) ? exp(sc - m) : 0.0f;
        pk[t] = e;
        const float lsum = wg_sum(e, red);                      /* its barriers also publish pk[] */
        const int n = min(CH, nv - s0);
        float a0 = 0.0f, a1 = 0.0f;
        for (int kk = 0; kk < n; ++kk) {
            const int si = s0 + kk;
            if (si == nv - 1) { a0 += pk[kk] * vcur[t]; a1 += pk[kk] * vcur[64 + t]; }
            else {
                const int slot = win ? (int)((start + si) % win) : si;
                const __global half* vr = vc + ((size_t)g * nslot + slot) * 128;
                a0 += pk[kk] * (float)vr[t];
                a1 += pk[kk] * (float)vr[64 + t];
            }
        }
        if (t == 0) { P[0] = m; P[1] = lsum; }
        P[2 + t] = a0; P[66 + t] = a1;
    }
}
/* router with the pre-FFN RMSNorm computed on the fly (each work-group reads the 2048-float residual) */
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void router_norm(const __global float4* w, const __global float* xr, const __global float* nw, __global float* y) {
    __local float r[16];
    const int row = get_group_id(0), t = get_local_id(0);
    /* subgroup reductions (2 barriers each), x and the norm weights read from the cache */
    float s = 0.0f;
    for (int i = t; i < 512; i += 64) { const float4 v = vload4(i, xr); s += dot(v, v); }
    const float inv = rsqrt(wg_sum(s, r) / 2048.0f + 1e-6f);
    const __global float4* wr = w + (size_t)row * 512;
    float acc = 0.0f;
    for (int i = t; i < 512; i += 64) acc += dot(wr[i], vload4(i, xr) * inv * vload4(i, nw));
    acc = wg_sum(acc, r);
    if (t == 0) y[row] = acc;
}
/* one launch, independent work-groups: WG 0 = top-6 (one subgroup); WGs 1..4 = pre-FFN RMSNorm of the residual and
   the Hadamard-512 (unscaled, as the expert kernel expects) of block b: xh (normed, half) for the shared expert,
   xrot for the expert gate_up (which then skips its own Hadamard) */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
void topk_prep(const __global float* logits, __global int* idx, __global float* rw, const __global float* xr,
               const __global float* nw, __global half* xh, __global float* xrot) {
    const int g = get_group_id(0), t = get_local_id(0);
    if (g == 0) {
        if (t < 16) {
            const int lane = t;
            float p[16];
            for (int j = 0; j < 16; ++j) p[j] = 1.0f / (1.0f + exp(-logits[lane * 16 + j]));
            float sv[6]; int si[6]; float sum = 0.0f;
            for (int r = 0; r < 6; ++r) {
                float bv = -2.0f; int bi = 0;
                for (int j = 0; j < 16; ++j) if (p[j] > bv) { bv = p[j]; bi = lane * 16 + j; }
                const float gm = sub_group_reduce_max(bv);
                const int gi = sub_group_reduce_min(bv == gm ? bi : (1 << 30));
                sv[r] = gm; si[r] = gi; sum += gm;
                for (int j = 0; j < 16; ++j) if (lane * 16 + j == gi) p[j] = -1.0f;
            }
            if (lane == 0)
                for (int r = 0; r < 6; ++r) { idx[r] = si[r]; rw[r] = sv[r] / (sum + 1e-20f) * 1.2400000095367432f; }
        }
        return;
    }
    __local float xs[512], rs[16];
    const int b = g - 1;
    float s = 0.0f;
    for (int i = t; i < 2048; i += 256) {
        const float v = xr[i];
        s += v * v;
        if ((i >> 9) == b) xs[i & 511] = v;
    }
    const float inv = rsqrt(wg_sum(s, rs) / 2048.0f + 1e-6f);        /* subgroup reduction */
    for (int j = t; j < 512; j += 256) {
        const float v = xs[j] * inv * nw[b * 512 + j];
        xs[j] = v;
        xh[b * 512 + j] = (half)v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = 1; h < 512; h <<= 1) {
        const int j = (t / h) * 2 * h + (t % h);
        const float a0 = xs[j], c0 = xs[j + h];
        xs[j] = a0 + c0; xs[j + h] = a0 - c0;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int j = t; j < 512; j += 256) xrot[b * 512 + j] = xs[j];
}

/* prompt queued on the GPU : next input token = prompt[pc], position += 1, pc += 1 (no host round trip) */
__kernel void prompt_next(const __global int* prompt, __global int* pc, __global int* tp, __global long* posb) {
    const int c = pc[0];
    tp[0] = prompt[c];
    posb[0] = posb[0] + 1;
    pc[0] = c + 1;
}

__kernel void swiglu(const __global half* gu, __global half* act, const int n) {
    const int i = get_global_id(0);
    const float g = (float)gu[i], u = (float)gu[n + i];
    act[i] = (half)(g / (1.0f + exp(-g)) * u);
}
"""

GEMV_ACC = r"""
/* same as gemv_q41, but the result is ADDED into a float vector (residual stream) */
__attribute__((reqd_work_group_size(16, KS, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_q41_acc(const __global uint* restrict w, const __global half* restrict d,
                           const __global half* restrict m, const __global half* restrict x,
                           __global float* restrict y, const int G) {
    const int lane = get_local_id(0), ks = get_local_id(1), nb = get_group_id(0);
    const int g0 = (G * ks) / KS, g1 = (G * (ks + 1)) / KS;
    float acc = 0.0f;
    for (int g = g0; g < g1; ++g) {
        const size_t blk = (size_t)nb * G + g;
        const uint4 v = intel_sub_group_block_read4(w + blk * 64);
        const float x0 = (float)x[g * 32 + lane], x1 = (float)x[g * 32 + 16 + lane];
        const float sx = sub_group_reduce_add(x0 + x1);
        const uint u[4] = {v.s0, v.s1, v.s2, v.s3};
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int j = k * 4 + b;
                s += (float)((u[k] >> (8 * b)) & 15u) * intel_sub_group_shuffle(x0, j)
                   + (float)((u[k] >> (8 * b + 4)) & 15u) * intel_sub_group_shuffle(x1, j);
            }
        }
        acc += (float)d[blk * 16 + lane] * s + (float)m[blk * 16 + lane] * sx;
    }
    __local float red[KS][16];
    red[ks][lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float t = 0.0f;
        for (int k = 0; k < KS; ++k) t += red[k][lane];
        y[nb * 16 + lane] += t;
    }
}

/* down projection whose input is SiLU(gate) * up computed on the fly from the gate+up GEMV output  */
__attribute__((reqd_work_group_size(16, KS, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_q41_acc_glu(const __global uint* restrict w, const __global half* restrict d,
                               const __global half* restrict m, const __global half* restrict gu,
                               __global float* restrict y, const int G, const int n) {
    const int lane = get_local_id(0), ks = get_local_id(1), nb = get_group_id(0);
    const int g0 = (G * ks) / KS, g1 = (G * (ks + 1)) / KS;
    float acc = 0.0f;
    for (int g = g0; g < g1; ++g) {
        const size_t blk = (size_t)nb * G + g;
        const uint4 v = intel_sub_group_block_read4(w + blk * 64);
        const int i0 = g * 32 + lane, i1 = g * 32 + 16 + lane;
        const float ga = (float)gu[i0], gb = (float)gu[i1];
        const float x0 = ga / (1.0f + exp(-ga)) * (float)gu[n + i0];
        const float x1 = gb / (1.0f + exp(-gb)) * (float)gu[n + i1];
        const float sx = sub_group_reduce_add(x0 + x1);
        const uint u[4] = {v.s0, v.s1, v.s2, v.s3};
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int j = k * 4 + b;
                s += (float)((u[k] >> (8 * b)) & 15u) * intel_sub_group_shuffle(x0, j)
                   + (float)((u[k] >> (8 * b + 4)) & 15u) * intel_sub_group_shuffle(x1, j);
            }
        }
        acc += (float)d[blk * 16 + lane] * s + (float)m[blk * 16 + lane] * sx;
    }
    __local float red[KS][16];
    red[ks][lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float t = 0.0f;
        for (int k = 0; k < KS; ++k) t += red[k][lane];
        y[nb * 16 + lane] += t;
    }
}


/* GEMV added into the residual, then the LAST work-group runs RMSNorm over the whole residual  */
#define GEMV_NORM_KERNEL(NAME, XDECL, XLOAD0, XLOAD1)                                                          \
__attribute__((reqd_work_group_size(16, KS, 1))) __attribute__((intel_reqd_sub_group_size(16)))                \
__kernel void NAME(const __global uint* restrict w, const __global half* restrict d,                           \
                   const __global half* restrict m, XDECL, __global float* y, const int G, const int n,       \
                   __global int* counter, const int nblocks, const __global float* add, const int do_add,    \
                   const __global float* nw, __global half* outh, __global float* outf,                       \
                   __global float* tmp) {                                                                     \
    __local float red[KS][16];                                                                                 \
    __local float rn[16 * KS];                                                                                 \
    __local int flag[1];                                                                                       \
    const int lane = get_local_id(0), ks = get_local_id(1), nb = get_group_id(0), tid = ks * 16 + lane;      \
    const int g0 = (G * ks) / KS, g1 = (G * (ks + 1)) / KS;                                                    \
    float acc = 0.0f;                                                                                          \
    for (int g = g0; g < g1; ++g) {                                                                            \
        const size_t blk = (size_t)nb * G + g;                                                                 \
        const uint4 v = intel_sub_group_block_read4(w + blk * 64);                                             \
        const int i0 = g * 32 + lane, i1 = g * 32 + 16 + lane;                                                 \
        const float x0 = XLOAD0, x1 = XLOAD1;                                                                  \
        const float sx = sub_group_reduce_add(x0 + x1);                                                        \
        const uint u[4] = {v.s0, v.s1, v.s2, v.s3};                                                            \
        float s = 0.0f;                                                                                        \
        _Pragma("unroll") for (int k = 0; k < 4; ++k) {                                                        \
            _Pragma("unroll") for (int b = 0; b < 4; ++b) {                                                    \
                const int j = k * 4 + b;                                                                       \
                s += (float)((u[k] >> (8 * b)) & 15u) * intel_sub_group_shuffle(x0, j)                         \
                   + (float)((u[k] >> (8 * b + 4)) & 15u) * intel_sub_group_shuffle(x1, j);                    \
            }                                                                                                  \
        }                                                                                                      \
        acc += (float)d[blk * 16 + lane] * s + (float)m[blk * 16 + lane] * sx;                                 \
    }                                                                                                          \
    red[ks][lane] = acc;                                                                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                              \
    if (ks == 0) {                                                                                             \
        float t = 0.0f;                                                                                        \
        for (int k = 0; k < KS; ++k) t += red[k][lane];                                                        \
        tmp[nb * 16 + lane] = t;                                                                               \
    }                                                                                                          \
    if (!last_wg(counter, nblocks, tid, flag)) return;                                                         \
    norm_epilogue(y, tmp, add, do_add, nw, outh, outf, tid, 16 * KS, rn);                                      \
}
GEMV_NORM_KERNEL(gemv_q41_acc_norm, const __global half* restrict x, (float)x[i0], (float)x[i1])
GEMV_NORM_KERNEL(gemv_q41_acc_glu_norm, const __global half* restrict x,
                 (float)x[i0] / (1.0f + exp(-(float)x[i0])) * (float)x[n + i0],
                 (float)x[i1] / (1.0f + exp(-(float)x[i1])) * (float)x[n + i1])


/* GEMV whose input is RMSNorm(residual) * w computed on the fly in every work-group (K = 2048), half output  */
__attribute__((reqd_work_group_size(16, KS, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_q41_norm(const __global uint* restrict w, const __global half* restrict d,
                            const __global half* restrict m, const __global float* restrict xr,
                            const __global float* restrict nw, __global half* restrict y, const int G) {
    __local float rs[KS];
    __local float red[KS][16];
    const int lane = get_local_id(0), ks = get_local_id(1), nb = get_group_id(0), tid = ks * 16 + lane;
    /* sum of squares per subgroup (no barrier), one combine step; x is read from the cache in the main loop */
    float s0 = 0.0f;
    for (int i = tid; i < 2048; i += 16 * KS) { const float v = xr[i]; s0 += v * v; }
    s0 = sub_group_reduce_add(s0);
    if (lane == 0) rs[ks] = s0;
    barrier(CLK_LOCAL_MEM_FENCE);
    float tot = 0.0f;
    for (int k = 0; k < KS; ++k) tot += rs[k];
    const float inv = rsqrt(tot / 2048.0f + 1e-6f);
    const int g0 = (G * ks) / KS, g1 = (G * (ks + 1)) / KS;
    float acc = 0.0f;
    for (int g = g0; g < g1; ++g) {
        const size_t blk = (size_t)nb * G + g;
        const uint4 v = intel_sub_group_block_read4(w + blk * 64);
        const int i0 = g * 32 + lane, i1 = g * 32 + 16 + lane;
        const float x0 = xr[i0] * inv * nw[i0], x1 = xr[i1] * inv * nw[i1];
        const float sx = sub_group_reduce_add(x0 + x1);
        const uint u[4] = {v.s0, v.s1, v.s2, v.s3};
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int j = k * 4 + b;
                s += (float)((u[k] >> (8 * b)) & 15u) * intel_sub_group_shuffle(x0, j)
                   + (float)((u[k] >> (8 * b + 4)) & 15u) * intel_sub_group_shuffle(x1, j);
            }
        }
        acc += (float)d[blk * 16 + lane] * s + (float)m[blk * 16 + lane] * sx;
    }
    red[ks][lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float t = 0.0f;
        for (int k = 0; k < KS; ++k) t += red[k][lane];
        y[nb * 16 + lane] = (half)t;
    }
}
/* shared/dense down projection (SiLU*up on the fly) added into the residual, plus the routed-expert output for the
   same rows (each work-group touches only its own 16 rows: no sync)  */
__attribute__((reqd_work_group_size(16, KS, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_q41_acc_glu_add(const __global uint* restrict w, const __global half* restrict d,
                                   const __global half* restrict m, const __global half* restrict gu,
                                   __global float* restrict y, const int G, const int n,
                                   const __global float* restrict add, const int do_add) {
    const int lane = get_local_id(0), ks = get_local_id(1), nb = get_group_id(0);
    const int g0 = (G * ks) / KS, g1 = (G * (ks + 1)) / KS;
    float acc = 0.0f;
    for (int g = g0; g < g1; ++g) {
        const size_t blk = (size_t)nb * G + g;
        const uint4 v = intel_sub_group_block_read4(w + blk * 64);
        const int i0 = g * 32 + lane, i1 = g * 32 + 16 + lane;
        const float ga = (float)gu[i0], gb = (float)gu[i1];
        const float x0 = ga / (1.0f + exp(-ga)) * (float)gu[n + i0];
        const float x1 = gb / (1.0f + exp(-gb)) * (float)gu[n + i1];
        const float sx = sub_group_reduce_add(x0 + x1);
        const uint u[4] = {v.s0, v.s1, v.s2, v.s3};
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int j = k * 4 + b;
                s += (float)((u[k] >> (8 * b)) & 15u) * intel_sub_group_shuffle(x0, j)
                   + (float)((u[k] >> (8 * b + 4)) & 15u) * intel_sub_group_shuffle(x1, j);
            }
        }
        acc += (float)d[blk * 16 + lane] * s + (float)m[blk * 16 + lane] * sx;
    }
    __local float red[KS][16];
    red[ks][lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float t = 0.0f;
        for (int k = 0; k < KS; ++k) t += red[k][lane];
        const int row = nb * 16 + lane;
        y[row] += t + (do_add ? add[row] : 0.0f);
    }
}


/* shared gate+up GEMV (work-groups 0..nblocks-1) + 6 extra work-groups that Hadamard-rotate (unscaled) the 6 expert
   activations SiLU(g)*u written by the expert gate_up, so the expert down skips its own butterfly */
__attribute__((reqd_work_group_size(16, KS, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_q41_rot6(const __global uint* restrict w, const __global half* restrict d,
                            const __global half* restrict m, const __global half* restrict x,
                            __global half* restrict y, const int G, const int nblocks,
                            const __global float* restrict act, __global float* restrict actrot) {
    __local float xs[512];                                   /* rotation scratch (local arrays: kernel scope) */
    __local float red[KS][16];
    const int lane = get_local_id(0), ks = get_local_id(1), nb = get_group_id(0), tid = ks * 16 + lane;
    if (nb >= nblocks) {                                     /* rotation work-group for expert e (uniform) */
        const int e = nb - nblocks;
        for (int j = tid; j < 512; j += 16 * KS) xs[j] = act[e * 512 + j];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int h = 1; h < 512; h <<= 1) {
            for (int q = tid; q < 256; q += 16 * KS) {
                const int j = (q / h) * 2 * h + (q % h);
                const float a0 = xs[j], c0 = xs[j + h];
                xs[j] = a0 + c0; xs[j + h] = a0 - c0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (int j = tid; j < 512; j += 16 * KS) actrot[e * 512 + j] = xs[j];
        return;
    }
    const int g0 = (G * ks) / KS, g1 = (G * (ks + 1)) / KS;
    float acc = 0.0f;
    for (int g = g0; g < g1; ++g) {
        const size_t blk = (size_t)nb * G + g;
        const uint4 v = intel_sub_group_block_read4(w + blk * 64);
        const float x0 = (float)x[g * 32 + lane], x1 = (float)x[g * 32 + 16 + lane];
        const float sx = sub_group_reduce_add(x0 + x1);
        const uint u[4] = {v.s0, v.s1, v.s2, v.s3};
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int j = k * 4 + b;
                s += (float)((u[k] >> (8 * b)) & 15u) * intel_sub_group_shuffle(x0, j)
                   + (float)((u[k] >> (8 * b + 4)) & 15u) * intel_sub_group_shuffle(x1, j);
            }
        }
        acc += (float)d[blk * 16 + lane] * s + (float)m[blk * 16 + lane] * sx;
    }
    red[ks][lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float t = 0.0f;
        for (int k = 0; k < KS; ++k) t += red[k][lane];
        y[nb * 16 + lane] = (half)t;
    }
}

/* f32 router with a float input */
__attribute__((reqd_work_group_size(16, KS, 1))) __attribute__((intel_reqd_sub_group_size(16)))
__kernel void router_f32(const __global float* restrict w, const __global float* restrict x,
                         __global float* restrict y, const int K) {
    const int lane = get_local_id(0), ks = get_local_id(1), nb = get_group_id(0);
    const int c0 = (K / 16 * ks) / KS * 16, c1 = (K / 16 * (ks + 1)) / KS * 16;
    float acc = 0.0f;
    for (int r = 0; r < 16; ++r) {
        const __global float* wr = w + (size_t)(nb * 16 + r) * K;
        float s = 0.0f;
        for (int c = c0 + lane; c < c1; c += 16) s += wr[c] * x[c];
        s = sub_group_reduce_add(s);
        if (lane == r) acc = s;
    }
    __local float red[KS][16];
    red[ks][lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float t = 0.0f;
        for (int k = 0; k < KS; ++k) t += red[k][lane];
        y[nb * 16 + lane] = t;
    }
}
"""


def _tok():
    """(encode, decode) from a Hugging Face `tokenizers` JSON for Flint's tokenizer (--tokenizer)"""
    import tokenizers
    t = tokenizers.Tokenizer.from_file(CFG["tokenizer"])
    return (lambda s: t.encode(s).ids), (lambda ids: t.decode(ids))


class FlintRT:
    def __init__(self, dev=0, max_ctx=2048, gguf=None, log=print, profile=False, fast=True, fuse=True,
                 fuse_norm=True, plan="k18"):
        import pyopencl as cl
        self.cl, self.max_ctx, self.log = cl, max_ctx, log
        devs = [d for p in cl.get_platforms() for d in p.get_devices(cl.device_type.GPU)]
        self.ctx = cl.Context([devs[dev]])
        self.profile, self.fast, self.fuse, self.fuse_norm = profile, fast, fuse and fast, fuse_norm
        self.plan_name = plan if fast else "k14"
        self.q = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE) if profile \
            else cl.CommandQueue(self.ctx)                           # in-order, no events
        log(f"device #{dev}: {devs[dev].name}")
        self.keep = []
        self._build()
        t0 = time.perf_counter()
        self._load(gguf)
        log(f"weights loaded in {time.perf_counter() - t0:.0f} s")
        self._plan()

    # ------------------------------------------------------------------ kernels
    def _build(self):
        cl = self.cl
        moe, at2, gv = _kernel_sources()
        self.KSG, self.KSD = moe["KSG"], moe["KSD"]
        o = "-cl-mad-enable"
        self.clstd = ""
        for std in ("-cl-std=CL3.0", "-cl-std=CL2.0", ""):
            try:
                self.p_rt = cl.Program(self.ctx, RT_SRC).build(f"{o} {std}")
                self.clstd = std
                break
            except Exception as ex:
                if not std:
                    raise
                self.log(f"  build with {std} failed ({str(ex).splitlines()[0][:120]}); trying the next")
        self.log(f"  RT kernels built with {self.clstd or 'the default OpenCL C version'}")
        helpers = RT_SRC[RT_SRC.index("#if defined(__OPENCL_C_VERSION__)"):RT_SRC.index("/* router (one work-group per row)")]
        self.p_g = {ks: cl.Program(self.ctx, gv["SRC"] + helpers + GEMV_ACC).build(f"{o} -DKS={ks} {self.clstd}")
                    for ks in (8, 16)}
        fix = lambda s, tot: s.replace("for (int i = 0; i < INPUT0_DIMS_SIZE; ++i) tot *= INPUT0_DIMS[i];", f"tot = {tot};")
        self.p_gu = cl.Program(self.ctx, fix(moe["CL_COMMON"] + moe["CL_GATE_UP"], 2048)).build(
            f"{o} -DKSG={self.KSG} -DKSD={self.KSD} -DINPUT0_TYPE=half -DINPUT1_TYPE=int -DINPUT3_TYPE=half "
            f"-DINPUT5_TYPE=half -DOUTPUT0_TYPE=float")
        gu_src = fix(moe["CL_COMMON"] + moe["CL_GATE_UP"], 2048)
        assert "        FWHT_LOCAL(xl, 4, tid, 16 * KSG);\n" in gu_src
        self.p_gu_rot = cl.Program(self.ctx, gu_src.replace("        FWHT_LOCAL(xl, 4, tid, 16 * KSG);\n",
                                                             "        /* input already Hadamard-rotated */\n")).build(
            f"{o} -DKSG={self.KSG} -DKSD={self.KSD} -DINPUT0_TYPE=float -DINPUT1_TYPE=int -DINPUT3_TYPE=half "
            f"-DINPUT5_TYPE=half -DOUTPUT0_TYPE=float")
        self.p_dn_pf = cl.Program(self.ctx, moe["CL_COMMON"] + DOWN_PF).build(f"{o} -DKSG={self.KSG} -DKSD={self.KSD}")
        dn_src = fix(moe["CL_COMMON"] + moe["CL_DOWN"], 3072)
        assert "        FWHT_LOCAL(al, 6, tid, 16 * KSD);\n" in dn_src
        self.p_dn_rot = cl.Program(self.ctx, dn_src.replace("        FWHT_LOCAL(al, 6, tid, 16 * KSD);\n",
                                                             "        /* input already Hadamard-rotated */\n")).build(
            f"{o} -DKSG={self.KSG} -DKSD={self.KSD} -DINPUT0_TYPE=float -DINPUT1_TYPE=int -DINPUT2_TYPE=float "
            f"-DINPUT4_TYPE=half -DOUTPUT0_TYPE=float")
        self.p_dn = cl.Program(self.ctx, fix(moe["CL_COMMON"] + moe["CL_DOWN"], 3072)).build(
            f"{o} -DKSG={self.KSG} -DKSD={self.KSD} -DINPUT0_TYPE=float -DINPUT1_TYPE=int -DINPUT2_TYPE=float "
            f"-DINPUT4_TYPE=half -DOUTPUT0_TYPE=float")
        self.p_at = {}
        # the old single-pass attention keeps all scores in local memory (NSLOT floats): only built for the
        # non-fast path, and it cannot hold more than ~16k positions on a 64 KB-SLM GPU
        for nslot, win, rope in (((WIN, WIN, 1), (self.max_ctx, 0, 0)) if not self.fast else ()):
            self.p_at[nslot] = cl.Program(self.ctx, at2["CL_COMMON"] + at2["CL_ATTN"]).build(
                f"{o} -DNSLOT={nslot} -DWIN={max(win, 1)} -DRING={int(win > 0)} -DROPE={rope} -DINPUT0_TYPE=half "
                f"-DINPUT1_TYPE=long -DINPUT2_TYPE=half -DINPUT3_TYPE=half -DINPUT4_TYPE=float -DINPUT5_TYPE=float "
                f"-DINPUT6_TYPE=float -DINPUT7_TYPE=float -DOUTPUT0_TYPE=half")

    # ------------------------------------------------------------------ buffers
    def buf(self, nbytes=None, arr=None):
        cl = self.cl
        if arr is not None:
            b = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(arr))
        else:
            b = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=int(nbytes))
        self.keep.append(b)
        return b

    def _cached(self, name, make):
        """packed arrays cached as .npy in --cache (the first load packs from the GGUF)"""
        os.makedirs(CFG["cache"], exist_ok=True)
        paths = [os.path.join(CFG["cache"], f"{name}.{i}.npy") for i in range(8)]
        if os.path.exists(paths[0]):
            out, i = [], 0
            while os.path.exists(paths[i]):
                out.append(np.load(paths[i], mmap_mode="r"))
                i += 1
            return out
        arrs = make()
        for i, a in enumerate(arrs):
            np.save(paths[i], a)
        return arrs

    def _load(self, gguf):
        mm, _, _, T = read_gguf(gguf)

        def f32(name):
            t = T[name]
            n = int(np.prod(t["dims"]))
            dt = {0: np.float32, 1: np.float16}[t["type"]]
            return np.asarray(mm[t["off"]:t["off"] + n * np.dtype(dt).itemsize]).view(dt).astype(np.float32)

        def q41(names_rows):
            raws = []
            for name, rows in names_rows:
                t = T[name]
                if t["type"] != 3:
                    raise ValueError(f"{name}: expected Q4_1 (type 3), got {t['type']}")
                K = t["dims"][0]
                raws.append(np.asarray(mm[t["off"]:t["off"] + rows * (K // 32) * 20]).reshape(rows, K // 32, 20))
            raw = np.concatenate(raws, 0)
            N, G, _ = raw.shape
            d = np.ascontiguousarray(raw[..., 0:2]).view(np.float16).reshape(N, G)
            m = np.ascontiguousarray(raw[..., 2:4]).view(np.float16).reshape(N, G)
            u = np.ascontiguousarray(raw[..., 4:20]).view(np.uint32).reshape(N, G, 4)
            return [np.ascontiguousarray(u.reshape(N // 16, 16, G, 4).transpose(0, 2, 3, 1)),
                    np.ascontiguousarray(d.reshape(N // 16, 16, G).transpose(0, 2, 1)),
                    np.ascontiguousarray(m.reshape(N // 16, 16, G).transpose(0, 2, 1))]

        def fc(name, parts):
            c, d, m = self._cached(name, lambda: q41(parts))
            return dict(w=self.buf(arr=c), d=self.buf(arr=d), m=self.buf(arr=m), N=c.shape[0] * 16, G=c.shape[1])

        t = T["token_embd.weight"]
        V = t["dims"][1]
        dt = {0: np.float32, 1: np.float16}[t["type"]]
        emb = np.asarray(mm[t["off"]:t["off"] + V * HID * np.dtype(dt).itemsize]).view(dt).astype(np.float16)
        self.V = V
        self.emb = self.buf(arr=emb)
        del emb
        self.invf = self.buf(arr=(1.0 / (10000.0 ** (np.arange(64, dtype=np.float32) * 2.0 / HD))).astype(np.float32))
        self.L = []
        for L in range(NL):
            b = f"blk.{L}."
            d = dict(sliding=bool(SLIDING[L]))
            d["attn_norm"] = self.buf(arr=f32(b + "attn_norm.weight"))
            d["ffn_norm"] = self.buf(arr=f32(b + "ffn_norm.weight"))
            d["qn"] = self.buf(arr=f32(b + "attn_q_norm.weight"))
            d["kn"] = self.buf(arr=f32(b + "attn_k_norm.weight"))
            d["sinks"] = self.buf(arr=f32(b + "attn_sinks.weight"))
            d["qkv"] = fc(f"L{L}.qkv", [(b + "attn_q.weight", 4096), (b + "attn_k.weight", 512), (b + "attn_v.weight", 512)])
            d["o"] = fc(f"L{L}.o", [(b + "attn_output.weight", 2048)])
            nslot = WIN if d["sliding"] else self.max_ctx
            d["nslot"] = nslot
            d["kc"] = self.buf(KVH * nslot * HD * 2)
            d["vc"] = self.buf(KVH * nslot * HD * 2)
            if L < 2:
                d["gu"] = fc(f"L{L}.gu", [(b + "ffn_gate.weight", 6144), (b + "ffn_up.weight", 6144)])
                d["down"] = fc(f"L{L}.down", [(b + "ffn_down.weight", 2048)])
                d["ffn"] = 6144
            else:
                d["router"] = self.buf(arr=f32(b + "ffn_gate_inp.weight").reshape(256, HID))
                d["gu"] = fc(f"L{L}.shgu", [(b + "ffn_gate_shexp.weight", 1024), (b + "ffn_up_shexp.weight", 1024)])
                d["down"] = fc(f"L{L}.shdown", [(b + "ffn_down_shexp.weight", 2048)])
                d["ffn"] = 1024
                packed = self._cached(f"L{L}.exps", lambda: [a for kind in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
                                                             for a in pack_for_gpu(raw_expert_tensor(mm, T, L, kind))])
                for key, a in zip(("wg", "sg", "wu", "su", "wd", "sd"), packed):
                    d[key] = self.buf(arr=a)
            self.L.append(d)
            if L % 8 == 0:
                self.log(f"  layer {L} loaded")
        self.out_norm = self.buf(arr=f32("output_norm.weight"))
        self.head = fc("head", [("output.weight", self.V)])

    # ------------------------------------------------------------------ the token plan
    def _plan(self):
        cl = self.cl
        self.tp = self.buf(arr=np.zeros(1, np.int32))
        self.posb = self.buf(arr=np.zeros(1, np.int64))
        self.x = self.buf(HID * 4)
        self.xh, self.xf = self.buf(HID * 2), self.buf(HID * 4)
        self.qkv = self.buf(5120 * 2)
        self.att = self.buf(4096 * 2)
        self.gub, self.act = self.buf(12288 * 2), self.buf(6144 * 2)
        self.rlog, self.idx, self.rw = self.buf(256 * 4), self.buf(6 * 4), self.buf(6 * 4)
        self.eact, self.moe = self.buf(6 * 512 * 4), self.buf(HID * 4)
        self.nsplit = {WIN: -(-WIN // 64), self.max_ctx: -(-self.max_ctx // 64)}
        self.part = self.buf(H * max(self.nsplit.values()) * 130 * 4)
        self.ntmp = self.buf(HID * 4)                                    # GEMV rows before the fused norm
        self.logits = self.buf(self.V * 2)
        self.logits_h = np.empty(self.V, np.float16)
        P = []

        self.tags = []
        cur = ["setup"]
        self._cur = None

        def k(prog, name, gws, lws, *args, role=None):
            kern = cl.Kernel(prog, name)
            kern.set_args(*args)
            P.append((kern, gws, lws))
            self.tags.append(f"{role or name} {self._cur or cur[0]}")

        def gemv(w, x, y, acc, ks=16, role="gemv"):
            k(self.p_g[ks], "gemv_q41_acc" if acc else "gemv_q41", (w["N"], ks), (16, ks),
              w["w"], w["d"], w["m"], x, y, np.int32(w["G"]), role=f"gemv {role}")

        def old_plan():                                                 # reference plan (kept for compare mode)
            pending = False
            k(self.p_rt, "embed", (HID,), None, self.emb, self.tp, self.x)
            for L, d in enumerate(self.L):
                cur[0] = "dense" if L < 2 else ("sliding" if d["sliding"] else "global")
                k(self.p_rt, "rmsnorm", (256,), (256,), self.x, self.moe, np.int32(pending), d["attn_norm"], self.xh, self.xf)
                gemv(d["qkv"], self.xh, self.qkv, False, ks=8, role="qkv")
                k(self.p_rt, "kv_store", (KVH * 16,), (16,), self.qkv, self.posb, d["kn"], self.invf, d["kc"], d["vc"],
                  np.int32(d["nslot"]), np.int32(d["sliding"]), np.int32(d["sliding"]))
                if self.fast:          # split-KV attention (many work-groups) + combine with the sink
                    ns = self.nsplit[d["nslot"]]
                    win = WIN if d["sliding"] else 0
                    k(self.p_rt, "attn_part", (H * 64, ns), (64, 1), self.qkv, self.posb, d["kc"], d["vc"], d["qn"],
                      self.invf, self.part, np.int32(d["nslot"]), np.int32(win), np.int32(d["sliding"]), np.int32(ns))
                    k(self.p_rt, "attn_combine", (H * 128,), (128,), self.part, d["sinks"], self.posb, self.att,
                      np.int32(win), np.int32(ns))
                else:
                    k(self.p_at[d["nslot"]], "flint_attn", (H * 256,), (256,), self.qkv, self.posb, d["kc"], d["vc"],
                      d["qn"], d["kn"], self.invf, d["sinks"], self.att)
                gemv(d["o"], self.att, self.x, True, role="o")
                k(self.p_rt, "rmsnorm", (256,), (256,), self.x, self.moe, np.int32(0), d["ffn_norm"], self.xh, self.xf)
                if L >= 2:
                    if self.fast:      # one work-group per router row; top-6 in one subgroup
                        k(self.p_rt, "router2", (256 * 64,), (64,), d["router"], self.xf, self.rlog)
                        k(self.p_rt, "topk6b", (16,), (16,), self.rlog, self.idx, self.rw)
                    else:
                        k(self.p_g[16], "router_f32", (256, 16), (16, 16), d["router"], self.xf, self.rlog, np.int32(HID))
                        k(self.p_rt, "topk6", (256,), (256,), self.rlog, self.idx, self.rw)
                    k(self.p_gu, "flint_gate_up", (6 * 32 * 16, self.KSG), (16, self.KSG), self.xh, self.idx,
                      d["wg"], d["sg"], d["wu"], d["su"], self.eact)
                    k(self.p_dn, "flint_down", (128 * 16, self.KSD), (16, self.KSD), self.eact, self.idx, self.rw,
                      d["wd"], d["sd"], self.moe)
                gemv(d["gu"], self.xh, self.gub, False, role="ffn gate+up")
                if self.fast:          # SiLU*up computed inside the down GEMV (one launch less per layer)
                    w = d["down"]
                    k(self.p_g[16], "gemv_q41_acc_glu", (w["N"], 16), (16, 16), w["w"], w["d"], w["m"], self.gub,
                      self.x, np.int32(w["G"]), np.int32(d["ffn"]), role="gemv ffn down+glu")
                else:
                    k(self.p_rt, "swiglu", (d["ffn"],), None, self.gub, self.act, np.int32(d["ffn"]))
                    gemv(d["down"], self.act, self.x, True, role="ffn down")
                pending = L >= 2
            cur[0] = "head"
            k(self.p_rt, "rmsnorm", (256,), (256,), self.x, self.moe, np.int32(pending), self.out_norm, self.xh, self.xf)
            gemv(self.head, self.xh, self.logits, False, role="lm head")

        self.plan_k16, self.variants = None, {}
        if self.plan_name == "k18":
            self._plan_k18(P, k)
            k18, ktags = list(P), list(self.tags)
            P.clear(); self.tags.clear(); self._cur = None
            old_plan()
            self.plan_k16 = list(P)
            self.variants = {"k18": k18}
            P[:] = k18; self.tags[:] = ktags
            return self._finish_plan(P)
        if self.plan_name == "k17" and self.fuse:
            self._plan_fused(P, k, fn=self.fuse_norm)
            fused, ftags = list(P), list(self.tags)
            for name, parts in (("none", (0, 0, 0)), ("attn", (1, 0, 0)), ("topk", (0, 1, 0)), ("norm", (0, 0, 1))):
                P.clear(); self.tags.clear(); self._cur = None
                self._plan_fused(P, k, *map(bool, parts))
                self.variants[name] = list(P)
            self.variants["all"] = fused
            P.clear(); self.tags.clear(); self._cur = None
            old_plan()
            self.plan_k16 = list(P)
            P[:] = fused; self.tags[:] = ftags
        else:
            old_plan()
        return self._finish_plan(P)

    def _plan_k18(self, P, k):
        """no cross-work-group sync. RMSNorm computed on the fly by the consumer (qkv GEMV, router, head);
        the pre-FFN norm + Hadamard of the expert input made by spare work-groups of the top-6 launch; KV write inside
        the attention part; routed output added row by row in the shared down GEMV. ~10 launches per MoE layer."""
        self.xrot = self.buf(HID * 4)
        self.eact_rot = self.buf(6 * 512 * 4)

        def gemv(w, x, y, ks=16, role="gemv"):
            k(self.p_g[ks], "gemv_q41", (w["N"], ks), (16, ks), w["w"], w["d"], w["m"], x, y, np.int32(w["G"]),
              role=f"gemv {role}")

        def gemv_norm(w, nw, y, ks=16, role="gemv"):
            k(self.p_g[ks], "gemv_q41_norm", (w["N"], ks), (16, ks), w["w"], w["d"], w["m"], self.x, nw, y,
              np.int32(w["G"]), role=f"gemv {role}")

        k(self.p_rt, "embed", (HID,), None, self.emb, self.tp, self.x, role="embed")
        for L, d in enumerate(self.L):
            self._cur = "dense" if L < 2 else ("sliding" if d["sliding"] else "global")
            gemv_norm(d["qkv"], d["attn_norm"], self.qkv, ks=8, role="qkv+norm")
            ns = self.nsplit[d["nslot"]]
            win = WIN if d["sliding"] else 0
            k(self.p_rt, "attn_part_kv", (H * 64, ns), (64, 1), self.qkv, self.posb, d["kc"], d["vc"], d["qn"], d["kn"],
              self.invf, self.part, np.int32(d["nslot"]), np.int32(win), np.int32(d["sliding"]), np.int32(ns),
              role="attn_part_kv")
            k(self.p_rt, "attn_combine", (H * 128,), (128,), self.part, d["sinks"], self.posb, self.att,
              np.int32(win), np.int32(ns), role="attn_combine")
            k(self.p_g[16], "gemv_q41_acc", (d["o"]["N"], 16), (16, 16), d["o"]["w"], d["o"]["d"], d["o"]["m"], self.att,
              self.x, np.int32(d["o"]["G"]), role="gemv o")
            if L >= 2:
                k(self.p_rt, "router_norm", (256 * 64,), (64,), d["router"], self.x, d["ffn_norm"], self.rlog,
                  role="router_norm")
                k(self.p_rt, "topk_prep", (5 * 256,), (256,), self.rlog, self.idx, self.rw, self.x, d["ffn_norm"],
                  self.xh, self.xrot, role="topk_prep")
                k(self.p_gu_rot, "flint_gate_up", (6 * 32 * 16, self.KSG), (16, self.KSG), self.xrot, self.idx,
                  d["wg"], d["sg"], d["wu"], d["su"], self.eact, role="flint_gate_up (rotated input)")
                # the shared gate+up launch also rotates the expert activation (6 extra work-groups);
                # the expert down then reads it pre-rotated (no butterfly in each of its 128 work-groups)
                w = d["gu"]
                k(self.p_g[16], "gemv_q41_rot6", (w["N"] + 6 * 16, 16), (16, 16), w["w"], w["d"], w["m"], self.xh,
                  self.gub, np.int32(w["G"]), np.int32(w["N"] // 16), self.eact, self.eact_rot,
                  role="shared gate+up (+expert act rotation)")
                k(self.p_dn_rot, "flint_down", (128 * 16, self.KSD), (16, self.KSD), self.eact_rot, self.idx, self.rw,
                  d["wd"], d["sd"], self.moe, role="flint_down (rotated input)")
            else:
                gemv_norm(d["gu"], d["ffn_norm"], self.gub, role="dense gate+up+norm")
            w = d["down"]
            k(self.p_g[16], "gemv_q41_acc_glu_add", (w["N"], 16), (16, 16), w["w"], w["d"], w["m"], self.gub, self.x,
              np.int32(w["G"]), np.int32(d["ffn"]), self.moe, np.int32(L >= 2), role="gemv down+glu+moe")
        self._cur = "head"
        gemv_norm(self.head, self.out_norm, self.logits, role="lm head+norm")

    def _finish_plan(self, P):
        cl = self.cl
        self.plan = P
        self.hist = self.buf(arr=np.zeros(65536, np.int32))
        self.cnt = self.buf(arr=np.zeros(1, np.int32))
        self.prompt_buf = self.buf(arr=np.zeros(self.max_ctx, np.int32))
        self.pc = self.buf(arr=np.zeros(1, np.int32))
        self.k_prompt = cl.Kernel(self.p_rt, "prompt_next")
        self.k_prompt.set_args(self.prompt_buf, self.pc, self.tp, self.posb)
        self.k_argmax = cl.Kernel(self.p_rt, "argmax_next")
        self.k_argmax.set_args(self.logits, np.int32(self.V), self.tp, self.posb, self.hist, self.cnt)
        self.log(f"token plan: {len(P)} launches")

    def _plan_fused(self, P, k, fa=True, ft=True, fn=True):
        """experimental plan (not used; wrong results): ~8 launches per MoE layer. Norms run in the last work-group of the GEMV before them, top-6 in
        the last router work-group, the KV write + attention combine inside the attention kernel."""
        self.cnt_k = self.buf(arr=np.zeros(64, np.int32))              # [0:32] attention heads, 32.. misc
        c_attn = self.cnt_k
        c_router, c_onorm, c_dnorm = (self.buf(arr=np.zeros(1, np.int32)) for _ in range(3))   # one counter each

        def gemv_norm(w, x, glu, n, add, do_add, nw, counter, role):
            name = "gemv_q41_acc_glu_norm" if glu else "gemv_q41_acc_norm"
            k(self.p_g[16], name, (w["N"], 16), (16, 16), w["w"], w["d"], w["m"], x, self.x, np.int32(w["G"]),
              np.int32(n), counter, np.int32(w["N"] // 16), add, np.int32(do_add), nw, self.xh, self.xf,
              self.ntmp, role=f"gemv {role}")

        def gemv(w, x, y, ks=16, role="gemv"):
            k(self.p_g[ks], "gemv_q41", (w["N"], ks), (16, ks), w["w"], w["d"], w["m"], x, y, np.int32(w["G"]),
              role=f"gemv {role}")

        def gemv_acc(w, x, y, glu, n, role):
            if glu:
                k(self.p_g[16], "gemv_q41_acc_glu", (w["N"], 16), (16, 16), w["w"], w["d"], w["m"], x, y,
                  np.int32(w["G"]), np.int32(n), role=f"gemv {role}")
            else:
                k(self.p_g[16], "gemv_q41_acc", (w["N"], 16), (16, 16), w["w"], w["d"], w["m"], x, y,
                  np.int32(w["G"]), role=f"gemv {role}")

        def rms(add_pending, w):
            k(self.p_rt, "rmsnorm", (256,), (256,), self.x, self.moe, np.int32(add_pending), w, self.xh, self.xf,
              role="rmsnorm")

        k(self.p_rt, "embed", (HID,), None, self.emb, self.tp, self.x, role="embed")
        if fn:
            rms(0, self.L[0]["attn_norm"])
        pending = False
        for L, d in enumerate(self.L):
            self._cur = "dense" if L < 2 else ("sliding" if d["sliding"] else "global")
            if not fn:
                rms(pending, d["attn_norm"])
            gemv(d["qkv"], self.xh, self.qkv, ks=8, role="qkv")
            ns = self.nsplit[d["nslot"]]
            win = WIN if d["sliding"] else 0
            if fa:
                k(self.p_rt, "attn_fused", (H * 64, ns), (64, 1), self.qkv, self.posb, d["kc"], d["vc"], d["qn"],
                  d["kn"], self.invf, d["sinks"], self.part, c_attn, self.att, np.int32(d["nslot"]), np.int32(win),
                  np.int32(d["sliding"]), np.int32(ns), role="attn_fused")
            else:
                k(self.p_rt, "kv_store", (KVH * 16,), (16,), self.qkv, self.posb, d["kn"], self.invf, d["kc"], d["vc"],
                  np.int32(d["nslot"]), np.int32(d["sliding"]), np.int32(d["sliding"]), role="kv_store")
                k(self.p_rt, "attn_part", (H * 64, ns), (64, 1), self.qkv, self.posb, d["kc"], d["vc"], d["qn"],
                  self.invf, self.part, np.int32(d["nslot"]), np.int32(win), np.int32(d["sliding"]), np.int32(ns),
                  role="attn_part")
                k(self.p_rt, "attn_combine", (H * 128,), (128,), self.part, d["sinks"], self.posb, self.att,
                  np.int32(win), np.int32(ns), role="attn_combine")
            if fn:
                gemv_norm(d["o"], self.att, False, 0, self.moe, 0, d["ffn_norm"], c_onorm, "o+norm")
            else:
                gemv_acc(d["o"], self.att, self.x, False, 0, "o")
                rms(0, d["ffn_norm"])
            if L >= 2:
                if ft:
                    k(self.p_rt, "router_topk", (256 * 64,), (64,), d["router"], self.xf, self.rlog, c_router,
                      self.idx, self.rw, role="router_topk")
                else:
                    k(self.p_rt, "router2", (256 * 64,), (64,), d["router"], self.xf, self.rlog, role="router2")
                    k(self.p_rt, "topk6b", (16,), (16,), self.rlog, self.idx, self.rw, role="topk6b")
                k(self.p_gu, "flint_gate_up", (6 * 32 * 16, self.KSG), (16, self.KSG), self.xh, self.idx,
                  d["wg"], d["sg"], d["wu"], d["su"], self.eact, role="flint_gate_up")
                k(self.p_dn, "flint_down", (128 * 16, self.KSD), (16, self.KSD), self.eact, self.idx, self.rw,
                  d["wd"], d["sd"], self.moe, role="flint_down")
            gemv(d["gu"], self.xh, self.gub, role="ffn gate+up")
            if fn:
                nxt_norm = self.L[L + 1]["attn_norm"] if L + 1 < NL else self.out_norm
                gemv_norm(d["down"], self.gub, True, d["ffn"], self.moe, int(L >= 2), nxt_norm, c_dnorm,
                          "down+glu+norm")
            else:
                gemv_acc(d["down"], self.gub, self.x, True, d["ffn"], "down+glu")
                pending = L >= 2
        self._cur = "head"
        if not fn:
            rms(pending, self.out_norm)
        gemv(self.head, self.xh, self.logits, role="lm head")

    def generate(self, prompt_ids, n_new, sync_every=16):
        """greedy decode with the next token chosen ON the GPU : after the prompt, the host only enqueues
        work (token N+1 is queued without waiting for token N) and collects the tokens every `sync_every` steps."""
        cl = self.cl
        if len(prompt_ids) + n_new > self.max_ctx:
            raise ValueError("prompt + new tokens > max_ctx")
        # the whole prompt is queued on the GPU (prompt_next kernel), no host sync per prompt token
        P_ = len(prompt_ids)
        cl.enqueue_copy(self.q, self.prompt_buf, np.asarray(prompt_ids, np.int32))
        cl.enqueue_copy(self.q, self.pc, np.array([1], np.int32))
        cl.enqueue_copy(self.q, self.tp, np.array([prompt_ids[0]], np.int32))
        cl.enqueue_copy(self.q, self.posb, np.array([0], np.int64))
        cl.enqueue_copy(self.q, self.cnt, np.zeros(1, np.int32))
        for i in range(P_ - 1):
            for kern, g, l in self.plan:
                cl.enqueue_nd_range_kernel(self.q, kern, g, l)
            cl.enqueue_nd_range_kernel(self.q, self.k_prompt, (1,), (1,))
            if (i + 1) % sync_every == 0:
                self.q.flush()
        for j in range(n_new):
            for kern, g, l in self.plan:
                cl.enqueue_nd_range_kernel(self.q, kern, g, l)
            cl.enqueue_nd_range_kernel(self.q, self.k_argmax, (256,), (256,))
            if (j + 1) % sync_every == 0:
                self.q.flush()
        out = np.empty(n_new, np.int32)
        cl.enqueue_copy(self.q, out, self.hist)                          # one sync at the end
        return out.tolist()

    def step(self, tok, pos, plan=None):
        """one token -> float32 logits [V]"""
        cl = self.cl
        if pos >= self.max_ctx:
            raise ValueError(f"position {pos} >= max_ctx {self.max_ctx}")
        cl.enqueue_copy(self.q, self.tp, np.array([tok], np.int32))
        cl.enqueue_copy(self.q, self.posb, np.array([pos], np.int64))
        evs = [cl.enqueue_nd_range_kernel(self.q, kern, g, l) for kern, g, l in (plan or self.plan)]
        cl.enqueue_copy(self.q, self.logits_h, self.logits)            # blocking read = one sync per token
        if self.profile:
            self.last_events = evs
        return self.logits_h.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "ppl", "prompts", "speed", "long", "profile", "compare", "prefill"])
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=1600)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--short", action="store_true", help="speed: only 256 tokens (positions ~9..265)")
    ap.add_argument("--gguf", required=True, help="Flint-40B PQ2_0 GGUF")
    ap.add_argument("--tokenizer", required=True, help="tokenizers JSON (tokenizer.json) for Flint's tokenizer")
    ap.add_argument("--data", default=None, help="text file for ppl/long/compare/prefill (e.g. WikiText-2 test)")
    ap.add_argument("--cache", default="flint_cache", help="folder for the repacked weights")
    ap.add_argument("--out", default="flint_rt_report.md")
    ap.add_argument("--reference", action="store_true", help="run the simpler reference plan (more launches)")
    args = ap.parse_args()
    args.old, args.k16, args.k17, args.nonorm = False, args.reference, False, False
    CFG.update(tokenizer=args.tokenizer, data=args.data, cache=args.cache)
    if args.mode in ("ppl", "long", "compare", "prefill") and not args.data:
        ap.error(f"{args.mode} needs --data (a plain text file, e.g. the WikiText-2 test split)")
    R = []

    def say(s=""):
        print(s, flush=True)
        R.append(s)
    say(f"# Flint-40B on the OpenCL runtime ({args.mode})")
    t0 = time.perf_counter()
    plan = "k16" if args.k16 else ("k17" if args.k17 else "k18")
    rt = FlintRT(args.dev, args.max_ctx, gguf=args.gguf, log=say, profile=(args.mode == "profile"),
                 fast=True, fuse=False, plan=plan)
    say(f"- kernels/plan: {'K14 (old)' if args.old else ('main' if rt.plan_name == 'k18' else 'reference')}")
    say(f"- ready in {time.perf_counter() - t0:.0f} s")
    enc, dec = _tok()
    try:
        if args.mode == "check":
            lg = rt.step(enc("Hello")[0], 0)
            say(f"**check: logits finite {bool(np.isfinite(lg).all())}, max {lg.max():.2f}, argmax {int(lg.argmax())} "
                f"= {dec([int(lg.argmax())])!r}**")
            if not args.old:       # GPU-side greedy must give exactly the host-side greedy tokens
                ids = enc(PROMPTS[0][1])
                for i, t in enumerate(ids):
                    lg = rt.step(t, i)
                host, nxt = [], int(lg.argmax())
                for j in range(16):
                    host.append(nxt)
                    lg = rt.step(nxt, len(ids) + j)
                    nxt = int(lg.argmax())
                gpu = rt.generate(ids, 16)
                say(f"**check greedy: GPU-side == host-side for 16 tokens: {gpu == host}** ({dec(gpu)!r})")
        elif args.mode in ("ppl", "long"):
            n = 128 if args.mode == "ppl" else args.tokens
            ids = enc(open(CFG["data"], encoding="utf-8").read())[:n + 1]
            nll, cor, bad, chunk = 0.0, 0, 0, []
            c_nll, c_cor, c_n = 0.0, 0, 0
            t1 = time.perf_counter()
            for t in range(len(ids) - 1):
                lg = rt.step(ids[t], t)
                if not np.isfinite(lg).all():
                    bad += 1
                    continue
                m = lg.max()
                l = float(m + np.log(np.exp(lg - m).sum()) - lg[ids[t + 1]])
                nll, c_nll = nll + l, c_nll + l
                ok = int(lg.argmax() == ids[t + 1])
                cor, c_cor, c_n = cor + ok, c_cor + ok, c_n + 1
                if args.mode == "long" and (c_n == 256 or t == len(ids) - 2):
                    say(f"- tokens up to {t + 1}: PPL {math.exp(c_nll / c_n):.2f}, top-1 {c_cor / c_n * 100:.1f}%")
                    c_nll, c_cor, c_n = 0.0, 0, 0
            n_ok = len(ids) - 1 - bad
            say(f"**{args.mode}: {len(ids) - 1} tokens, PPL {math.exp(nll / max(1, n_ok)):.2f}, top-1 "
                f"{cor / max(1, n_ok) * 100:.2f}%, NaN steps {bad}, {(len(ids) - 1) / (time.perf_counter() - t1):.2f} tok/s "
                f"(incl. logits to host + softmax in numpy)**")
        elif args.mode == "prompts" and not args.old:
            sp = []
            for name, text in PROMPTS:
                ids = enc(text)
                rt.generate(ids, 4)                                        # warm-up
                t1 = time.perf_counter()
                out = rt.generate(ids, 64)
                dt = time.perf_counter() - t1
                tps = 64 / dt
                sp.append(tps)
                say(f"- **{name}**: `{dec(out).strip()[:160]!r}` ({tps:.1f} tok/s incl. the prompt)")
            say(f"**prompts (GPU-side greedy, pipelined): median {np.median(sp):.1f} tok/s**")
        elif args.mode == "prompts":
            sp = []
            for name, text in PROMPTS:
                ids = enc(text)
                for i, t in enumerate(ids):
                    lg = rt.step(t, i)
                nxt, out = int(lg.argmax()), []
                t1 = time.perf_counter()
                for j in range(64):
                    out.append(nxt)
                    lg = rt.step(nxt, len(ids) + j)
                    nxt = int(lg.argmax())
                tps = 64 / (time.perf_counter() - t1)
                sp.append(tps)
                say(f"- **{name}**: `{dec(out).strip()[:160]!r}` ({tps:.1f} tok/s)")
            say(f"**prompts: decode median {np.median(sp):.1f} tok/s**")
        elif args.mode == "compare":
            # K17 vs K16 plan on the same weights and KV history: per token, run K16, then K17 on the same token
            if rt.plan_k16 is None:
                raise SystemExit("compare needs the fused (default) runtime")
            say(f"- OpenCL C for the fused kernels: {rt.clstd or 'default (1.2)'}")
            all_ids = enc(open(CFG["data"], encoding="utf-8").read())
            for name in [v for v in ("none", "attn", "topk", "norm", "all", "k18") if v in rt.variants]:
                n = 600 if name in ("all", "k18") else 48
                ids, worst, agree, rels = all_ids[:n], 0.0, 0, []
                for t, tok in enumerate(ids):
                    a = rt.step(tok, t, plan=rt.plan_k16).copy()
                    b = rt.step(tok, t, plan=rt.variants[name])
                    rel = float(np.abs(a - b).max() / (np.abs(a).max() + 1e-9))
                    rels.append(rel)
                    worst = max(worst, rel)
                    agree += int(a.argmax() == b.argmax())
                if name == "all" and not rt.fuse_norm:
                    name_shown = "all (without norm)"
                else:
                    name_shown = name
                say(f"**compare plan '{name_shown}' vs reference: {n} tokens, worst rel diff {worst:.2e}, "
                    f"median {np.median(rels):.2e}, tokens > 5%: {sum(r > 0.05 for r in rels)}, same argmax {agree}/{n}**")
        elif args.mode == "profile":
            # GPU time per kernel kind (summed per token) at a short and a long context, vs wall time per token
            ids = enc("The history of the city of Paris begins with")
            for i, t in enumerate(ids):
                lg = rt.step(t, i)
            nxt, pos = int(lg.argmax()), len(ids)
            for label, target in (("short (pos ~10-40)", 40), ("long (pos ~1000-1030)", 1030)):
                while pos < target - 30:                                    # move to the context length (no timing)
                    lg = rt.step(nxt, pos)
                    nxt, pos = int(lg.argmax()), pos + 1
                per, walls, gsum = {}, [], []
                for _ in range(30):
                    t1 = time.perf_counter()
                    lg = rt.step(nxt, pos)
                    walls.append((time.perf_counter() - t1) * 1e3)
                    nxt, pos = int(lg.argmax()), pos + 1
                    tot = 0.0
                    for e, tag in zip(rt.last_events, rt.tags):
                        dt = (e.profile.end - e.profile.start) * 1e-6
                        per[tag] = per.get(tag, 0.0) + dt
                        tot += dt
                    gsum.append(tot)
                say(f"\n**{label}: wall {np.median(walls):.2f} ms/token (profiling on), GPU sum {np.median(gsum):.2f} ms/token**")
                for tag, v in sorted(per.items(), key=lambda kv: -kv[1])[:16]:
                    say(f"- {tag}: {v / 30:.3f} ms")
        elif args.mode == "prefill":
            ids = enc(open(CFG["data"], encoding="utf-8").read())[:512]
            rt.generate(ids[:16], 1)                                      # warm-up
            for n in (64, 512):
                t1 = time.perf_counter()
                out = rt.generate(ids[:n], 1)
                dt = time.perf_counter() - t1
                say(f"**prefill: {n}-token prompt -> first new token in {dt:.2f} s ({n / dt:.1f} prompt tok/s)** "
                    f"(next token {dec(out)!r})")
        elif args.mode == "speed" and not args.old:
            ids = enc("The history of the city of Paris begins with")
            n = 256 if args.short else min(args.max_ctx - len(ids), 2039)
            rt.generate(ids, 8)
            t1 = time.perf_counter()
            out = rt.generate(ids, n)
            dt = time.perf_counter() - t1
            say(f"**speed (GPU-side greedy, pipelined): {n} tokens in {dt:.1f} s = {n / dt:.2f} tok/s "
                f"({dt / n * 1e3:.2f} ms/token), positions {len(ids)}..{len(ids) + n}**")
            say(f"- first tokens: `{dec(out[:24])!r}`")
        elif args.mode == "speed":
            ids = enc("The history of the city of Paris begins with")
            for i, t in enumerate(ids):
                lg = rt.step(t, i)
            nxt, pos, n = int(lg.argmax()), len(ids), 0
            t1 = time.perf_counter()
            while time.perf_counter() - t1 < args.seconds and pos < args.max_ctx:
                lg = rt.step(nxt, pos)
                nxt, pos, n = int(lg.argmax()), pos + 1, n + 1
            dt = time.perf_counter() - t1
            say(f"**speed: {n} tokens in {dt:.0f} s = {n / dt:.2f} tok/s ({dt / n * 1e3:.1f} ms/token), "
                f"positions {len(ids)}..{pos}**")
    finally:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write("\n".join(R) + "\n\n")
        print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
