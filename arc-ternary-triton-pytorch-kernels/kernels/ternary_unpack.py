"""Triton unpack kernels for PQ2_0 ternary expert weights (Intel Arc / XPU, Windows or Linux).

PQ2_0 code layout. One expert matrix W[e, n, :] (output row n of expert e, K = G*128 input columns) is stored as
  codes : uint8   [E, N, G, 32]   four 2-bit codes per byte
  sc    : float16 [E, N, G]       one scale per block of 128 columns
and decodes as

    W[e, n, g*128 + j*4 + s] = (((codes[e, n, g, j] >> (2*s)) & 3) - 1) * sc[e, n, g]      j in 0..31, s in 0..3

so every 2-bit code maps to -1, 0, +1 (code 3 would be +2 and does not occur in valid data), times the block scale.
Output is fp32 and bit-identical to the plain torch reference (shift, mask, subtract 1, multiply by the fp16 scale
converted to fp32). One pass reads the packed bytes and writes the fp32 matrix.

Two entry points:
  unpack_group_triton(codes, sc, e0, e1)   contiguous expert range e0..e1-1       -> fp32 [e1-e0, N, K]
  unpack_ids_triton(codes, sc, eids)       arbitrary expert ids, gathered in-kernel -> fp32 [len(eids), N, K]
codes must be contiguous.
"""
import torch, triton, triton.language as tl

# ---------------------------------------------------------------- unpack: codes+scales -> fp32 W in one pass
@triton.jit
def _unpack_gather_k(codes, sc, out, R, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < R
    col = tl.arange(0, 128)
    b = tl.load(codes + rows[:, None] * 32 + (col // 4)[None, :], mask=rm[:, None], other=0).to(tl.int32)
    c = ((b >> ((col % 4) * 2)[None, :]) & 3) - 1
    s = tl.load(sc + rows, mask=rm, other=0.0).to(tl.float32)
    tl.store(out + rows[:, None] * 128 + col[None, :], c.to(tl.float32) * s[:, None], mask=rm[:, None])


@triton.jit
def _unpack_reshape_k(codes, sc, out, R, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < R
    sh = tl.arange(0, 4) * 2
    b = tl.load(codes + rows[:, None] * 32 + tl.arange(0, 32)[None, :], mask=rm[:, None], other=0).to(tl.int32)
    c = tl.reshape(((b[:, :, None] >> sh[None, None, :]) & 3) - 1, (BR, 128))
    s = tl.load(sc + rows, mask=rm, other=0.0).to(tl.float32)
    col = tl.arange(0, 128)
    tl.store(out + rows[:, None] * 128 + col[None, :], c.to(tl.float32) * s[:, None], mask=rm[:, None])


def unpack_group_triton(codes, sc, e0=0, e1=None, out=None, variant="gather", BR=32, num_warps=4):
    """== wbase_group(d, kind, e0, e1) (fp32): codes uint8 [E,N,G,32], sc fp16 [E,N,G] -> W fp32 [g,N,G*128].
    out: optional preallocated fp32 buffer with >= g*N*G*128 elements (a view of it is returned)."""
    e1 = codes.shape[0] if e1 is None else e1
    c = codes[e0:e1]
    s = sc[e0:e1]
    g, N, G, _ = c.shape
    R = g * N * G
    if out is None:
        o = torch.empty(g, N, G * 128, dtype=torch.float32, device=codes.device)
    else:
        o = out.view(-1)[:R * 128].view(g, N, G * 128)
    k = _unpack_gather_k if variant == "gather" else _unpack_reshape_k
    k[(triton.cdiv(R, BR),)](c, s, o, R, BR=BR, num_warps=num_warps)
    return o


# ---------------------------------------------------------------- unpack with an expert-id list (gather inside the kernel)
@triton.jit
def _unpack_ids_k(codes, sc, eids, out, R, NG, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < R
    i = rows // NG
    rem = rows - i * NG
    e = tl.load(eids + i, mask=rm, other=0).to(tl.int64)
    srow = e * NG + rem                                   # row (expert, n, g) in the [E*N*G] layout
    sh = tl.arange(0, 4) * 2
    b = tl.load(codes + srow[:, None] * 32 + tl.arange(0, 32)[None, :], mask=rm[:, None], other=0).to(tl.int32)
    c = tl.reshape(((b[:, :, None] >> sh[None, None, :]) & 3) - 1, (BR, 128))
    s = tl.load(sc + srow, mask=rm, other=0.0).to(tl.float32)
    col = tl.arange(0, 128)
    tl.store(out + rows[:, None] * 128 + col[None, :], c.to(tl.float32) * s[:, None], mask=rm[:, None])


def unpack_ids_triton(codes, sc, eids, BR=16, num_warps=4):
    """W[i] = unpack(codes[eids[i]], sc[eids[i]]) as fp32 [g, N, G*128]; bit-identical to wbase for each id.
    codes uint8 [E,N,G,32] (contiguous), sc fp16 [E,N,G], eids int32/int64 [g] on the same device."""
    E, N, G, _ = codes.shape
    g = eids.shape[0]
    NG = N * G
    R = g * NG
    o = torch.empty(g, N, G * 128, dtype=torch.float32, device=codes.device)
    _unpack_ids_k[(triton.cdiv(R, BR),)](codes, sc, eids, o, R, NG, BR=BR, num_warps=num_warps)
    return o
