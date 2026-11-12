"""Flint-40B (ternary PQ2_0 experts + Q4_1 dense parts) as a PyTorch model for TRAINING on an Arc A770 (XPU).

Same math as flint_rt.py (the OpenCL runtime): RMSNorm eps 1e-6; per-head q/k RMSNorm; neox RoPE (base 10000) on the
sliding layers only; GQA 32/4 with learned sinks; sliding window 513; dense FFN on layers 0-1; MoE layers: f32 router
-> sigmoid -> top-6 normalized x1.24, experts on Hadamard-512-rotated input (unscaled FWHT, x 1/sqrt(512)), plus a
shared expert. Frozen weights stay packed on the GPU (2-bit expert codes, Q4_1 nibbles) and are unpacked per use.
Trainable: one multiplier per expert output row (gate/up/down), starting at 1.0.

What works on this driver (R5a): one layer forward+backward at <= 256 tokens. So training is done chunk by chunk
(<= 256 tokens) and layer by layer: a no-grad forward stores every layer's input, then the backward walks the
layers in reverse, recomputing one layer at a time with grad enabled (only one layer's graph exists at a time).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flint_rt as F               # GGUF reader, PQ2_0 layout, SLIDING pattern (numpy only)

HID, NL, H, KVH, HD, WIN = 2048, 49, 32, 4, 128, 513
E, TOPK, EFF, BLK = 256, 6, 512, 128
INV512 = 512 ** -0.5
KINDS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


try:
    sys.path.insert(0, os.path.join(HERE, ".."))
    from kernels import ternary_unpack as _fk
    _TK = _fk.unpack_group_triton
    _TKI = _fk.unpack_ids_triton
    _TK_ERR = None
except Exception as _e:  # fallback: old torch unpack
    _TK = None
    _TKI = None
    print('[flint_torch_tk] WARNING: triton unpack unavailable, using torch unpack: ' + repr(_e), flush=True)

class FlintTorch:
    def __init__(self, gguf, device="xpu", dtype="float32", log=print, layers=None):
        import torch
        self.torch, self.dev, self.log = torch, torch.device(device), log
        self.dt = getattr(torch, dtype)
        mm, _, _, T = F.read_gguf(gguf)
        self.T, self.mm = T, mm
        t = self.torch

        def f32(name):
            tt = T[name]
            n = int(np.prod(tt["dims"]))
            dt = {0: np.float32, 1: np.float16}[tt["type"]]
            return t.from_numpy(np.asarray(mm[tt["off"]:tt["off"] + n * np.dtype(dt).itemsize]).view(dt)
                                .astype(np.float32).copy()).to(self.dev)

        def q41(name, rows=None):
            tt = T[name]
            K = tt["dims"][0]
            N = rows or tt["dims"][1]
            raw = np.asarray(mm[tt["off"]:tt["off"] + N * (K // 32) * 20]).reshape(N, K // 32, 20)
            d = np.ascontiguousarray(raw[..., 0:2]).view(np.float16).reshape(N, K // 32)
            m = np.ascontiguousarray(raw[..., 2:4]).view(np.float16).reshape(N, K // 32)
            q = np.ascontiguousarray(raw[..., 4:20])
            return {"q": t.from_numpy(q.copy()).to(self.dev), "d": t.from_numpy(d.copy()).to(self.dev),
                    "m": t.from_numpy(m.copy()).to(self.dev), "N": N, "K": K}

        te = T["token_embd.weight"]
        self.V = te["dims"][1]
        edt = {0: np.float32, 1: np.float16}[te["type"]]
        self.emb = np.asarray(mm[te["off"]:te["off"] + self.V * HID * np.dtype(edt).itemsize]).view(edt).reshape(self.V, HID)
        self.invf = (1.0 / (10000.0 ** (t.arange(64, dtype=t.float32) * 2.0 / HD))).to(self.dev)
        self.shifts2 = t.tensor([0, 2, 4, 6], dtype=t.int32, device=self.dev)
        self.lut4 = (((t.arange(256, dtype=t.int32).unsqueeze(-1) >> self.shifts2.cpu()) & 3) - 1).to(self.dev, self.dt)
        self.Hm = self._hadamard()
        self.L = []
        self.params = {}
        for L in range(NL):
            if layers is not None and L not in layers:
                self.L.append(None)
                continue
            b = f"blk.{L}."
            d = {"sliding": bool(F.SLIDING[L])}
            d["attn_norm"], d["ffn_norm"] = f32(b + "attn_norm.weight"), f32(b + "ffn_norm.weight")
            d["qn"], d["kn"], d["sinks"] = f32(b + "attn_q_norm.weight"), f32(b + "attn_k_norm.weight"), \
                f32(b + "attn_sinks.weight")
            d["q"], d["k"], d["v"] = q41(b + "attn_q.weight", 4096), q41(b + "attn_k.weight", 512), \
                q41(b + "attn_v.weight", 512)
            d["o"] = q41(b + "attn_output.weight", 2048)
            if L < 2:
                d["gate"], d["up"] = q41(b + "ffn_gate.weight", 6144), q41(b + "ffn_up.weight", 6144)
                d["down"] = q41(b + "ffn_down.weight", 2048)
            else:
                rt = T[b + "ffn_gate_inp.weight"]
                d["router"] = t.from_numpy(np.asarray(mm[rt["off"]:rt["off"] + 256 * HID * 4]).view(np.float32)
                                           .reshape(256, HID).copy()).to(self.dev)
                d["gate"], d["up"] = q41(b + "ffn_gate_shexp.weight", 1024), q41(b + "ffn_up_shexp.weight", 1024)
                d["down"] = q41(b + "ffn_down_shexp.weight", 2048)
                for kind in KINDS:
                    raw = F.raw_expert_tensor(mm, T, L, kind)                     # [E,N,G,34]
                    sc = np.ascontiguousarray(raw[..., :2]).view(np.float16).reshape(raw.shape[:3])
                    d[kind] = (t.from_numpy(np.ascontiguousarray(F.LUT_CLAMP[raw[..., 2:]])).to(self.dev),
                               t.from_numpy(sc.copy()).to(self.dev))
                    N = raw.shape[1]
                    p = t.nn.Parameter(t.ones(E, N, 1, device=self.dev))
                    self.params[f"L{L}.{kind}"] = p
                    d["mult_" + kind] = p
            self.L.append(d)
            if L % 8 == 0:
                log(f"  layer {L} on the GPU")
        self.out_norm = f32("output_norm.weight")
        self.head = q41("output.weight", self.V)

    # ------------------------------------------------------------------ unpacking
    def _hadamard(self):
        t = self.torch
        Hm = t.tensor([[1.0]])
        while Hm.shape[0] < 512:
            Hm = t.cat([t.cat([Hm, Hm], 1), t.cat([Hm, -Hm], 1)], 0)
        return Hm.to(self.dev, self.dt)

    def deq41(self, w, r0=0, r1=None):
        """Q4_1 rows r0..r1: byte j of a 32-block = weight j (low nibble) and weight j+16 (high nibble); w = q*d + m"""
        t = self.torch
        r1 = w["N"] if r1 is None else r1
        q = w["q"][r0:r1].to(t.int32)
        lo, hi = (q & 15).to(self.dt), (q >> 4).to(self.dt)
        vals = t.cat([lo, hi], -1)                                                # [rows, G, 32]
        out = vals * w["d"][r0:r1].unsqueeze(-1).to(self.dt) + w["m"][r0:r1].unsqueeze(-1).to(self.dt)
        return out.reshape(r1 - r0, w["K"])

    def _make_explin(self):
        """y = x @ (W * mult)^T for one expert matrix, where W (unpacked 2-bit codes x block scales) is NOT kept for
        the backward pass: it is unpacked again there. Saving W for 250+ experts per layer filled the 16 GB card (T2)."""
        t = self.torch

        class ExpLin(t.autograd.Function):
            # mult scales whole output rows, so y = (x @ W^T) * mult^T. Keeping z = x @ W^T (tokens x rows, small)
            # makes the multiplier gradient free in backward (it used to redo the matmul) and spares the full-size
            # W * mult temporary in forward. Only the multipliers train, so no weight gradient is ever needed.
            @staticmethod
            def forward(ctx, x, mult, wfn):
                W = wfn()
                ctx.wfn = wfn
                z = x @ W.t()
                ctx.save_for_backward(z, mult)
                return z * mult.t().to(z.dtype)

            @staticmethod
            def backward(ctx, gy):
                z, mult = ctx.saved_tensors
                W = ctx.wfn()
                gx = (gy * mult.t().to(gy.dtype)) @ W
                gm = (gy.float() * z.float()).sum(0).unsqueeze(-1)
                return gx, gm.to(mult.dtype), None
        return ExpLin.apply

    def _make_explin_group(self):
        """grouped version: x [g,T,K] (one row of T padded tokens per expert), mult [g,N,1], wfn() -> W [g,N,K].
        Same idea as ExpLin: W is unpacked again in backward, z = x @ W^T is kept (small) for the multiplier grad."""
        t = self.torch

        class ExpLinG(t.autograd.Function):
            @staticmethod
            def forward(ctx, x, mult, wfn):
                W = wfn()
                ctx.wfn = wfn
                z = t.bmm(x, W.transpose(1, 2))
                ctx.save_for_backward(z, mult)
                return z * mult.transpose(1, 2).to(z.dtype)

            @staticmethod
            def backward(ctx, gy):
                z, mult = ctx.saved_tensors
                W = ctx.wfn()
                gx = t.bmm(gy * mult.transpose(1, 2).to(gy.dtype), W)
                gm = (gy.float() * z.float()).sum(1).unsqueeze(-1)
                return gx, gm.to(mult.dtype), None
        return ExpLinG.apply

    def wbase_group(self, d, kind, e0, e1):
        """unpacked expert matrices e0..e1-1 (a contiguous slice: no gather, slicing has a safe backward)"""
        codes, sc = d[kind]
        if _TK is not None and self.dt == self.torch.float32 and codes.is_contiguous():
            return _TK(codes, sc, e0, e1, variant='reshape', BR=16, num_warps=4)   # one-pass Triton unpack, bit-identical
        codes, sc = d[kind]
        v = ((codes[e0:e1].to(self.torch.int32).unsqueeze(-1) >> self.shifts2) & 3).to(self.dt) - 1   # [g,N,G,32,4] (S1: shifts, not a 16-bit LUT gather)
        g, N, G = e1 - e0, codes.shape[1], codes.shape[2]
        return (v.reshape(g, N, G, BLK) * sc[e0:e1].unsqueeze(-1).to(self.dt)).reshape(g, N, G * BLK)

    def wbase(self, d, kind, e):
        """unpacked expert matrix without the trainable multiplier"""
        t = self.torch
        codes, sc = d[kind]
        v = ((codes[e].to(t.int32).unsqueeze(-1) >> self.shifts2) & 3).to(self.dt) - 1   # [N,G,32,4] codes -> -1/0/+1
        N, G = codes.shape[1], codes.shape[2]
        return (v.reshape(N, G, BLK) * sc[e].unsqueeze(-1).to(self.dt)).reshape(N, G * BLK)

    def explin(self, x, d, kind, e):
        if not hasattr(self, "_explin"):
            self._explin = self._make_explin()
        return self._explin(x, d["mult_" + kind][e], lambda: self.wbase(d, kind, e))

    def deq_expert(self, d, kind, e):
        t = self.torch
        codes, sc = d[kind]
        v = ((codes[e].to(t.int32).unsqueeze(-1) >> self.shifts2) & 3).to(self.dt) - 1   # [N,G,32,4]
        N, G = codes.shape[1], codes.shape[2]
        s = sc[e].to(t.float32) * d["mult_" + kind][e]                                   # [N,G] x [N,1]
        return (v.reshape(N, G, BLK) * s.unsqueeze(-1).to(self.dt)).reshape(N, G * BLK)

    # ------------------------------------------------------------------ pieces
    def rms(self, x, w, eps=1e-6):
        t = self.torch
        xf = x.float()
        return xf * t.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w

    def h512(self, x):
        return (x.reshape(-1, 512) @ self.Hm).reshape(x.shape)

    def rope(self, x, pos):                                                       # x [T, heads, 128], pos [T]
        t = self.torch
        ang = pos.float().unsqueeze(-1) * self.invf                               # [T, 64]
        c, s = t.cos(ang).unsqueeze(1), t.sin(ang).unsqueeze(1)
        a, b = x[..., :64], x[..., 64:]
        return t.cat([a * c - b * s, b * c + a * s], -1)

    def attention(self, d, xn, pos, past):
        """xn [T,2048] normed input, pos [T] absolute positions, past = (k [S,4,128], v, kpos [S]) or None.
        Returns (out [T,2048], (k, v, kpos) of THIS chunk, to append to the cache by the caller)."""
        t = self.torch
        import torch.utils.checkpoint
        T_ = xn.shape[0]
        xb = xn.to(self.dt)
        q = (xb @ self.deq41(d["q"]).t()).float().view(T_, H, HD)
        k = (xb @ self.deq41(d["k"]).t()).float().view(T_, KVH, HD)
        v = (xb @ self.deq41(d["v"]).t()).float().view(T_, KVH, HD)
        q = q * t.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * d["qn"]
        k = k * t.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * d["kn"]
        if d["sliding"]:
            q, k = self.rope(q, pos), self.rope(k, pos)
        if past is not None:
            K, Vv, kpos = t.cat([past[0], k], 0), t.cat([past[1], v], 0), t.cat([past[2], pos], 0)
        else:
            K, Vv, kpos = k, v, pos
        qg = q.view(T_, KVH, 8, HD).permute(1, 2, 0, 3)                         # [4, 8, T, 128]
        kT = K.permute(1, 2, 0).unsqueeze(1)                                     # [4, 1, 128, S]
        Vp = Vv.permute(1, 0, 2).unsqueeze(1)                                    # [4, 1, S, 128]
        sink_w = d["sinks"].view(KVH, 8, 1, 1)
        S_ = K.shape[0]
        QB = self.att_qb(T_, S_)
        if QB >= T_:
            o = self._att_block(qg, kT, Vp, kpos, pos, sink_w, d["sliding"])
        else:
            use_ckpt = t.is_grad_enabled() and (qg.requires_grad or kT.requires_grad)
            outs = []
            for q0 in range(0, T_, QB):
                q1 = min(q0 + QB, T_)
                args = (qg[:, :, q0:q1], kT, Vp, kpos, pos[q0:q1], sink_w, d["sliding"])
                if use_ckpt:        # K5: keep only the block inputs; the score block is recomputed in backward
                    outs.append(t.utils.checkpoint.checkpoint(self._att_block, *args, use_reentrant=False,
                                                              preserve_rng_state=False))
                else:
                    outs.append(self._att_block(*args))
            o = t.cat(outs, 2)
        o = o.permute(2, 0, 1, 3).reshape(T_, H * HD)
        out = (o.to(self.dt) @ self.deq41(d["o"]).t()).float()
        return out, (k.detach(), v.detach(), pos)

    ATT_QB = 256                # K5: query rows per attention block (0 = unblocked, the K4 path)
    ATT_FULL = 32 * 2 ** 20     # K5: score grids up to this many elements (256 MB fp32) stay unblocked (faster)
    ATT_BUDGET = 32 * 2 ** 20   # K5: cap on QB*S*32 score elements per block (shrinks QB for long keys)

    def att_qb(self, T_, S_):
        if not self.ATT_QB or 32 * T_ * S_ <= self.ATT_FULL:
            return T_
        qb = self.ATT_QB
        while qb > 32 and qb * S_ * 32 > self.ATT_BUDGET:
            qb //= 2
        return qb

    def _att_block(self, qb, kT, Vp, kpos, pb, sink_w, sliding):
        """scores -> masked softmax with the sink -> output for one block of query rows (fp32). Rows are independent,
        so blocking is exact; masks are built per block with masked_fill (no boolean-mask indexing)."""
        t = self.torch
        sc = (qb @ kT) * (HD ** -0.5)                                            # [4, 8, QB, S]
        allowed = kpos.unsqueeze(0) <= pb.unsqueeze(1)                           # [QB, S] causal
        if sliding:
            allowed = allowed & (kpos.unsqueeze(0) > pb.unsqueeze(1) - WIN)
        sc = sc.masked_fill(~allowed, float("-inf"))
        sink = sink_w.expand(KVH, 8, qb.shape[2], 1)
        p = t.softmax(t.cat([sc, sink], -1), -1)[..., :-1]                       # sink takes part, then drops out
        return p @ Vp

    def ffn_dense(self, d, xn):
        Fn = self.torch.nn.functional
        xb = xn.to(self.dt)
        g, u = xb @ self.deq41(d["gate"]).t(), xb @ self.deq41(d["up"]).t()
        return (Fn.silu(g.float()) * u.float()).to(self.dt) @ self.deq41(d["down"]).t()

    MT_BUCKET = 16          # padded tokens per expert are rounded up to this (zeros, route weight 0)
    MOE_GROUP = 16          # experts per batched launch (0 = the old one-expert-at-a-time loop)

    def moe(self, d, xn):
        return self.moe_grouped(d, xn) if self.MOE_GROUP else self.moe_loop(d, xn)

    def route(self, d, xn):
        t = self.torch
        with t.no_grad():
            p = t.sigmoid(xn.float() @ d["router"].t())
            pc = p.cpu().numpy()
        idx = np.argpartition(-pc, TOPK - 1, axis=1)[:, :TOPK]
        w = np.take_along_axis(pc, idx, 1)
        return idx, w / (w.sum(1, keepdims=True) + 1e-20) * 1.24

    memprobe = None
    MOE_ACC = False             # K5: accumulate group outputs with index_add per group (tiny summation-order change)
    _rc = None                  # K4: per-layer routing cache of the current train_chunk (None = off)
    _rcphase = None
    _cur_L = -1
    last_rc_bytes = 0

    def wbase_ids(self, d, kind, eids):
        """unpacked fp32 expert matrices for the expert ids in eids (int64 tensor [g]); bit-identical to wbase"""
        codes, sc = d[kind]
        if _TKI is not None and self.dt == self.torch.float32 and codes.is_contiguous():
            return _TKI(codes, sc, eids, BR=16, num_warps=4)
        t = self.torch
        v = ((codes.index_select(0, eids).to(t.int32).unsqueeze(-1) >> self.shifts2) & 3).to(self.dt) - 1
        g, N, G = eids.shape[0], codes.shape[1], codes.shape[2]
        return (v.reshape(g, N, G, BLK) * sc.index_select(0, eids).unsqueeze(-1).to(self.dt)).reshape(g, N, G * BLK)

    def _make_explin_k4(self):
        """y = (x @ W^T) * mult[eids]. W is unpacked again in backward. The multiplier gradient is accumulated straight
        into the parameter's .grad with index_add_ (expert ids are unique), so no E-sized gradient is built per group."""
        t = self.torch

        class ExpLinK(t.autograd.Function):
            @staticmethod
            def forward(ctx, x, multp, eids, wfn):
                W = wfn()
                ctx.wfn = wfn
                z = t.bmm(x, W.transpose(1, 2))
                mult = multp.detach().index_select(0, eids)
                ctx.p, ctx.eids = multp, eids
                ctx.save_for_backward(z, mult)
                return z * mult.transpose(1, 2).to(z.dtype)

            @staticmethod
            def backward(ctx, gy):
                z, mult = ctx.saved_tensors
                W = ctx.wfn()
                gx = t.bmm(gy * mult.transpose(1, 2).to(gy.dtype), W)
                gm = (gy.float() * z.float()).sum(1).unsqueeze(-1)
                p = ctx.p
                if p.grad is None:
                    p.grad = t.zeros_like(p)
                p.grad.index_add_(0, ctx.eids, gm.to(p.dtype))
                return gx, None, None, None
        return ExpLinK.apply

    def _build_route(self, d, xn):
        """routing + expert-sorted group index tensors of one layer, uploaded with ONE host->GPU copy.
        Experts are ordered by token count (descending), empty ones skipped, groups of MOE_GROUP consecutive experts,
        padded size per group = the largest member's count rounded up to MT_BUCKET."""
        t = self.torch
        idx, w = self.route(d, xn)
        ef = idx.ravel()
        order = np.argsort(ef, kind="stable")
        tok_sorted, w_sorted = (order // TOPK), w.ravel()[order].astype(np.float32)
        counts = np.bincount(ef, minlength=E)
        starts = np.concatenate([[0], np.cumsum(counts)])
        nz = np.nonzero(counts)[0]
        nz = nz[np.argsort(-counts[nz], kind="stable")]
        GE = self.MOE_GROUP
        groups, pp, ww, ee = [], [], [], []
        p0 = 0
        for s in range(0, len(nz), GE):
            ids = nz[s:s + GE]
            g = len(ids)
            mt = -(-int(counts[ids[0]]) // self.MT_BUCKET) * self.MT_BUCKET
            P = np.zeros((g, mt), np.int64)
            Wt = np.zeros((g, mt), np.float32)
            for j, e in enumerate(ids):
                a, b = starts[e], starts[e + 1]
                P[j, :b - a], Wt[j, :b - a] = tok_sorted[a:b], w_sorted[a:b]
            groups.append((p0, g, mt, s))
            pp.append(P.ravel()); ww.append(Wt.ravel()); ee.append(ids.astype(np.int64))
            p0 += g * mt
        n_p = p0
        wcat = np.concatenate(ww)
        nw2 = (wcat.size + 1) // 2
        wpad = np.zeros(nw2 * 2, np.float32)
        wpad[:wcat.size] = wcat
        ecat = np.concatenate(ee)
        buf = np.empty(n_p + nw2 + ecat.size, np.int64)
        buf[:n_p] = np.concatenate(pp)
        buf[n_p:n_p + nw2] = wpad.view(np.int64)
        buf[n_p + nw2:] = ecat
        db = t.from_numpy(buf).to(self.dev)                                          # the single H2D copy
        return {"groups": groups, "P": db[:n_p], "W": db[n_p:n_p + nw2].view(t.float32)[:wcat.size],
                "E": db[n_p + nw2:], "bytes": int(buf.nbytes)}

    def moe_grouped(self, d, xn):
        """K4: expert-sorted groups (little padding, few launches), Triton unpack by expert id, one H2D per layer,
        routing computed once in the no-grad forward and reused by the recompute"""
        t = self.torch
        Fn = t.nn.functional
        if not hasattr(self, "_explin_k"):
            self._explin_k = self._make_explin_k4()
        T = xn.shape[0]
        rc = self._rc
        prep = None
        if rc is not None and self._rcphase == "reuse":
            prep = rc.pop(self._cur_L, None)
        if prep is None:
            prep = self._build_route(d, xn)
            if rc is not None and self._rcphase == "store":
                rc[self._cur_L] = prep
        xh = self.h512(xn.to(self.dt))
        Pall, Wall, Eall = prep["P"], prep["W"], prep["E"]
        vals = []
        acc = t.zeros(T, HID, device=self.dev) if self.MOE_ACC else None
        mp = self.memprobe
        if mp: mp(f'L{self._cur_L} moe start rows={Pall.shape[0]} groups={[(g, mt) for _, g, mt, _ in prep["groups"]]}')
        for p0, g, mt, s in prep["groups"]:
            Pt = Pall[p0:p0 + g * mt]
            Wg = Wall[p0:p0 + g * mt].view(g, mt, 1)
            eid = Eall[s:s + g]
            xe = xh.index_select(0, Pt).view(g, mt, -1)
            mk = lambda kind, eid=eid: (lambda: self.wbase_ids(d, kind, eid))          # bind now: runs again in backward
            gt = self._explin_k(xe, d["mult_ffn_gate_exps"], eid, mk("ffn_gate_exps")) * INV512
            u = self._explin_k(xe, d["mult_ffn_up_exps"], eid, mk("ffn_up_exps")) * INV512
            act = (Fn.silu(gt.float()) * u.float()).to(self.dt)
            y = self._explin_k(self.h512(act), d["mult_ffn_down_exps"], eid, mk("ffn_down_exps")) * INV512
            v = (y.float() * Wg).reshape(-1, HID)
            if acc is not None:      # K5: add each group straight into the output, no list of group outputs + cat
                acc = acc.index_add(0, Pt, v)
            else:
                vals.append(v)
            if mp: mp(f'  group g={g} mt={mt} done')
        r = acc if acc is not None else t.zeros(T, HID, device=self.dev).index_add(0, Pall, t.cat(vals))
        if mp: mp('  moe cat+index_add done')
        return r

    def moe_loop(self, d, xn):
        t = self.torch
        Fn = t.nn.functional
        with t.no_grad():
            p = t.sigmoid(xn.float() @ d["router"].t())
            pc = p.cpu().numpy()
        idx = np.argpartition(-pc, TOPK - 1, axis=1)[:, :TOPK]
        w = np.take_along_axis(pc, idx, 1)
        w = w / (w.sum(1, keepdims=True) + 1e-20) * 1.24
        xh = self.h512(xn.to(self.dt))
        toks, vals = [], []
        for e in np.unique(idx).tolist():
            tok_np, slot_np = np.nonzero(idx == e)
            tok = t.from_numpy(tok_np).to(self.dev)
            xe = xh.index_select(0, tok)
            gt = self.explin(xe, d, "ffn_gate_exps", e) * INV512
            u = self.explin(xe, d, "ffn_up_exps", e) * INV512
            act = (Fn.silu(gt.float()) * u.float()).to(self.dt)
            y = self.explin(self.h512(act), d, "ffn_down_exps", e) * INV512
            we = t.from_numpy(w[tok_np, slot_np].astype(np.float32)).to(self.dev).unsqueeze(-1)
            toks.append(tok)
            vals.append(y.float() * we)
        tok_all, val_all = t.cat(toks), t.cat(vals)
        return t.zeros(xn.shape[0], HID, device=self.dev).index_add(0, tok_all, val_all)

    trace = None                                                                  # set to a function to get markers

    def _tr(self, msg):
        if self.trace:
            self.torch.xpu.synchronize()
            self.trace(f"{msg} (GPU {self.torch.xpu.memory_allocated() / 2**30:.2f} GiB, "
                       f"reserved {self.torch.xpu.memory_reserved() / 2**30:.2f} GiB)")

    def layer(self, L, x, pos, past):
        """one transformer layer: x [T,2048] f32 residual -> (x_out, kv of this chunk)"""
        self._cur_L = L
        d = self.L[L]
        a, kv = self.attention(d, self.rms(x, d["attn_norm"]), pos, past)
        self._tr(f"  layer {L} attention")
        x = x + a
        xn = self.rms(x, d["ffn_norm"])
        f = self.ffn_dense(d, xn).float()
        self._tr(f"  layer {L} dense/shared ffn")
        if L >= 2:
            f = f + self.moe(d, xn)
            self._tr(f"  layer {L} moe")
        return x + f, kv

    def embed(self, ids):
        return self.torch.from_numpy(self.emb[np.asarray(ids)].astype(np.float32)).to(self.dev)

    HEAD_SLICE = 16384                                                            # vocab rows per head slice

    def logits(self, x):
        """no-grad logits, computed in vocab slices (the full float32 head would need ~1.2 GB at once)"""
        t = self.torch
        with t.no_grad():
            xn = self.rms(x, self.out_norm).to(self.dt)
            return t.cat([(xn @ self.deq41(self.head, r0, min(r0 + self.HEAD_SLICE, self.V)).t()).float()
                          for r0 in range(0, self.V, self.HEAD_SLICE)], -1)

    def deq41_rows(self, w, rows):
        """Q4_1 dequant of selected rows (index_select only: index assignment crashed this driver)"""
        t = self.torch
        q = w["q"].index_select(0, rows).to(t.int32)
        vals = t.cat([(q & 15).to(self.dt), (q >> 4).to(self.dt)], -1)
        out = vals * w["d"].index_select(0, rows).unsqueeze(-1).to(self.dt) \
            + w["m"].index_select(0, rows).unsqueeze(-1).to(self.dt)
        return out.reshape(rows.shape[0], w["K"])

    def head_loss_grad(self, x, tgt, msk):
        """masked mean cross-entropy of the next-token prediction and its gradient w.r.t. x (the final hidden state),
        done by hand in vocab slices so the head is never held whole: pass 1 = log-sum-exp, pass 2 = softmax @ W,
        minus the target rows (gathered with index_select; no index assignment, which crashed this driver)."""
        t = self.torch
        n = msk.sum()
        xr = x.detach().requires_grad_(True)
        xn = self.rms(xr, self.out_norm)                                          # norm keeps autograd (cheap)
        xd = xn.detach().to(self.dt)
        with t.no_grad():
            Wt = self.deq41_rows(self.head, tgt)                                  # [T, 2048] target rows
            tl = (xd.float() * Wt.float()).sum(-1)                                # target logits
            m = t.full((x.shape[0],), -float("inf"), device=self.dev)
            se = t.zeros(x.shape[0], device=self.dev)
            for r0 in range(0, self.V, self.HEAD_SLICE):
                lg = (xd @ self.deq41(self.head, r0, min(r0 + self.HEAD_SLICE, self.V)).t()).float()
                mn = t.maximum(m, lg.max(-1).values)
                se = se * t.exp(m - mn) + t.exp(lg - mn.unsqueeze(-1)).sum(-1)
                m = mn
            lse = m + t.log(se)
            loss = ((lse - tl) * msk).sum() / n
            wgt = (msk / n).unsqueeze(-1)
            gxn = -(wgt * Wt.float())                                             # the "- one-hot" part
            for r0 in range(0, self.V, self.HEAD_SLICE):
                W = self.deq41(self.head, r0, min(r0 + self.HEAD_SLICE, self.V))
                p = t.exp((xd @ W.t()).float() - lse.unsqueeze(-1))
                gxn += ((p * wgt).to(self.dt) @ W).float()
        xn.backward(gxn)                                                          # through the final RMSNorm
        return float(loss.item()), xr.grad

    # ------------------------------------------------------------------ forward / train step
    def forward_chunk(self, ids, pos0, cache):
        """no-grad forward of one chunk; cache = list per layer of (k, v, kpos) or None; returns (inputs per layer,
        final hidden). The cache is extended in place (sliding layers keep only their last WIN-1 positions)."""
        t = self.torch
        pos = t.arange(pos0, pos0 + len(ids), device=self.dev)
        x = self.embed(ids)
        ins = []
        with t.no_grad():
            for L in range(NL):
                ins.append(x if self.INS_DT is None else x.to(self.INS_DT))   # T4: optional half-precision store
                x, kv = self.layer(L, x, pos, cache[L])
                cache[L] = self._append(cache[L], kv, self.L[L]["sliding"])
        return ins, x, pos

    INS_DT = None               # T4: dtype the stored layer inputs are kept in (None = float32, exact)

    def _append(self, old, kv, sliding):
        t = self.torch
        if old is None:
            k, v, p = kv
        else:
            k, v, p = t.cat([old[0], kv[0]]), t.cat([old[1], kv[1]]), t.cat([old[2], kv[2]])
        if sliding and k.shape[0] > WIN - 1:
            k, v, p = k[-(WIN - 1):], v[-(WIN - 1):], p[-(WIN - 1):]
        return (k, v, p)

    def head_loss(self, x, tgt, msk):
        """masked mean next-token cross-entropy, no gradient (for validation)"""
        t = self.torch
        with t.no_grad():
            xd = self.rms(x, self.out_norm).to(self.dt)
            tl = (xd.float() * self.deq41_rows(self.head, tgt).float()).sum(-1)
            m = t.full((x.shape[0],), -float("inf"), device=self.dev)
            se = t.zeros(x.shape[0], device=self.dev)
            for r0 in range(0, self.V, self.HEAD_SLICE):
                lg = (xd @ self.deq41(self.head, r0, min(r0 + self.HEAD_SLICE, self.V)).t()).float()
                mn = t.maximum(m, lg.max(-1).values)
                se = se * t.exp(m - mn) + t.exp(lg - mn.unsqueeze(-1)).sum(-1)
                m = mn
            return float((((m + t.log(se)) - tl) * msk).sum().item()), float(msk.sum().item())

    def eval_example(self, ids, prompt_tokens, chunk):
        """(summed loss, counted tokens) on the answer part of one example; forward only"""
        t = self.torch
        tg = ids[1:] + [0]
        mask = [1.0 if (i + 1 >= prompt_tokens and i < len(ids) - 1) else 0.0 for i in range(len(ids))]
        cache, tot, cnt = [None] * NL, 0.0, 0.0
        for s in range(0, len(ids), chunk):
            sl = slice(s, s + chunk)
            _, x, _ = self.forward_chunk(ids[sl], s, cache)
            if sum(mask[sl]):
                l, n = self.head_loss(x, t.from_numpy(np.asarray(tg[sl], np.int64)).to(self.dev),
                                      t.from_numpy(np.asarray(mask[sl], np.float32)).to(self.dev))
                tot, cnt = tot + l, cnt + n
        return tot, cnt

    def train_chunk(self, ids, targets, mask, pos0, cache, trace=None):
        """forward + layer-by-layer backward of one chunk (<= 256 tokens). targets/mask: next-token ids and 0/1 loss
        mask per position. The cache holds the PAST chunks only; it is extended at the end. Returns the mean loss."""
        t = self.torch
        past = list(cache)
        self._tr("chunk: no-grad forward starts")
        self._rc, self._rcphase = {}, "store"                  # K4: keep each layer's routing for the recompute
        ins, xL, pos = self.forward_chunk(ids, pos0, cache)       # also extends `cache` for the next chunk
        self._rcphase = "reuse"
        self.last_rc_bytes = sum(v["bytes"] for v in self._rc.values())
        self._tr("chunk: no-grad forward done")
        tgt = t.from_numpy(np.asarray(targets, np.int64)).to(self.dev)
        msk = t.from_numpy(np.asarray(mask, np.float32)).to(self.dev)
        n = float(msk.sum().item())
        if n == 0:
            self._rc = None
            return None
        loss, g = self.head_loss_grad(xL, tgt, msk)
        self._tr("chunk: head loss + gradient done")
        first = min(L for L in range(NL) if self.L[L] is not None and "mult_ffn_gate_exps" in self.L[L])
        for L in reversed(range(first, NL)):                 # layers below the first trainable one need no backward
            x_in = ins[L].detach().float().requires_grad_(L > first)
            out, _ = self.layer(L, x_in, pos, past[L])
            self._tr(f"backward layer {L}: recomputed")
            out.backward(g)
            g = x_in.grad if L > first else None
            self._tr(f"backward layer {L}: done")
        self._rc = None
        return loss
