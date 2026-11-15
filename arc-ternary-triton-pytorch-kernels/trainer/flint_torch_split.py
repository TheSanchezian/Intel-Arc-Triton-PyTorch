"""Two-GPU model split (pipeline) of the Flint-40B trainer, built AROUND flint_torch.FlintTorch (which is not modified).

Layout (K = split point, chosen by auto_split_at() unless given):
  device 0 : token embedding lookup (the embedding table itself stays in host RAM, as in FlintTorch) + layers 0..K-1
  device 1 : layers K..48 + final RMSNorm + output head + loss
Two FlintTorch instances are built, one per half (FlintTorch(layers=...) skips the other half's layers). Each half owns its
frozen packed weights, its trainable multipliers (fp32), their grads and Adam state, its KV cache and its routing cache.
Only the hidden state [T,2048] fp32 crosses the boundary: forward activation device0 -> device1 and its gradient back,
through host memory (x.cpu().to(dev)): a blocking D2H copy also guarantees every kernel that produced x has finished,
so no cross-device P2P / cross-queue ordering is relied on (torch.xpu.synchronize does not wait for Triton kernels).

Device handling: every call into a half runs under torch.xpu.device(idx) so Triton kernels (flint_kernels.*) launch on
that half's device; all tensors in FlintTorch are created with an explicit device=self.dev, flint_kernels allocates its
outputs with device=codes.device and keeps no module-level buffers/streams, so no change to those files is needed.
The backward of a custom autograd.Function (wfn() -> Triton unpack) runs on the autograd worker of the node's stream
device, which is the half's device.

The class mirrors the FlintTorch API that flint_train_p2.py uses: params, forward_chunk, eval_example, train_chunk and
the tunables MOE_GROUP / MT_BUCKET / ATT_QB / HEAD_SLICE / INS_DT / trace (setting them propagates to both halves).
"""
import contextlib
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint_rt as F
import flint_torch as FT

NL = FT.NL
HID, KVH, HD = FT.HID, FT.KVH, FT.HD
GiB = 2 ** 30
MULT_ADAM_BYTES = 12 * 2 ** 20          # per MoE layer: 3 multiplier tensors (3.0 MB fp32) x (param + grad + 2 Adam moments)


def layer_weight_bytes(gguf):
    """bytes of each layer's tensors in the GGUF (= packed bytes on the GPU) and of the head (output + output_norm)"""
    mm, _, _, T = F.read_gguf(gguf)
    items = sorted((t["off"], n) for n, t in T.items())
    sizes = {}
    for i, (off, n) in enumerate(items):
        end = items[i + 1][0] if i + 1 < len(items) else len(mm)
        sizes[n] = end - off
    per = [0] * NL
    for n, s in sizes.items():
        if n.startswith("blk."):
            per[int(n.split(".")[1])] += s
    head = sizes.get("output.weight", 0) + sizes.get("output_norm.weight", 0)
    return per, head


def auto_split_at(gguf, kv_tokens=16384, chunk=1024, head_extra=0.5 * GiB):
    """K that minimises the larger of the two devices' estimated footprints. Per layer: packed weights + stored layer input
    of one chunk + (MoE) multipliers/grads/Adam + (full-attention layers) KV cache for kv_tokens tokens. Device 1 also
    carries the head and head_extra (the sliced head-loss temporaries). Returns (K, info)."""
    per, head = layer_weight_bytes(gguf)
    cost = []
    for L in range(NL):
        c = per[L] + chunk * HID * 4
        if L >= 2:
            c += MULT_ADAM_BYTES
        if not F.SLIDING[L]:
            c += kv_tokens * KVH * HD * 4 * 2
        cost.append(c)
    best = None
    for K in range(3, NL - 1):                       # device 0 keeps >= layers 0..2 (it must own a trainable layer)
        a, b = sum(cost[:K]), sum(cost[K:]) + head + head_extra
        if best is None or max(a, b) < best[0]:
            best = (max(a, b), K, a, b)
    return best[1], {"dev0_gib": best[2] / GiB, "dev1_gib": best[3] / GiB, "head_gib": head / GiB,
                     "layer_gib": [round(x / GiB, 4) for x in per[:3]]}


class FlintSplit:
    _PROPAGATE = ("MOE_GROUP", "MT_BUCKET", "ATT_QB", "ATT_FULL", "ATT_BUDGET", "HEAD_SLICE", "INS_DT", "MOE_ACC",
                  "trace", "memprobe")

    def __init__(self, gguf, devices=("xpu:0", "xpu:1"), split_at=0, dtype="float32", log=print, kv_tokens=16384,
                 force_hop=False):
        import torch
        self.torch, self.log = torch, log
        self.devs = [torch.device(d) for d in devices]
        self.force_hop = force_hop                       # also hop through the host when both halves share a device
        if split_at:
            self.K, self.split_info = int(split_at), {"manual": True}
        else:
            self.K, self.split_info = auto_split_at(gguf, kv_tokens=kv_tokens)
        K = self.K
        assert 3 <= K <= NL - 1, f"split point {K} out of range"
        log(f"model split: device {self.devs[0]} = embedding + layers 0..{K - 1}; device {self.devs[1]} = layers "
            f"{K}..{NL - 1} + final norm + head + loss   {self.split_info}")
        self.halves = []
        for i, layers in enumerate((set(range(K)), set(range(K, NL)))):
            with self._guard(self.devs[i]):
                h = FT.FlintTorch(gguf, str(self.devs[i]), dtype, log=lambda s, i=i: log(f"[dev{i}] {s}"), layers=layers)
            self.halves.append(h)
        A, B = self.halves
        A.head, A.out_norm = None, None                  # device 0 never runs the head: give its copy back
        # Host RAM: FlintTorch keeps the GGUF memory map (11 GB of file pages stay in the process working set) and reads the
        # embedding rows through it. Copy the embedding (0.57 GB fp16) into ordinary RAM and drop both maps, so the
        # working set shrinks to ~1-2 GB (the training PC has 32 GB and shares it with other jobs).
        A.emb = np.ascontiguousarray(A.emb)
        B.emb = None
        for h in self.halves:
            h.mm, h.T = None, None
        self.params = {**A.params, **B.params}           # layer-ascending order, same as one FlintTorch
        self.groups = [list(A.params.values()), list(B.params.values())]
        self.V = B.V
        assert any("mult_ffn_gate_exps" in A.L[L] for L in range(K) if A.L[L] is not None), "device 0 has no trainable layer"

    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)
        if k in self._PROPAGATE and "halves" in self.__dict__:
            for h in self.halves:
                setattr(h, k, v)

    # ------------------------------------------------------------------ helpers
    def _guard(self, dev):
        if dev.type == "xpu":
            return self.torch.xpu.device(dev.index if dev.index is not None else self.torch.xpu.current_device())
        return contextlib.nullcontext()

    def _hop(self, x, dev):
        """move a tensor to another device through host memory (exact for fp32)"""
        if x.device == dev and not self.force_hop:
            return x
        return x.detach().cpu().to(dev)

    def owner(self, L):
        return self.halves[0] if L < self.K else self.halves[1]

    def memory_report(self):
        t = self.torch
        out = []
        for h in self.halves:
            if h.dev.type == "xpu":
                out.append((t.xpu.memory_allocated(h.dev) / GiB, t.xpu.memory_reserved(h.dev) / GiB,
                            t.xpu.max_memory_allocated(h.dev) / GiB))
            else:
                out.append((0.0, 0.0, 0.0))
        return out

    def reset_peak(self):
        for h in self.halves:
            if h.dev.type == "xpu":
                self.torch.xpu.reset_peak_memory_stats(h.dev)

    # ------------------------------------------------------------------ forward
    def _forward(self, ids, pos0, cache):
        t = self.torch
        A, B = self.halves
        K, n = self.K, len(ids)
        ins = [None] * NL
        with t.no_grad():
            with self._guard(A.dev):
                posA = t.arange(pos0, pos0 + n, device=A.dev)
                x = A.embed(ids)
                for L in range(K):
                    ins[L] = x if self.INS_DT is None else x.to(self.INS_DT)
                    x, kv = A.layer(L, x, posA, cache[L])
                    cache[L] = A._append(cache[L], kv, A.L[L]["sliding"])
            x = self._hop(x, B.dev)
            with self._guard(B.dev):
                posB = t.arange(pos0, pos0 + n, device=B.dev)
                for L in range(K, NL):
                    ins[L] = x if self.INS_DT is None else x.to(self.INS_DT)
                    x, kv = B.layer(L, x, posB, cache[L])
                    cache[L] = B._append(cache[L], kv, B.L[L]["sliding"])
        return ins, x, (posA, posB)

    INS_DT = None

    def forward_chunk(self, ids, pos0, cache):
        """no-grad forward of one chunk; same contract as FlintTorch.forward_chunk (cache: per-layer (k, v, kpos) lives on
        the layer's own device). Returns (inputs per layer, final hidden on device 1, (pos0, pos1))."""
        return self._forward(ids, pos0, cache)

    # ------------------------------------------------------------------ eval / train
    def eval_example(self, ids, prompt_tokens, chunk):
        t = self.torch
        B = self.halves[1]
        tg = ids[1:] + [0]
        mask = [1.0 if (i + 1 >= prompt_tokens and i < len(ids) - 1) else 0.0 for i in range(len(ids))]
        cache, tot, cnt = [None] * NL, 0.0, 0.0
        for s in range(0, len(ids), chunk):
            sl = slice(s, s + chunk)
            _, x, _ = self._forward(ids[sl], s, cache)
            if sum(mask[sl]):
                with self._guard(B.dev):
                    l, n = B.head_loss(x, t.from_numpy(np.asarray(tg[sl], np.int64)).to(B.dev),
                                       t.from_numpy(np.asarray(mask[sl], np.float32)).to(B.dev))
                tot, cnt = tot + l, cnt + n
        return tot, cnt

    def train_chunk(self, ids, targets, mask, pos0, cache, trace=None):
        """Same contract as FlintTorch.train_chunk (<= ~1024 tokens per chunk, cache = PAST chunks only, extended at the
        end; returns the mean loss or None when the chunk has no answer tokens)."""
        t = self.torch
        A, B = self.halves
        K = self.K
        past = list(cache)
        for h in self.halves:
            h._rc, h._rcphase = {}, "store"                  # K4: each half keeps its layers' routing for the recompute
        ins, xL, (posA, posB) = self._forward(ids, pos0, cache)
        for h in self.halves:
            h._rcphase = "reuse"
            h.last_rc_bytes = sum(v["bytes"] for v in h._rc.values())
        self.last_rc_bytes = sum(h.last_rc_bytes for h in self.halves)
        with self._guard(B.dev):
            tgt = t.from_numpy(np.asarray(targets, np.int64)).to(B.dev)
            msk = t.from_numpy(np.asarray(mask, np.float32)).to(B.dev)
            n = float(msk.sum().item())
            if n == 0:
                for h in self.halves:
                    h._rc = None
                return None
            loss, g = B.head_loss_grad(xL, tgt, msk)             # loss on device 1, g = d loss / d final hidden
        first = min(L for L in range(K) if A.L[L] is not None and "mult_ffn_gate_exps" in A.L[L])
        for L in reversed(range(first, NL)):
            h, pos = (A, posA) if L < K else (B, posB)
            with self._guard(h.dev):
                x_in = ins[L].detach().float().requires_grad_(L > first)
                out, _ = h.layer(L, x_in, pos, past[L])
                out.backward(g)
                g = x_in.grad if L > first else None
            ins[L] = None                                        # this layer's stored input is no longer needed
            if L == K and g is not None:                         # boundary: gradient goes back to device 0
                g = self._hop(g, A.dev)
        for h in self.halves:
            h._rc = None
        return loss


# ---------------------------------------------------------------------- optimizer / clipping over per-device groups
class SplitAdam:
    """One torch.optim.Adam per device group (each device's params, grads and Adam state stay on that device).
    state_dict()/load_state_dict() use the SAME format as one Adam over list(model.params.values()) (state indices =
    global parameter index, one param group), so run2's save.pt loads here and a split save.pt loads in flint_train_p2."""

    def __init__(self, torch, groups, lr):
        self.torch = torch
        self.groups = [list(g) for g in groups]
        self.opts = [torch.optim.Adam(g, lr=lr) if g else None for g in self.groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            if o is not None:
                o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            if o is not None:
                o.step()

    def state_dict(self):
        state, pg, off = {}, None, 0
        for o, g in zip(self.opts, self.groups):
            if o is None:
                continue
            sd = o.state_dict()
            for k, v in sd["state"].items():
                state[k + off] = {kk: (vv.detach().cpu().clone() if hasattr(vv, "detach") else vv) for kk, vv in v.items()}
            if pg is None:
                pg = copy.deepcopy(sd["param_groups"][0])
                pg["params"] = []
            pg["params"] += [i + off for i in sd["param_groups"][0]["params"]]
            off += len(g)
        return {"state": state, "param_groups": [pg]}

    def load_state_dict(self, sd):
        off = 0
        for o, g in zip(self.opts, self.groups):
            n = len(g)
            if o is not None:
                grp = copy.deepcopy(sd["param_groups"][0])
                grp["params"] = list(range(n))
                sub = {"state": {k - off: v for k, v in sd["state"].items() if off <= k < off + n}, "param_groups": [grp]}
                o.load_state_dict(sub)
            off += n


def clip_grad_norm_split(torch, groups, max_norm):
    """global L2 grad-norm clip over params on several devices (per-device norms -> host -> one coefficient)"""
    sq = 0.0
    for g in groups:
        gs = [p.grad for p in g if p.grad is not None]
        if gs:
            n = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x.float()) for x in gs]))
            sq += float(n.item()) ** 2
    total = sq ** 0.5
    coef = min(1.0, max_norm / (total + 1e-6))
    if coef < 1.0:
        with torch.no_grad():
            for g in groups:
                for p in g:
                    if p.grad is not None:
                        p.grad.mul_(coef)
    return total
