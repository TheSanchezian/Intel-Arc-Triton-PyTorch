"""LoRA (Low-Rank Adaptation) for Flint-40B PyTorch model (single-device and two-GPU split).

Wrap FlintTorch without editing flint_torch.py (runtime monkeypatching or subclassing).
All LoRA weights, activations, and optimizer states are strictly fp32 (no fp16/bf16).

Formulation:
  y = x @ W^T + (alpha / r) * (x @ A^T) @ B^T
  where A ~ Kaiming uniform (or normal), B = 0 at init.
  Since B = 0 at initialization, output delta is 0 and base model outputs are exactly preserved.

Targets:
  - Attention q/k/v/o projections in all 49 layers (default ON)
  - Dense FFN in layers 0-1 (default ON)
  - Shared expert gate/up/down in layers 2-48 (default ON)
  - Routed experts: default OFF; option for ONE shared low-rank adapter per projection kind
    (ffn_gate_exps, ffn_up_exps, ffn_down_exps) applied to all experts in a layer.

Multipliers:
  Base per-row expert multipliers (L{L}.{gate|up|down}) and LoRA adapters are both trainable by default.
  --freeze-mult freezes the multipliers so only LoRA trains.
"""

import math
import os
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Architecture constants from flint_torch.py
HID, NL, H, KVH, HD, WIN = 2048, 49, 32, 4, 128, 513
E, TOPK, EFF, BLK = 256, 6, 512, 128
INV512 = 512 ** -0.5

# Architectural shapes for linear projections: (in_features, out_features)
PROJECTION_SHAPES = {
    # Attention projections (all 49 layers: 0..48)
    "q": (2048, 4096),
    "k": (2048, 512),
    "v": (2048, 512),
    "o": (4096, 2048),
    # Dense FFN projections (layers 0..1)
    "dense_gate": (2048, 6144),
    "dense_up": (2048, 6144),
    "dense_down": (6144, 2048),
    # Shared expert projections (layers 2..48)
    "shexp_gate": (2048, 1024),
    "shexp_up": (2048, 1024),
    "shexp_down": (1024, 2048),
    # Routed experts shared adapter (layers 2..48)
    "exp_gate": (2048, 512),
    "exp_up": (2048, 512),
    "exp_down": (512, 2048),
}


class LoRALinear(nn.Module):
    """Low-rank adapter module: (alpha / r) * (x @ A^T) @ B^T.

    A: [r, in_features] initialized with Kaiming uniform.
    B: [out_features, r] initialized with zeros.
    Always kept in fp32.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: float = 32.0,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = float(alpha)
        self.scaling = (self.alpha / self.r) if self.r > 0 else 0.0

        self.A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform init on A
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # Exact zero init on B ensures delta is identically zero at step 0
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = (alpha/r) * (x @ A^T) @ B^T
        xf = x.float()
        # Matmul supports arbitrary leading dimensions [..., in_features]
        return ((xf @ self.A.t()) @ self.B.t()) * self.scaling

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, r={self.r}, alpha={self.alpha}, scaling={self.scaling}"


def parse_targets(targets: Union[str, List[str], Set[str]]) -> Set[str]:
    """Parse target specification into canonical tokens: 'attn', 'dense', 'shexp'."""
    if isinstance(targets, str):
        tokens = [t.strip().lower() for t in targets.split(",") if t.strip()]
    else:
        tokens = [str(t).strip().lower() for t in targets]
    res = set()
    for t in tokens:
        if t in ("attn", "attention", "qkv", "qkvo"):
            res.add("attn")
        elif t in ("dense", "ffn_dense", "dense_ffn"):
            res.add("dense")
        elif t in ("shexp", "shared_expert", "shared"):
            res.add("shexp")
        elif t == "all":
            res.update(["attn", "dense", "shexp"])
        else:
            res.add(t)
    return res


def count_lora_params(
    r: int = 16,
    targets: Union[str, List[str], Set[str]] = "attn,dense,shexp",
    lora_experts: str = "off",
    nl: int = NL,
) -> Tuple[int, Dict[str, int]]:
    """Compute theoretical trainable LoRA parameters from architecture dimensions.

    Returns:
        (total_params, breakdown_dict)
    """
    tgt = parse_targets(targets)
    breakdown = {}
    total = 0

    # 1. Attention: q, k, v, o on all nl layers
    if "attn" in tgt or any(k in tgt for k in ("q", "k", "v", "o")):
        per_layer_attn = 0
        for proj in ("q", "k", "v", "o"):
            if "attn" in tgt or proj in tgt:
                inf, outf = PROJECTION_SHAPES[proj]
                p = r * (inf + outf)
                per_layer_attn += p
        attn_total = per_layer_attn * nl
        breakdown["attention"] = attn_total
        total += attn_total

    # 2. Dense FFN: gate, up, down on layers 0..1 (2 layers)
    num_dense = min(2, nl)
    if num_dense > 0 and ("dense" in tgt or any(k in tgt for k in ("dense_gate", "dense_up", "dense_down"))):
        per_layer_dense = 0
        for proj in ("dense_gate", "dense_up", "dense_down"):
            if "dense" in tgt or proj in tgt:
                inf, outf = PROJECTION_SHAPES[proj]
                per_layer_dense += r * (inf + outf)
        dense_total = per_layer_dense * num_dense
        breakdown["dense_ffn"] = dense_total
        total += dense_total

    # 3. Shared expert: gate, up, down on layers 2..nl-1
    num_shexp = max(0, nl - 2)
    if num_shexp > 0 and ("shexp" in tgt or any(k in tgt for k in ("shexp_gate", "shexp_up", "shexp_down"))):
        per_layer_shexp = 0
        for proj in ("shexp_gate", "shexp_up", "shexp_down"):
            if "shexp" in tgt or proj in tgt:
                inf, outf = PROJECTION_SHAPES[proj]
                per_layer_shexp += r * (inf + outf)
        shexp_total = per_layer_shexp * num_shexp
        breakdown["shared_expert"] = shexp_total
        total += shexp_total

    # 4. Routed experts (shared low-rank adapter per projection kind across all experts)
    if lora_experts == "shared" and num_shexp > 0:
        per_layer_exp = 0
        for proj in ("exp_gate", "exp_up", "exp_down"):
            inf, outf = PROJECTION_SHAPES[proj]
            per_layer_exp += r * (inf + outf)
        exp_total = per_layer_exp * num_shexp
        breakdown["routed_experts_shared"] = exp_total
        total += exp_total

    return total, breakdown


# ----------------------------------------------------------------------
# Monkeypatched Layer / Forward / Backward Wrappers
# ----------------------------------------------------------------------

def _lora_attention(self, d, xn, pos, past):
    """FlintTorch.attention replacement with additive LoRA adapters on q, k, v, o."""
    t = self.torch
    import torch.utils.checkpoint
    T_ = xn.shape[0]
    xb = xn.to(self.dt)

    q = (xb @ self.deq41(d["q"]).t()).float()
    if "lora_q" in d:
        q = q + d["lora_q"](xb)
    q = q.view(T_, H, HD)

    k = (xb @ self.deq41(d["k"]).t()).float()
    if "lora_k" in d:
        k = k + d["lora_k"](xb)
    k = k.view(T_, KVH, HD)

    v = (xb @ self.deq41(d["v"]).t()).float()
    if "lora_v" in d:
        v = v + d["lora_v"](xb)
    v = v.view(T_, KVH, HD)

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
            if use_ckpt:
                outs.append(t.utils.checkpoint.checkpoint(self._att_block, *args, use_reentrant=False,
                                                          preserve_rng_state=False))
            else:
                outs.append(self._att_block(*args))
        o = t.cat(outs, 2)
    o = o.permute(2, 0, 1, 3).reshape(T_, H * HD)
    out = (o.to(self.dt) @ self.deq41(d["o"]).t()).float()
    if "lora_o" in d:
        out = out + d["lora_o"](o)
    return out, (k.detach(), v.detach(), pos)


def _lora_ffn_dense(self, d, xn):
    """FlintTorch.ffn_dense replacement with additive LoRA adapters on gate, up, down."""
    Fn = self.torch.nn.functional
    xb = xn.to(self.dt)
    g = xb @ self.deq41(d["gate"]).t()
    if "lora_gate" in d:
        g = g + d["lora_gate"](xb)
    u = xb @ self.deq41(d["up"]).t()
    if "lora_up" in d:
        u = u + d["lora_up"](xb)
    act = (Fn.silu(g.float()) * u.float()).to(self.dt)
    out = act @ self.deq41(d["down"]).t()
    if "lora_down" in d:
        out = out + d["lora_down"](act)
    return out


def _lora_moe_grouped(self, d, xn):
    """FlintTorch.moe_grouped replacement with optional shared routed-expert LoRA."""
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
        mk = lambda kind, eid=eid: (lambda: self.wbase_ids(d, kind, eid))
        gt = self._explin_k(xe, d["mult_ffn_gate_exps"], eid, mk("ffn_gate_exps")) * INV512
        if "lora_exp_gate" in d:
            gt = gt + d["lora_exp_gate"](xe)
        u = self._explin_k(xe, d["mult_ffn_up_exps"], eid, mk("ffn_up_exps")) * INV512
        if "lora_exp_up" in d:
            u = u + d["lora_exp_up"](xe)
        act = (Fn.silu(gt.float()) * u.float()).to(self.dt)
        hact = self.h512(act)
        y = self._explin_k(hact, d["mult_ffn_down_exps"], eid, mk("ffn_down_exps")) * INV512
        if "lora_exp_down" in d:
            y = y + d["lora_exp_down"](hact)
        v = (y.float() * Wg).reshape(-1, HID)
        if acc is not None:
            acc = acc.index_add(0, Pt, v)
        else:
            vals.append(v)
        if mp: mp(f'  group g={g} mt={mt} done')
    r = acc if acc is not None else t.zeros(T, HID, device=self.dev).index_add(0, Pall, t.cat(vals))
    if mp: mp('  moe cat+index_add done')
    return r


def _lora_moe_loop(self, d, xn):
    """FlintTorch.moe_loop replacement with optional shared routed-expert LoRA."""
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
        if "lora_exp_gate" in d:
            gt = gt + d["lora_exp_gate"](xe)
        u = self.explin(xe, d, "ffn_up_exps", e) * INV512
        if "lora_exp_up" in d:
            u = u + d["lora_exp_up"](xe)
        act = (Fn.silu(gt.float()) * u.float()).to(self.dt)
        hact = self.h512(act)
        y = self.explin(hact, d, "ffn_down_exps", e) * INV512
        if "lora_exp_down" in d:
            y = y + d["lora_exp_down"](hact)
        we = t.from_numpy(w[tok_np, slot_np].astype(np.float32)).to(self.dev).unsqueeze(-1)
        toks.append(tok)
        vals.append(y.float() * we)
    tok_all, val_all = t.cat(toks), t.cat(vals)
    return t.zeros(xn.shape[0], HID, device=self.dev).index_add(0, tok_all, val_all)


def _find_first_trainable_layer(model, nl: int) -> int:
    """Find lowest layer index that owns trainable parameters (multipliers or LoRA)."""
    freeze_mult = getattr(model, "freeze_mult", False)
    lora_params = getattr(model, "lora_params", {})
    first = None
    for L in range(nl):
        layer_dict = model.L[L] if hasattr(model, "L") and L < len(model.L) else None
        if layer_dict is None:
            continue
        # Check multiplier trainable
        if not freeze_mult and "mult_ffn_gate_exps" in layer_dict:
            p = layer_dict["mult_ffn_gate_exps"]
            if p.requires_grad:
                first = L
                break
        # Check LoRA params in layer L
        prefix = f"L{L}."
        if any(p.requires_grad for name, p in lora_params.items() if name.startswith(prefix)):
            first = L
            break
    return 0 if first is None else first


def _lora_train_chunk(self, ids, targets, mask, pos0, cache, trace=None):
    """FlintTorch.train_chunk replacement ensuring backward walks all trainable layers (down to layer 0)."""
    t = self.torch
    past = list(cache)
    self._tr("chunk: no-grad forward starts")
    self._rc, self._rcphase = {}, "store"
    ins, xL, pos = self.forward_chunk(ids, pos0, cache)
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

    # Find the lowest layer that has trainable parameters (multipliers or LoRA)
    first = _find_first_trainable_layer(self, NL)

    for L in reversed(range(first, NL)):
        x_in = ins[L].detach().float().requires_grad_(L > first)
        out, _ = self.layer(L, x_in, pos, past[L])
        self._tr(f"backward layer {L}: recomputed")
        out.backward(g)
        g = x_in.grad if L > first else None
        self._tr(f"backward layer {L}: done")
    self._rc = None
    return loss


def _split_lora_train_chunk(self, ids, targets, mask, pos0, cache, trace=None):
    """FlintSplit.train_chunk replacement ensuring backward walks all trainable layers on both devices."""
    t = self.torch
    A, B = self.halves
    K = self.K
    past = list(cache)
    for h in self.halves:
        h._rc, h._rcphase = {}, "store"
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
        loss, g = B.head_loss_grad(xL, tgt, msk)

    # Lowest trainable layer on half A (device 0)
    first = _find_first_trainable_layer(A, K)

    for L in reversed(range(first, NL)):
        h, pos = (A, posA) if L < K else (B, posB)
        with self._guard(h.dev):
            x_in = ins[L].detach().float().requires_grad_(L > first)
            out, _ = h.layer(L, x_in, pos, past[L])
            out.backward(g)
            g = x_in.grad if L > first else None
        ins[L] = None
        if L == K and g is not None:
            g = self._hop(g, A.dev)

    for h in self.halves:
        h._rc = None
    return loss


# ----------------------------------------------------------------------
# Application Entry Points
# ----------------------------------------------------------------------

import types
import numpy as np


def _apply_lora_to_flint_torch(
    model,
    r: int = 16,
    alpha: float = 32.0,
    targets: Union[str, List[str], Set[str]] = "attn,dense,shexp",
    lora_experts: str = "off",
    freeze_mult: bool = False,
) -> Dict[str, nn.Parameter]:
    """Applies LoRA to a single FlintTorch instance."""
    tgt = parse_targets(targets)
    dev = model.dev
    lora_params: Dict[str, nn.Parameter] = {}

    nl = len(model.L)
    for L in range(nl):
        d = model.L[L]
        if d is None:
            continue

        # 1. Attention targets: q, k, v, o
        if "attn" in tgt or any(k in tgt for k in ("q", "k", "v", "o")):
            for proj in ("q", "k", "v", "o"):
                if "attn" in tgt or proj in tgt:
                    inf, outf = PROJECTION_SHAPES[proj]
                    mod = LoRALinear(inf, outf, r=r, alpha=alpha, device=dev, dtype=torch.float32)
                    d[f"lora_{proj}"] = mod
                    lora_params[f"L{L}.attn_{proj}.A"] = mod.A
                    lora_params[f"L{L}.attn_{proj}.B"] = mod.B

        # 2. Dense FFN targets (layers 0..1)
        if L < 2 and ("dense" in tgt or any(k in tgt for k in ("dense_gate", "dense_up", "dense_down"))):
            for proj in ("gate", "up", "down"):
                canon = f"dense_{proj}"
                if "dense" in tgt or canon in tgt:
                    inf, outf = PROJECTION_SHAPES[canon]
                    mod = LoRALinear(inf, outf, r=r, alpha=alpha, device=dev, dtype=torch.float32)
                    d[f"lora_{proj}"] = mod
                    lora_params[f"L{L}.ffn_{proj}.A"] = mod.A
                    lora_params[f"L{L}.ffn_{proj}.B"] = mod.B

        # 3. Shared expert targets (layers 2..48)
        if L >= 2 and ("shexp" in tgt or any(k in tgt for k in ("shexp_gate", "shexp_up", "shexp_down"))):
            for proj in ("gate", "up", "down"):
                canon = f"shexp_{proj}"
                if "shexp" in tgt or canon in tgt:
                    inf, outf = PROJECTION_SHAPES[canon]
                    mod = LoRALinear(inf, outf, r=r, alpha=alpha, device=dev, dtype=torch.float32)
                    d[f"lora_{proj}"] = mod
                    lora_params[f"L{L}.ffn_{proj}_shexp.A"] = mod.A
                    lora_params[f"L{L}.ffn_{proj}_shexp.B"] = mod.B

        # 4. Routed experts shared adapter (layers 2..48)
        if L >= 2 and lora_experts == "shared":
            for proj in ("gate", "up", "down"):
                inf, outf = PROJECTION_SHAPES[f"exp_{proj}"]
                mod = LoRALinear(inf, outf, r=r, alpha=alpha, device=dev, dtype=torch.float32)
                d[f"lora_exp_{proj}"] = mod
                lora_params[f"L{L}.ffn_{proj}_exps.A"] = mod.A
                lora_params[f"L{L}.ffn_{proj}_exps.B"] = mod.B

    # Multipliers handling
    model.freeze_mult = bool(freeze_mult)
    if freeze_mult and hasattr(model, "params"):
        for p in model.params.values():
            p.requires_grad = False

    # Bind monkeypatched execution methods
    model.attention = types.MethodType(_lora_attention, model)
    model.ffn_dense = types.MethodType(_lora_ffn_dense, model)
    model.moe_grouped = types.MethodType(_lora_moe_grouped, model)
    model.moe_loop = types.MethodType(_lora_moe_loop, model)
    model.train_chunk = types.MethodType(_lora_train_chunk, model)

    model.lora_params = lora_params
    return lora_params


def apply_lora(
    model,
    r: int = 16,
    alpha: float = 32.0,
    targets: Union[str, List[str], Set[str]] = "attn,dense,shexp",
    lora_experts: str = "off",
    freeze_mult: bool = False,
) -> Dict[str, nn.Parameter]:
    """Wraps model (FlintTorch or FlintSplit) with LoRA adapters at runtime.

    Args:
        model: FlintTorch or FlintSplit instance
        r: rank (default 16)
        alpha: alpha scaling factor (default 32.0)
        targets: comma-separated or list of target groups ('attn', 'dense', 'shexp')
        lora_experts: 'off' or 'shared' (one adapter per projection kind across all experts)
        freeze_mult: if True, freeze expert row multipliers (requires_grad = False)

    Returns:
        dict of {name: Parameter} for all trainable LoRA parameters.
    """
    # Check if model is FlintSplit (two-GPU split wrapper)
    if hasattr(model, "halves") and isinstance(model.halves, (list, tuple)) and len(model.halves) == 2:
        model.freeze_mult = bool(freeze_mult)
        if freeze_mult and hasattr(model, "params"):
            for p in model.params.values():
                p.requires_grad = False

        half0_lora = _apply_lora_to_flint_torch(
            model.halves[0], r=r, alpha=alpha, targets=targets, lora_experts=lora_experts, freeze_mult=freeze_mult
        )
        half1_lora = _apply_lora_to_flint_torch(
            model.halves[1], r=r, alpha=alpha, targets=targets, lora_experts=lora_experts, freeze_mult=freeze_mult
        )

        model.lora_params = {**half0_lora, **half1_lora}
        model.lora_groups = [list(half0_lora.values()), list(half1_lora.values())]
        model.train_chunk = types.MethodType(_split_lora_train_chunk, model)
        return model.lora_params

    # Single-device FlintTorch (or test mock layer)
    return _apply_lora_to_flint_torch(
        model, r=r, alpha=alpha, targets=targets, lora_experts=lora_experts, freeze_mult=freeze_mult
    )


# ----------------------------------------------------------------------
# Optional Subclass Wrapping
# ----------------------------------------------------------------------

try:
    import flint_torch as _FT

    class FlintTorchLoRA(_FT.FlintTorch):
        """FlintTorch subclass with native LoRA initialization."""

        def __init__(
            self,
            gguf: str,
            device: str = "xpu",
            dtype: str = "float32",
            log=print,
            layers=None,
            r: int = 16,
            alpha: float = 32.0,
            targets: Union[str, List[str], Set[str]] = "attn,dense,shexp",
            lora_experts: str = "off",
            freeze_mult: bool = False,
        ):
            super().__init__(gguf, device=device, dtype=dtype, log=log, layers=layers)
            apply_lora(
                self,
                r=r,
                alpha=alpha,
                targets=targets,
                lora_experts=lora_experts,
                freeze_mult=freeze_mult,
            )
except Exception:
    pass
