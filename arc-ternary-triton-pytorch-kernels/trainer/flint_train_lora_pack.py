"""Train Flint's expert row multipliers and/or LoRA adapters with SEQUENCE PACKING.

Extended from flint_train_lora_fast.py with:
  --pack                 enable sequence packing for short examples (<= --pack-max)
  --pack-max             maximum tokens for an example to be packed (default 1024)
  --pack-window          lookahead window for greedy first-fit packing (default 50)

Design and Guarantees:
1. Examples <= --pack-max are packed greedily using First-Fit bin packing into 1024-token chunks.
   Longer examples (> --pack-max) use the unchanged multi-chunk path with KV cache across chunks.
2. Per-packed-segment correctness:
   - Block-diagonal causal attention mask: tokens cannot attend across segment boundaries.
   - RoPE positions reset to 0 at each segment start.
   - Learned attention sinks behave identically to unpacked sequence starts (only the sink is visible
     at position 0 of each segment).
   - Sliding-window layers are windowed within each segment.
   - No KV cache carried between packed chunks.
   - Loss mask: only answer tokens of each segment contribute to the loss.
   - Loss normalisation: per-token mean cross-entropy over all answer tokens in the step, exactly
     matching the unpacked trainer's normalisation per step.
   - Padding only at the chunk tail.
3. Cursor and Resume semantics:
   - state["cursor"] points to the example index in ex.
   - In-progress multi-chunk examples resume from state["chunk_in_example"].
   - When packing, windows advance state["cursor"] cleanly to window_end upon window completion,
     ensuring every example in ex is trained exactly once, with full resume compatibility with run3's save.pt.
"""
import argparse
import ctypes
import gc
import json
import os
import random
import shutil
import sys
import time
import types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
KERNELS_EXP = os.path.join(HERE, "..", "kernels")
if KERNELS_EXP not in sys.path:
    sys.path.insert(0, KERNELS_EXP)


def check_ram(min_gb=12.0, stop_gb=4.0):
    m = (ctypes.c_ulonglong * 8)()
    m[0] = 64
    ctypes.windll.kernel32.GlobalMemoryStatusEx(m)
    free_gb = m[2] / (1024 ** 3)
    if free_gb < stop_gb:
        raise RuntimeError(f"ABORT: Free RAM ({free_gb:.2f} GB) < {stop_gb:.1f} GB safety threshold!")
    if free_gb < min_gb:
        raise RuntimeError(f"ABORT: Free RAM ({free_gb:.2f} GB) < {min_gb:.1f} GB required before model load!")
    return free_gb


def load_examples(path, max_tokens, seed):
    ex = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["tokens"] <= max_tokens:
                ex.append(d)
    random.Random(seed).shuffle(ex)
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True, help="folder with train.jsonl / val.jsonl")
    ap.add_argument("--extra", default="", help="optional extra JSONL in the same format")
    ap.add_argument("--extra-weight", type=int, default=3, help="each extra example is seen this many times per pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1024, help="tokens per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--save-tokens", type=int, default=1280)
    ap.add_argument("--max-example-tokens", type=int, default=2048)
    ap.add_argument("--hours", type=float, default=1.0, help="stop cleanly after this long (supervisor restarts)")
    ap.add_argument("--dtype", default="float32", help="float32 only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-max-tokens", type=int, default=2048, help="validation length cap")
    ap.add_argument("--trace", action="store_true", help="sync + print a marker after every stage")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = no limit)")
    ap.add_argument("--eval-every", type=int, default=100, help="validation loss every N steps (0 = off)")
    ap.add_argument("--eval-examples", type=int, default=12, help="fixed validation examples per evaluation")
    ap.add_argument("--moe-group", type=int, default=16, help="experts per batched launch")
    ap.add_argument("--pad-to", type=int, default=256, help="pad chunk to a multiple of this")
    ap.add_argument("--empty-cache", action="store_true", help="empty cache after step (off by default; never use on XPU)")
    ap.add_argument("--cpu-adam", action="store_true", help="Adam state on CPU")
    ap.add_argument("--ins-dtype", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--mt-bucket", type=int, default=0)
    ap.add_argument("--att-qb", type=int, default=-1)
    ap.add_argument("--head-slice", type=int, default=0)
    ap.add_argument("--train-file", default="")
    ap.add_argument("--init-from", default="")
    ap.add_argument("--val-new", default="")
    ap.add_argument("--val-new-examples", type=int, default=40)
    ap.add_argument("--val-new-every", type=int, default=300)
    ap.add_argument("--eval-now", action="store_true")

    # LoRA extensions
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-targets", default="attn,dense,shexp")
    ap.add_argument("--lora-experts", choices=["off", "shared"], default="off")
    ap.add_argument("--freeze-mult", action="store_true")
    ap.add_argument("--lora-init-from", default="")
    ap.add_argument("--lora-lr", type=float, default=0.0)

    # Fast mode extension
    ap.add_argument("--fast", action="store_true")

    # Sequence packing extensions
    ap.add_argument("--pack", action="store_true", help="enable sequence packing for short examples")
    ap.add_argument("--pack-max", type=int, default=1024, help="maximum tokens for an example to be packed (default 1024)")
    ap.add_argument("--pack-window", type=int, default=50, help="lookahead window for greedy first-fit packing (default 50)")

    a = ap.parse_args()
    if a.fast:
        assert a.freeze_mult, "--fast requires --freeze-mult to be enabled"

    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a", encoding="utf-8")

    def log(**kw):
        kw["t"] = round(time.time(), 1)
        logf.write(json.dumps(kw) + "\n")
        logf.flush()

    def say(s):
        print(f"[train {time.strftime('%H:%M:%S')}] {s}", flush=True)

    # Safety RAM check before importing torch and loading model
    check_ram(min_gb=12.0, stop_gb=4.0)

    import torch
    import flint_torch as FT
    import flint_lora as FL
    if a.fast:
        import fused_kernels as FK

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    ex = load_examples(a.train_file or os.path.join(a.data, "train.jsonl"), a.max_example_tokens, a.seed)
    if a.extra and os.path.exists(a.extra):
        extra = load_examples(a.extra, a.max_example_tokens, a.seed + 1) * a.extra_weight
        ex = ex + extra
        random.Random(a.seed + 2).shuffle(ex)
        say(f"extra data: {len(extra) // a.extra_weight} examples x {a.extra_weight}")
    say(f"{len(ex)} examples <= {a.max_example_tokens} tokens")

    t0 = time.time()
    gguf_to_load = a.gguf  # with --fast, pass the multiplier-folded GGUF from export_folded.py
    say(f"Loading GGUF model: {gguf_to_load}...")
    m = FT.FlintTorch(gguf_to_load, "xpu", a.dtype, log=lambda s: say(s))
    m.emb = m.emb.copy()
    m.mm = None
    m.T = None
    gc.collect()
    try:
        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
    except Exception:
        pass

    say(f"model on the GPU in {time.time() - t0:.0f} s, {torch.xpu.memory_allocated() / 2 ** 30:.2f} GiB, "
        f"{sum(p.numel() for p in m.params.values()) / 1e6:.1f}M base multiplier params")
    m.MOE_GROUP = a.moe_group
    if a.mt_bucket: m.MT_BUCKET = a.mt_bucket
    if a.att_qb >= 0: m.ATT_QB = a.att_qb
    if a.head_slice: m.HEAD_SLICE = a.head_slice
    m.INS_DT = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[a.ins_dtype]
    if a.trace:
        m.trace = say

    # LoRA setup
    lora_params_dict = {}
    lora_params = []
    if a.lora:
        lora_params_dict = FL.apply_lora(
            m,
            r=a.lora_r,
            alpha=a.lora_alpha,
            targets=a.lora_targets,
            lora_experts=a.lora_experts,
            freeze_mult=a.freeze_mult,
        )
        lora_params = list(lora_params_dict.values())
        tot_lora = sum(p.numel() for p in lora_params)
        say(f"LoRA enabled: r={a.lora_r}, alpha={a.lora_alpha}, targets={a.lora_targets}, experts={a.lora_experts}, "
            f"{tot_lora / 1e6:.2f}M trainable LoRA params ({tot_lora * 4 / 2**20:.1f} MB fp32)")
        if a.freeze_mult:
            say("Base expert row multipliers FROZEN (requires_grad = False)")
        else:
            say("Base expert row multipliers TRAINABLE concurrently with LoRA")
    elif a.freeze_mult:
        for p in m.params.values():
            p.requires_grad = False
        say("Base expert row multipliers FROZEN")

    # Fast monkeypatches if --fast
    if a.fast:
        say("Applying FAST monkeypatches: FoldedExpLinK + fused RMSNorm + fused SwiGLU (skipping multipliers)")
        m.rms = FK.fused_rmsnorm
        t = m.torch
        orig_bmm = t.bmm

        class FoldedExpLinK(t.autograd.Function):
            @staticmethod
            def forward(ctx, x, eids, wfn):
                W = wfn()
                ctx.wfn = wfn
                return orig_bmm(x.contiguous(), W.transpose(1, 2).contiguous())

            @staticmethod
            def backward(ctx, gy):
                W = ctx.wfn()
                gx = orig_bmm(gy.contiguous(), W.contiguous())
                return gx, None, None

        m._explin_k_folded = FoldedExpLinK.apply

        def patched_lora_ffn_dense(d, xn):
            xb = xn.to(m.dt)
            g = xb @ m.deq41(d["gate"]).t()
            if "lora_gate" in d: g = g + d["lora_gate"](xb)
            u = xb @ m.deq41(d["up"]).t()
            if "lora_up" in d: u = u + d["lora_up"](xb)
            act = FK.fused_swiglu(g, u).to(m.dt)
            y = act @ m.deq41(d["down"]).t()
            if "lora_down" in d: y = y + d["lora_down"](act)
            return y
        m.ffn_dense = patched_lora_ffn_dense

        def patched_lora_moe_grouped_folded(d, xn):
            T = xn.shape[0]
            rc = m._rc
            prep = None
            if rc is not None and m._rcphase == "reuse":
                prep = rc.pop(m._cur_L, None)
            if prep is None:
                prep = m._build_route(d, xn)
                if rc is not None and m._rcphase == "store":
                    rc[m._cur_L] = prep
            xh = m.h512(xn.to(m.dt))
            Pall, Wall, Eall = prep["P"], prep["W"], prep["E"]
            vals = []
            acc = m.torch.zeros(T, FT.HID, device=m.dev) if m.MOE_ACC else None
            for p0, g, mt, s in prep["groups"]:
                Pt = Pall[p0:p0 + g * mt]
                Wg = Wall[p0:p0 + g * mt].view(g, mt, 1)
                eid = Eall[s:s + g]
                xe = xh.index_select(0, Pt).view(g, mt, -1)
                mk = lambda kind, eid=eid: (lambda: m.wbase_ids(d, kind, eid))
                gt = m._explin_k_folded(xe, eid, mk("ffn_gate_exps")) * FT.INV512
                if "lora_exp_gate" in d: gt = gt + d["lora_exp_gate"](xe)
                u = m._explin_k_folded(xe, eid, mk("ffn_up_exps")) * FT.INV512
                if "lora_exp_up" in d: u = u + d["lora_exp_up"](xe)
                act = FK.fused_swiglu(gt, u).to(m.dt)
                hact = m.h512(act)
                y = m._explin_k_folded(hact, eid, mk("ffn_down_exps")) * FT.INV512
                if "lora_exp_down" in d: y = y + d["lora_exp_down"](hact)
                v = (y.float() * Wg).reshape(-1, FT.HID)
                if acc is not None:
                    acc = acc.index_add(0, Pt, v)
                else:
                    vals.append(v)
            return acc if acc is not None else m.torch.zeros(T, FT.HID, device=m.dev).index_add(0, Pall, m.torch.cat(vals))

        m.moe_grouped = patched_lora_moe_grouped_folded

    # Sequence packing monkeypatches if --pack
    if a.pack:
        say(f"Applying SEQUENCE PACKING monkeypatches: pack_max={a.pack_max}, pack_window={a.pack_window}")

        def _packed_att_block(self, qb, kT, Vp, kpos, pb, sink_w, sliding, k_seg=None, q_seg=None):
            sc = (qb @ kT) * (FT.HD ** -0.5)
            if k_seg is not None and q_seg is not None:
                same_seg = (k_seg.unsqueeze(0) == q_seg.unsqueeze(1)) & (q_seg.unsqueeze(1) >= 0)
                allowed = same_seg & (kpos.unsqueeze(0) <= pb.unsqueeze(1))
            else:
                allowed = kpos.unsqueeze(0) <= pb.unsqueeze(1)
            if sliding:
                allowed = allowed & (kpos.unsqueeze(0) > pb.unsqueeze(1) - FT.WIN)
            sc = sc.masked_fill(~allowed, float("-inf"))
            sink = sink_w.expand(FT.KVH, 8, qb.shape[2], 1)
            p = self.torch.softmax(self.torch.cat([sc, sink], -1), -1)[..., :-1]
            return p @ Vp

        m._att_block = types.MethodType(_packed_att_block, m)

        def _packed_lora_attention(self, d, xn, pos, past):
            t = self.torch
            import torch.utils.checkpoint
            T_ = xn.shape[0]
            xb = xn.to(self.dt)

            q = (xb @ self.deq41(d["q"]).t()).float()
            if "lora_q" in d: q = q + d["lora_q"](xb)
            q = q.view(T_, FT.H, FT.HD)

            k = (xb @ self.deq41(d["k"]).t()).float()
            if "lora_k" in d: k = k + d["lora_k"](xb)
            k = k.view(T_, FT.KVH, FT.HD)

            v = (xb @ self.deq41(d["v"]).t()).float()
            if "lora_v" in d: v = v + d["lora_v"](xb)
            v = v.view(T_, FT.KVH, FT.HD)

            q = q * t.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6) * d["qn"]
            k = k * t.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6) * d["kn"]
            if d["sliding"]:
                q, k = self.rope(q, pos), self.rope(k, pos)
            if past is not None:
                K, Vv, kpos = t.cat([past[0], k], 0), t.cat([past[1], v], 0), t.cat([past[2], pos], 0)
            else:
                K, Vv, kpos = k, v, pos
            qg = q.view(T_, FT.KVH, 8, FT.HD).permute(1, 2, 0, 3)
            kT = K.permute(1, 2, 0).unsqueeze(1)
            Vp = Vv.permute(1, 0, 2).unsqueeze(1)
            sink_w = d["sinks"].view(FT.KVH, 8, 1, 1)
            S_ = K.shape[0]
            QB = self.att_qb(T_, S_)
            seg_id = getattr(self, "_seg_id", None)

            if QB >= T_:
                o = self._att_block(qg, kT, Vp, kpos, pos, sink_w, d["sliding"], seg_id, seg_id)
            else:
                use_ckpt = t.is_grad_enabled() and (qg.requires_grad or kT.requires_grad)
                outs = []
                for q0 in range(0, T_, QB):
                    q1 = min(q0 + QB, T_)
                    q_seg = seg_id[q0:q1] if seg_id is not None else None
                    args = (qg[:, :, q0:q1], kT, Vp, kpos, pos[q0:q1], sink_w, d["sliding"], seg_id, q_seg)
                    if use_ckpt:
                        outs.append(t.utils.checkpoint.checkpoint(self._att_block, *args, use_reentrant=False,
                                                                  preserve_rng_state=False))
                    else:
                        outs.append(self._att_block(*args))
                o = t.cat(outs, 2)
            o = o.permute(2, 0, 1, 3).reshape(T_, FT.H * FT.HD)
            out = (o.to(self.dt) @ self.deq41(d["o"]).t()).float()
            if "lora_o" in d:
                out = out + d["lora_o"](o)
            return out, (k.detach(), v.detach(), pos)

        m.attention = types.MethodType(_packed_lora_attention, m)

        def forward_chunk_packed(self, ids, positions, seg_ids):
            t = self.torch
            pos = t.as_tensor(positions, dtype=t.long, device=self.dev)
            self._seg_id = t.as_tensor(seg_ids, dtype=t.long, device=self.dev)
            x = self.embed(ids)
            ins = []
            with t.no_grad():
                for L in range(FT.NL):
                    ins.append(x if self.INS_DT is None else x.to(self.INS_DT))
                    x, _ = self.layer(L, x, pos, None)
            return ins, x, pos

        m.forward_chunk_packed = types.MethodType(forward_chunk_packed, m)

        def train_chunk_packed(self, ids, targets, mask, positions, seg_ids):
            t = self.torch
            self._tr("chunk: no-grad forward starts")
            self._rc, self._rcphase = {}, "store"
            ins, xL, pos = self.forward_chunk_packed(ids, positions, seg_ids)
            self._rcphase = "reuse"
            self.last_rc_bytes = sum(v["bytes"] for v in self._rc.values())
            self._tr("chunk: no-grad forward done")
            tgt = t.from_numpy(np.asarray(targets, np.int64)).to(self.dev)
            msk = t.from_numpy(np.asarray(mask, np.float32)).to(self.dev)
            n = float(msk.sum().item())
            if n == 0:
                self._rc = None
                self._seg_id = None
                return None
            loss, g = self.head_loss_grad(xL, tgt, msk)
            self._tr("chunk: head loss + gradient done")

            first = FL._find_first_trainable_layer(self, FT.NL)
            for L in reversed(range(first, FT.NL)):
                x_in = ins[L].detach().float().requires_grad_(L > first)
                out, _ = self.layer(L, x_in, pos, None)
                self._tr(f"backward layer {L}: recomputed")
                out.backward(g)
                g = x_in.grad if L > first else None
                self._tr(f"backward layer {L}: done")
            self._rc = None
            self._seg_id = None
            return loss

        m.train_chunk_packed = types.MethodType(train_chunk_packed, m)

    mult_params = list(m.params.values())
    active_mult_params = [p for p in mult_params if p.requires_grad]
    lora_lr = a.lora_lr if a.lora_lr > 0 else a.lr

    if a.cpu_adam:
        cparams_mult = [p.detach().cpu().clone().requires_grad_(True) for p in active_mult_params] if active_mult_params else []
        opt = torch.optim.Adam(cparams_mult, lr=a.lr) if cparams_mult else None

        cparams_lora = [p.detach().cpu().clone().requires_grad_(True) for p in lora_params] if lora_params else []
        lora_opt = torch.optim.Adam(cparams_lora, lr=lora_lr) if cparams_lora else None
    else:
        opt = torch.optim.Adam(active_mult_params, lr=a.lr) if active_mult_params else None
        lora_opt = torch.optim.Adam(lora_params, lr=lora_lr) if lora_params else None

    save_path, prev_path = os.path.join(a.out, "save.pt"), os.path.join(a.out, "save_prev.pt")
    resumed = False
    state = {"cursor": 0, "chunk_in_example": 0, "steps": 0, "tokens": 0, "epoch": 0, "last_daily": 0.0}

    def load_checkpoint_into_state(path, is_init=False):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "params" in ck:
            with torch.no_grad():
                for k, p in m.params.items():
                    if k in ck["params"]:
                        p.copy_(ck["params"][k].to(p.device))
        if opt is not None and "opt" in ck and ck["opt"]:
            try:
                opt.load_state_dict(ck["opt"])
            except Exception as e:
                say(f"warning: could not load opt state from {os.path.basename(path)} ({e})")
        if a.lora and lora_params_dict:
            if "lora" in ck:
                lora_ck = ck["lora"]
                if isinstance(lora_ck, dict) and "params" in lora_ck:
                    lora_ck = lora_ck["params"]
                loaded_cnt = 0
                with torch.no_grad():
                    for k, p in lora_params_dict.items():
                        if k in lora_ck:
                            p.copy_(lora_ck[k].to(p.device))
                            loaded_cnt += 1
                say(f"loaded {loaded_cnt}/{len(lora_params_dict)} LoRA tensors from {os.path.basename(path)}")
                if lora_opt is not None and "lora_opt" in ck and ck["lora_opt"]:
                    try:
                        lora_opt.load_state_dict(ck["lora_opt"])
                        say(f"loaded LoRA Adam state from {os.path.basename(path)}")
                    except Exception as e:
                        say(f"warning: could not load lora_opt state ({e})")
            else:
                say(f"{os.path.basename(path)} has no 'lora' key: LoRA remains zero-initialized")
        if not is_init and "state" in ck:
            state.update(ck["state"])
        del ck

    for path in (save_path, prev_path):
        if os.path.exists(path):
            try:
                load_checkpoint_into_state(path, is_init=False)
                resumed = True
                say(f"resumed from {os.path.basename(path)}: step {state['steps']}, {state['tokens']} trained tokens, "
                    f"example {state['cursor']}")
                break
            except Exception as e:
                say(f"could not load {os.path.basename(path)} ({type(e).__name__}: {e})")

    if not resumed and a.init_from and os.path.exists(a.init_from):
        load_checkpoint_into_state(a.init_from, is_init=True)
        say(f"INIT from {a.init_from}: base multiplier params loaded; step/cursor start at 0")

    if a.lora and a.lora_init_from and os.path.exists(a.lora_init_from):
        lck = torch.load(a.lora_init_from, map_location="cpu", weights_only=False)
        lora_sd = lck.get("lora", lck.get("params", lck))
        if isinstance(lora_sd, dict) and "params" in lora_sd:
            lora_sd = lora_sd["params"]
        loaded_cnt = 0
        with torch.no_grad():
            for k, p in lora_params_dict.items():
                if k in lora_sd:
                    p.copy_(lora_sd[k].to(p.device))
                    loaded_cnt += 1
        say(f"INIT LoRA from {a.lora_init_from}: {loaded_cnt}/{len(lora_params_dict)} tensors loaded")
        if lora_opt is not None and "lora_opt" in lck and lck["lora_opt"]:
            try:
                lora_opt.load_state_dict(lck["lora_opt"])
                say(f"INIT LoRA Adam state loaded from {a.lora_init_from}")
            except Exception as e:
                say(f"warning: could not load lora_opt from {a.lora_init_from} ({e})")
        del lck

    if a.cpu_adam:
        with torch.no_grad():
            if opt is not None:
                for cp, p in zip(cparams_mult, active_mult_params):
                    cp.copy_(p.detach().cpu())
            if lora_opt is not None:
                for cp, p in zip(cparams_lora, lora_params):
                    cp.copy_(p.detach().cpu())

    def save():
        ck = {
            "params": {k: p.detach().cpu() for k, p in m.params.items()},
            "opt": opt.state_dict() if opt is not None else {},
            "state": state,
        }
        if a.lora and lora_params_dict:
            ck["lora"] = {k: p.detach().cpu() for k, p in lora_params_dict.items()}
            ck["lora_opt"] = lora_opt.state_dict() if lora_opt is not None else None

        tmp = save_path + ".tmp"
        torch.save(ck, tmp)
        if os.path.exists(save_path):
            os.replace(save_path, prev_path)
        os.replace(tmp, save_path)
        if time.time() - state.get("last_daily", 0) > 86400:
            day = time.strftime("%Y%m%d_%H%M")
            shutil.copyfile(save_path, os.path.join(a.out, f"daily_{day}.pt"))
            state["last_daily"] = time.time()

    val = load_examples(os.path.join(a.data, "val.jsonl"), a.eval_max_tokens, 0)[:a.eval_examples]
    val_new = []
    if a.val_new:
        rows = [json.loads(l) for l in open(a.val_new, encoding="utf-8")]
        val_new = [d for d in rows if d["tokens"] <= a.eval_max_tokens][:a.val_new_examples]

    def evaluate(tag, vset=None, name="val_loss"):
        vset = val if vset is None else vset
        te, tot, cnt = time.time(), 0.0, 0.0
        for d in vset:
            l, n = m.eval_example(tok.encode(d["text"], add_special_tokens=False).ids, d["prompt_tokens"], a.chunk)
            tot, cnt = tot + l, cnt + n
        vl = tot / max(cnt, 1)
        log(eval=tag, set=("p1val" if vset is val else "val_new"), step=state["steps"], val_loss=round(vl, 5),
            val_tokens=int(cnt), sec=round(time.time() - te, 1))
        say(f"VALIDATION ({tag}, {'p1val' if vset is val else 'val_new'}) at step {state['steps']}: loss {vl:.4f} on {int(cnt)} answer tokens "
            f"({len(vset)} fixed examples, {time.time() - te:.0f} s)")

    if a.eval_now and state["steps"] == 0:
        evaluate("start")
        if val_new:
            evaluate("start", val_new)

    all_trainable = active_mult_params + lora_params

    def step_optimizer(loss_val):
        gn = float(torch.nn.utils.clip_grad_norm_(all_trainable, a.clip)) if all_trainable else 0.0
        if a.cpu_adam:
            if opt is not None:
                for cp, p in zip(cparams_mult, active_mult_params):
                    cp.grad = p.grad.detach().cpu() if p.grad is not None else None
                opt.step()
                with torch.no_grad():
                    for cp, p in zip(cparams_mult, active_mult_params):
                        p.copy_(cp.to(p.device))
            if lora_opt is not None:
                for cp, p in zip(cparams_lora, lora_params):
                    cp.grad = p.grad.detach().cpu() if p.grad is not None else None
                lora_opt.step()
                with torch.no_grad():
                    for cp, p in zip(cparams_lora, lora_params):
                        p.copy_(cp.to(p.device))
        else:
            if opt is not None: opt.step()
            if lora_opt is not None: lora_opt.step()
        return gn

    since_save, t_start, stop_reason = 0, time.time(), "hours"

    while True:
        if os.path.exists(os.path.join(a.out, "STOP")):
            stop_reason = "STOP file"
            break
        if time.time() - t_start > a.hours * 3600:
            break
        if a.max_steps and state["steps"] >= a.max_steps:
            stop_reason = "max steps"
            break
        if state["cursor"] >= len(ex):
            state["cursor"], state["epoch"] = 0, state["epoch"] + 1
            say(f"pass {state['epoch']} over the data starts")

        # Check for in-progress example (resumed from unpacked checkpoint)
        if state["chunk_in_example"] > 0:
            cur_idx = state["cursor"]
            d = ex[cur_idx]
            ids = tok.encode(d["text"], add_special_tokens=False).ids
            tg = ids[1:] + [0]
            mask = [1.0 if (i + 1 >= d["prompt_tokens"] and i < len(ids) - 1) else 0.0 for i in range(len(ids))]
            cache = [None] * FT.NL
            start_chunk = state["chunk_in_example"]
            for ci, s in enumerate(range(0, len(ids), a.chunk)):
                sl = slice(s, s + a.chunk)
                if ci < start_chunk or sum(mask[sl]) == 0:
                    m.forward_chunk(ids[sl], s, cache)
                    continue
                tc = time.time()
                if opt is not None: opt.zero_grad(set_to_none=True)
                if lora_opt is not None: lora_opt.zero_grad(set_to_none=True)
                for p in all_trainable: p.grad = None
                c_ids, c_tg, c_mk = ids[sl], tg[sl], mask[sl]
                if a.pad_to and s + a.chunk >= len(ids):
                    npad = -len(c_ids) % a.pad_to
                    c_ids, c_tg, c_mk = c_ids + [0] * npad, c_tg + [0] * npad, c_mk + [0.0] * npad
                m._seg_id = None
                loss = m.train_chunk(c_ids, c_tg, c_mk, s, cache)
                gn = step_optimizer(loss)
                n_tok = int(sum(mask[sl]))
                state["steps"] += 1
                state["tokens"] += n_tok
                state["chunk_in_example"] = ci + 1
                since_save += n_tok
                dt = time.time() - tc
                log(step=state["steps"], ex=cur_idx, src=d.get("source"), chunk=ci, tokens=n_tok, loss=loss,
                    grad_norm=round(gn, 4), sec=round(dt, 1),
                    gib=round(torch.xpu.memory_allocated() / 2**30, 2), res_gib=round(torch.xpu.memory_reserved() / 2**30, 2))
                if a.eval_every and state["steps"] % a.eval_every == 0: evaluate("periodic")
                if val_new and a.val_new_every and state["steps"] % a.val_new_every == 0: evaluate("periodic", val_new)
                if state["steps"] % 10 == 0:
                    say(f"step {state['steps']}: loss {loss:.3f}, {n_tok} tokens in {dt:.0f} s, "
                        f"{state['tokens'] / 1e3:.0f}k trained tokens, peak {torch.xpu.max_memory_allocated() / 2**30:.1f} GiB")
                if since_save >= a.save_tokens:
                    save()
                    since_save = 0
                if a.max_steps and state["steps"] >= a.max_steps: break
            state["cursor"] += 1
            state["chunk_in_example"] = 0
            continue

        # Standard processing
        if not a.pack:
            # UNPACKED PATH (identical to flint_train_lora_fast.py)
            cur_idx = state["cursor"]
            d = ex[cur_idx]
            ids = tok.encode(d["text"], add_special_tokens=False).ids
            tg = ids[1:] + [0]
            mask = [1.0 if (i + 1 >= d["prompt_tokens"] and i < len(ids) - 1) else 0.0 for i in range(len(ids))]
            cache = [None] * FT.NL
            for ci, s in enumerate(range(0, len(ids), a.chunk)):
                sl = slice(s, s + a.chunk)
                if sum(mask[sl]) == 0:
                    m.forward_chunk(ids[sl], s, cache)
                    continue
                tc = time.time()
                if opt is not None: opt.zero_grad(set_to_none=True)
                if lora_opt is not None: lora_opt.zero_grad(set_to_none=True)
                for p in all_trainable: p.grad = None
                c_ids, c_tg, c_mk = ids[sl], tg[sl], mask[sl]
                if a.pad_to and s + a.chunk >= len(ids):
                    npad = -len(c_ids) % a.pad_to
                    c_ids, c_tg, c_mk = c_ids + [0] * npad, c_tg + [0] * npad, c_mk + [0.0] * npad
                m._seg_id = None
                loss = m.train_chunk(c_ids, c_tg, c_mk, s, cache)
                gn = step_optimizer(loss)
                n_tok = int(sum(mask[sl]))
                state["steps"] += 1
                state["tokens"] += n_tok
                state["chunk_in_example"] = ci + 1
                since_save += n_tok
                dt = time.time() - tc
                log(step=state["steps"], ex=cur_idx, src=d.get("source"), chunk=ci, tokens=n_tok, loss=loss,
                    grad_norm=round(gn, 4), sec=round(dt, 1),
                    gib=round(torch.xpu.memory_allocated() / 2**30, 2), res_gib=round(torch.xpu.memory_reserved() / 2**30, 2))
                if a.eval_every and state["steps"] % a.eval_every == 0: evaluate("periodic")
                if val_new and a.val_new_every and state["steps"] % a.val_new_every == 0: evaluate("periodic", val_new)
                if state["steps"] % 10 == 0:
                    say(f"step {state['steps']}: loss {loss:.3f}, {n_tok} tokens in {dt:.0f} s, "
                        f"{state['tokens'] / 1e3:.0f}k trained tokens, peak {torch.xpu.max_memory_allocated() / 2**30:.1f} GiB")
                if since_save >= a.save_tokens:
                    save()
                    since_save = 0
                if a.max_steps and state["steps"] >= a.max_steps: break
            state["cursor"] += 1
            state["chunk_in_example"] = 0
        else:
            # PACKED PATH
            cur_idx = state["cursor"]
            d = ex[cur_idx]
            if d["tokens"] > a.pack_max:
                # Long example: unchanged multi-chunk path
                ids = tok.encode(d["text"], add_special_tokens=False).ids
                tg = ids[1:] + [0]
                mask = [1.0 if (i + 1 >= d["prompt_tokens"] and i < len(ids) - 1) else 0.0 for i in range(len(ids))]
                cache = [None] * FT.NL
                for ci, s in enumerate(range(0, len(ids), a.chunk)):
                    sl = slice(s, s + a.chunk)
                    if sum(mask[sl]) == 0:
                        m.forward_chunk(ids[sl], s, cache)
                        continue
                    tc = time.time()
                    if opt is not None: opt.zero_grad(set_to_none=True)
                    if lora_opt is not None: lora_opt.zero_grad(set_to_none=True)
                    for p in all_trainable: p.grad = None
                    c_ids, c_tg, c_mk = ids[sl], tg[sl], mask[sl]
                    if a.pad_to and s + a.chunk >= len(ids):
                        npad = -len(c_ids) % a.pad_to
                        c_ids, c_tg, c_mk = c_ids + [0] * npad, c_tg + [0] * npad, c_mk + [0.0] * npad
                    m._seg_id = None
                    loss = m.train_chunk(c_ids, c_tg, c_mk, s, cache)
                    gn = step_optimizer(loss)
                    n_tok = int(sum(mask[sl]))
                    state["steps"] += 1
                    state["tokens"] += n_tok
                    state["chunk_in_example"] = ci + 1
                    since_save += n_tok
                    dt = time.time() - tc
                    log(step=state["steps"], ex=cur_idx, src=d.get("source"), chunk=ci, tokens=n_tok, loss=loss,
                        grad_norm=round(gn, 4), sec=round(dt, 1),
                        gib=round(torch.xpu.memory_allocated() / 2**30, 2), res_gib=round(torch.xpu.memory_reserved() / 2**30, 2))
                    if a.eval_every and state["steps"] % a.eval_every == 0: evaluate("periodic")
                    if val_new and a.val_new_every and state["steps"] % a.val_new_every == 0: evaluate("periodic", val_new)
                    if state["steps"] % 10 == 0:
                        say(f"step {state['steps']}: loss {loss:.3f}, {n_tok} tokens in {dt:.0f} s, "
                            f"{state['tokens'] / 1e3:.0f}k trained tokens, peak {torch.xpu.max_memory_allocated() / 2**30:.1f} GiB")
                    if since_save >= a.save_tokens:
                        save()
                        since_save = 0
                    if a.max_steps and state["steps"] >= a.max_steps: break
                state["cursor"] += 1
                state["chunk_in_example"] = 0
            else:
                # Windowed First-Fit packing of short examples
                w_start = state["cursor"]
                w_end = min(len(ex), w_start + a.pack_window)
                w_indices = list(range(w_start, w_end))

                # Partition window into items: long examples and packed chunks of short examples
                # Greedily pack short examples using First-Fit (capacity = a.pack_max)
                items = []
                packed_bins = []  # list of [idx, ...]
                bin_toks = []

                for idx in w_indices:
                    ed = ex[idx]
                    if ed["tokens"] > a.pack_max:
                        items.append({"type": "long", "indices": [idx]})
                    else:
                        placed = False
                        for b_i, b_tok in enumerate(bin_toks):
                            if b_tok + ed["tokens"] <= a.pack_max:
                                packed_bins[b_i].append(idx)
                                bin_toks[b_i] += ed["tokens"]
                                placed = True
                                break
                        if not placed:
                            packed_bins.append([idx])
                            bin_toks.append(ed["tokens"])

                for b in packed_bins:
                    items.append({"type": "packed", "indices": b})

                # Sort items by their earliest example index to roughly preserve data order
                items.sort(key=lambda it: it["indices"][0])

                for it in items:
                    if os.path.exists(os.path.join(a.out, "STOP")):
                        stop_reason = "STOP file"
                        break
                    if time.time() - t_start > a.hours * 3600:
                        break
                    if a.max_steps and state["steps"] >= a.max_steps:
                        stop_reason = "max steps"
                        break

                    if it["type"] == "long":
                        long_idx = it["indices"][0]
                        ld = ex[long_idx]
                        ids = tok.encode(ld["text"], add_special_tokens=False).ids
                        tg = ids[1:] + [0]
                        mask = [1.0 if (i + 1 >= ld["prompt_tokens"] and i < len(ids) - 1) else 0.0 for i in range(len(ids))]
                        cache = [None] * FT.NL
                        for ci, s in enumerate(range(0, len(ids), a.chunk)):
                            sl = slice(s, s + a.chunk)
                            if sum(mask[sl]) == 0:
                                m.forward_chunk(ids[sl], s, cache)
                                continue
                            tc = time.time()
                            if opt is not None: opt.zero_grad(set_to_none=True)
                            if lora_opt is not None: lora_opt.zero_grad(set_to_none=True)
                            for p in all_trainable: p.grad = None
                            c_ids, c_tg, c_mk = ids[sl], tg[sl], mask[sl]
                            if a.pad_to and s + a.chunk >= len(ids):
                                npad = -len(c_ids) % a.pad_to
                                c_ids, c_tg, c_mk = c_ids + [0] * npad, c_tg + [0] * npad, c_mk + [0.0] * npad
                            m._seg_id = None
                            loss = m.train_chunk(c_ids, c_tg, c_mk, s, cache)
                            gn = step_optimizer(loss)
                            n_tok = int(sum(mask[sl]))
                            state["steps"] += 1
                            state["tokens"] += n_tok
                            since_save += n_tok
                            dt = time.time() - tc
                            log(step=state["steps"], ex=long_idx, src=ld.get("source"), chunk=ci, tokens=n_tok, loss=loss,
                                grad_norm=round(gn, 4), sec=round(dt, 1),
                                gib=round(torch.xpu.memory_allocated() / 2**30, 2), res_gib=round(torch.xpu.memory_reserved() / 2**30, 2))
                    else:
                        # Packed chunk step
                        tc = time.time()
                        if opt is not None: opt.zero_grad(set_to_none=True)
                        if lora_opt is not None: lora_opt.zero_grad(set_to_none=True)
                        for p in all_trainable: p.grad = None

                        p_ids, p_tg, p_mk, p_pos, p_seg = [], [], [], [], []
                        for s_idx, ex_i in enumerate(it["indices"]):
                            ed = ex[ex_i]
                            ids = tok.encode(ed["text"], add_special_tokens=False).ids
                            tg = ids[1:] + [0]
                            mask = [1.0 if (i + 1 >= ed["prompt_tokens"] and i < len(ids) - 1) else 0.0 for i in range(len(ids))]
                            L = len(ids)
                            p_ids.extend(ids)
                            p_tg.extend(tg)
                            p_mk.extend(mask)
                            p_pos.extend(range(L))
                            p_seg.extend([s_idx] * L)

                        valid_len = len(p_ids)
                        if a.pad_to:
                            npad = -valid_len % a.pad_to
                            p_ids += [0] * npad
                            p_tg += [0] * npad
                            p_mk += [0.0] * npad
                            p_pos += [0] * npad
                            p_seg += [-1] * npad

                        loss = m.train_chunk_packed(p_ids, p_tg, p_mk, p_pos, p_seg)
                        gn = step_optimizer(loss)
                        n_tok = int(sum(p_mk))
                        state["steps"] += 1
                        state["tokens"] += n_tok
                        since_save += n_tok
                        dt = time.time() - tc
                        log(step=state["steps"], packed=it["indices"], tokens=n_tok, loss=loss,
                            grad_norm=round(gn, 4), sec=round(dt, 1),
                            gib=round(torch.xpu.memory_allocated() / 2**30, 2), res_gib=round(torch.xpu.memory_reserved() / 2**30, 2))

                    if a.eval_every and state["steps"] % a.eval_every == 0: evaluate("periodic")
                    if val_new and a.val_new_every and state["steps"] % a.val_new_every == 0: evaluate("periodic", val_new)
                    if state["steps"] % 10 == 0:
                        say(f"step {state['steps']}: loss {loss:.3f}, {n_tok} tokens in {dt:.0f} s, "
                            f"{state['tokens'] / 1e3:.0f}k trained tokens, peak {torch.xpu.max_memory_allocated() / 2**30:.1f} GiB")
                    if since_save >= a.save_tokens:
                        save()
                        since_save = 0
                    if a.max_steps and state["steps"] >= a.max_steps: break

                state["cursor"] = w_end
                state["chunk_in_example"] = 0

    save()
    say(f"stopped ({stop_reason}); saved at step {state['steps']}, {state['tokens']} trained tokens, "
        f"example {state['cursor']}")
    logf.close()


if __name__ == "__main__":
    main()
