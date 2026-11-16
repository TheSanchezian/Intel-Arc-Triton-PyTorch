"""Export trained Flint-40B expert row multipliers into a new GGUF by folding them into the fp16 block scales.

W = code * scale * mult[e,n]  ==  code * (scale * mult[e,n]); only the 2-byte fp16 scale at the start of each 34-byte
PQ2_0 block is rewritten (same offsets), nothing else in the file changes.
Usage: python export_folded.py SAVE.pt SOURCE.gguf OUTPUT.gguf
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flint_rt as F

if len(sys.argv) != 4:
    sys.exit("usage: python export_folded.py SAVE.pt SOURCE.gguf OUTPUT.gguf")
SAVE = sys.argv[1]
SRC = sys.argv[2]
DST = sys.argv[3]
KINDS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
t0 = time.time()
say = lambda s: print(f"[export {time.time() - t0:6.0f}s] {s}", flush=True)

if not os.path.exists(DST):
    say(f"copying {SRC} -> {DST}")
    shutil.copyfile(SRC, DST + ".part")
    os.replace(DST + ".part", DST)
else:
    say("destination exists, not copying again")
assert os.path.getsize(DST) == os.path.getsize(SRC), "size mismatch after copy"
say(f"copy done, {os.path.getsize(DST)} bytes")

ck = torch.load(SAVE, map_location="cpu", weights_only=False)
P = ck["params"]
say(f"loaded {len(P)} multiplier tensors, state={ck['state']}")

# open the copy writable, using the reader's table (it opens read-only, so map r+ ourselves with the same offsets)
_, _, _, T = F.read_gguf(DST)
mm = np.memmap(DST, np.uint8, "r+")

n_changed, mn, mx, sm, cnt = 0, np.inf, -np.inf, 0.0, 0
max_rel_all, max_rel_normal = 0.0, 0.0
n_nonfinite_new, n_subnormal_new, n_zero_new, n_nonfinite_old, n_mult_bad = 0, 0, 0, 0, 0
n_scales = 0
tiny = float(np.finfo(np.float16).tiny)       # smallest normal fp16 (6.1e-5)
per = []
for L in range(2, F.NL if hasattr(F, "NL") else 49):
    for kind in KINDS:
        key = f"L{L}.{kind}"
        mult = P[key].numpy().astype(np.float32)               # [E, N, 1]
        raw = F.raw_expert_tensor(mm, T, L, kind)               # [E,N,G,34] view onto the r+ memmap
        assert mult.shape[:2] == raw.shape[:2], (key, mult.shape, raw.shape)
        if not np.all(np.isfinite(mult)):
            n_mult_bad += int((~np.isfinite(mult)).sum())
        mn, mx = min(mn, float(mult.min())), max(mx, float(mult.max()))
        sm += float(mult.astype(np.float64).sum()); cnt += mult.size
        E_, N_, G_, _ = raw.shape
        lay_rel = 0.0
        for e0 in range(0, E_, 32):
            e1 = min(e0 + 32, E_)
            blk = raw[e0:e1]
            old = np.ascontiguousarray(blk[..., :2]).view(np.float16).reshape(e1 - e0, N_, G_)
            exact = old.astype(np.float64) * mult[e0:e1].astype(np.float64)       # [g,N,G] (mult [g,N,1])
            new = exact.astype(np.float16)                                         # round-to-nearest-even fp16
            n_nonfinite_old += int((~np.isfinite(old)).sum())
            nf = ~np.isfinite(new)
            n_nonfinite_new += int(nf.sum())
            absn = np.abs(new.astype(np.float64))
            n_subnormal_new += int(((absn < tiny) & (absn > 0)).sum())
            n_zero_new += int(((new == 0) & (exact != 0)).sum())
            ok = np.isfinite(new) & (exact != 0)
            rel = np.zeros_like(exact)
            rel[ok] = np.abs(new.astype(np.float64)[ok] - exact[ok]) / np.abs(exact[ok])
            max_rel_all = max(max_rel_all, float(rel.max()))
            okn = ok & (np.abs(exact) >= tiny)
            if okn.any():
                r_n = float(rel[okn].max()); max_rel_normal = max(max_rel_normal, r_n); lay_rel = max(lay_rel, r_n)
            n_scales += new.size
            blk[..., :2] = new.view(np.uint8).reshape(e1 - e0, N_, G_, 2)
        n_changed += 1
    mm.flush()
    say(f"layer {L} done")
mm.flush()
del mm

stats = dict(tensors_changed=n_changed, scales_rewritten=n_scales, mult_min=mn, mult_mean=sm / cnt, mult_max=mx,
             mult_nonfinite=n_mult_bad, orig_scale_nonfinite=n_nonfinite_old, new_scale_nonfinite=n_nonfinite_new,
             new_scale_subnormal=n_subnormal_new, new_scale_underflow_to_zero=n_zero_new,
             max_rel_err_all=max_rel_all, max_rel_err_normal_range=max_rel_normal,
             fp16_eps_half=2.0 ** -11, seconds=round(time.time() - t0, 1))
say(json.dumps(stats))
json.dump(stats, open(os.path.splitext(DST)[0] + "_export_stats.json", "w"), indent=1)

