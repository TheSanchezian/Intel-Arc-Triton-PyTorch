"""Phase-2 trainer on TWO GPUs (model split / pipeline), so training examples can be longer than 4096 tokens.

Same CLI, data handling, loss masking, save format (save.pt / save_prev.pt / daily_*.pt: params + Adam state + data position)
and STOP / --hours behaviour as flint_train_p2.py; a save.pt written by either trainer resumes in the other.
Extra flags: --devices (default xpu:0,xpu:1), --split-at K (0 = auto-balance), --kv-tokens (sizing hint for the auto split),
--force-hop. Run it through start_flint_train_split.bat / flint_supervisor_split.py (both GPUs visible).

  layers 0..K-1 + embedding lookup on device 0, layers K..48 + final norm + head + loss on device 1 (see flint_torch_split.py).
"""
import argparse
import json
import os
import random
import shutil
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


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
    ap.add_argument("--data", required=True, help="folder with train.jsonl / val.jsonl (flint_data_build.py)")
    ap.add_argument("--extra", default="", help="optional extra JSONL in the same format (e.g. verified Gemma data)")
    ap.add_argument("--extra-weight", type=int, default=3, help="each extra example is seen this many times per pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=256, help="tokens per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--save-tokens", type=int, default=1280)
    ap.add_argument("--max-example-tokens", type=int, default=2048)
    ap.add_argument("--hours", type=float, default=1.0, help="stop cleanly after this long (supervisor restarts)")
    ap.add_argument("--dtype", default="float32", help="float32 matched the runtime (T1); bfloat16 did not")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-max-tokens", type=int, default=2048, help="validation examples are picked at this length cap (kept at 2048 so scores stay comparable)")
    ap.add_argument("--trace", action="store_true", help="sync + print a marker after every stage (crash hunting)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = no limit)")
    ap.add_argument("--eval-every", type=int, default=100, help="validation loss every N steps (0 = off)")
    ap.add_argument("--eval-examples", type=int, default=12, help="fixed validation examples per evaluation")
    ap.add_argument("--moe-group", type=int, default=32, help="experts per batched launch (0 = old one-by-one loop)")
    ap.add_argument("--pad-to", type=int, default=256, help="pad the last chunk of an example to a multiple of this")
    ap.add_argument("--empty-cache", action="store_true",
                    help="hand cached GPU memory back after every step (P2: crashed the driver; off by default)")
    ap.add_argument("--cpu-adam", action="store_true", help="accepted for CLI compatibility; NOT supported in split mode (ignored)")
    ap.add_argument("--ins-dtype", default="none", choices=["none", "bf16", "fp16"], help="K5/T4: store layer inputs in half precision")
    ap.add_argument("--mt-bucket", type=int, default=0, help="K5: padded tokens per expert bucket (0 = module default)")
    ap.add_argument("--att-qb", type=int, default=-1, help="K5/T1: attention query block (-1 = module default, 0 = off)")
    ap.add_argument("--head-slice", type=int, default=0, help="K5/T2: vocab rows per head slice (0 = module default)")
    ap.add_argument("--train-file", default="", help="P2: train JSONL to use instead of <data>/train.jsonl")
    ap.add_argument("--init-from", default="", help="P2: save.pt to take params + Adam state from when this run has no save of its own (step/cursor restart at 0)")
    ap.add_argument("--val-new", default="", help="P2: extra validation JSONL (phase 2 held-out rows)")
    ap.add_argument("--val-new-examples", type=int, default=40, help="P2: first N rows of --val-new used (file is ordered so any first-N is stratified)")
    ap.add_argument("--val-new-every", type=int, default=300, help="P2: val_new evaluation every N steps (0 = only at start)")
    ap.add_argument("--eval-now", action="store_true", help="evaluate once before step 1 (the baseline), then train")
    ap.add_argument("--devices", default="xpu:0,xpu:1", help="SPLIT: devices of the two halves (first = embedding + low layers)")
    ap.add_argument("--split-at", type=int, default=0, help="SPLIT: first layer on the second device (0 = auto-balance by memory)")
    ap.add_argument("--kv-tokens", type=int, default=16384, help="SPLIT: example length the auto split sizes the KV cache for")
    ap.add_argument("--force-hop", action="store_true", help="SPLIT: hop through host memory even if both halves share a device (tests)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a", encoding="utf-8")

    def log(**kw):
        kw["t"] = round(time.time(), 1)
        logf.write(json.dumps(kw) + "\n")
        logf.flush()

    def say(s):
        print(f"[train {time.strftime('%H:%M:%S')}] {s}", flush=True)

    import torch
    import flint_torch as FT
    import flint_torch_split as FS
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    ex = load_examples(a.train_file or os.path.join(a.data, "train.jsonl"), a.max_example_tokens, a.seed)
    if a.extra and os.path.exists(a.extra):
        extra = load_examples(a.extra, a.max_example_tokens, a.seed + 1) * a.extra_weight
        ex = ex + extra
        random.Random(a.seed + 2).shuffle(ex)
        say(f"extra data: {len(extra) // a.extra_weight} examples x {a.extra_weight}")
    say(f"{len(ex)} examples <= {a.max_example_tokens} tokens")
    if a.cpu_adam:
        say("WARNING: --cpu-adam is not supported in split mode; ignored (Adam state stays on each GPU)")

    t0 = time.time()
    m = FS.FlintSplit(a.gguf, tuple(s.strip() for s in a.devices.split(",")), a.split_at, a.dtype, log=lambda s: say(s),
                      kv_tokens=a.kv_tokens, force_hop=a.force_hop)
    mr = m.memory_report()
    say(f"model on the GPUs in {time.time() - t0:.0f} s, split at layer {m.K}: "
        f"dev0 {mr[0][0]:.2f} GiB, dev1 {mr[1][0]:.2f} GiB, "
        f"{sum(p.numel() for p in m.params.values()) / 1e6:.1f}M trainable")
    m.MOE_GROUP = a.moe_group
    if a.mt_bucket: m.MT_BUCKET = a.mt_bucket
    if a.att_qb >= 0: m.ATT_QB = a.att_qb
    if a.head_slice: m.HEAD_SLICE = a.head_slice
    m.INS_DT = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[a.ins_dtype]
    params = list(m.params.values())
    if a.trace:
        m.trace = say
    opt = FS.SplitAdam(torch, m.groups, a.lr)            # one Adam per device, run2-compatible state dict
    save_path, prev_path = os.path.join(a.out, "save.pt"), os.path.join(a.out, "save_prev.pt")
    resumed = False
    state = {"cursor": 0, "chunk_in_example": 0, "steps": 0, "tokens": 0, "epoch": 0, "last_daily": 0.0}
    for path in (save_path, prev_path):                 # resume from the newest readable save
        if os.path.exists(path):
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                with torch.no_grad():
                    for k, p in m.params.items():
                        p.copy_(ck["params"][k].to(p.device))
                opt.load_state_dict(ck["opt"])
                state.update(ck["state"])
                resumed = True
                say(f"resumed from {os.path.basename(path)}: step {state['steps']}, {state['tokens']} trained tokens, "
                    f"example {state['cursor']}")
                break
            except Exception as e:                       # a damaged save: try the previous one
                say(f"could not load {os.path.basename(path)} ({type(e).__name__}: {e})")

    if not resumed and a.init_from and os.path.exists(a.init_from):   # P2: start phase 2 from the phase 1 result
        ck = torch.load(a.init_from, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for k, p in m.params.items():
                p.copy_(ck["params"][k].to(p.device))
        opt.load_state_dict(ck["opt"])
        say(f"INIT from {a.init_from}: params + Adam state of phase 1 step {ck['state']['steps']}; phase 2 step/cursor start at 0")
        del ck

    def save():
        ck = {"params": {k: p.detach().cpu() for k, p in m.params.items()}, "opt": opt.state_dict(), "state": state}
        tmp = save_path + ".tmp"
        torch.save(ck, tmp)
        if os.path.exists(save_path):
            os.replace(save_path, prev_path)
        os.replace(tmp, save_path)
        if time.time() - state.get("last_daily", 0) > 86400:
            day = time.strftime("%Y%m%d_%H%M")
            shutil.copyfile(save_path, os.path.join(a.out, f"daily_{day}.pt"))
            state["last_daily"] = time.time()

    val = load_examples(os.path.join(a.data, "val.jsonl"), a.eval_max_tokens, 0)[:a.eval_examples]   # K5: fixed val set (same 12 as before)

    val_new = load_examples(a.val_new, a.eval_max_tokens, 0)[:a.val_new_examples] if a.val_new else []
    if a.val_new:                                       # load_examples shuffles: restore the file order (stratified prefix)
        rows = [json.loads(l) for l in open(a.val_new, encoding="utf-8")]
        val_new = [d for d in rows if d["tokens"] <= a.eval_max_tokens][:a.val_new_examples]

    def evaluate(tag, vset=None, name="val_loss"):
        vset = val if vset is None else vset
        te, tot, cnt = time.time(), 0.0, 0.0
        for d in vset:
            l, n = m.eval_example(tok.encode(d["text"], add_special_tokens=False).ids, d["prompt_tokens"], a.chunk)
            tot, cnt = tot + l, cnt + n
        vl = tot / max(cnt, 1)
        log(eval=tag, set=("p1val" if vset is val else "val_new"), step=state["steps"], val_loss=round(vl, 5), val_tokens=int(cnt), sec=round(time.time() - te, 1))
        say(f"VALIDATION ({tag}, {'p1val' if vset is val else 'val_new'}) at step {state['steps']}: loss {vl:.4f} on {int(cnt)} answer tokens "
            f"({len(vset)} fixed examples, {time.time() - te:.0f} s)")

    if a.eval_now and state["steps"] == 0:              # baseline once; not again at every 4-hour restart
        evaluate("start")
        if val_new:
            evaluate("start", val_new)
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
        d = ex[state["cursor"]]
        ids = tok.encode(d["text"], add_special_tokens=False).ids
        tg = ids[1:] + [0]
        mask = [1.0 if (i + 1 >= d["prompt_tokens"] and i < len(ids) - 1) else 0.0 for i in range(len(ids))]
        cache = [None] * FT.NL
        start_chunk = state["chunk_in_example"]
        for ci, s in enumerate(range(0, len(ids), a.chunk)):
            sl = slice(s, s + a.chunk)
            if ci < start_chunk or sum(mask[sl]) == 0:   # resumed mid-example, or prompt only: just fill the cache
                m.forward_chunk(ids[sl], s, cache)
                continue
            tc = time.time()
            opt.zero_grad(set_to_none=True)
            for p in params:
                p.grad = None
            c_ids, c_tg, c_mk = ids[sl], tg[sl], mask[sl]
            if a.pad_to and s + a.chunk >= len(ids):     # S1: last chunk padded at the END (mask 0, causal: nothing
                npad = -len(c_ids) % a.pad_to            # reads it, no later chunk uses the cache) -> fixed GPU shapes
                c_ids, c_tg, c_mk = c_ids + [0] * npad, c_tg + [0] * npad, c_mk + [0.0] * npad
            loss = m.train_chunk(c_ids, c_tg, c_mk, s, cache)
            gn = FS.clip_grad_norm_split(torch, m.groups, a.clip)
            opt.step()
            if a.empty_cache:                            # P2: with this on, the driver crashed (access violation,
                for h in m.halves:                       # DEVICE_LOST) within minutes; off by default
                    with m._guard(h.dev):
                        torch.xpu.synchronize()
                        torch.xpu.empty_cache()
            n_tok = int(sum(mask[sl]))
            state["steps"] += 1
            state["tokens"] += n_tok
            state["chunk_in_example"] = ci + 1
            since_save += n_tok
            dt = time.time() - tc
            mr = m.memory_report()
            log(step=state["steps"], ex=state["cursor"], src=d.get("source"), chunk=ci, tokens=n_tok, loss=loss,
                grad_norm=round(gn, 4), sec=round(dt, 1),
                gib=round(max(x[0] for x in mr), 2), res_gib=round(max(x[1] for x in mr), 2),
                gib0=round(mr[0][0], 2), gib1=round(mr[1][0], 2), res0=round(mr[0][1], 2), res1=round(mr[1][1], 2))
            if a.eval_every and state["steps"] % a.eval_every == 0:
                evaluate("periodic")
            if val_new and a.val_new_every and state["steps"] % a.val_new_every == 0:
                evaluate("periodic", val_new)
            if state["steps"] % 10 == 0:
                say(f"step {state['steps']}: loss {loss:.3f}, {n_tok} tokens in {dt:.0f} s, "
                    f"{state['tokens'] / 1e3:.0f}k trained tokens, peak {mr[0][2]:.1f}/{mr[1][2]:.1f} GiB (dev0/dev1)")
            if since_save >= a.save_tokens:
                save()
                since_save = 0
            if a.max_steps and state["steps"] >= a.max_steps:
                break
        state["cursor"] += 1
        state["chunk_in_example"] = 0
    save()
    mr = m.memory_report()
    say(f"stopped ({stop_reason}); saved at step {state['steps']}, {state['tokens']} trained tokens, "
        f"peak {mr[0][2]:.1f}/{mr[1][2]:.1f} GiB (dev0/dev1)")


if __name__ == "__main__":
    main()
