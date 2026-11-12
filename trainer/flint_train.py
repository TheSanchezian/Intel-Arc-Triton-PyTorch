"""Train Flint's expert row multipliers on the data pool (PyTorch XPU), resumable, with save points.

  python flint_train.py --gguf ... --tokenizer ... --data DATA_DIR --out RUN_DIR
         [--chunk 256] [--lr 1e-4] [--save-tokens 1280] [--max-example-tokens 2048] [--hours 1.0]

- One optimizer step per chunk (<= 256 tokens) that contains answer tokens; prompt-only chunks just fill the cache.
- Loss only on the assistant part (from the example's prompt_tokens on), including the end token (teaches stopping).
- Save point every --save-tokens trained tokens: params + Adam state + data position, written to a temp file then
  renamed (a crash during a save cannot destroy the last good save); the previous save is kept as save_prev.pt.
- Stops cleanly after --hours (the supervisor restarts it, which also clears driver state) or when a file named
  STOP appears in --out. Every step is logged to train_log.jsonl; a long-term checkpoint is kept once a day.
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
    ap.add_argument("--chunk", type=int, default=256, help="tokens per step (R5a: 512+ hangs the driver)")
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
    ap.add_argument("--cpu-adam", action="store_true", help="K5/T3: Adam state + master params on the CPU")
    ap.add_argument("--ins-dtype", default="none", choices=["none", "bf16", "fp16"], help="K5/T4: store layer inputs in half precision")
    ap.add_argument("--mt-bucket", type=int, default=0, help="K5: padded tokens per expert bucket (0 = module default)")
    ap.add_argument("--att-qb", type=int, default=-1, help="K5/T1: attention query block (-1 = module default, 0 = off)")
    ap.add_argument("--head-slice", type=int, default=0, help="K5/T2: vocab rows per head slice (0 = module default)")
    ap.add_argument("--eval-now", action="store_true", help="evaluate once before step 1 (the baseline), then train")
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
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    ex = load_examples(os.path.join(a.data, "train.jsonl"), a.max_example_tokens, a.seed)
    if a.extra and os.path.exists(a.extra):
        extra = load_examples(a.extra, a.max_example_tokens, a.seed + 1) * a.extra_weight
        ex = ex + extra
        random.Random(a.seed + 2).shuffle(ex)
        say(f"extra data: {len(extra) // a.extra_weight} examples x {a.extra_weight}")
    say(f"{len(ex)} examples <= {a.max_example_tokens} tokens")

    t0 = time.time()
    m = FT.FlintTorch(a.gguf, "xpu", a.dtype, log=lambda s: say(s))
    say(f"model on the GPU in {time.time() - t0:.0f} s, {torch.xpu.memory_allocated() / 2 ** 30:.2f} GiB, "
        f"{sum(p.numel() for p in m.params.values()) / 1e6:.1f}M trainable")
    m.MOE_GROUP = a.moe_group
    if a.mt_bucket: m.MT_BUCKET = a.mt_bucket
    if a.att_qb >= 0: m.ATT_QB = a.att_qb
    if a.head_slice: m.HEAD_SLICE = a.head_slice
    m.INS_DT = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[a.ins_dtype]
    params = list(m.params.values())
    if a.trace:
        m.trace = say
    if a.cpu_adam:                                       # K5/T3: Adam state + master copy live on the CPU
        cparams = [p.detach().cpu().clone().requires_grad_(True) for p in params]
        opt = torch.optim.Adam(cparams, lr=a.lr)
    else:
        opt = torch.optim.Adam(params, lr=a.lr)
    save_path, prev_path = os.path.join(a.out, "save.pt"), os.path.join(a.out, "save_prev.pt")
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
                say(f"resumed from {os.path.basename(path)}: step {state['steps']}, {state['tokens']} trained tokens, "
                    f"example {state['cursor']}")
                break
            except Exception as e:                       # a damaged save: try the previous one
                say(f"could not load {os.path.basename(path)} ({type(e).__name__}: {e})")

    if a.cpu_adam:
        with torch.no_grad():
            for cp, p in zip(cparams, params):
                cp.copy_(p.detach().cpu())               # master copy = (possibly resumed) GPU params

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

    def evaluate(tag):
        te, tot, cnt = time.time(), 0.0, 0.0
        for d in val:
            l, n = m.eval_example(tok.encode(d["text"], add_special_tokens=False).ids, d["prompt_tokens"], a.chunk)
            tot, cnt = tot + l, cnt + n
        vl = tot / max(cnt, 1)
        log(eval=tag, step=state["steps"], val_loss=round(vl, 5), val_tokens=int(cnt), sec=round(time.time() - te, 1))
        say(f"VALIDATION ({tag}) at step {state['steps']}: loss {vl:.4f} on {int(cnt)} answer tokens "
            f"({len(val)} fixed examples, {time.time() - te:.0f} s)")

    if a.eval_now and state["steps"] == 0:              # baseline once; not again at every 4-hour restart
        evaluate("start")
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
            gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
            if a.cpu_adam:
                for cp, p in zip(cparams, params):
                    cp.grad = p.grad.detach().cpu()
                opt.step()
                with torch.no_grad():
                    for cp, p in zip(cparams, params):
                        p.copy_(cp.to(p.device))
            else:
                opt.step()
            if a.empty_cache:                            # P2: with this on, the driver crashed (access violation,
                torch.xpu.synchronize()                  # DEVICE_LOST) within minutes; the creep is handled by
                torch.xpu.empty_cache()                  # restarting every 2 hours instead
            n_tok = int(sum(mask[sl]))
            state["steps"] += 1
            state["tokens"] += n_tok
            state["chunk_in_example"] = ci + 1
            since_save += n_tok
            dt = time.time() - tc
            log(step=state["steps"], ex=state["cursor"], src=d.get("source"), chunk=ci, tokens=n_tok, loss=loss,
                grad_norm=round(gn, 4), sec=round(dt, 1),
                gib=round(torch.xpu.memory_allocated() / 2**30, 2), res_gib=round(torch.xpu.memory_reserved() / 2**30, 2))
            if a.eval_every and state["steps"] % a.eval_every == 0:
                evaluate("periodic")
            if state["steps"] % 10 == 0:
                say(f"step {state['steps']}: loss {loss:.3f}, {n_tok} tokens in {dt:.0f} s, "
                    f"{state['tokens'] / 1e3:.0f}k trained tokens, peak {torch.xpu.max_memory_allocated() / 2**30:.1f} GiB")
            if since_save >= a.save_tokens:
                save()
                since_save = 0
            if a.max_steps and state["steps"] >= a.max_steps:
                break
        state["cursor"] += 1
        state["chunk_in_example"] = 0
    save()
    say(f"stopped ({stop_reason}); saved at step {state['steps']}, {state['tokens']} trained tokens, "
        f"peak {torch.xpu.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
