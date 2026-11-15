"""Supervisor for the TWO-GPU split trainer (flint_train_split.py). Same behaviour as flint_supervisor_p2.py (restart after each
clean --hours stop and after crashes, wait + GPU check, stop when STOP is in the run folder) but both GPUs are visible.
  python flint_supervisor_split.py --gpus 0,1 --out PATH\\TO\\run -- <trainer args without --out>
Refuses to start while flint_train_p2 / flint_supervisor_p2 / another split trainer is running: two trainers on one run
folder would corrupt save.pt (and the GPUs are shared). Log: <out>\\supervisor.log"""
import os
import subprocess
import sys
import time

GPU_CHECK = ("import torch\n"
             "for i in range(torch.xpu.device_count()):\n"
             "    x = torch.randn(512, 512, device=f'xpu:{i}')\n"
             "    print(i, float((x @ x).sum()))\n")


def other_trainers():
    """command lines of other trainer / supervisor processes (this process and its venv launcher parent excluded)"""
    me = {os.getpid(), os.getppid()}
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
          "'flint_train_p2\\.py|flint_supervisor_p2\\.py|flint_train_split\\.py|flint_supervisor_split\\.py' -and "
          "$_.Name -notmatch 'powershell' } | ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    out = []
    for l in r.stdout.splitlines():
        if "|" in l and int(l.split("|", 1)[0]) not in me:
            out.append(l)
    return out


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: flint_supervisor_split.py --gpus 0,1 --out DIR -- <trainer args>")
    mine, trainer = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    gpus = mine[mine.index("--gpus") + 1] if "--gpus" in mine else "0,1"
    out = mine[mine.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, ZE_AFFINITY_MASK=gpus)
    here = os.path.dirname(os.path.abspath(__file__))
    logp = os.path.join(out, "supervisor.log")

    def note(s):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
        print(line, flush=True)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    busy = other_trainers()
    if busy:
        note("REFUSING to start, another trainer is running (stop it first): " + " || ".join(l[:200] for l in busy))
        sys.exit(2)
    crashes = []
    note(f"split supervisor started (GPUs {gpus}, run folder {out})")
    while not os.path.exists(os.path.join(out, "STOP")):
        cmd = [sys.executable, os.path.join(here, "flint_train_split.py"), "--out", out] + trainer
        t0 = time.time()
        with open(os.path.join(out, "trainer_console.log"), "a", encoding="utf-8") as con:
            rc = subprocess.call(cmd, env=env, stdout=con, stderr=subprocess.STDOUT, cwd=here)
        mins = (time.time() - t0) / 60
        if rc == 0:
            note(f"trainer stopped cleanly after {mins:.0f} min; restarting")
            crashes.clear()
            continue
        crashes = [t for t in crashes if time.time() - t < 3600] + [time.time()]
        note(f"trainer CRASHED (exit {rc}) after {mins:.0f} min; crash {len(crashes)} in the last hour")
        wait = 60 if len(crashes) < 4 else 1800          # many crashes in a row: let the driver/PC settle
        time.sleep(wait)
        chk = subprocess.call([sys.executable, "-c", GPU_CHECK], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        note(f"GPU check after the crash (both GPUs): {'OK' if chk == 0 else 'FAILED (exit %d)' % chk}")
        if chk != 0:
            time.sleep(600)
    note("STOP file found: supervisor ends")


if __name__ == "__main__":
    main()
