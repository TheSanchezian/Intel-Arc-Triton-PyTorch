"""Keeps flint_train.py running for weeks: restarts it after each clean hourly stop, and after crashes (waits, checks
the GPU, then resumes from the last save). Stops when a file named STOP is in the run folder.
  python flint_supervisor.py --gpu 1 --out RUN_DIR -- <flint_train.py arguments without --out>
Log: <out>\\supervisor.log"""
import os
import subprocess
import sys
import time


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: flint_supervisor.py --gpu N --out DIR -- <trainer args>")
    mine, trainer = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    gpu = mine[mine.index("--gpu") + 1] if "--gpu" in mine else "1"
    out = mine[mine.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, ZE_AFFINITY_MASK=gpu)
    here = os.path.dirname(os.path.abspath(__file__))
    logp = os.path.join(out, "supervisor.log")

    def note(s):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
        print(line, flush=True)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    crashes = []
    note(f"supervisor started (GPU {gpu}, run folder {out})")
    while not os.path.exists(os.path.join(out, "STOP")):
        cmd = [sys.executable, os.path.join(here, "flint_train.py"), "--out", out] + trainer
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
        chk = subprocess.call([sys.executable, "-c",
                               "import torch; x=torch.randn(512,512,device='xpu'); print(float((x@x).sum()))"],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        note(f"GPU check after the crash: {'OK' if chk == 0 else 'FAILED (exit %d)' % chk}")
        if chk != 0:
            time.sleep(600)
    note("STOP file found: supervisor ends")


if __name__ == "__main__":
    main()
