#!/usr/bin/env python
"""S2: watch the orphaned c075 FT app (client killed at step ~421/500).

Polls every 60 s until either:
  - the s2_c075 adapter appears in the ft-adapters volume (app survived to
    completion and committed), or
  - no extra bag-sft-s2 ephemeral app remains (the orphan was reaped without
    saving), or
  - 20 minutes elapse.
Prints a single verdict line: ADAPTER_SAVED | APP_STOPPED_NO_ADAPTER | TIMEOUT
"""
import os
import pathlib
import subprocess
import time

WT = pathlib.Path(__file__).resolve().parents[2]
env = dict(os.environ)
env.pop("VIRTUAL_ENV", None)


def adapter_saved():
    r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", "s2_c075"],
                       cwd=WT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and "adapter_config.json" in r.stdout


def n_ephemeral_sft_apps():
    r = subprocess.run(["uv", "run", "modal", "app", "list"],
                       cwd=WT, env=env, capture_output=True, text=True)
    return sum(1 for line in r.stdout.splitlines()
               if "bag-sft-s2" in line and "ephemeral" in line)


def main():
    deadline = time.time() + 20 * 60
    while time.time() < deadline:
        if adapter_saved():
            print("VERDICT: ADAPTER_SAVED", flush=True)
            return
        n = n_ephemeral_sft_apps()
        print(f"  poll: ephemeral bag-sft-s2 apps={n}, adapter not yet saved", flush=True)
        # the 14B FT owns one ephemeral app; if only <=1 remains, the c075 orphan is gone
        if n <= 1:
            print("VERDICT: APP_STOPPED_NO_ADAPTER", flush=True)
            return
        time.sleep(60)
    print("VERDICT: TIMEOUT", flush=True)


if __name__ == "__main__":
    main()
