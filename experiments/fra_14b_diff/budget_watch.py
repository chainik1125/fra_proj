"""Budget watchdog for the FRA-diff overnight campaign. HARD $300 (GPU+API) cap.

Run every ~15 min (cron). Integrates GPU burn-rate over time into a persisted
accumulator (terminated pods' cost is not recoverable from a point query, so we
integrate rate × dt each tick), adds the analyst-reported API/judge spend, and
**terminates ALL `fra-diff-*` pods if the total exceeds the cap**.

SAFETY: only ever terminates pods whose name starts with `fra-diff-`. NEVER
touches dmitry-fra-* / jamie-* / aniket-* (teammates) or any other pod.

State: /tmp/fra_diff_spend.json   {gpu_usd, last_ts}
API spend (written by results-analyst): /tmp/fra_diff_api_spend.txt  (a float, USD)
Stop flag (written here on breach): /tmp/fra_diff_STOP
env: RP_API_KEY_MATS
"""
import json, os, ssl, time, urllib.request
from pathlib import Path
try:
    import certifi
    _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = ssl.create_default_context()

CAP_USD = 300.0
PREFIX = "fra-diff-"
PROTECTED = ("dmitry-fra-", "jamie-", "aniket-")   # NEVER terminate
DEFAULT_HR = 2.8       # $/hr fallback if costPerHr absent (H100 secure)
RUNAWAY_S = 12 * 3600  # per-pod 12h runaway kill
STATE = Path("/tmp/fra_diff_spend.json")
APISPEND = Path("/tmp/fra_diff_api_spend.txt")
STOP = Path("/tmp/fra_diff_STOP")
GQL = "https://api.runpod.io/graphql"


def gql(query):
    req = urllib.request.Request(GQL, data=json.dumps({"query": query}).encode(),
        headers={"Authorization": f"Bearer {os.environ['RP_API_KEY_MATS']}",
                 "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30, context=_CTX))


def terminate(pid):
    gql(f'mutation {{ podTerminate(input:{{podId:"{pid}"}}) }}')


def main():
    now = time.time()
    pods = gql("query { myself { pods { id name desiredStatus costPerHr runtime { uptimeInSeconds } } } }")["data"]["myself"]["pods"]
    running = [p for p in pods if p["desiredStatus"] == "RUNNING"]
    campaign = [p for p in running if p["name"].startswith(PREFIX)]
    # SAFETY assertion: never act on protected pods
    burn = 0.0
    for p in campaign:
        try: rate = float(p.get("costPerHr") or DEFAULT_HR)
        except Exception: rate = DEFAULT_HR
        burn += rate

    st = json.loads(STATE.read_text()) if STATE.exists() else {"gpu_usd": 0.0, "last_ts": now}
    dt_hr = max(0.0, (now - st["last_ts"]) / 3600.0)
    st["gpu_usd"] = st.get("gpu_usd", 0.0) + burn * dt_hr
    st["last_ts"] = now
    STATE.write_text(json.dumps(st))

    api_usd = 0.0
    if APISPEND.exists():
        try: api_usd = float(APISPEND.read_text().strip() or 0)
        except Exception: api_usd = 0.0
    total = st["gpu_usd"] + api_usd

    print(f"[budget] running fra-diff pods={len(campaign)} burn=${burn:.2f}/hr | "
          f"GPU=${st['gpu_usd']:.2f} API=${api_usd:.2f} TOTAL=${total:.2f}/{CAP_USD:.0f}")

    # per-pod 12h runaway kill (campaign pods only)
    for p in campaign:
        up = (p.get("runtime") or {}).get("uptimeInSeconds") or 0
        if up > RUNAWAY_S:
            terminate(p["id"]); print(f"[runaway-kill] {p['name']} up={up//3600}h")

    # HARD cap
    if total >= CAP_USD:
        STOP.write_text(f"budget breach ${total:.2f} at {int(now)}")
        for p in campaign:
            assert p["name"].startswith(PREFIX) and not p["name"].startswith(PROTECTED)
            terminate(p["id"]); print(f"[BUDGET-KILL] {p['name']}")
        print(f"[BUDGET-STOP] total ${total:.2f} >= ${CAP_USD:.0f} — terminated {len(campaign)} fra-diff pods. STOP flag set.")
    return total


if __name__ == "__main__":
    main()
