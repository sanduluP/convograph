"""
Unscripted supersession evaluation — content nobody wrote for this pipeline.

Runs the repo's tmp/ transcript slices (real GroupMemBench meeting excerpts,
curated by module 2's author with DOCUMENTED ground truth — see tmp/README.md)
through the full text pipeline and scores supersession behavior both ways:

    01 baseline     ordinary discussion, no reversal   -> expect 0 invalidated
    02 progression  a % moves forward, NOT a reversal  -> expect ~0 (soft)
    03 reversal ⭐   three-bucket split abandoned       -> expect >= 1
    04 reversal ⭐   backup-owners pushback             -> expect >= 1

This tests detection AND false positives on unseen content. Rendering is off
(no FLUX/caption cost); extraction runs on the configured SAIA backend, so a
full run makes a few hundred LLM calls and takes ~15-30 min.

    ./.venv/bin/python tests/corpus_eval.py          # server on :8700
"""
import asyncio
import json
import os
import sys

import httpx

BASE = "http://localhost:8700"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SLICES = [
    ("01 baseline",    "tmp/01_baseline_regulatory_compliance.txt",      "== 0"),
    ("02 progression", "tmp/02_progression_compliance_review_gate.txt",  "~ 0"),
    ("03 reversal ⭐",  "tmp/03_reversal_finance_aml_three_bucket_split.txt", ">= 1"),
    ("04 reversal ⭐",  "tmp/04_reversal_technology_backup_owners.txt",   ">= 1"),
]


async def run_slice(client: httpx.AsyncClient, name: str, path: str) -> dict:
    text = open(os.path.join(REPO, path)).read()
    r = await client.post(f"{BASE}/api/session", json={
        "title": f"corpus eval — {name}",
        "settings": {"render": False},
    })
    sid = r.json()["id"]
    await client.post(f"{BASE}/api/session/{sid}/text", json={"text": text})
    await client.post(f"{BASE}/api/session/{sid}/end")
    print(f"  {name}: session {sid} running...", flush=True)
    while True:
        await asyncio.sleep(15)
        meta = (await client.get(f"{BASE}/api/session/{sid}/meta")).json()
        if meta["state"] in ("ended", "failed"):
            break
    # final graph state from the last snapshot on disk
    snaps = sorted(f for f in os.listdir(os.path.join(REPO, "web/output", sid))
                   if f.startswith("snap_ep"))
    if not snaps:
        return {"name": name, "sid": sid, "state": meta["state"],
                "facts": 0, "invalidated": [], "episodes": 0}
    g = json.load(open(os.path.join(REPO, "web/output", sid, snaps[-1])))
    inv = [e["fact"] for e in g["edges"] if e.get("invalid_at")]
    return {"name": name, "sid": sid, "state": meta["state"],
            "facts": len(g["edges"]), "invalidated": inv,
            "episodes": len(snaps)}


async def main() -> None:
    # optional slice filter: ./corpus_eval.py 03 04
    wanted = sys.argv[1:]
    slices = [s for s in SLICES if not wanted or any(s[0].startswith(w)
                                                     for w in wanted)]
    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        for i, (name, path, _) in enumerate(slices):
            if i:
                print("  (pacing 120 s between slices — SAIA rate limits)",
                      flush=True)
                await asyncio.sleep(120)
            results.append(await run_slice(client, name, path))

    print("\n================ CORPUS EVALUATION ================")
    print(f"{'slice':<16} {'expect':<7} {'got':<4} {'facts':<6} "
          f"{'episodes':<9} verdict")
    failures = 0
    for (name, _, expect), res in zip(slices, results):
        got = len(res["invalidated"])
        if expect == "== 0":
            ok = got == 0
        elif expect == ">= 1":
            ok = got >= 1
        else:                      # "~ 0": progression — report, don't fail
            ok = True
        failures += 0 if ok else 1
        print(f"{name:<16} {expect:<7} {got:<4} {res['facts']:<6} "
              f"{res['episodes']:<9} {'OK' if ok else 'MISS'}  ({res['sid']})")
        for f in res["invalidated"]:
            print(f"    ✗ {f[:90]}")
    print("===================================================")
    print("PASS" if failures == 0 else f"{failures} slice(s) missed expectation")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
