#!/usr/bin/env python3
"""exp_jury_r1v2_cluster_zveto — per-question z-scored verifier as cluster veto+reweight.

Pure reanalysis of cached real R1 data (exp_bet_jury_r1_verifier_gain/results.json):
200 GSM8K questions x 8 real sampled candidates with real verifier scores. Zero model calls,
zero new tokens. is_smoke=false because every number derives from real executed runs.

Pre-registered (MATH.md / kill #2332):
  - Tune (alpha, tau) on EVEN-index questions only; report ODD-index (held-out, 100 qs).
  - SUPPORTED iff held-out acc >= held-out SC + 3pp AND win-flips > 5.
  - Otherwise KILLED.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "exp_bet_jury_r1_verifier_gain" / "results.json"
OUT = HERE / "results.json"

VETO_Z = -1.0
ALPHA_GRID = [i / 20 for i in range(21)]           # 0.00 .. 1.00
TAU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # raw vscore std gate


def zscores(vs):
    n = len(vs)
    mu = sum(vs) / n
    var = sum((v - mu) ** 2 for v in vs) / n
    sd = var ** 0.5
    if sd == 0:
        return [0.0] * n, 0.0
    return [(v - mu) / sd for v in vs], sd


def jury_pick(q, alpha, tau):
    """Return predicted answer string for one question record."""
    cands = q["cands"]
    vs = [c["vscore"] for c in cands]
    zs, sd = zscores(vs)
    if sd < tau:
        return q["sc_pred"]  # verifier is guessing -> pure SC fallback
    clusters = defaultdict(list)  # pred -> list of z
    for c, z in zip(cands, zs):
        clusters[c["pred"]].append(z)
    items = [(pred, len(zl) / len(cands), max(zl)) for pred, zl in clusters.items()]
    kept = [it for it in items if it[2] >= VETO_Z]
    if not kept:
        kept = items  # all vetoed -> restore all
    # deterministic tie-break: score desc, vote desc, maxz desc, pred lexical
    kept.sort(key=lambda it: (-(alpha * it[1] + (1 - alpha) * it[2]), -it[1], -it[2], it[0]))
    return kept[0][0]


def z_top1_pick(q):
    cands = q["cands"]
    zs, _ = zscores([c["vscore"] for c in cands])
    best = max(range(len(cands)), key=lambda i: zs[i])
    return cands[best]["pred"]


def acc(details, picker):
    return sum(1 for q in details if picker(q) == q["gt"]) / len(details)


def main():
    t0 = time.time()
    data = json.loads(SRC.read_text())
    details = data["details"]
    assert len(details) == 200, f"expected 200 cached questions, got {len(details)}"

    even = details[0::2]   # tuning half
    odd = details[1::2]    # held-out half (reported)

    # --- tune on even half ---
    grid = []
    for alpha in ALPHA_GRID:
        for tau in TAU_GRID:
            a = acc(even, lambda q, al=alpha, t=tau: jury_pick(q, al, t))
            grid.append({"alpha": alpha, "tau": tau, "tune_acc": a})
    # tie-break toward SC-like configs (higher alpha, then lower tau)
    grid.sort(key=lambda g: (-g["tune_acc"], -g["alpha"], g["tau"]))
    best = grid[0]
    alpha, tau = best["alpha"], best["tau"]

    # --- held-out evaluation ---
    sc_holdout = sum(1 for q in odd if q["sc_ok"]) / len(odd)
    bon_holdout = sum(1 for q in odd if q["bon_ok"]) / len(odd)
    ztop1_holdout = acc(odd, z_top1_pick)
    jury_holdout = acc(odd, lambda q: jury_pick(q, alpha, tau))

    win_flips = sum(
        1 for q in odd if (not q["sc_ok"]) and jury_pick(q, alpha, tau) == q["gt"]
    )
    loss_flips = sum(
        1 for q in odd if q["sc_ok"] and jury_pick(q, alpha, tau) != q["gt"]
    )

    # also report tuning-half numbers for transparency (NOT gated on)
    sc_tune = sum(1 for q in even if q["sc_ok"]) / len(even)
    jury_tune = best["tune_acc"]

    gain = jury_holdout - sc_holdout
    gate_acc = jury_holdout >= sc_holdout + 0.03
    gate_flips = win_flips > 5
    all_pass = bool(gate_acc and gate_flips)
    verdict = "supported" if all_pass else "killed"

    results = {
        "experiment_id": "exp_jury_r1v2_cluster_zveto",
        "config": {
            "source": str(SRC),
            "split": "even=tune(100), odd=heldout(100)",
            "alpha_grid": ALPHA_GRID,
            "tau_grid": TAU_GRID,
            "veto_z": VETO_Z,
            "tuned_alpha": alpha,
            "tuned_tau": tau,
            "new_tokens": 0,
        },
        "accuracy_heldout": {
            "self_consistency_8": sc_holdout,
            "bon_8_raw_vscore": bon_holdout,
            "z_top1": ztop1_holdout,
            "jury_zveto": jury_holdout,
        },
        "accuracy_tune_half": {"self_consistency_8": sc_tune, "jury_zveto": jury_tune},
        "gain_jury_minus_sc_heldout": gain,
        "win_flips_heldout": win_flips,
        "loss_flips_heldout": loss_flips,
        "kill_criteria": [
            {
                "id": 2332,
                "gate": "heldout >= SC+3pp AND win_flips > 5",
                "heldout_acc": jury_holdout,
                "sc_plus_3pp": sc_holdout + 0.03,
                "pass": all_pass,
            }
        ],
        "top_grid_configs": grid[:10],
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "total_wall_clock_sec": time.time() - t0,
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["accuracy_heldout"], indent=2))
    print(f"tuned alpha={alpha} tau={tau} | gain={gain:+.3f} "
          f"win_flips={win_flips} loss_flips={loss_flips} -> {verdict}")


if __name__ == "__main__":
    main()
