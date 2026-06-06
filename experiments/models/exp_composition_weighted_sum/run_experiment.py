"""exp_composition_weighted_sum — preempt-structural stub.

Not executed empirically. KILLED by triple-fire: method-dependent-redundancy
(3rd instance post-promotion; anchors F#731+F#732+this), F#666-pure (21st),
§5 tautological-inter-variant-delta (13th). See MATH.md for 3 independent
theorems (F#664 fixed-algebraic preempt, F#164 learned-weight impossibility,
F#137/F#643 tautological-duplicate).
"""
import json
from pathlib import Path

results = {
    "experiment_id": "exp_composition_weighted_sum",
    "verdict": "KILLED",
    "preempt_structural": True,
    "triple_fire": True,
    "all_pass": False,
    "is_smoke": False,
    "kill_criteria": {
        "1896": {"result": "fail", "reason": "inter-variant-delta across 3 branches collapses to F#664/F#164/F#137 priors"},
        "1897": {"result": "inconclusive", "reason": "learned branch un-evaluable (F#164 divergence); target-unbound (F#666-pure)"},
    },
    "antipattern_fires": [
        "method-dependent-redundancy (3rd instance, post-promotion)",
        "F#666-pure target-unbound (21st reuse, 2 KCs)",
        "§5 tautological-inter-variant-delta (13th reuse)",
        "F#664 preempt-category (2nd reuse)",
        "F#643 tautological-duplicate KC (K1896 vs F#137)",
    ],
    "references": [137, 157, 164, 496, 543, 643, 664, 22, 544, 731, 732],
}

out = Path(__file__).parent / "results.json"
out.write_text(json.dumps(results, indent=2))
print(f"Wrote {out}. Verdict: KILLED (preempt-structural, triple-fire).")
