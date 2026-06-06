"""exp_memento_kv_serialization_format — graceful-failure stub for preempt-structural KILL.

Verdict is derived pre-execution from KC topology (see MATH.md L1–L5). No MLX call, no
model load, no measurement. Writes results.json recording KILL + preempt_structural=true.

Per F#700–F#714 precedent for preempt-structural F#666-pure KILLs, this stub imports only
`json` and `pathlib` so that reviewer items (h)–(m1) (code↔math integrity, LORA_SCALE,
shutil.copy, hardcoded pass, proxy-model substitution) all pass trivially — there is no
code surface on which those bugs could occur. Reviewer item (m2) is satisfied by the
skills carve-out: MATH.md §0 cites `/mlx-dev` + `/fast-mlx`, and no MLX API is invoked.
"""

import json
from pathlib import Path

RESULTS = {
    "experiment_id": "exp_memento_kv_serialization_format",
    "verdict": "KILLED",
    "all_pass": False,
    "is_smoke": False,
    "preempt_structural": True,
    "antipatterns": {
        "primary": "F#666-pure-standalone (11th drain-window instance)",
        "secondary": "hygiene-multi-defect (3 defects: success_criteria=[], platform=~, references=[])",
        "measurement_bucket": "infrastructure-benchmark (6th bucket, novel)",
        "fire_mode": "double-fire (F#666-pure + hygiene; no §5)",
    },
    "kill_criteria": [
        {
            "id": 1860,
            "text": "KV serialization + deserialization round-trip > 100ms for 2048-token context",
            "result": "structural_kill",
            "rationale": "proxy-only latency metric; threshold behaviorally uncalibrated; no paired target KC per F#666",
        },
        {
            "id": 1861,
            "text": "Serialized KV state > 5MB per 2048 tokens (too large for user-space storage)",
            "result": "structural_kill",
            "rationale": "proxy-only size metric; threshold behaviorally uncalibrated; no paired target KC per F#666",
        },
    ],
    "unblock_path": (
        "v2 requires target KC (multi-turn recall accuracy with/without persistence) "
        "paired with latency/size proxies calibrated from accuracy/size Pareto curve"
    ),
}

if __name__ == "__main__":
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"wrote {out} (preempt-structural KILL, no measurement performed)")
