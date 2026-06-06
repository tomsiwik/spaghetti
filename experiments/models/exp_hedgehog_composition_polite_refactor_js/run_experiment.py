"""run_experiment.py — exp_hedgehog_composition_polite_refactor_js (PREEMPT-KILL).

Preempt-killed per Finding #669 (4th+ reuse, see F#671, F#672, F#687). No MLX
code is written because all three parent adapters are target-unverified:
  - exp_hedgehog_behavior_adapter_politeness  → PROVISIONAL (F#683, design-only)
  - exp_hedgehog_procedural_adapter_refactor  → PROVISIONAL (F#684, design-only)
  - exp_hedgehog_domain_adapter_js            → OPEN (never run)

Every KC in this child transitively requires trained Hedgehog adapters from
all three parents; composing over nonexistent or untrained adapters produces
unidentifiable samples (vacuous PASS or FAIL). See MATH.md §1 for the theorem.

This scaffold always writes a well-formed `results.json` encoding the
preempt-KILL verdict and structurally-untestable KCs.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    """Return results dict encoding preempt-KILL.

    No MLX import, no model load, no training, no composition. The verdict is
    structural: 3 parents target-unverified ⇒ child unidentifiable.
    """
    return {
        "experiment_id": "exp_hedgehog_composition_polite_refactor_js",
        "verdict": "KILLED",
        "kill_reason": "preempt-child-parents-target-unverified-triple",
        "finding_reference": "F#669 (4th+ reuse)",
        "parent_experiments": [
            {"id": "exp_hedgehog_behavior_adapter_politeness", "status": "provisional", "finding": "F#683"},
            {"id": "exp_hedgehog_procedural_adapter_refactor", "status": "provisional", "finding": "F#684"},
            {"id": "exp_hedgehog_domain_adapter_js", "status": "open", "finding": "none-never-run"},
        ],
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1794,
                "text": "K1 structural: composed per-layer cos vs ideal teacher > 0.70 on JS refactor prompts",
                "result": "untested",
                "reason": "preempt-blocked: 0/3 Hedgehog adapters exist (all 3 parents target-unverified)",
            },
            {
                "id": 1795,
                "text": "K2 target: auto-judge scores for all 3 axes (politeness +15pp AND refactor >= LoRA AND JS-idiomatic >= base+JS-LoRA), n>=100",
                "result": "untested",
                "reason": "preempt-blocked: no adapters to compose; triple-axis judge over nonexistent deltas is undefined",
            },
            {
                "id": 1796,
                "text": "K3 target ablation-polite (F#666 pair K2): removing polite drops politeness >=15pp; refactor/JS drop <3pp",
                "result": "untested",
                "reason": "preempt-blocked: ablation of a nonexistent adapter is a no-op; 'diagonal dominance' is undefined",
            },
            {
                "id": 1797,
                "text": "K4 target ablation-refactor: removing refactor drops refactor >=10pp; polite/JS drop <3pp",
                "result": "untested",
                "reason": "preempt-blocked: same structural defect as K3",
            },
            {
                "id": 1798,
                "text": "K5 target non-interference: on non-JS non-refactor polite tasks composition matches polite-alone within 2pp",
                "result": "untested",
                "reason": "preempt-blocked: composition target requires all three trained adapters — requirement unmet",
            },
        ],
        "unblock_condition": (
            "All three parents reach status=supported with target KCs verified at full scale: "
            "(1) exp_hedgehog_behavior_adapter_politeness K2 (auto-judge politeness Δ); "
            "(2) exp_hedgehog_procedural_adapter_refactor K2/K3 (Fowler-judge refactor quality); "
            "(3) exp_hedgehog_domain_adapter_js K2 (HumanEval-JS / JS-nuance benchmark). "
            "Impl-companions already filed at P3 for the two PROVISIONAL parents. "
            "Domain-JS parent has no `_impl` because its own status is open, not provisional — "
            "it can be claimed directly when the queue reorders."
        ),
        "platform_skills_invoked": [
            "/mlx-dev (noted, not used — no code path)",
            "/fast-mlx (noted, not used)",
        ],
        "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
        "notes": (
            "No MLX code executed. Structural preempt-KILL per F#669; 4th+ reuse — reviewer.md §5 "
            "canonical KILL (preempt-structural) clause applies. No _impl companion (unblock is "
            "parent-external via 3 separate parent paths)."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[preempt-kill] Wrote {out} — verdict=KILLED, reason=preempt F#669 (3 parents unverified)")


if __name__ == "__main__":
    main()
