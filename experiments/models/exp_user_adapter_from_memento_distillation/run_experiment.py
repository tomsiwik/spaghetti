"""run_experiment.py — exp_user_adapter_from_memento_distillation (PREEMPT-KILL).

Preempt-killed per Finding #669 (5th+ reuse, see F#671, F#672, F#687, F#688).
No MLX code is written because both parent experiments are target-unverified:
  - exp_memento_gemma4_replication        → PROVISIONAL (F#685, design-only)
  - exp_memento_cross_session_persistence → OPEN (never run)

Every KC in this child transitively requires (a) a memento-rehydrated teacher
(from P_R) and/or (b) a certified 50-session user memento buffer (from P_X).
Missing either operand produces vacuous PASS (init-artifact co-occurrence) or
vacuous FAIL (pipeline plumbing noise), i.e. an unidentifiable sample per
F#669. See MATH.md §1 for the dual-parent disjunctive theorem.

This scaffold always writes a well-formed `results.json` encoding the
preempt-KILL verdict and structurally-untestable KCs.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    """Return results dict encoding preempt-KILL.

    No MLX import, no model load, no training, no composition. The verdict is
    structural: 2 parents target-unverified (disjunctive) ⇒ child unidentifiable.
    """
    return {
        "experiment_id": "exp_user_adapter_from_memento_distillation",
        "verdict": "KILLED",
        "kill_reason": "preempt-child-parents-target-unverified-dual-disjunctive",
        "finding_reference": "F#669 (5th+ reuse)",
        "parent_experiments": [
            {
                "id": "exp_memento_gemma4_replication",
                "status": "provisional",
                "finding": "F#685",
                "role": "P_R — memento-rehydrated teacher signal (Δθ_MEMENTO + block-mask inference loop)",
            },
            {
                "id": "exp_memento_cross_session_persistence",
                "status": "open",
                "finding": "none-never-run",
                "role": "P_X — 50-session user memento buffer (serialization + compaction + rehydration latency)",
            },
        ],
        "sibling_composition_dependency": {
            "id": "exp_hedgehog_composition_polite_refactor_js",
            "status": "killed",
            "finding": "F#688",
            "note": "K3 (4-way composition) inherits the triple-Hedgehog-parent preempt-KILL; a partial unblock via pair composition is possible as redesign, currently out of scope.",
        },
        "all_pass": False,
        "is_smoke": False,
        "kill_criteria": [
            {
                "id": 1807,
                "text": "K1 training-cost: wall-clock training from 50-session memento buffer < 30 min on M5 Pro",
                "result": "untested",
                "reason": "preempt-blocked: training runs against undefined teacher (P_R PROVISIONAL) on uncertified buffer (P_X OPEN); wall-clock is measurable but is a measurement of init-artifact plumbing, not of the claimed mechanism.",
            },
            {
                "id": 1808,
                "text": "K2 target (pair K1 per F#666): user-style held-out auto-judge |student − teacher| ≤ 5pp",
                "result": "untested",
                "reason": "preempt-blocked: teacher_judge requires memento-rehydrated teacher (P_R). Without Δθ_MEMENTO, 'teacher' is base+prepended-tokens — a different teacher than the design specifies. KC becomes proxy-without-ground-truth.",
            },
            {
                "id": 1809,
                "text": "K3 target composition: 4-way {user, polite, refactor, JS} drop < 3pp on each axis",
                "result": "untested",
                "reason": "preempt-blocked: composition requires meaningfully-trained user-adapter (K2 SUPPORTED — fails above) AND {ΔW_polite, ΔW_refactor, ΔW_js}. Latter come from exp_hedgehog_composition_polite_refactor_js (KILLED preempt F#688, 3/3 Hedgehog parents unverified).",
            },
            {
                "id": 1810,
                "text": "K4 target non-interference: MMLU drop < 2pp ∧ HumanEval drop < 2pp with user-adapter attached",
                "result": "untested",
                "reason": "preempt-blocked: 'user-adapter' available is one trained against unidentifiable teacher. Drop measurable but not a claim about personalization non-interference — only about arbitrary rank-6 perturbation on (v_proj, o_proj). Scope-scrambled.",
            },
            {
                "id": 1811,
                "text": "K5 target structural privacy: white-box reconstruction L2 of any memento from weights > fixed threshold",
                "result": "untested",
                "reason": "preempt-blocked: requires B_user to be a real certified user buffer (P_X SUPPORTED). Without P_X, 'memento' to reconstruct is a placeholder; no ground truth for what was 'in' the buffer. Reconstruction error is undefined.",
            },
        ],
        "unblock_condition": (
            "Both parents reach status=supported with target KCs verified at full scale: "
            "(1) exp_memento_gemma4_replication K2 (GSM8K drop <5pp ∧ MMLU drop <3pp) and K3 (KV-channel ablation ≥10pp) target-SUPPORTED; "
            "(2) exp_memento_cross_session_persistence K1-K4 target-SUPPORTED (latency + accuracy + compaction + pickle round-trip). "
            "Additionally for K3 specifically: {ΔW_polite, ΔW_refactor, ΔW_js} available via resolution of F#688's triple-Hedgehog-parent preempt. "
            "P_R impl-companion already filed (exp_memento_gemma4_replication_impl, P3). "
            "P_X has no _impl because its status is open (not provisional) — can be claimed directly when queue reorders."
        ),
        "platform_skills_invoked": [
            "/mlx-dev (cited, not used — no code path)",
            "/fast-mlx (cited, not used)",
        ],
        "base_model": "mlx-community/gemma-4-e4b-it-4bit (per F#627, not loaded)",
        "notes": (
            "No MLX code executed. Structural preempt-KILL per F#669; 5th+ reuse — reviewer.md §5 "
            "canonical KILL (preempt-structural) clause applies. Dual-parent disjunctive sub-case: "
            "strictly sharper than single-parent (F#687), weaker than triple-parent (F#688) only in "
            "parent count. No _impl companion (unblock is parent-external via P_R's _impl at P3 and "
            "P_X's own queue entry). No scope-swap attempted (e.g. dropping memento teacher to train "
            "on raw SFT pairs would be antipattern-t — explicitly forbidden per MATH.md §0 scope lock)."
        ),
    }


def main() -> None:
    """Entry point — never raises, always writes results.json."""
    results = build_results()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[preempt-kill] Wrote {out} — verdict=KILLED, reason=preempt F#669 (2 parents disjunctive-unverified)")


if __name__ == "__main__":
    main()
