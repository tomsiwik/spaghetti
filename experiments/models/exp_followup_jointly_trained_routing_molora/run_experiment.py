"""exp_followup_jointly_trained_routing_molora — NOT EXECUTED.

Preemptive KILL on structural grounds (F#431 + F#305 + F#312 + F#193 + F#340).
See MATH.md / PAPER.md / REVIEW-adversarial.md for the proof chain.

No run was performed. This file exists only to satisfy the 6/6 artifact
requirement and to document the design that would have been run had the
preempt not held.

Would-have-run design (not executed):
  - Train MoLoRA-style router jointly with N=5 LoRA adapters on
    math/code/medical/legal/finance real NLP corpora (Gemma 4 base).
  - Eval per-token routing accuracy on held-out split with same 5
    domains; compare to F#431's TF-IDF nearest-centroid baseline
    (A_TF-IDF = 96.6%).
  - Measure Δ = A_MoLoRA − A_TF-IDF.
  - K1551: fire if Δ >= 3pp.

Preempt: Δ bounded above by min(ceiling=3.4pp, MLP-only=3.3pp) and
equals 0 on full-sequence mixed-domain per F#305. K1551 structurally
unreachable.
"""
raise SystemExit(
    "Preemptive KILL (structural). Do not run. See MATH.md section "
    "'Theorem 1 (Ceiling incompatibility)' for the proof."
)
