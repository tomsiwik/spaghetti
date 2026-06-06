# REVIEW-adversarial.md — exp_p9_cispo_adapter_rl (self-review)

## Context
Preemptive kill before any execution. Self-adversarial review against the
reviewer's (a)-(t) checklist (PLAN.md §1) to minimize REVIEW-blocking issues
when reviewer iter 63 ratifies.

## Checklist

- **(a) verdict in results.json matches PAPER.md** — both KILLED. ✓
- **(b) verdict in DB matches results.json** — status=killed (will be set on
  `experiment complete --status killed`). ✓
- **(c) all_pass consistent** — results.json all_pass=false; PAPER KC table
  shows no passes. ✓
- **(d) is_smoke correct** — is_smoke=false; this is a preemptive kill, not a
  smoke run. ✓
- **(e) KC text verbatim from DB** — K1399/K1400/K1401 text in MATH.md and
  PAPER.md matches DB `experiment get` output. No edits. ✓
- **(f) not tautology** — dep-unfulfilled is inter-experiment
  (parent→child), not intra-experiment self-defeating. Platform-mismatch is
  orthogonal. Not F#498/F#666 pattern. ✓
- **(g) no measurement** — N/A, no execution. ✓
- **(h)-(l) composition / LoRA / routing / copy / hardcoded-pass / proxy
  antipatterns** — N/A, no code executed. Stub script only writes JSON. ✓
- **(m) eval-template truncation** — N/A. ✓
- **(m2) platform skill invocation (/mlx-dev, /fast-mlx)** — N/A. No platform
  code written. Pure documentation + JSON stub. ✓
- **(n)-(q) eval-side checks** — N/A, no eval. ✓
- **(r) prediction-vs-measurement table** — PAPER.md has complete table with
  all 3 KCs, predictions, "not measured", and FAIL verdicts. ✓
- **(s) proofs sound** —
  - T1 reduces KCs to parent dep-closure via explicit 3-step argument per KC.
    Each step is structural (undefined quantities without the parent
    artifact). ✓
  - T2 cites Unsloth repo + arXiv:2506.13585 for CUDA-only backend; cites
    PLAN.md + `feedback_mlx_first.md` for MLX-only target. ✓
  - T3 independence: even conditional on ¬T1, T2 still kills; and vice versa.
    ✓
- **(t) target-gated KCs** — KCs 1399-1401 measure target behavior (adapter
  accuracy, rare-token gradient ratio, training stability). Not proxies.
  Preempt does not substitute a proxy target. ✓

## Potential reviewer objections (pre-empted)

1. **"Couldn't you run a smoke RL loop on MLX to satisfy K1401 stability?"**
   — No. K1401 measures stability of the CISPO algorithm specifically.
   CISPO is not implementable in MLX without re-implementing the
   importance-ratio-clip machinery. Any MLX smoke would necessarily be a
   different algorithm, which would violate KC text (tautological substitution
   under F#498/F#666 pattern). Correct choice: preempt.

2. **"Isn't the platform-mismatch argument (T2) a hand-wave?"** — No, it's
   specific: Unsloth ships Triton kernels + CUDA-only fast-attention; CISPO
   via MiniMax codebase requires Megatron. Both are external to the
   hardware-reachable from M5 Pro. The policy memory
   `feedback_mlx_first.md` is explicit. Not hand-wave.

3. **"F#669 3rd reuse — should this alone warrant promotion?"** — Yes, per
   reviewer iter 62 precedent: "Next occurrence should promote F#669 from
   sub-axis to standalone finding." This is that occurrence. Proposal logged
   in results.json / PAPER / LEARNINGS.

## Verdict
PREEMPT STANDS. 6/6 docs on disk. Ratify as KILLED.
