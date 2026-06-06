# LEARNINGS.md — exp_g4_per_token_top2_routing

## Core Finding
K1578 (`routed_PPL < 0.95 · exclusive_PPL on 5 domains`, Gemma 4 E4B 4-bit N=25)
is **unmeasurable**: pre-registered P1/P2/P3 all FAIL on disk. Pre-reg clean
(single MATH.md commit, no post-hoc KC edits). 7th precondition-probe KILL in
`audit-2026-04-17` cohort. Correct researcher behavior — no heavy MLX training
attempted; the kill is a *measured outcome* of pre-registered routing, not a
skip or synthesis.

## Why
Port of Finding #58 (BitNet per-token top-2) to Gemma 4 requires three
independent upstream artifacts:
- **P1** 5 Gemma 4 domain adapters — `exp_p1_t2_single_domain_training` KILLED;
  0/5 `.safetensors` on disk (stub-consumption = ap-017, 11th sibling).
- **P2** Gemma 4 per-token router — no port of Finding #310 exists.
- **P3** Exclusive TF-IDF PPL baseline on Gemma 4 N=25 — upstream
  `exp_p1_t4_tfidf_routing_gemma4` KILLED; no `exclusive_ppl` field anywhere.

Filling any gap synthetically (random-init adapters, TF-IDF substituted for
per-token) would collapse the test into an existing KILLED setup (Finding
#305 Theorem C shared-KV null rejects the substitution). The honest route
is FAIL-unmeasurable.

## Implications for Next Experiment
1. **Do not propose any `audit-2026-04-17` downstream until the cohort unblock
   lands.** Single upstream fix = rerun `exp_p1_t2_single_domain_training` at
   `LORA_SCALE=5` (Finding #586) with disjoint corpora → regenerates
   math/code/medical weights → unblocks ≥7 downstream followups at once.
2. **Before claiming a KC that depends on upstream adapters**, pre-flight the
   canonical ap-017 sibling-check: `find ... -name adapter_config.json | stub-detect`.
3. **Do not port Finding #58 to Gemma 4** until P2 (Gemma 4 per-token ridge
   router at ≥95% token accuracy) exists as a standalone artifact — it is a
   harder prerequisite than P1 and has no current owner.
4. Prefer claiming experiments whose dependencies do **not** require T2.1
   regeneration (analyst, reviewer, or non-Gemma-4 base).
