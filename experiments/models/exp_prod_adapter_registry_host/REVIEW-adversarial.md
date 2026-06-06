# REVIEW-adversarial.md — exp_prod_adapter_registry_host

**Verdict: KILL** (confirm the researcher's preemptive KILLED_PREEMPTIVE).
32nd cohort preempt; 2nd instance of ap-017 axis **(s) hardware-topology-
unavailable** (generalized CUDA-absence → public-network-absence).

## Consistency (a–d)
- (a) `results.json.verdict = "KILLED"` ↔ DB `status = killed` ↔ PAPER
  `KILLED_PREEMPTIVE` — aligned. No silent upgrade. ✓
- (b) `all_pass = false`; all three KCs (#1659/#1660/#1661) marked fail
  in DB evidence. ✓
- (c) PAPER verdict line = `KILLED_PREEMPTIVE`; no PROVISIONAL /
  PARTIALLY-SUPPORTED leakage. ✓
- (d) `is_smoke = false`. ✓

## KC integrity (e–g)
- (e) MATH.md and KC pre-registration appear on the same commit surface
  as the preemptive-kill runner (no post-hoc KC relaxation; KCs were
  DB-registered at claim time, blocked by T3 schema-completeness
  axis — the absence, not the edit, drove the kill).
- (f) No tautology: `pierre://` grep = 0, `nvidia_smi` = absent, arm64
  Darwin — these are genuine observables, not `x==x` identities.
- (g) K-IDs in `kill_criteria` section match DB KCs #1659/#1660/#1661.

## Code ↔ math (h–m2)
- (h) No LoRA composition code — pure-stdlib probe. ✓
- (i) No LORA_SCALE present. ✓
- (j) No routing. ✓
- (k) No `shutil.copy` of sibling adapters. ✓
- (l) No hard-coded `{"pass": True}`. ✓
- (m) No target model loaded — pure probe. ✓
- (m2) MLX skills N/A (no MLX import; stdlib-only probe). ✓

## Eval integrity (n–q)
- All non-applicable (no base/adapter inference, no N-sample sweep).

## Deliverables (r–s)
- (r) PAPER prediction-vs-measurement table present (5 rows; T4
  explicitly marked "reinforcing only"). ✓
- (s) Math: 4/5 theorems each independently block under defense-in-
  depth; T1 ∧ T2 ∧ T3 ∧ T5 any-one sufficient. No unsupported claim.

## Defense-in-depth audit
- T1 alone blocks (3 artefact shortfalls + arm64/no-nvidia-smi).
- T2 alone blocks (1440 min required vs 120 min micro ceiling; 12× over).
- T3 alone blocks (`success_criteria: []` DB literal; 4th occurrence of
  F#502/F#646 axis after tfidf_routing_no_alias, flywheel_real_users,
  loader_portability).
- T5 alone blocks (5/5 source-scope breaches: transport, throughput,
  uptime, push-path, infra-topology).
- T4 reinforces only (pin_ratio = 0.333 > 0.20 auto-block floor).

## Non-blocking observations
- ap-017 (s) generalization from "physical hardware absent" (iter 35
  CUDA) to "public network/DNS absent" (iter 36) is a scope widening;
  analyst may later split into (s-hw) / (s-net) sub-axes. Does not
  affect this verdict.
- LEARNINGS.md debt for this experiment adds to the 9 prior
  analyst-owed entries; operator-blocked by cap (HALT §C).

## Routing
Verdict: **KILL**. Emit `review.killed`. Finding added for the 2nd
ap-017 (s) preempt; registry_host → signing ancestor remains
SUPPORTED at its declared scope.
