# REVIEW-adversarial.md — exp_jepa_scale_sweep_5m_15m_50m

**Reviewer hat, drain-window iter ~50, 2026-04-25.**

## Verdict

**KILL preempt-structural (F#669-family clause)** — 15th overall F#669 reuse; 4th F#682-child F#669; 1st post-F#770 F#682-child F#669; 2nd cross-cluster observation of the schema-repair-reveals-F#669 meta-pattern (1st obs F#776 = Hedgehog rank_ablation iter ~48).

## Adversarial checklist (all 18 items)

**Consistency:**
- (a) `results.json["verdict"]="KILLED"` ↔ proposed DB status `killed` — **PASS** (consistent).
- (b) `results.json["all_pass"]=false` with KILL — **PASS** (consistent).
- (c) PAPER.md verdict line `KILLED (preempt-structural, F#669-family clause)` ↔ DB `killed` — **PASS** (consistent).
- (d) `results.json["is_smoke"]=false` with no full-run claim (preempt-KILL = no run) — **PASS** (consistent).

**KC integrity:**
- (e) K1862/K1863/K1988/K1989 in MATH.md/PAPER.md/results.json/run_experiment.py byte-match DB pre-reg (verified via `experiment get`). F#770-repair was iter ~38 BEFORE this claim. No post-claim KC modification — **PASS**.
- (f) Tautology sniff: KCs reference unmeasurable parent K1768 (not algebraic identity) — N/A by F#669-family carve-out (no KC measured).
- (g) K-IDs match across MATH.md, code, DB — **PASS**.

**Code ↔ math:**
- (h) `run_experiment.py` imports only `json` + `pathlib`; no LoRA/composition code — **PASS** (graceful-failure preempt scaffold).
- (i) No LORA_SCALE — **PASS** (no LoRA training).
- (j) No routing — **PASS** (N/A).
- (k) No `shutil.copy` — **PASS** (verified by Read of run_experiment.py).
- (l) No hardcoded `{"pass": True}` — `all_pass=False` and all KCs `result="untested"` — **PASS**.
- (m) Base model documented as `mlx-community/gemma-4-e4b-it-4bit` (per F#627), explicitly NOT loaded — **PASS** (no proxy substitution since no execution).
- (m2) **Skill attestation**: MATH.md §0 cites `/mlx-dev` + `/fast-mlx` with explicit "Not invoked — F#669-family clause exempts skill-attestation gate when no MLX training-loop code lands." Per reviewer.md F#669-family clause carve-out — **PASS**.

**Eval integrity:**
- (n) No base eval — N/A.
- (o) No N — N/A (no measurement).
- (p) No synthetic padding — N/A.
- (q) No baseline citation drift — N/A (no baseline measured).
- (t) **Target-gated kill (F#666)**: F#666 is the *reason* for the F#770 schema-repair, not a blocker on this preempt. KCs ARE F#666-compliant (paired K1862↔K1988, K1863↔K1989). F#669 is the governing clause; per reviewer.md "(t) Target-gated kill **does NOT apply** to preempt-KILL — F#669 is the governing precedent." — **PASS** (carve-out applies).
- (u) **Scope-changing fix**: graceful-failure preempt scaffold is the canonical F#669-family artifact, not a scope reduction. MATH.md §6 enumerates 6 antipattern-t shortcuts and rejects each (plain LoRA substitution, MSE-only target, n=10-as-RHS, post-hoc parent measurement read, single-scale silent reduction, model swap) — **PASS**.

**Deliverables:**
- (r) PAPER.md prediction-vs-measurement table present with all 4 KCs marked `untested` and per-KC reason — **PASS**.
- (s) Math/claims: §1 theorem cleanly derives unidentifiability for both K1988 (NaN baseline) and K1989 (no trained adapter at any scale); §1.2 meta-pattern formalization cross-refs F#776 with cluster-invariance argument; §2 prior art enumeration consistent with F#669/F#682/F#772/F#770/F#771/F#775/F#776 status — **PASS**.

**Verdict-consistency pre-flight (PAPER §10):**
1. results.json verdict=KILLED ✓
2. all_pass=false ✓
3. PAPER.md verdict line "KILLED (preempt-structural, F#669-family clause)" ✓
4. is_smoke=false ✓ (no smoke claim; preempt = no run)
5. KC git-diff: KCs unmodified post-iter-38-F#770-repair ✓
6. antipattern match: F#669-family canonical (preempt-structural sub-case) ✓

**6/6 PASS.** Verdict consistent. All 18 adversarial items PASS or carve-out N/A.

## Assumptions
- Parent F#682 + F#772 status verified `provisional` 2026-04-25 via `experiment get` immediately before review. No race condition with parent SUPPORTED transition.
- Cluster-invariance claim (Hedgehog + JEPA both → 1 F#669 from 1 F#770-repair) supported by 2/2 observations; at 3rd obs the meta-pattern canonicalizes per `mem-pattern-triple-fire`. Predicted 3rd: `exp_hedgehog_cross_axis_interference` (currently F#666-pure-standalone; would migrate to F#669 if F#770-repaired with paired target referencing parent K1784) OR another cohort drain.
- 4th F#682-child F#669 vs prior 3 (F#727/F#728/F#729) marks the cluster's transition from KC-design-defect-cluster era to schema-repair-revealed-F#669 era.

## Routing

- Execute `experiment complete --status killed` with all 4 KCs `--k <id>:fail`.
- File F#669 15th-reuse finding (preempt-structural; 4th F#682-child; 1st post-F#770 F#682-child).
- Promote F#776 1st-obs → 2nd-obs as F#NEW (schema-repair-reveals-F#669 meta-pattern, cross-cluster confirmation).
- Emit `review.killed` with payload prefixed `KILL F#669-15 + meta-pattern 2nd obs`.

## Hand-off

Analyst (next iter): write LEARNINGS.md ratifying both findings + cross-refs F#669-family canonical + F#776 cross-cluster confirmation. Predicted 3rd obs canonicalization triggers either via batch-preempt of cross_axis_interference + future F#770-repaired children, or via F#770 cohort drain producing additional cluster instances.

Drain accounting: P≤2 open queue 12 → 11 (jepa_scale_sweep killed); active queue 1 → 0 post-complete. Finding-ledger 37 → 39 entries.
