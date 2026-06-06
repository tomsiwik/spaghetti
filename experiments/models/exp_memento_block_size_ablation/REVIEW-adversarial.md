# REVIEW-adversarial.md — exp_memento_block_size_ablation

## Verdict: KILL (preempt-structural, F#669 reuse, 2nd MEMENTO-cluster child)

Routine preempt-structural KILL per reviewer.md §5 "preempt-structural sub-case" clause + F#669 family (≥9 reuses). Child-KC transitivity on target-unverified parent is structurally unidentifiable; additional compounding factor (hyperparameter sweep over N=4 block sizes) strictly strengthens the gate.

## Checklist

**Consistency (highest priority):**
- (a) `results.json["verdict"]="KILLED"` ↔ proposed status `killed` — **PASS**
- (b) `all_pass=false` ↔ KILLED — **PASS**
- (c) PAPER.md verdict line = "KILLED (preempt, F#669 — ≥9 reuses; 2nd MEMENTO-cluster child)" — **PASS**
- (d) `is_smoke=false` ↔ full-claim — **PASS** (preempt-structural, not smoke)

**KC integrity:**
- (e) Git diff: no pre-reg mutation — KC IDs K1904/K1905 retained verbatim from DB. **PASS**
- (f) Tautology: KCs are `untested` (preempt-blocked) — no algebraic identity, no vacuous PASS. **PASS**
- (g) KC text in results.json matches MATH.md §3 table. **PASS**

**Code ↔ math:**
- (h) `run_experiment.py` imports `json` + `pathlib` only; no composition math. **PASS**
- (i) No `LORA_SCALE` hard-coded; no LoRA. **PASS**
- (j) No per-sample vs batched routing; no routing. **PASS**
- (k) No `shutil.copy`; no file aliasing. **PASS**
- (l) No hardcoded `{"pass": True}`; all KCs `result="untested"`. **PASS**
- (m) No proxy-model substitution; §6 explicitly rejects 3 common proxy shortcuts (base KV chunking, text-chunking, truncation-as-block-size). **PASS**
- (m2) Skills cited: §0 notes `/mlx-dev` + `/fast-mlx` "noted, not used — no code path" — honest disclosure, matches reviewer.md §5 preempt-structural requirement #1 pattern (skill citation without MLX code landing). **PASS**

**Eval integrity:**
- (n) No base-eval pathology; no eval run. **N/A**
- (o) No headline n; preempt-structural. **N/A**
- (p) No synthetic padding; no N. **N/A**
- (t) Target-gated kill — **CARVE-OUT APPLIES**: preempt-structural KILL is excluded from (t) per reviewer.md §5 F#669-family clause last bullet ("F#666 does NOT apply to preempt-KILL — F#669 is the governing precedent; no KC measured"). K1905 is a quasi-target by construction anyway (paper-accuracy gate), so the KC set is F#666-compliant in form.
- (u) No scope-changing fix; graceful-failure stub is canonical preempt-structural artifact. §6 explicitly enumerates rejected scope-swaps. **PASS**

**Deliverables:**
- (r) PAPER.md prediction-vs-measurement table present (both rows "not measured, untested"). **PASS**
- (s) Math errors: none detected. §1 theorem correctly derives unidentifiability from parent PROVISIONAL + sweep-N-requirement. §1.2 hyperparameter-sweep sub-axis is a 1st-observation candidate (honestly flagged as not yet promotion-eligible).

## Preempt-structural artifact pattern compliance

Per reviewer.md §5 F#669-family clause, required pattern:
1. MATH.md §1 theorem deriving transitivity → **PRESENT** (§1 + §1.1 F#666 gating + §1.2 sweep sub-axis).
2. `run_experiment.py` graceful-failure (no MLX, always writes results.json) → **PRESENT** (json+pathlib only).
3. PAPER.md "KILLED (preempt, F#669)" verdict line + prediction-vs-measurement "not measured" + Unblock path section → **PRESENT** (all three).
4. No `_impl` companion → **CONFIRMED** (§5 of MATH.md + PAPER.md "Follow-up filed: None" + results.json `impl_follow_up_filed=false`).

## Assumptions / judgment calls

- Accepted §1.2 "hyperparameter-sweep-strictly-stronger-than-single-config" as a legitimate 1st-observation sub-axis candidate. It is a formal tightening of F#669, not a novel antipattern claim; promotion threshold (3rd instance) correctly cited. Non-blocking for this verdict.
- MEMENTO paper citation (Kontonis et al., arxiv:2604.09852, Apr 2026, released Qwen3/Phi-4/Olmo 3 only) is consistent with parent F#685 and sibling F#699 precedent — not independently verified, but internally consistent and not load-bearing for the preempt argument (parent's PROVISIONAL status alone suffices).

## Route

- `experiment complete exp_memento_block_size_ablation --status killed --dir micro/models/exp_memento_block_size_ablation/ --k 1904:inconclusive --k 1905:inconclusive --evidence "preempt-structural KILL per F#669 (≥9 reuses); parent exp_memento_gemma4_replication PROVISIONAL per F#685; sweep over {128,256,512,1024} requires N=4 parent _impl runs (strictly stronger than single-config); both KCs untested (antecedent unsatisfiable)" --source results.json`
- `experiment finding-add --title "..." --status killed ...`
- Emit `review.killed`
