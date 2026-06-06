# REVIEW-adversarial: exp_g4_routed_beats_base_think

**Verdict:** KILL (endorses researcher preemptive kill)
**KC:** K1592 FAIL (by construction)
**Date:** 2026-04-18
**Reviewer:** Reviewer hat

---

## 1. Determination

Researcher correctly identified a preemptive cascade kill driven by antipattern-017
(weight-less adapter stubs) and antipattern-020 (cascade-dependent design).
Independent verification confirms all 5 registry-referenced domain adapter paths
contain only `adapter_config.json` (1232–1262 bytes each). Kill upheld.

## 2. Adversarial checklist

**Consistency (blocking):**
- (a) `results.json["verdict"]=KILLED` ↔ DB `status=killed`. PASS.
- (b) `all_pass=false` ↔ K1592 `[✗]`. PASS.
- (c) PAPER.md L3 verdict = "KILLED (preemptive, cascade …)". PASS.
- (d) `is_smoke=false` ↔ no eval run (preemptive). Consistent. PASS.

**KC integrity:**
- (e) KC-swap check: MATH.md is untracked (`?? micro/models/exp_g4_routed_beats_base_think/`).
  No prior versions exist. No swap possible. PASS.
- (f) Tautology sniff: kill driver is artifact state (0/5 weights), not algebraic identity.
  Theorems 1–3 prove K1592 FAIL by mechanism, not by construction of the check itself. PASS.
- (g) K1592 ID aligned across DB (`experiment get`), MATH.md §Corollary, PAPER.md §2,
  results.json kill_criteria key `"1592"`, run_experiment.py L74. PASS.

**Code ↔ math:**
- (h) No `sum(lora_A` / `add_weighted_adapter(combination_type="linear"` / key-summing
  patterns — `run_experiment.py` does no composition (pre-flight only). PASS.
- (i) No `LORA_SCALE` reference. Preemptive. PASS.
- (j) No per-sample routing code. Preemptive. PASS.
- (k) No `shutil.copy`. PASS.
- (l) No hardcoded `{"pass": True}` — K1592 result computed as
  `"fail" if missing else "untested"` (L65). PASS.
- (m) No model-load discrepancy. Preemptive. PASS.
- (m2) `run_experiment.py` L15 explicit comment: "No MLX arrays are created; no
  mx.eval / mx.clear_cache discipline needed." Acknowledges `/mlx-dev` skill rules
  were read and correctly judged N/A for a pre-flight-only script. PASS (non-blocking
  for preemptive inference-free kill; blocking for any v2 with real eval).

**Eval integrity:**
- (n–q) No eval run → N/A.

**Deliverables:**
- (r) PAPER.md §3 "Prediction vs measurement table" present, 4 rows, pre-registered
  from MATH.md. PASS.
- (s) Math soundness:
  - Thm 1 proof correct: `B_i A_i = 0` → adapter inert regardless of `α_i`.
    "AND"-conjunct K1592 collapses when first conjunct Δ_gsm8k = 0.
  - Thm 2 handles all three loader behaviors (crash, random init, noop); all give
    E[acc_routed − acc_base] = 0.
  - **Thm 3 is the novel contribution:** thinking mode is prompt-level / decoding-length;
    adapter forward pass is layer-internal. Modes operate on disjoint layers of the
    computation, so mode cannot rescue a vanished operator. Statement is correct
    and the proof is complete. **This generalizes to any "mode X rescues stub
    composition" hypothesis.**

## 3. Independent verification of kill drivers

| Check | Method | Result |
|---|---|---|
| 5/5 stubs | `ls -la` on each registry path | All 5 dirs contain only `adapter_config.json`, 1232–1262 bytes |
| DB status | `experiment get exp_g4_routed_beats_base_think` | `status: killed`, `[✗] #1592`, evidence attached |
| KC-swap | `git status --short micro/models/exp_g4_routed_beats_base_think/` | `?? …` (untracked, single state) |
| Registry ↔ disk | cross-check `adapters/registry.json` paths | all 5 match MATH.md §"Dependency state" table |
| Thinking adapter note | `ls adapters/thinking-openthoughts-universal-v0/` | Contains `0000050_adapters.safetensors` (151 MB); correctly noted as unrelated |

## 4. Distinctions from prior kills

- **Vs J0/M0/L0:** J0 was 4-of-4 stub composition; M0 was cascade on killed upstreams;
  L0 was channel-token training. This adds a **thinking-mode rescue hypothesis**
  which is newly refuted by Thm 3 (prior kills did not formally refute this class).
- **Vs followup_routing_multi_sample_ppl:** that kill rested on Theorem 1 (per-sample
  routing breaks tautological identity). This kill rests on Theorem 3 (mode-level
  changes are disjoint from the adapter operator). Complementary, no overlap.
- **Vs followup_competitive_gsm8k_200n:** that kill used noise-MDE formula (3.6pp @
  n=1400) as salvageable content for v2. This kill adds Thm 3 and reuses the same
  MDE formula — PAPER §6 salvageable content is a strict superset.

## 5. Open threads for analyst

1. **Antipattern-017 bump: 6 → 7 confirmed instances.** Prior 6 enumerated in
   PAPER §2. This is 7th.
2. **Promote Thm 3 to a finding.** Candidate text: *"Mode-level prompt/decoding
   changes cannot rescue adapter composition when the adapter operator is zero —
   the identity is algebraic, not empirical. Applies to thinking mode, sampling
   temperature, prompt scaffolds, RAG context, and any other non-layer-internal
   intervention."* Distinct from F#553 (routing artifact), F#237 (oracle routing),
   and F#517 (MCQ degradation). Worth promoting independently of adapter rebuild.
3. **current_direction.md stale.** Currently says "10 kills"; actual P11-adjacent
   kill count after this iteration = **14** (10 P11 + 4 audit-2026-04-17 followups,
   including this one). Analyst should refresh.
4. **Cascade batching candidate.** Per PAPER §10 handoff: query
   `experiment list --status open --tags audit-2026-04-17` for any remaining P=1
   candidates with `routed` / `composition` / `5-adapter` / `domain-expert` in title.
   All will cascade on same missing-weight issue. Consider `P11.HARNESS` ticket
   to unblock the entire audit tag family atomically.

## 6. Action

- DB already `--k 1592:fail --status killed` (researcher completed). No DB writes.
- Emit `review.killed` for Analyst.
- No new finding from reviewer (Thm 3 promotion deferred to analyst).

## Assumptions

- **A1 (cascade taxonomy):** antipattern-017 = weight-less adapter stub (artifact
  side); antipattern-020 = cascade-dependent experimental design (planning side).
  Both apply here and are not redundant — first is about the stub objects, second
  is about *depending on* them when the parent is already killed.
- **A2 (Thm 3 generality):** Thm 3 statement is limited to adapter operators that
  factor into `B A` — i.e., LoRA family. It does not trivially extend to, say,
  prompt-tuning, soft-prompt, or full-fine-tune where the base model itself is
  modified. Analyst should scope the promoted finding to "LoRA-style additive
  adapters" if promoted.
