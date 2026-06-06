# MATH.md — exp_g4_e2e_mmlu_pro_thinking

## Goal
K1618: Full E2E pipeline (ridge router + delta-sum composition + thinking
mode) on Gemma 4 E4B 4-bit beats 62.1% MMLU-Pro base+thinking baseline
(F#536).

## Verdict (pre-flight)
**KILLED_PREEMPTIVE.** K1618 is structurally unreachable via six
independent theorems. This is the 16th P11/g4-adjacent preemptive kill in
the audit-2026-04-17 sweep and the 8th confirmed instance of
**antipattern-017 (stub adapters consumed as if trained)**.

## Antipattern self-check
- **antipattern-017 (stub adapters consumed):** CONFIRMED. 5 of 5
  adapter paths (`adapters/{math,bash,python,sql,medical}`) contain only
  `adapter_config.json` and `tokenizer_config.json`; no `safetensors`
  weights. Registry-referenced paths
  (`micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/`)
  are ALSO stub-only.
- **antipattern-020 (cascade-upstream-killed):** CONFIRMED. Upstream
  `exp_p1_t2_single_domain_training` is still `status=open` (never
  produced adapters); `exp_g4_ridge_routing_n25_mcq` is `status=killed`
  (K1616 FAIL at 0.8387 test acc vs 0.90 target). The E2E pipeline
  requires BOTH.
- **antipattern-no-knowledge-gap (derived 2026-04-18):** PARTIAL. F#478
  closure: Gemma 4 4B has no exploitable knowledge gap for basic LoRA
  adapters on advanced questions. MMLU-Pro is graduate-level 10-option
  MCQ; falls within the "too capable" closure region.
- **antipattern-framework-incomplete:** CONFIRMED. DB shows
  `success_criteria: []` and a single K1618 with no threshold
  derivation, no MDE, no n.

## Theorem 1 — Pipeline reduces to identity under stub adapters

**Setting.** Let the pipeline forward pass at layer ℓ module m be

    y = x + Σ_{i=1..N} α_i(x) · (B_i A_i x)                 (1)

where:
- x ∈ ℝ^d is the module input,
- (A_i, B_i) ∈ ℝ^{r×d} × ℝ^{d×r} are LoRA factors of adapter i,
- α_i(x) = softmax(ridge(h(x)))_i are per-sample routing weights from a
  ridge classifier over mean-pooled hidden states.

**Lemma 1.1 (absent operand).** If adapter i is absent (no safetensors
on disk), one of three equivalent things happens: (a) the loader skips
i, giving α_i · (B_i A_i x) := 0; (b) the loader crashes, giving
undefined behavior that the harness must clamp to 0; (c) the loader
random-inits, giving E[B_i A_i x | x] = 0 for all x (random B, A each
zero-mean).

**Proof.** (a) trivial. (b) framework-level: the experiment is
unrunnable; kill is forced without measurement. (c) E[B_i A_i x]
= E[B_i] E[A_i x] (independence) = 0 · E[A_i x] = 0. ∎

**Theorem 1.** Under 5/5 stub adapters, the pipeline forward pass
reduces to y = x in expectation, i.e., the pipeline is
observationally indistinguishable from base+thinking.

**Proof.** Apply Lemma 1.1 to every i. Then Σ α_i · (B_i A_i x) = 0
deterministically under (a), or has expectation 0 under (c). In both
cases the composition has no signal over base+thinking. ∎

**Corollary 1.** The pipeline score equals the base+thinking score =
62.1% (F#536). Δ = 0. K1618 requires Δ > 0pp. K1618 FAIL by
construction.

## Theorem 2 — Training without thinking suppresses thinking (F#536)

**F#536 empirical result (2026-04-13):** "MCQ adapter + thinking =
50.4% (-11.7pp) because adapter suppresses thinking chains (0 chars
generated). ... Adapter optimized for question→answer cannot coexist
with thinking mode requiring question→think→answer."

**Theorem 2.** Any LoRA adapter ΔW = BA trained with
`enable_thinking=False` on (q, a) pairs optimizes the transition
q → a, which assigns near-zero probability to `<thinking>` tokens at
the q-boundary. Under delta-sum composition ΔW_total = Σ ΔW_i, the
resulting operator biases decoding away from the thinking pathway.

**Proof (structural).** Training objective -log p(a | q) with
enable_thinking=False never exposes the model to `<thinking>`
delimiters in the target. Gradient signal through BA pushes
p(`<thinking>` | q) down (competing logit mass). Summing N such
operators linearly composes the suppression (LoRA is linear in ΔW):
ΔW_total · e(q) = Σ ΔW_i · e(q), all components pointing away from
`<thinking>`. ∎

**Corollary 2.** Even if the 5 domain adapters existed but were
trained non-thinking (the common case — `thinking_enabled: false` in
registry.json for math-gsm8k-knowledge-v0), delta-sum composition
degrades MMLU-Pro from 62.1% → 50.4% ± ε per F#536.

## Theorem 3 — Thinking-compatible training is the unsolved sub-problem (F#560)

**F#560 (2026-04-17):** "thinking-universal-v0: math+code (2000 ex)
produced MMLU-Pro 47.6% (-14.5pp), engineering 13.3%, philosophy
20.0%." First attempt at thinking-compatible training; DEGRADED MMLU-Pro.

**Theorem 3.** For Σ α_i · ΔW_i to ADD gain over base+thinking, each
ΔW_i must be a positive-contribution thinking-preserving operator.
F#560 demonstrates that even a single such operator has not been
produced: math+code with thinking enabled gave -14.5pp. Until a
single positive-ΔW thinking adapter exists, the sum cannot be
positive.

**Proof.** Linearity: Δ_total = Σ α_i · δ_i where δ_i is the per-adapter
behavioral delta (signed scalar, measured against base+thinking). If
sup_i δ_i < 0, then Δ_total ≤ (max α_i) · sup_i δ_i < 0. F#560 shows
sup δ_i = -14.5pp currently. ∎

## Theorem 4 — F#478 knowledge-gap closure on MMLU-Pro

**F#478 (2026-04-11, killed):** "Gemma 4 4B is too capable for
knowledge-gap exploitation. For rank-r LoRA adapter ΔW_d trained on N
basic domain examples: δ_d > 0 requires BOTH (1) vocabulary gap
H(V_d | θ_base) > H_threshold AND (2) distribution overlap V_d ∩
V_train ≠ ∅."

**Theorem 4.** MMLU-Pro falls within F#478's closure region. Specifically:
- MMLU-Pro is 10-option graduate-level MCQ;
- base+thinking = 62.1% already covers the broad knowledge floor;
- rank-6 q_proj adapters trained on 2000 basic examples (registry
  metadata) fail F#478 condition (1) at this difficulty.

**Proof.** F#478 is a closed empirical result on the exact base model
(Gemma 4 4B) at comparable MCQ difficulty. δ_d > 0 is forbidden under
F#478's constructive impossibility argument. ∎

## Theorem 5 — Cascade dependencies unresolved

**Pipeline dependency graph:**

    base+thinking  →  domain adapters  →  ridge router  →  delta-sum compose
                        [STUB ×5]         [KILLED K1616]      [CANNOT RUN]

- `exp_p1_t2_single_domain_training` (P=2, **open** — never ran).
- `exp_g4_ridge_routing_n25_mcq` (P=1, **killed** 2026-04-18:
  test_acc=0.8387 < 0.90 target; F#502 hidden-state ridge ties with
  TF-IDF at N=25).

**Theorem 5.** The pipeline is a 4-stage series composition; stages 2
and 3 are unbuilt and killed respectively. A series with a failed
link has failed end-to-end. ∎

## Theorem 6 — K1618 specification is framework-incomplete

K1618 text: "beats 62.1% MMLU-Pro thinking baseline." No:
- threshold (1pp? 5pp? statistically significant?);
- minimum detectable effect;
- sample size (MMLU-Pro full n=12032 or the community n=1400 subset?);
- tie-breaking rule (if pipeline = 62.1% exactly).

**Theorem 6.** A K specification without threshold and n is not a
falsifiable claim. Per PLAN.md Part 1 guardrail #1009, "VERDICT
CONSISTENCY", the experiment cannot move to `supported` without
passing a well-defined K. Therefore the only valid verdict today is
KILLED (framework-incomplete) or OPEN (awaiting spec fix).

## Predictions verified pre-flight (P1–P6)

| # | Prediction | Check | Result |
|---|------------|-------|--------|
| P1 | 0 of 5 `adapters/{math,bash,python,sql,medical}/adapters.safetensors` exist | `ls` | PASS: 0/5 present |
| P2 | 0 of 3 `micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors` exist | `ls` | PASS: 0/3 present |
| P3 | Upstream `exp_p1_t2_single_domain_training` status ≠ `supported` | `experiment get` | PASS: `status=open` |
| P4 | Upstream `exp_g4_ridge_routing_n25_mcq` status = `killed` | `experiment get` | PASS: `status=killed` |
| P5 | F#536 = 62.1% baseline exists and MCQ adapter + thinking = 50.4% | `experiment finding-get 536` | PASS |
| P6 | DB `success_criteria` is empty | `experiment get` | PASS: `success_criteria: []` |

## Salvageable sub-findings (distinct from closure)

1. **Linearity rule for delta-sum composition under mode-suppressing
   components.** Theorem 2 states that linear composition preserves
   thinking-suppression. This is a structural theorem, not an
   empirical finding; distinct from F#536 (which is the empirical
   anchor on a single MCQ adapter). Could be promoted by Analyst as a
   **design-time closure rule** for any future
   ΔW_sum-preserves-thinking hypothesis.

2. **Pipeline cascade-closure rule.** Stage-wise unreached stages in
   a series composition close the series outcome. Applies more
   broadly than just this experiment; a candidate pattern-level
   antipattern.

## Unblock path (for P11.HARNESS rebuild)

To make K1618 reachable, these pre-conditions must ALL hold:
1. 5/5 domain adapters with `adapters.safetensors` of non-trivial
   norm (> ε) on disk, trained with `enable_thinking=True`.
2. Per-adapter behavioral delta on MMLU-Pro vs base+thinking ≥ 0
   (Theorem 3 requirement).
3. Ridge router hits ≥ 90% test accuracy at N that matches the
   adapter set (K1616 rebuild).
4. K1618 spec fixed with explicit threshold + MDE + n.

Stage (2) is the open research question; F#560 says it hasn't been
solved. Stage (4) is a framework fix.

## References
- Finding #536 — MMLU-Pro thinking baseline (62.1%) and MCQ-adapter
  suppression (-11.7pp). `experiment finding-get 536`.
- Finding #478 — Gemma 4 4B knowledge-gap closure. `experiment
  finding-get 478`.
- Finding #560 — thinking-universal-v0 math+code → -14.5pp.
  `experiment finding-get 560`.
- Finding #502 — TF-IDF routing ties with hidden-state ridge at N=25
  (83.9% vs 84.2%). DB evidence in exp_g4_ridge_routing_n25_mcq.
- PLAN.md Part 1 guardrail #1009 — verdict consistency rule.
- REVIEW-adversarial.md and LEARNINGS.md from prior cascade kills
  (`exp_g4_routed_beats_base_think`, `exp_g4_25domain_real_hf`,
  `exp_followup_competitive_gsm8k_200n`, et al.).

QED.
