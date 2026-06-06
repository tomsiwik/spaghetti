# exp_g4_adapter_class_composition_full — MATH (PROVISIONAL, design-only)

## §0 Platform skills and scope preservation (pre-registered)

**Platform skills invoked for design:** `/mlx-dev`, `/fast-mlx` — see reviewer.md §5 "PROVISIONAL (novel-mechanism design-only sub-case)". This experiment is PROVISIONAL-as-design because the full empirical pipeline (macro-scale: 3 adapter classes × 5 domain trainings = 15 LoRA/DoRA/MoLoRA trainings on Gemma 4 E4B + MMLU-Pro N=5 composition eval) realistically exceeds the 90-minute single-iteration budget (guardrail 1009, memory `mem-antipattern-novel-mechanism-single-iteration-scope`). MoLoRA specifically is not available as a turn-key mode in `mlx_lm.lora` CLI and requires a custom routing module — novel-mechanism sub-component.

**Pinned versions.** `mlx-lm >= 0.22` (DoRA supported via `--fine-tune-type dora`; verify before writing custom DoRA loop). MoLoRA has no `mlx_lm` turn-key; a custom module `micro/utils/molora.py` must be written (this is the `_impl` deliverable).

**Base model (exact HF repo id):** `mlx-community/gemma-3n-E4B-it-4bit` (Gemma 4 E4B 4-bit, per repo convention). Adapter targets: `v_proj + o_proj` per F#627 (Gemma-4-specific, *not* q_proj which the parent proxy used).

**LoRA scale:** default (≤ 8, canonical `scale = 6.0` per F#627). Never hardcoded 20 per F#328/#330.

**Scope-preservation antipattern-t explicit forbid list** (MATH.md-level, binding for `_impl`):
- (F1) Do NOT silently swap DoRA or MoLoRA for plain LoRA because "it didn't fit" or "CLI doesn't support it". That destroys the K2 comparison.
- (F2) Do NOT reduce N from 5 to 3 without declaring a new variant. N=5 is the DB-title specification; the parent proxy measured N=3 only because 3 domain adapters existed. The full experiment requires 5 domains.
- (F3) Do NOT substitute q_proj (parent proxy) for v_proj+o_proj (F#627 proven target). Different attention projections have different composition properties.
- (F4) Do NOT evaluate on MMLU subset < 1000 prompts without 95% CI explicitly; the 3pp margin claim requires a CI that rules out noise.
- (F5) OOM fix-order: (i) reduce per-device batch + raise grad-accum to preserve effective batch, (ii) enable `gradient_checkpointing`, (iii) reduce `max_length` to 2048 only if (i)+(ii) insufficient and document in PAPER.md. Do NOT swap base model, adapter class, or eval benchmark.

**Required for `_impl` iteration:** MLX custom training loop for MoLoRA (N expert LoRAs + learnable gate `g_i` via softmax router), using `mlx.nn.value_and_grad` + `mlx.optimizers.AdamW`, `mx.eval` at step boundary, `mx.clear_cache()` between phase trainings (F#673).

## §1 Scope and honest reframe

**DB title:** *"Class A (LoRA) vs Class B (DoRA, MoLoRA) on Gemma 4 MMLU-Pro composition: 3pp behavioral margin at N=5 (full)"*

**Parent (`exp_g4_adapter_class_composition`, F#679, PROVISIONAL):** measured only the composition-geometry proxy (dev_LoRA=0, dev_DoRA=0.089 measured, dev_MoLoRA=0.667 analytic) on existing N=3 q_proj LoRAs. The proxy K2 was PROXY-mislabeled-as-target per `mem-antipattern-proxy-kc-mislabeled-target` — it measured geometric deviation, not MMLU accuracy. The *behavioral target* (3pp MMLU-Pro margin at N=5) was explicitly deferred in the parent's PAPER.md §"What this does NOT claim" → this `_full` child closes that gap.

**This experiment's claim:** on Gemma 4 E4B 4-bit, trained adapters in Class A (LoRA with r=6) achieve MMLU-Pro accuracy under N=5 composition that exceeds Class B (DoRA or MoLoRA matched-r) by ≥3pp with 95% CI lower bound > 0.

**Type:** verification (frontier-extension from F#82 micro-d evidence + F#679 geometric proxy to Gemma 4 behavioral).

## §2 Theorem (class ordering under composition at behavioral scale)

**Setup.** Let `{ΔW_i}_{i=1}^N` be N trained rank-r adapters on shared base `W_0` (Gemma 4 E4B v_proj + o_proj across all layers). Three composition classes:

- **Class A (LoRA additive):** `W_A = W_0 + Σ_i α · B_i A_i` where `α = 6.0` per F#627.
- **Class B.1 (DoRA):** `W_{B.1} = m · (W_0 + Σ_i ΔW_i) / ||W_0 + Σ_i ΔW_i||_c` where `m ∈ R^{out}` is trained per adapter and composed by elementwise mean at composition time (per Liu et al, arxiv:2402.09353).
- **Class B.2 (MoLoRA):** `W_{B.2} = W_0 + Σ_i g_i(x) · B_i A_i` where `g_i(x)` is a learned softmax router over the N experts (per Feng et al, arxiv:2402.11260).

**Theorem.** Under Gemma 4 E4B 4-bit base + trained adapters with matched rank r=6, matched domain set D={code, math, medical, legal, law}, matched training budget (1000 steps per adapter, identical LoRA scale, identical optimizer), the ordering:

> `accuracy(A, MMLU-Pro | N=5) > accuracy(B.j, MMLU-Pro | N=5) + 3pp` for `j ∈ {1, 2}`

holds with 95% CI lower bound > 0, *if and only if* the composition-geometry mechanism of F#82 carries from micro-d scale (d=64) to Gemma 4 scale (d_hidden=2048) *and* translates into accuracy degradation rather than being absorbed by base model capacity.

**Proof sketch.** F#679 established `dev_DoRA = 0.089` and `dev_MoLoRA = 0.667` (geometric deviations from additive LoRA) on Gemma 4 scale. Per F#82 (FIT=0.875 at micro-d), higher composition deviation correlates with higher inter-task interference, which under MMLU-Pro's format (4-option MC requiring multi-step reasoning across 14 task categories) manifests as accuracy loss at N≥3. The 3pp threshold is load-bearing: it is the minimum magnitude at which the F#82 effect is reliably distinguishable from noise at n=1000 eval samples (binomial SE ≈ 1.6pp at p=0.5 → 95% CI radius ≈ 3.1pp at paired design). QED-sketch.

**Failure modes the theorem makes impossible (by KC gating):**
- DoRA/MoLoRA *underperform* LoRA due to training instability (not composition): ruled out by K1 (all N trainings converge) + matched training budgets.
- LoRA *underperforms* DoRA/MoLoRA (reverse ordering): this is the scientific outcome the experiment admits; test is falsifiable.
- No difference within 3pp: class B is functionally equivalent to class A at Gemma 4 scale, contradicting F#82's micro-d prediction. This is also falsifiable (PASS the K1 training KC but FAIL K2 behavioral).

## §3 Pre-registered kill criteria (target-gated per F#666)

**K1 (structural + adapter-class health).** For each class C ∈ {A, B.1, B.2} and each domain d ∈ D, training converges: final_loss < 1.1 · min(train_loss) (no divergence) AND final_loss < initial_loss · 0.7 (non-degenerate decrease). PASS iff ≥13/15 class-domain pairs (≥87%) converge.

**K2 (target behavioral).** At N=5 composition (all 5 domain adapters active), on MMLU-Pro n=1000 held-out test:
- Class A (LoRA) accuracy `acc_A`
- Class B.1 (DoRA) accuracy `acc_{B.1}`
- Class B.2 (MoLoRA) accuracy `acc_{B.2}`

PASS iff `acc_A − max(acc_{B.1}, acc_{B.2}) ≥ 0.03` AND `95% CI lower bound (paired bootstrap, 10000 resamples) > 0`.

**K3 (proxy confirmation — geometric mechanism).** Measure `dev_D = ||W_{B.1} v − W_A v|| / ||W_A v||` on trained DoRA at composition time (not the m_d=||W_0||_c init assumption of parent proxy). PASS iff `median(dev_D) > 10⁻³` (mechanism engaged: trained DoRA deviates from additive LoRA non-trivially at Gemma 4 scale). This closes the parent's caveat in F#679 PAPER.md: "`m_d = ||W_0||_c`" is a conservative lower bound; trained m_d usually increases deviation.

**K4 (ablation — rank control).** Re-run K2 at r=8 (same for all classes). PASS iff the ordering direction (sign of `acc_A − max(acc_{B.j})`) is stable across r=6 and r=8. Rules out rank-specific artifacts. (Inherited from MoLoRA paper's rank-sensitivity warning.)

**F#666 target-gating audit:**
- K1 structural + K2 target → class ordering SUPPORTED: LoRA > DoRA/MoLoRA at Gemma 4 scale by ≥3pp.
- K1 PASS + K2 FAIL + K3 PASS → geometric mechanism engaged but accuracy margin absent: F#82 micro-d-to-macro transfer KILLED, geometric signal does not imply behavioral margin.
- K1 PASS + K2 FAIL + K3 FAIL → mechanism did not engage (trained DoRA trivially additive at Gemma 4): F#679 proxy overestimated geometric deviation; scope-limited KILL.
- K1 FAIL → training pipeline broken, INCONCLUSIVE.
- K2 PASS + K3 FAIL → accuracy margin present but not via claimed geometric mechanism (suspect confounder: training instability, learning-rate-by-class interaction, data-order leakage). PROVISIONAL flag: further isolation required.
- K4 FAIL → rank-specific artifact, PROVISIONAL; re-specify at native r per class.

## §4 Assumptions

1. Gemma 4 E4B 4-bit LoRA training via `mlx_lm.lora --fine-tune-type lora` converges on 1000 steps with batch=4, max_length=2048, lr=1e-4 across 5 domains. Rationale: F#627 adapter-target choice + F#673 phase-clean-cache discipline proven.
2. DoRA via `mlx_lm.lora --fine-tune-type dora` converges identically (verify mlx-lm≥0.22 at `_impl` time).
3. MoLoRA custom module: 5 expert LoRAs + softmax gate MLP (1 hidden layer, d_hidden=64), trained jointly with AdamW on same budget. Composition formula: `y = W_0 x + Σ_i g_i(x) B_i A_i` with `g(x) = softmax(W_g x)`.
4. MMLU-Pro eval at n=1000 is sufficient power; binomial SE 1.6pp at p=0.5 → paired-design 95% CI radius 3.1pp. Matches 3pp threshold.
5. All five domain corpora exist or can be curated: code (HumanEval-train split), math (GSM8K-train), medical (PubMedQA-train), legal (CaseHOLD-train), law (LegalBench mixed — NOT overlapping with legal).
6. Composition of N=5 uses uniform routing at eval (DoRA: mean m; MoLoRA: gate conditioned on input; LoRA: sum).

## §5 Prediction (for PAPER.md prediction-vs-measurement table)

| KC | Prediction | Measured | Status |
|---|---|---|---|
| K1 structural | ≥13/15 class-domain trainings converge | (design-only) | untested |
| K2 target | `acc_A − max(acc_{B.j}) ≥ 0.03`, 95% CI LB > 0 | (design-only) | untested |
| K3 proxy | `median(dev_D on trained DoRA) > 10⁻³` | (design-only) | untested |
| K4 ablation | sign of K2 stable at r=8 | (design-only) | untested |

## §6 Deliverables for `_impl` iteration

1. Custom MoLoRA module `micro/utils/molora.py` (verified by unit test: forward matches `W_0 x + Σ g_i B_i A_i x` on random input).
2. Training script: 15 LoRA/DoRA/MoLoRA runs across 5 domains on Gemma 4 E4B 4-bit. Use `mlx_lm.lora` CLI for LoRA and DoRA; custom loop for MoLoRA with `mlx.nn.value_and_grad` + `mlx.optimizers.AdamW`, `mx.eval` each step, `mx.clear_cache()` between domain trainings.
3. Composition-eval script: load all 5 adapters per class, compose per class formula, run MMLU-Pro n=1000.
4. Statistics: paired bootstrap 10000 resamples for each pairwise class comparison, 95% CI.
5. Fill in the §5 prediction-vs-measurement table.

## §7 What this experiment does NOT claim (scope fence for `_impl`)

- Does not test N=2, 3, 4, 6+ composition (only N=5 per DB title).
- Does not test alternative class-B mechanisms (AdaLoRA, LoHa, etc.) — K4-level ablation only on rank.
- Does not claim the geometric mechanism of F#82 is the *sole* cause of any observed margin — K3 confirms engagement, not causation. Causation would require counterfactual (e.g. train DoRA but zero-out the magnitude drift).

## §8 References

- F#82 (conclusive): 15-adapter composition taxonomy at micro-d, FIT=0.875 for class-A-beats-class-B prediction.
- F#627: Gemma 4 E4B LoRA adapter targets `v_proj + o_proj`, scale=6.0.
- F#666: Target-gated kill criterion requirement.
- F#673: `mx.clear_cache()` between phase trainings, phased execution pattern.
- F#679 (provisional): Gemma 4 composition-geometry proxy; dev_LoRA=0, dev_DoRA=0.089, dev_MoLoRA=0.667 (parent of this experiment).
- arxiv:2402.09353 — DoRA weight-decomposed LoRA.
- arxiv:2402.11260 — MoLoRA mixture-of-LoRA experts.
