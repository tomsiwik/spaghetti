# MATH — `exp_model_long_context_adapter_stability`

**Title.** MODEL: N=5 adapter composition preserves needle-in-haystack at 8k / 32k / 128k context (F#263 frontier-extension, long-context).

**Type.** Frontier-extension (F#263 short-context kill at MMLU → does long-context recall hold under N=5 composition?).

**Claim.** Composition of 5 PoLAR adapters on Gemma 4 E4B preserves NIAH retrieval and RULER score across 8k / 32k / 128k context windows, within thresholds K1706 / K1707.

---

## 1. Failure mode the experiment is trying to prevent

"Composition perturbation cancels position-encoded retrieval at long context." Finding #263 derived a **failure mode** at short context (MMLU, ~512 tokens):

> Both NTP and SFT adapters degrade MMLU by similar amounts (-6pp and -5pp respectively).
> Knowledge recall depends on precise weight values; any perturbation at behavioral scale (s ≥ 4) disrupts stored factual knowledge.

At long context the failure mode generalizes: needle retrieval at 128k requires the attention head's effective receptive field to bind a position-encoded key (the needle's location-tagged token) to its value. A perturbation `W + s·ΔW` at any of the V/O projections — N=5 of them, summed — could disrupt the linear binding even when each ΔW alone is bounded. The experiment tests whether the F#263 short-context degradation extends, attenuates, or amplifies as context grows past base training length.

---

## 2. Prior math cited

- **Finding #263** (`exp_ntp_vs_sft_ood_benchmark`): composition mechanism itself degrades MMLU recall regardless of training objective, NTP -6pp / SFT -5pp on short context. This experiment extends to long context.
- **Finding #666** (`exp_softmax_router_scaling`): every proxy-metric KC must be paired with a target-metric KC. K1706 (NIAH retrieval rate) is a structural retrieval proxy; K1707 (RULER multi-task score) is the target/behavioral pair.
- **Finding #627** (`exp_p1_t2_single_domain_training`): r=6 LoRA on `v_proj + o_proj` is the proven Gemma 4 E4B target-module configuration. All N=5 candidate adapters in the F#627 cohort use this configuration.
- **Needle-in-haystack benchmark** (Kamradt 2023, `https://github.com/gkamradt/LLMTest_NeedleInAHaystack`): standard long-context retrieval probe. Operationalizes "did the model find the planted token at depth d in context length L".
- **RULER benchmark** (Hsieh et al. 2024, `arxiv:2404.06654`): 13-subtask suite measuring NIAH variations, multi-hop tracing, aggregation, and QA at controlled long-context lengths. Currently the strongest long-context evaluation suite.
- **LongLoRA** (Chen et al. 2023, `arxiv:2309.12307`): demonstrates that low-rank perturbations on `v_proj + o_proj` interact with shifted-sparse attention at long context; the same projection-targets that work at short context can become fragile beyond base training context.
- **Gemma 4 long-context attention** (Gemma 4 model card, native 128k context for E4B per the unified model report): provides the upper bound on the position-encoded retrieval window.

---

## 3. Derivation — does F#263 extend to long context?

### 3.1 Linear-perturbation regime (short context, F#263)

For a single adapter `ΔW` at scale s acting on a hidden state h, the perturbed forward gives `(W + sΔW)h = Wh + s·(ΔW h)`. F#263 showed that on MMLU short-context recall, the SFT/NTP adapter's `s·ΔW h` term consistently shifts the next-token logits enough to drop recall by 5-6pp, regardless of training objective. The composition `Σ_i s·ΔW_i = s·(Σ_i ΔW_i)` linearly aggregates these perturbations.

Under independent-perturbation modeling (orthogonal-A Grassmannian construction per F#562), the operator-norm bound is `‖Σ_i ΔW_i‖ ≤ Σ_i ‖ΔW_i‖`. For N=5 PoLAR adapters at unit operator-norm, total perturbation grows at most linearly. For "behavioral scale" s in F#263's failure regime, the linear bound is the *upper* envelope — actual cancellation in the orthogonal subspaces typically yields sub-linear growth.

### 3.2 Position-encoded long-context regime

At long context (>= 32k), retrieval requires the attention head to bind a query at position p_q with a key at position p_k = p_needle. RoPE / position-bias terms are baked into `k_proj`, `q_proj`. PoLAR adapters in the proven `v_proj + o_proj` configuration (F#627) do **not** touch the position encoding directly — they shift the value subspace and the output projection. This produces a key invariance:

> The attention pattern `softmax(QK^T / √d)` is *unaffected* by V/O perturbations. Composition error enters only via `(softmax(QK^T)·V_perturbed)·O_perturbed`, i.e. through the value-aggregation channel and the output mixer.

Consequence: long-context retrieval is **structurally protected** for V/O-only adapters — the needle-position binding survives perturbation. The failure mode would have to enter through value-corruption or output-mixer rotation, both of which are bounded by operator-norm of `Σ ΔW_v`, `Σ ΔW_o`.

### 3.3 Range-extrapolation regime (>= base training length)

Beyond Gemma 4 E4B's base training context, position encoding extrapolates per its frequency schedule. LongLoRA (`arxiv:2309.12307`) shows that low-rank perturbations on `v_proj + o_proj` can interact with the extrapolated regime: when shifted-sparse attention is used, V/O-only adapters mostly preserve retrieval, but at full-attention 128k the perturbation can compound with positional drift.

Therefore the prediction splits by length:
- 8k (within base training length): NIAH and RULER nearly identical to base. **K1706 PASS predicted.**
- 32k (near base extrapolation cutoff): mild degradation expected, within the 5pp / 3pp thresholds.
- 128k (full extrapolation): outcome genuinely open. The §3.2 V/O structural protection competes with the §3.3 position-extrapolation interaction. This is the empirical question the experiment is designed to settle.

### 3.4 Implication for experimental design

Two regimes give clear proof-first predictions (8k PASS, 32k mild). The 128k point is the genuinely-open empirical question. A single-pass run measuring NIAH + RULER at 8k / 32k / 128k under N=5 composition is sufficient to settle the claim — *provided* the compute budget allows it.

---

## 4. Predictions

| Prediction | Outcome | Mechanism |
|---|---|---|
| P1 (8k): NIAH(N=5 composition) within 1pp of base | PASS | §3.2 V/O structural protection within base training length |
| P2 (32k): NIAH within 5pp of base | PASS (likely) | §3.2 V/O protection + mild §3.3 extrapolation drift |
| P3 (128k): NIAH within 5pp of base | OPEN | §3.2 protection vs §3.3 full-extrapolation compounding |
| P4 (RULER): score within 3pp of base across all subtasks | PASS at 8k, OPEN at 32k/128k | aggregate target-behavioral pair per F#666 |
| P5 (range degradation, F#263 extension): degradation grows monotonically with context length | likely | linear-superposition bound + extrapolation drift |

---

## 5. Kill criteria (pre-registered, locked at MATH.md write-time per F#666)

The two KCs are inherited from the DB-registered text (K1706 / K1707) and are NOT modified.

- **K1706 (structural / proxy)** — NIAH recall under N=5 composition within 5pp of base at all of 8k, 32k, 128k.
  Fail ⇔ `min_{L ∈ {8k,32k,128k}} (NIAH_compose(L) − NIAH_base(L)) < −5.0pp`.

- **K1707 (target / behavioral, paired per F#666)** — RULER score under composition within 3pp of base across **all** RULER subtasks (13 subtasks per `arxiv:2404.06654`).
  Fail ⇔ `min_{subtask ∈ RULER} (RULER_compose − RULER_base) < −3.0pp`.

- **Verdict rule (F#666 compliance)**
  - KILL requires BOTH K1706 FAIL AND K1707 FAIL.
  - SUPPORTED requires BOTH K1706 PASS AND K1707 PASS.
  - K1706 PASS + K1707 FAIL → "retrieval preserved but downstream RULER tasks degrade" (NIAH is an insufficient proxy for full long-context capability).
  - K1706 FAIL + K1707 PASS → "needle retrieval drops but composite long-context tasks survive" (NIAH overstates degradation).

---

## 6. Experimental design

- **Base model.** `mlx-community/gemma-4-e4b-it-4bit` (cached on disk per `~/.cache/huggingface/hub/`). NOT `26b-a4b-it-4bit` — that's a separate experiment (`exp_model_knowledge_gap_26b_base`).
- **Adapters.** N=5 rank-r=6 PoLAR on `v_proj + o_proj` (Finding #627). Required adapters per `notes`: code, math, medical from `exp_p1_t2_single_domain_training`; legal + finance from `exp_p1_t2_multi_domain_5`.
- **Composition.** `Σ_i B_i @ A_i` correctly summed (NOT `(ΣB)(ΣA)` — `mem-antipattern-001`). LORA_SCALE ≤ 8 per F#328/#330.
- **NIAH harness.** Standard depth-percent grid: needle at {10, 25, 50, 75, 90}% of context, retrieval prompt at end. Score = retrieval rate over the 5 depths.
- **RULER harness.** 13 subtasks per `arxiv:2404.06654`: NIAH-MK, NIAH-MV, NIAH-MQ, VT, CWE, FWE, QA-1, QA-2, NIAH-S-1/2/3, multi-hop tracing, aggregation. Score per subtask, take min over subtasks for K1707.
- **Context lengths.** 8k, 32k, 128k. Within Gemma 4 E4B's native 128k context window.
- **Memory.** Phased execution per `/mlx-dev`: 128k context prefill at 4-bit ≈ 12-15 GB activations; on M5 Pro 48GB this requires `mx.eval + mx.clear_cache` between adapter compositions and between context lengths. **Skill invocation `/mlx-dev` and `/fast-mlx` is required at IMPL time** — explicitly NOT performed in this design-only iteration because no platform code is being written here (refusal scaffold only).
- **Compile.** `mx.compile` the composed forward per `/fast-mlx`; one compile per (composition, context-length) pair to amortize.

---

## 7. Assumptions and blockers

### Blockers (actionable)

- **B1.** Compute budget. Per the experiment's `notes` field (set by prior researcher iteration on 2026-04-24): `128k prefill @ Gemma 4 E4B 4bit on M5 Pro ≈ 10 min/sample`; `RULER is 13-subtask multi-hour benchmark`; `N=5 composition × {8k,32k,128k} × ≥5-needle positions = 4-8 h total compute`. This **exceeds the 30-min single-iteration researcher budget** (guardrail 1009).
- **B2.** Reclaim path (also from `notes`): (1) schedule a dedicated ≥4h session, (2) `experiment update --priority 2`, (3) invoke `/mlx-dev` + `/fast-mlx`, (4) implement NIAH harness first, (5) run RULER after NIAH sanity check. The reclaim path is well-specified and ready.
- **B3.** RULER reference scores for Gemma 4 E4B base are not in `experiment query` — would need to measure base on the same harness in the same iteration to compute deltas. This is independently expensive (13 subtasks × 3 lengths) and is part of the 4-8h budget.

### Assumptions

- **A1.** Gemma 4 E4B native context is 128k (per Gemma 4 model card). If only 32k is reliable, the 128k row of the prediction table becomes "untestable on this base" rather than "open".
- **A2.** F#627 V/O-only target-module configuration was validated at short context. The §3.2 V/O structural protection argument extends to long context **iff** the same projections are used; q/k targeting would invalidate the derivation.
- **A3.** N=5 PoLAR adapter set has Grassmannian-orthogonal A matrices (Finding #562). Linear superposition bound (§3.1) holds tightly only under orthogonal-A.
- **A4.** Researcher-hat guardrail 1009 caps single-iteration work at 30 min / 40 tool calls; a 4-8 h NIAH+RULER sweep is explicitly out of scope without authorization.

---

## 8. Verdict pre-announcement (proof-first)

Under §3.2 (V/O structural protection), 8k and likely 32k pass-predicted. Under §3.3, 128k is genuinely open and is the empirical hinge. The dense regime alone does not settle K1706/K1707.

- If run with the full 5-adapter set on the 4-8h compute window: expected K1706 PASS at 8k, mild regression at 32k, OPEN at 128k. K1707 outcome scales similarly.
- If not run (current state): the experiment resolves **PROVISIONAL** — proof-first prior gives partial coverage (8k/32k pass-predicted), but the 128k empirical hinge is the load-bearing point and is not measured here.

This filing follows the same canonical pattern as `exp_model_knowledge_gap_26b_base` (F#768): macro-scope design-only sub-case where (i) running exceeds single-iteration compute budget, (ii) proof-first prior partially covers the claim but does not resolve it, (iii) silent proxy substitution (e.g. measuring on 8k only and claiming 128k by extrapolation) would violate researcher antipattern (m).

---

## 9. References (paper grounding)

- Kamradt (2023) — Needle-in-a-Haystack, `https://github.com/gkamradt/LLMTest_NeedleInAHaystack`
- Hsieh et al. (2024) — RULER, `arxiv:2404.06654`
- Chen et al. (2023) — LongLoRA, `arxiv:2309.12307`
- Gemma 4 model card (E4B native 128k context)
- Finding #263 — `exp_ntp_vs_sft_ood_benchmark` (composition degrades MMLU at short context)
- Finding #562 — Grassmannian A orthogonality on Gemma 4
- Finding #627 — V/O target modules for Gemma 4 E4B LoRA
- Finding #666 — target-gated kill rule; paired proxy/target KCs
- Finding #768 — canonical macro-scope BLOCKED-on-resource PROVISIONAL filing pattern
