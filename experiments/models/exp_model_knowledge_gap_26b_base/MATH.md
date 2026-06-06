# MATH — `exp_model_knowledge_gap_26b_base`

**Title.** MODEL: Knowledge-gap reopens on Gemma 4 26B-A4B? (Finding #478 retest at larger base)

**Type.** Frontier-extension (F#478 impossibility at 4B → does it hold on 26B-A4B MoE?).

**Claim.** At least one domain adapter trained on Gemma 4 26B-A4B yields ≥5pp MMLU-Pro gain over base (vs. Finding #478 kill on 4B).

---

## 1. Failure mode the experiment is trying to prevent
"Base model too capable for adapter to lift domain accuracy" (F#478's failure mode). Finding #478 derived an **impossibility structure** on Gemma 4 4B:

> For rank-r LoRA adapter `ΔW_d` trained on N basic domain examples:
> `δ_d > 0` requires BOTH
>   (1) vocabulary gap `H(V_d | θ_base) > H_threshold`, AND
>   (2) distribution overlap `V_d ∩ V_train ≠ ∅`.
> Gemma 4 4B fails (1); P1 T2 adapters fail (2). (F#478)

The present experiment extends F#478 to a larger instance of the same architecture family. Per PLAN.md Part 2, `mlx-community/gemma-4-26b-a4b-it-4bit` is the Pierre production base; if its knowledge gap is still closed for basic-tier adapters, the "adapters as format amplifiers" interpretation (F#478 impossibility-structure) extends monotonically with capacity.

---

## 2. Prior math cited

- **Finding #478** (`exp_p4_b1_hard_question_eval`): knowledge-gap impossibility on Gemma 4 4B; introduces the `H(V_d|θ_base) > H_threshold` condition.
- **Finding #666** (`exp_softmax_router_scaling`): every proxy-metric KC must be paired with a target-metric KC. `accuracy_MMLU_Pro` is a task-accuracy (target) metric; behavioral-eval KC provides the structural pair.
- **Scaling-law monotonicity** (Kaplan et al. 2020, `arxiv:2001.08361`; Hoffmann et al. 2022, `arxiv:2203.15556`): conditional entropy of domain content decreases monotonically with model capacity on an aligned training distribution. This motivates the **monotonic extension** of F#478 to 26B.
- **MoE routing gap caveat** (Fedus et al. 2022, `arxiv:2101.03961`; Zhou et al. 2022, `arxiv:2202.09368`): expert routing in MoE models leaves "niche" regions where effective capacity per token is the active-parameter count (4B in a 26B-A4B), not the full count. This is the **only non-analogy mechanism** under which a reopened gap is defensible.

---

## 3. Derivation — does F#478 extend?

### 3.1 Monotonic extension (dense case)
For a dense instruction-tuned language model with parameters θ of size M, and domain corpus D_d with vocabulary V_d, denote the conditional entropy

> `H(V_d | θ) = E_{x ~ V_d} [-log p_θ(x | context)]`.

Scaling-law monotonicity implies `H(V_d | θ_M) ≤ H(V_d | θ_M')` whenever M ≥ M' and training data support is aligned. Therefore:

> If Gemma 4 4B (M≈4B) fails the condition `H(V_d|θ_base) > H_threshold`, Gemma 4 26B on the same domain corpus fails **more strongly**.

Consequence: in the dense-capacity limit, F#478 kill is **strictly monotonically stronger** at 26B than at 4B. K1702 (≥5pp MMLU-Pro gain) is predicted **FAIL**.

### 3.2 MoE-niche counter-mechanism (necessary escape hatch)
Gemma 4 26B is **26B-A4B** — 4B active parameters per token via expert routing. For a domain d whose routed experts' union has effective capacity `M_eff(d) = |⋃ E_d| * 4B`, the per-domain effective capacity can be **less than** the full 26B. If certain domains route to a narrow subset of experts (say 2 of 16), `M_eff ≤ 8B`, between 4B and 26B.

This creates a **non-monotonic capacity curve** in d. The F#478 impossibility structure then reopens iff **∃d : M_eff(d) ≤ 4B AND d was underserved by training**. No prior evidence identifies such a d on Gemma 4 26B; hypothesis is speculative pending measurement.

### 3.3 Implication for experimental design
The experiment's only defensible epistemic target is the **MoE-niche counter-mechanism** (§3.2). The dense monotonic case (§3.1) is proof-first KILLED. Running the experiment therefore measures:
- whether any domain in {code, math, medical, legal, finance} routes to a narrow expert set, AND
- whether a rank-r LoRA adapter on such a domain lifts MMLU-Pro by ≥5pp.

If no domain routes narrowly, K1702 will FAIL **for §3.1 reasons** (monotonic dense extension) and the finding is simply a confirmation of F#478 at larger scale.

---

## 4. Predictions

| Prediction | Outcome | Mechanism |
|---|---|---|
| P1 (dense): K1702 FAIL | `max_d δ_d < 5pp` | F#478 §3.1 monotonic extension |
| P2 (MoE-niche): K1702 PASS iff ∃d with `M_eff(d) ≤ 4B` | measurable only by running | §3.2 expert-routing measurement |
| P3 (target-gated per F#666): K1703 (behavioral) aligns with K1702 | same direction | target-metric pair |

---

## 5. Kill criteria (pre-registered, locked)

- **K1702 (structural / proxy)** — At least 1 domain adapter on Gemma 4 26B-A4B achieves `MMLU_Pro(base+adapter) − MMLU_Pro(base) ≥ +5.0pp`.
  Fail ⇔ `max_d (δ_d) < 5.0pp` across {code, math, medical, legal, finance}.

- **K1703 (target / behavioral, paired per F#666)** — Same domain-d adapter that passes K1702 shows domain-specific behavioral improvement on held-out eval: win-rate vs base ≥ 60% on N=30 held-out prompts rated by adversarial-judge.
  Fail ⇔ `win_rate < 60%`.

- **Verdict rule (F#666 compliance)**
  - KILL requires BOTH K1702 FAIL AND K1703 FAIL.
  - SUPPORTED requires BOTH K1702 PASS AND K1703 PASS.
  - K1702 PASS + K1703 FAIL → "format-only lift on MMLU" (keyword/notation artifact per F#478).
  - K1702 FAIL + K1703 PASS → "behavioral lift without MMLU movement" (MMLU insufficient proxy).

---

## 6. Experimental design

- **Base model.** `mlx-community/gemma-4-26b-a4b-it-4bit`. **NOT cached on disk as of 2026-04-23.**
- **Adapter.** rank-r LoRA, r=6, target `v_proj + o_proj` (Finding #627 on Gemma 4).
- **Domains.** code, math, medical (match F#478 set); optionally legal + finance if corpora are available (F#1629 flagged finance/legal train.jsonl missing in `exp_p1_t2_single_domain_training`).
- **Eval.** MMLU-Pro (structural) + N=30 per-domain held-out behavioral prompts (target).
- **Training.** 500 steps, lr=5e-5, effective batch 8, `enable_thinking=True` throughout.
- **Memory.** Gemma 4 26B-A4B-4bit ~13.5GB static + activations + LoRA deltas. Phased execution per `/mlx-dev`: `mx.eval + mx.clear_cache` between base-forward / adapter-forward / eval phases on M5 Pro 48GB.
- **Compile.** `mx.compile` the forward per `/fast-mlx`; no recompile between domains.

---

## 7. Assumptions and blockers

### Blockers (actionable)
- **B1.** Base model not present in `~/.cache/huggingface/hub/`. Estimated 13-15 GB download. In the current iteration the researcher did **not** initiate the download because (a) the F#478 monotonic prior (§3.1) is strongly predictive of kill, (b) the MoE-niche mechanism (§3.2) requires an orthogonal measurement first (expert-routing distribution per domain) that is itself a separate experiment, and (c) downloading + training 5 adapters at full N on 26B-A4B is out of the single-iteration (30 min, 40 tool call) budget per researcher hat guardrails.
- **B2.** Finance + legal training data absent in `exp_p1_t2_single_domain_training` (per F#1629 from the rank-complexity sibling); kept at 3 domains if not provided.

### Assumptions
- **A1.** Instruction-tuned 26B-A4B was pretrained on the same style of data as 4B (Google Gemma series). If the data mix differs materially (e.g. more medical/code), §3.1 holds *a fortiori*.
- **A2.** MoE expert routing is token-level not sample-level. Per-domain niche measurement must aggregate routing over a domain's tokens, not whole-sample labels.
- **A3.** LORA_SCALE ≤ 8 (safe per F#328/#330). No scale ablation in this run.

---

## 8. Verdict pre-announcement (proof-first)

Under §3.1 (monotonic extension of F#478), **P1 is proof-predicted FAIL**; the only route to K1702 PASS is the §3.2 MoE-niche mechanism, which needs an orthogonal expert-routing measurement not performed here.

- If run with the 26B model: expected KILL on K1702 (F#478 monotonic extension) + K1703 (target pair).
- If not run (current state): the experiment resolves **PROVISIONAL** — proof-first prior strongly predicts kill, but empirical verification is required before the claim can be filed as a finding of "adapter lift monotonically closes with capacity on Gemma 4 family."

---

## 9. References (paper grounding)

- Kaplan et al. (2020) — scaling laws for LM, `arxiv:2001.08361`
- Hoffmann et al. (2022) — Chinchilla, `arxiv:2203.15556`
- Fedus et al. (2022) — Switch Transformer, `arxiv:2101.03961`
- Zhou et al. (2022) — Mixture-of-Experts scaling, `arxiv:2202.09368`
- Finding #478 — `exp_p4_b1_hard_question_eval` (Gemma 4 4B knowledge-gap closed)
- Finding #627 — Gemma 4 LoRA target modules `v_proj + o_proj`
- Finding #666 — target-gated kill rule; paired proxy/target KCs
