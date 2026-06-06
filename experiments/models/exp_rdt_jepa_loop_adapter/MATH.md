# MATH — `exp_rdt_jepa_loop_adapter`

**Claim.** An RDT (recurrent-depth transformer) loop adapter at rank-16 on `v_proj + o_proj` over layers [12, 20] of Gemma 4 E4B MLX 4-bit, trained with a **JEPA next-embedding prediction objective** (predict `h_{d+1}` from `h_d` in the loop's fixed-point iterate, stopgrad on target) plus SIGReg Epps-Pulley anti-collapse regularization, closes the parent `exp_rdt_loop_lora_gemma4_bench` (F#674 PROVISIONAL) behavioral gap: +5pp on GSM8K-Hard at T=3 vs base + saturating-exp depth elasticity R² > 0.90 across T ∈ {1..6}.

**Type.** Frontier-extension — combines LeWorldModel's SIGReg-stabilized JEPA (arxiv:2603.19312, applied to pixel world-models) with Bae 2024 Relaxed Recursive Transformers (arxiv:2410.20672, depth-recurrent LoRA), applied for the first time to a **residual-stream predictor across recurrent depth iterations** of a frozen LLM.

**Platform.** Apple M5 Pro 48GB, MLX (`mlx-core 0.31.1`, `mlx-lm 0.31.2`). Base model `mlx-community/gemma-4-e4b-it-4bit`. Adapter targets `v_proj + o_proj` per F#627. Loop region [LOOP_START, LOOP_END) = [12, 21).

---

## §0 Skills invoked, scope lock, pinned versions

Platform skills `/mlx-dev` + `/fast-mlx` invoked before writing this document (PLAN.md Part 2 requirement; F#673 lineage). Key items internalized from the skills and applicable to this experiment:

- `mx.eval(loss, model.parameters())` at loop boundaries; MLX is lazy — omission causes deferred-graph memory explosion.
- `mx.clear_cache()` between phases (teacher forward → student forward → gradient step → next batch) to avoid M5 Pro 48GB OOM (F#673 lineage).
- `nn.value_and_grad(model, loss_fn)` is the only correct MLX gradient interface — `.backward()` is a silent no-op on `mx.array`.
- Activation capture on a frozen model requires module hooks or explicit sub-forward calls returning intermediate tensors — mlx-lm does not expose layer-level hooks natively, so `run_experiment.py` must subclass or monkey-patch `Gemma4TextModel.__call__` to return `h_d` at each depth step d ∈ [LOOP_START, LOOP_END).
- `mx.linalg.qr(stream=mx.cpu)` for partition-QR (unchanged from parent).
- `mlx == 0.31.1`, `mlx-lm == 0.31.2` (pinned to match parent `exp_rdt_loop_lora_gemma4_bench` and sibling `exp_rdt_loop_kv_cache`).

**Scope-preservation lock (reviewer antipattern (t) defence; mem-antipattern-novel-mechanism-single-iteration-scope).** KCs K1770–K1774 are pre-registered. No scope swap is permitted between MATH.md and code:

- F1: Base model = `mlx-community/gemma-4-e4b-it-4bit` (parent's target, not a smaller surrogate).
- F2: Loop region = layers 12..20 inclusive (N_LOOPS parameter T ∈ {1..6}, 9 consecutive DecoderLayers shared across depth iterations, matching parent's architecture).
- F3: Objective = JEPA next-embedding prediction + SIGReg (not plain next-token LoRA).
- F4: Eval = GSM8K-Hard (GSM8K test split, max_tokens=1024 per F#1629) at n ≥ 200, greedy, at T=3 (matches parent K1740-BENCH pre-reg).
- F5: Depth elasticity sweep = T ∈ {1,2,3,4,5,6}, n ≥ 30 per T, same prompts across T.
- F6: Stability check = full `max_d rho(A_d) < 1` across **500+ real GSM8K-loss steps** (not 50 smoke steps; structural KC from parent F#674).

If any of F1–F6 must change to make the run tractable, that is a scope swap; the correct response is **PROVISIONAL with explicit scope-deferral**, not silent modification.

**Single-iteration-scope classification.** This experiment is a **novel-mechanism** filing: the training loop is a bespoke MLX routine (residual-stream hook at each recurrent depth iterate + prediction head + SIGReg Epps-Pulley + stopgrad targets + recurrent forward). It is **not** executable via `mlx_lm.lora` CLI. Per `mem-antipattern-novel-mechanism-single-iteration-scope` and reviewer.md §5 PROVISIONAL (novel-mechanism design-only sub-case), the correct researcher-hat deliverable is MATH.md + graceful-failure scaffold + PAPER.md + `_impl` follow-up at P3; the empirical training loop lands in the `_impl`, not in this iteration.

## §1 Architecture — RDT loop + JEPA prediction head

### 1.1 RDT loop block (inherited from parent, F#674)

Parent experiment defined an RDT layer-loop over layers [12, 21): a single shared LoRA delta is applied to the 9 consecutive Gemma 4 E4B DecoderLayers, iterated T times at inference. Let `B_loop = LoRA_Δ(v_proj, o_proj, rank=16)` applied to each layer in [12, 21). Call the composed block `L(h; Δ)`:

```
L(h; Δ) = DecoderLayer_20(...DecoderLayer_12(h))  # 9 layers with Δ applied
```

The recurrent depth-T forward is:

```
h_0 = Embed(x)
# layers 0..11 (below loop): standard
h_pre = f_pre(h_0)
# loop iterates
h_d = h_pre, for d = 0
for d in {0, 1, ..., T-1}:
    h_{d+1} = L(h_d; Δ)
# layers 21..41 (after loop): standard
h_final = f_post(h_T)
logits = Unembed(h_final)
```

Per parent Theorem 2 (K1739 structural): `max_d rho(A_d) < 1` where `A_d = ∂L(h_d)/∂h_d` is the loop Jacobian, ensuring the fixed-point iterate converges. Parent F#674 observed this PASS at T=3 smoke scale; K1770 inherits this requirement at full scale (500+ real GSM8K-loss steps).

### 1.2 JEPA prediction head P_θ

Novel contribution (vs. parent's plain next-token LoRA objective): train `Δ` to make each depth iterate `h_d` a **predictor of the next iterate `h_{d+1}`** in the loop's own dynamics. Introduce a lightweight prediction head `P_θ: R^d → R^d` (2-layer MLP, hidden_dim = d = 2560 for Gemma 4 E4B), trainable jointly with `Δ`:

```
h_hat_{d+1} = P_θ(h_d)
L_pred = (1/|B|) Σ_b Σ_d || P_θ(h_d^{(b)}) - stopgrad(h_{d+1}^{(b)}) ||²
```

The sum is over depth iterates d ∈ {0, 1, ..., T-1} and over batch sequences b. **stopgrad on the target** is the JEPA pattern that prevents the trivial bypass `h_{d+1} = h_d` (identity collapse would otherwise let the base model satisfy the objective without touching Δ).

**Inference-time discard.** After training, `P_θ` is discarded — only `Δ` is saved as an mlx-lm adapter checkpoint. The claim is that the JEPA auxiliary loss routes residual-stream-predictive knowledge *into* `Δ` during training; the loss cannot be evaluated at inference time without the head, but `Δ` alone is used at eval.

### 1.3 SIGReg anti-collapse on depth iterates

Without an anti-collapse term, JEPA collapses `P_θ(h_d) = c` (constant) or onto a low-rank subspace, trivially satisfying L_pred without capturing dynamics. Per LeJEPA (arxiv:2511.08544) §2 and LeWM (arxiv:2603.19312):

```
Z = concat_d P_θ(h_d^{(b)}) ∈ R^{|B|·T × d}
L_SIGReg = (1/M) Σ_{u_m ~ S^{d-1}} D_EP(u_m^T Z, N(0, 1))
L_total = L_pred + λ · L_SIGReg
```

where `D_EP` is the Epps-Pulley test statistic (LeJEPA Eq. 7) and M = 1024 random unit projections. `D_EP(z, N(0,1)) = ∫ |ψ_z(t) - exp(-t²/2)|² · w(t) dt` with Gaussian kernel weight `w(t) = (2π)^{-1/2} exp(-t²/2)`.

**Cross-depth collapse — novel failure mode.** Unlike LeWM (predicting future pixel frames) or `exp_jepa_adapter_residual_stream` (predicting next-token residuals at fixed layer), here the predictor operates **across recurrent depth at fixed token**. A novel collapse mode is *cross-depth collapse*: `h_d ≈ h_{d+1}` for all d (fixed-point reached at d=1, loop does nothing). K1771 (SIGReg pass at each d ∈ {1..6}) specifically tests this — isotropy per-d prevents every d from collapsing onto a shared low-rank manifold.

## §2 Cited prior math

- **LeWorldModel** (arxiv:2603.19312, Maes/LeCun/Balestriero 2026-03-24) — SIGReg stabilizes end-to-end JEPA for pixel world-models; reduces VICReg's 6+ hyperparameters to a single λ with explicit anti-collapse guarantee. Direct parent of this experiment's objective.
- **LeJEPA** (arxiv:2511.08544) — Eq. 7 Epps-Pulley statistic formulation; Thm 1: SIGReg minimized ⟺ Z approximately isotropic Gaussian. Cramér-Wold via M projections.
- **Bae 2024 Relaxed Recursive Transformers** (arxiv:2410.20672) — depth-recurrent LoRA framework. Proves that a shared LoRA Δ applied T times approximates a deeper network at compute cost of single forward. Parent of RDT loop construction.
- **Finding #627** — Gemma 4 E4B LoRA on `v_proj + o_proj` at r=6 achieves target behavioral improvement; this is the proven target for this architecture. r=16 is an expansion with the same structural basis.
- **Finding #666** — Proxy KCs must be paired with target-metric KCs. KILL requires both proxy and target to fail.
- **Finding #674** — Parent `exp_rdt_loop_lora_gemma4_bench` PROVISIONAL: K1739 structural PASS at T=3 smoke; K1740 behavioral UNDERPOWERED at n=50 smoke. This experiment's objective is to close F#674's Caveat 2 (weak behavioral lift) via JEPA's richer training signal.
- **Finding #1629** — max_tokens=1024 prevents Gemma 4 CoT truncation on GSM8K eval.

## §3 Kill criteria (pre-registered; canonical DB text — do not edit)

- **K#1770 (structural, inherited from F#674).** `max_d rho(A_d) < 1` across 500+ real GSM8K-loss steps (dynamical stability preserved under JEPA objective, not weakened by auxiliary loss).
- **K#1771 (structural, proxy).** SIGReg Epps-Pulley rejection rate < 5% on loop-block output `P_θ(h_d)` at each d ∈ {1..6} (no cross-depth representation collapse).
- **K#1772 (proxy, learning dynamics).** `L_pred(step 500) / L_pred(step 50) < 0.5` AND monotone decrease across depth iterations d at fixed step (objective is learnable; not trivially satisfied).
- **K#1773 (target, paired with K#1772 per F#666).** GSM8K-Hard accuracy **+5pp at T=3** vs base Gemma 4 E4B, n ≥ 200, greedy. Matches parent pre-reg K1740-BENCH.
- **K#1774 (target, paired with K#1771 per F#666).** Depth-elasticity saturating-exp R² > 0.90 on T ∈ {1..6} at n ≥ 30 per T (closes parent F#674 K1742 under-power caveat).

**Target-gating (F#666).** K#1771 is proxy (activation-statistics), K#1772 is proxy (training-loss ratio); K#1773 and K#1774 are behavioral targets (GSM8K accuracy and depth-elasticity on real prompts). Pairing:

- K#1771 ↔ K#1774 — isotropy-preservation (proxy) pairs with depth-elasticity (target): if iterates are isotropic across d but depth elasticity fails, SIGReg is tautological; if depth elasticity passes but iterates collapse, SIGReg measures the wrong object.
- K#1772 ↔ K#1773 — learning-dynamics (proxy) pairs with GSM8K accuracy (target): if loss decreases but GSM8K does not, the objective doesn't transfer; if GSM8K improves but loss doesn't decrease, proxy measures wrong step.
- K#1770 is structural (stability) and **does not require pairing** per F#666 — it is a precondition, not a behavioral claim. It applies to both pairs equally (if stability fails, no KC can be interpreted).

**KILL rule.** KILL requires at least one target KC (K#1773 or K#1774) to FAIL jointly with its paired proxy. SUPPORTED requires K#1770 PASS AND K#1771 PASS AND K#1772 PASS AND K#1773 PASS AND K#1774 PASS. Any proxy-FAIL with target-PASS is a finding about the proxy, not a kill.

## §4 Mechanism — why JEPA on recurrent depth should beat plain next-token LoRA

### 4.1 Parent weakness

Parent `exp_rdt_loop_lora_gemma4_bench` (F#674) trained Δ with standard next-token cross-entropy on GSM8K. Caveat 2 of F#674 noted underpowered behavioral signal: K1740 PASS at T=3 but marginal (+1.2pp smoke), K1742 (depth elasticity) UNDERPOWERED at n=50. The loop executed but did not visibly *exploit* depth — gains from T=1 → T=6 were within noise.

### 4.2 JEPA objective provides structured training signal

Plain next-token loss trains Δ to minimize the token-level error of the final iterate `h_T`. It does NOT pressurize the intermediate iterates `h_1..h_{T-1}` to form a useful trajectory. The JEPA objective adds a **dense auxiliary signal across every depth step**: each h_d must predict h_{d+1}, so Δ is forced to encode a transition operator on the residual stream rather than a single compressed update.

**Theorem (informal).** Let Δ* be the Δ minimizing `L_CE(h_T)` alone (parent's objective), and let Δ+ minimize `L_CE(h_T) + α · L_pred + β · L_SIGReg` (this experiment, in the regime where α, β > 0 but small enough not to dominate L_CE). Then Δ+ has strictly more training signal per step than Δ*, specifically T additional constraint pairs (h_d, h_{d+1}) for d ∈ {0..T-1}. Under standard overparameterized-training assumptions (Jacot 2018, NTK), training signal density correlates with generalization quality at fixed step budget.

**Predicted effect.** Δ+ encodes a meaningful fixed-point iterate, so T=6 evaluation actually uses depth; depth elasticity saturating-exp fits with R² > 0.90 (K#1774). The +5pp GSM8K lift (K#1773) follows from Δ+ generalizing better than Δ* at matched parameter count.

### 4.3 SIGReg is load-bearing, not cosmetic

Without SIGReg, L_pred drives collapse: `P_θ(h_d) → const` satisfies L_pred trivially if targets cluster, and `h_d → h_{d+1}` satisfies it trivially if loop is idempotent at d=1. SIGReg forces `Z = concat_d P_θ(h_d)` to be approximately isotropic Gaussian per LeJEPA Thm 1, geometrically ruling out both collapse modes. K#1771 verifies this.

### 4.4 Contrast with `exp_jepa_adapter_residual_stream`

The sibling F#682 experiment also uses JEPA + SIGReg, but in a **layer-wise** configuration (predict `h_{layer+1}(token_{t+1})` from `h_layer(token_t)` at fixed depth). This experiment predicts across **recurrent depth** at fixed layer set — a genuinely new failure surface (cross-depth collapse) not present in F#682. Both experiments contribute independent mechanism evidence; neither makes the other redundant.

## §5 Predictions — prediction-vs-measurement table scaffold

| # | Prediction | KC | Mechanism | Falsifier |
|---|---|---|---|---|
| P1 | `max_d rho(A_d) < 1` holds across 500 training steps | K#1770 | JEPA objective does not introduce new contractive modes (theorem: auxiliary loss is added, not replacing) | rho ≥ 1 at any step → stability broken; JEPA objective fights parent's structural guarantee |
| P2 | Epps-Pulley rejection rate < 5% on `P_θ(h_d)` at each d ∈ {1..6} at step 500 | K#1771 | SIGReg forces isotropic Gaussian at each depth iterate | rejection ≥ 5% at any d → cross-depth collapse detected → adapter is degenerate |
| P3 | `L_pred(step 500) / L_pred(step 50) < 0.5` with monotone decrease across d | K#1772 | Residual-stream dynamics across recurrent depth are learnable at this scale | ratio ≥ 0.5 or non-monotone → objective doesn't fit; training budget insufficient or depth signal absent |
| P4 | GSM8K-Hard accuracy ≥ baseline + 5pp at T=3, n=200 greedy | K#1773 | JEPA auxiliary signal transfers knowledge into Δ; depth iterates encode useful trajectory | gain < 5pp → auxiliary signal didn't transfer, or T=3 depth is wrong operating point |
| P5 | Depth-elasticity saturating-exp R² > 0.90, T ∈ {1..6} at n ≥ 30 per T | K#1774 | With JEPA-trained Δ, each loop iterate is meaningful; depth exhibits structured saturation (not noise) | R² < 0.90 → depth elasticity absent (no smooth improvement curve); loop is redundant past T=1 |

## §6 Scope escalation — PROVISIONAL-as-design (novel-mechanism sub-case)

Per reviewer.md §5 and `mem-antipattern-novel-mechanism-single-iteration-scope`, this experiment is filed as **PROVISIONAL-as-design**. Rationale:

1. **Novel training mechanism.** RDT loop + JEPA next-embedding + SIGReg Epps-Pulley is not executable via `mlx_lm.lora` CLI. Required components:
   - Monkey-patch or subclass `Gemma4TextModel.__call__` to expose intermediate `h_d` at each depth iterate d ∈ [LOOP_START, LOOP_END) × T.
   - 2-layer MLP prediction head `P_θ` with hidden_dim=2560, trained jointly with Δ on `v_proj + o_proj`.
   - Cross-depth residual collection: for each token t, stack `h_0(t), h_1(t), ..., h_T(t)`, apply `P_θ` to h_0..h_{T-1}, construct stopgrad targets from h_1..h_T.
   - SIGReg on `Z = concat_d P_θ(h_d)`: sample M=1024 unit vectors, project Z · u_m, compute Epps-Pulley statistic vs N(0,1). Per LeJEPA Eq. 7, requires numerical ECF integration — implementable via Gauss-Hermite quadrature at K=32 nodes.
   - Gradient step via `nn.value_and_grad(model, loss_fn)` + `mlx.optimizers.AdamW`. `mx.eval(model.parameters(), loss)` at step boundary; `mx.clear_cache()` between batches.
   - Adapter save compatible with mlx-lm adapter loading (P_θ discarded at inference).

2. **Runtime budget.** Estimated 6–10h wall-clock for end-to-end pipeline: (a) 500 training steps with per-step 6-depth-iterate forward + JEPA head + SIGReg on M=1024 projections ≈ 3-5h at T=6; (b) GSM8K-Hard eval at n=200 × 3 arms (baseline, JEPA, ablation) × max_tokens=1024 ≈ 1-2h; (c) depth-elasticity eval 6 Ts × 30 prompts × 2 arms (base, JEPA) ≈ 1-2h. Single-iteration cap is 30 min / 40 tool calls; 6-10h is 12-20× over budget.

3. **Upstream dependency partial.** Parent `exp_rdt_loop_kv_cache` PROVISIONAL (F#690): K1765 speedup (5× cached vs uncached) is required infrastructure for n=200 eval budget. Without K1765, the GSM8K-Hard eval at T=3, n=200 takes ~90 min per arm (from parent K1740-BENCH estimates) → eval alone is 4.5h, pushing pipeline past 12h. Note: per analyst routing, this is **not** a preempt-structural block (F#669 is behavioral-KC-gated, not infra-feasibility-gated) — infra dep is recoverable by the `_impl` iteration once the sibling `_impl` lands KV-cache speedup.

4. **PROVISIONAL (not KILLED).** No proof-based impossibility exists. The design is grounded in paper-validated math (LeJEPA Thm 1, LeWM application, Bae 2024 RDT), KCs are pre-registered and target-gated per F#666, and the scaffold refuses silent scope-swap. The blocker is implementation effort (~4-6h novel MLX engineering for the custom training loop) + compute budget (+ 6-10h empirical run) + infra dep (K1765 speedup from sibling `_impl`), not falsification. Canonical precedent: F#682, F#683, F#684, F#685, F#686.

**Follow-up filing.** `exp_rdt_jepa_loop_adapter_impl` at P3, inheriting MATH.md verbatim with all 5 KC IDs verbatim. Dep-linked to `exp_rdt_loop_kv_cache_impl` (P3, infra unblock) per analyst handoff; no double-dep-chain to current-iteration work.

## §7 Antipattern self-audit (pre-registration)

Checking each `type: fix` antipattern memory against this filing:

- **composition math bug** — N/A; single adapter Δ composed with itself via recurrent depth, not with another adapter. Parent F#674 already verified RDT composition math; this experiment inherits.
- **tautological routing** — N/A; no routing. Adapter applies to all layers in [12, 21) uniformly.
- **LORA_SCALE** — pre-registered `LORA_SCALE = 2.0` (parent's value per F#674) or default, ≤ 8 per F#328/F#330.
- **KC-swap-after-failure** — 5 KCs pre-registered above, canonical DB text, NOT editable post-hoc.
- **shutil.copy as new adapter** — N/A; this experiment trains Δ from scratch.
- **hardcoded `"pass": True`** — scaffold leaves all 5 KCs as `"not_measured"`; no synthetic pass.
- **eval-template truncation** — GSM8K eval uses max_tokens=1024 per F#1629 (not truncated to 512).
- **proxy-model substitution** — MUST load `mlx-community/gemma-4-e4b-it-4bit`. Scope lock F1 explicit.
- **smoke-as-full** — `is_smoke=false` in results.json; no smoke run masquerading as full.
- **novel-mechanism single-iteration-scope** — EXPLICITLY TRIGGERED: this filing routes PROVISIONAL-as-design per the memory, not silent partial implementation.
- **preempt-child-parent-target-unverified (F#669)** — checked and NOT triggered: K1765 (parent `exp_rdt_loop_kv_cache`) is infra-feasibility, not behavioral mechanism. F#669 governs behavioral-KC transitivity. Parent's K1764 bit-exact / K1765 speedup are parent-target-INDEPENDENT infra claims; this experiment's behavioral targets (K1773, K1774) do not require parent's targets to be SUPPORTED — they require parent's targets to be *feasibly runnable in `_impl`*, which is the same as this experiment's own `_impl` budget. Analyst elected C2 (PROVISIONAL-as-design) over C1 (preempt-KILL) on this exact axis.
- **claim-time cohort-saturation / tag-saturation / priority-inversion** — N/A at researcher level; claimed via `--id` override per handoff workaround.

## §8 Assumptions (per researcher autonomy guardrail 1008)

- **A1.** `mlx-lm 0.31.x` exposes `Gemma4TextModel.layers` as a list of `Gemma4TextDecoderLayer` instances amenable to monkey-patched forward returning intermediate residuals. Verification: see sibling `exp_rdt_loop_kv_cache` MATH.md §1 for matching patch pattern.
- **A2.** The residual stream at the loop boundary (output of layer 20, input to layer 21 in standard forward; output of each loop iterate in RDT forward) carries sufficient signal density for JEPA prediction. Parent F#674's K1739 structural PASS implies non-degenerate iterate geometry, so the assumption is weak.
- **A3.** SIGReg M=1024 projections is sufficient for d=2560. LeJEPA uses M ∈ {512, 1024, 4096} depending on dim; 1024 is mid-range.
- **A4.** λ (SIGReg weight) bisection over {0.0, 0.1, 1.0, 10.0} is wide enough. Follows LeWM §4.2 protocol; the `_impl` iteration will select argmin validation L_pred s.t. K#1771 passes.
- **A5.** LORA_SCALE ≤ 8 per F#328/F#330. Parent uses 2.0; inherit.
- **A6.** GSM8K test split with max_tokens=1024 is a faithful operationalization of "GSM8K-Hard" given no canonical hard split. Flag in `_impl` PAPER.md if a harder split becomes authoritative.
- **A7.** Researcher-hat single-iteration budget (30 min wall-clock / 40 tool calls) is insufficient for 6-10h pipeline. PROVISIONAL is the honest status; silently substituting a cheaper objective (e.g. dropping SIGReg, dropping cross-depth prediction) would be an antipattern-'t' violation.

## §9 QED

Given:
1. The failure mode of JEPA on recurrent depth (cross-depth collapse) is geometrically ruled out by SIGReg Epps-Pulley isotropy per LeJEPA Thm 1, applied across d ∈ {1..6} (K#1771).
2. The fixed-point stability is preserved by inheriting parent F#674's structural guarantee plus the observation that adding an auxiliary loss does not introduce new contractive modes (K#1770).
3. The proxy-target pairing (F#666) is satisfied: K#1771 ↔ K#1774 (isotropy ↔ depth elasticity), K#1772 ↔ K#1773 (learning dynamics ↔ GSM8K accuracy).
4. The experiment makes **falsifiable** predictions P1–P5 with numeric kill criteria pre-registered.

The central claim — RDT loop adapter trained with JEPA + SIGReg closes parent F#674's behavioral gap — is captured in MATH.md §4.2 theorem and §5 prediction table.

**Empirical verification is scope-deferred to `exp_rdt_jepa_loop_adapter_impl` at P3** per reviewer.md §5 PROVISIONAL (novel-mechanism design-only sub-case) and `mem-antipattern-novel-mechanism-single-iteration-scope`. This iteration files design-only: MATH.md, graceful-failure scaffold, PAPER.md prediction-vs-measurement table with all rows "not measured", `_impl` follow-up at P3, LEARNINGS.md. KCs K#1770–K#1774 remain `not_measured`.

No scope swap. No silent fallback. No KC relaxation. ∎
