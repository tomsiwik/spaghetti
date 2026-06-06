# MATH.md — Gemma 4 attention head importance for adapter routing

## 1. Failure mode (disease)

Adapters could touch all heads uniformly (structural spread) instead of
concentrating on a task-relevant subset. If so, head-level sparsification
(pruning / gated routing per head) cannot accelerate inference without
behavior loss. We want a mathematical check for whether head importance is
concentrated AND functionally different across tasks — not just numerically
different due to random LoRA initialization.

## 2. Prior math (grounding)

- **Michel et al. 2019, arxiv:1905.10650** — "Are Sixteen Heads Really Better
  than One?" — showed large fractions of attention heads in a transformer can
  be pruned at inference with minimal loss, establishing that head importance
  IS non-uniform in standard transformers.
- **Voita et al. 2019, arxiv:1905.09418** — different heads encode different
  syntactic / positional roles; importance is functional, not random.
- **F#3 (LoRA orthogonality is structural)** — different-task LoRA deltas in
  this codebase are near-orthogonal in the per-weight sense; we are extending
  that claim to a per-head decomposition of ΔW_q.
- **F#627 (Gemma 4 E4B proven adapter targets)** — `v_proj + o_proj` is the
  recommended target. The adapters available here target `self_attn.q_proj`
  (exp_p1_t2_single_domain_training, pre-F#627). `q_proj` is still
  architecturally sound for head-importance analysis because its output
  factors cleanly into `(num_heads × head_dim)`; the concentration structure
  probed by this experiment is independent of whether the chosen target is
  optimal for downstream task accuracy.

## 3. Theorem (head-wise decomposition of q_proj LoRA delta)

Let the q_proj weight at layer l be `W_q^{(l)} ∈ R^{H × D}` where
`H = num_heads × head_dim = 8 × 256 = 2048` and `D = hidden_size = 2560`
(standard MLX layout: `y = x @ W_q^T`; the 2048 axis is the fan-out).

The LoRA delta at layer l is
```
ΔW_q^{(l)} = scale · A^{(l)} B^{(l)}   with   A ∈ R^{D × r}, B ∈ R^{r × H}
```
(rank `r = 6`, `scale = 6` per adapter_config.json).

Reshape fan-out axis to heads: view ΔW_q^{(l)} as a 3-tensor of shape
`(D, num_heads, head_dim) = (2560, 8, 256)`. Define the per-head mass

```
μ_{l,h} := ‖ΔW_q^{(l)}[:, h, :]‖_F².
```

These `μ_{l,h}` are non-negative and partition the total adapter energy:
```
Σ_{l,h} μ_{l,h} = ‖ΔW_q‖_F² = Σ_l ‖ΔW_q^{(l)}‖_F².
```

**Lemma (rank-bounded support).** Because `ΔW_q^{(l)}` has rank ≤ r = 6, the
per-head contribution `μ_{l,h}` is upper-bounded by the top-r singular
masses projected onto that head's slice. In particular, at most `r` linearly
independent directions of the fan-out space receive non-zero coupling per
layer, so any concentration measured at the head granularity is a direct
consequence of how the training signal selected those r directions — not
an artefact of noise.

## 4. Failure-impossible construction

Consider the (proxy, target) pair:

- **Proxy KC (K1, structural).** Let `C_20(dom)` = fraction of total energy
  `Σ μ_{l,h}` concentrated in the top-20% heads by `μ` for domain `dom ∈
  {code, math, medical}`. Concentration exists iff `mean(C_20) > 0.50`
  (mean across 3 domains).
- **Target KC (K2, functional / cross-domain).** Let `T_dom` = index-set of
  top-20% heads per domain. Functional specialization holds iff the
  average pairwise Jaccard overlap
  `J̄ = mean_{a≠b} |T_a ∩ T_b| / |T_a ∪ T_b|  <  0.60`.

**Why this pairing resolves F#666.** A proxy-only concentration claim (K1
alone) is known to be weak: random rank-6 B-matrices can concentrate energy
on a few fan-out directions by chance. The functional test (K2) rules out
that structural-only outcome: if concentration is the same head set across
all 3 tasks, K2 fails and the concentration is likely an initialization or
architecture artefact (not exploitable for task-aware routing). If
different tasks concentrate on different heads, the concentration has
functional meaning and a routing / pruning strategy can exploit it.

Decision rule (target-gated per Finding #666):

| K1 | K2 | Verdict |
|----|----|---------|
| PASS | PASS | SUPPORTED — concentration exists AND is functional. Head-level routing is a viable optimization path. |
| FAIL | FAIL | KILLED — heads are uniformly and identically engaged; no head-level exploitation is possible with `q_proj` adapters at rank 6. |
| PASS | FAIL | PROVISIONAL (proxy-only): concentration is structural artefact, not functional. Not a supported claim; not a clean kill. |
| FAIL | PASS | PROVISIONAL (tautological proxy): heads differ across tasks but not concentrated enough to exploit. |

## 5. Predictions (behavioral consequences)

- **P1.** If `μ_{l,h}` shows bimodal concentration (few large, many small),
  `C_20` ≥ 0.55 on every domain individually.
- **P2.** If adapter training selects semantically meaningful head subsets,
  `code`'s top set and `medical`'s top set should overlap no more than
  chance (`0.2 × 0.2 / (0.4 − 0.04) ≈ 0.11` for independent 20%-sets). An
  upper bound of 0.60 admits substantial overlap — a conservative threshold
  so a PASS genuinely means different concentration patterns.
- **P3.** Because rank is fixed at r = 6 and num_heads = 8, the per-layer
  `μ_{l,h}` vector has effective dimension ≤ 6, meaning at least
  ⌈(8 − 6)/8⌉ · 42 = some heads per layer receive zero coupling. This is a
  structural prediction that will manifest as hard zeros in the per-head
  heatmap for some (l, h) pairs.

## 6. Kill criteria (pre-registered, target-gated per F#666)

- **K1 (proxy / structural).**
  Metric: `C_20_mean = mean over {code, math, medical} of top-20% head energy share`.
  PASS iff `C_20_mean > 0.50`. FAIL iff ≤ 0.50.
- **K2 (target / functional).**
  Metric: `J̄ = mean pairwise Jaccard overlap of top-20% head sets across 3 domains`.
  PASS iff `J̄ < 0.60`. FAIL iff ≥ 0.60.

Overall verdict:
- `all_pass = K1.PASS AND K2.PASS` → SUPPORTED.
- `all_fail = K1.FAIL AND K2.FAIL` → KILLED.
- Mixed → PROVISIONAL (see §4 table).

## 7. Assumptions / limitations

- Gemma 4 E4B 4bit base, rank-6 LoRA on `self_attn.q_proj`, adapters from
  `exp_p1_t2_single_domain_training` (pre-F#627). Different targets
  (`v_proj + o_proj` per F#627) are not covered by this experiment; the
  conclusion transfers only if future v_proj analyses reproduce the head
  structure.
- `q_proj` output folds heads on the fan-out axis. Per-head reshape is
  `(D, num_heads, head_dim)`. We assume the standard MLX layout where
  `y = x @ W^T` and the contiguous 2048 axis corresponds to
  `num_heads × head_dim` in head-major order, as loaded from the HF config
  (`num_attention_heads=8, head_dim=256`). If a future MLX layout swaps
  this, the reshape must be revalidated.
- Absolute head pruning (zero-out + eval) is deferred to a follow-up
  experiment. This experiment is weight-space profiling only; its target
  metric (K2) is a cross-domain variation test, which is a behavioral
  surrogate (different tasks → different concentrated heads), not a direct
  behavior measurement. Follow-up experiment `exp_g4_head_ablation_ppl`
  will provide the direct PPL-based validation.
- Pre-registration: the thresholds `0.50` (K1) and `0.60` (K2) are fixed
  before running the experiment and MUST NOT be relaxed after seeing data.
