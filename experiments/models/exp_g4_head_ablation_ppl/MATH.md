# MATH.md — Behavioral PPL ablation of top-20% vs bottom-20% q_proj head slabs

Follow-up to F#742 / `exp_g4_attention_head_importance_ranking`. F#742 found
weak **structural** head-mass concentration (C_20 ≈ 0.335, < 0.50 threshold)
but real **functional** cross-domain specialization (J̄ = 0.349, < 0.60). It
never measured behavior. This experiment closes the proxy→target gap by
measuring whether the F#742 head-mass ranking predicts a **behavioral** effect:
does zeroing the top-20% heads degrade domain perplexity (PPL) substantially
more than zeroing the bottom-20%?

## 1. Failure mode (disease)

If the per-head Frobenius mass `μ_{l,h}` ranking from F#742 is a weight-space
artefact (random rank-6 B-matrices placing energy on arbitrary fan-out
directions), then removing the "important" heads will hurt the model no more
than removing the "unimportant" ones. The structural ranking would then be
behaviorally meaningless and no head-sparse serving path is viable. The
symptom to avoid: declaring "heads matter" from a weight-norm proxy alone.

## 2. Prior math (grounding)

- **Michel et al. 2019, arxiv:1905.10650** ("Are Sixteen Heads Really Better
  than One?") — established that attention-head importance in transformers is
  non-uniform and measurable by **ablation**: zeroing high-importance heads
  degrades loss far more than zeroing low-importance heads. Our K#1967 ratio
  test is the direct adapter-space analogue of their importance-ablation
  experiment.
- **F#742 (parent, this codebase)** — supplies the per-domain per-head mass
  matrix `μ_{l,h}` (42×8 per domain), the head-slab decomposition, the base
  model (`mlx-community/gemma-4-e4b-it-4bit`), and the adapter source
  (`exp_p1_t2_single_domain_training`, rank 6, scale 6.0, target
  `self_attn.q_proj`). We **reuse** that ranking verbatim — no recomputation.
- **F#627** — recommends `v_proj+o_proj` targets; these adapters are `q_proj`
  (pre-F#627). Conclusion transfers only to `q_proj`; a v_proj follow-up is the
  success unlock.

## 3. Head-slab mass definition (reused verbatim from F#742)

For layer `l`, the q_proj LoRA delta is
```
ΔW_q^{(l)} = scale · A^{(l)} B^{(l)},   A ∈ R^{D×r}, B ∈ R^{r×H_l},
D = hidden = 2560,  r = 6,  scale = 6.0,
H_l = num_heads · head_dim_l = 8 · 256 = 2048 (sliding) or 8 · 512 = 4096 (full).
```
Full-attention layers: indices {5,11,17,23,29,35,41} (head_dim=512); the other
35 layers are sliding (head_dim=256). Reshape the fan-out axis to
`(D, num_heads=8, head_dim_l)` and define the per-head mass
```
μ_{l,h} := ‖ΔW_q^{(l)}[:, h, :]‖_F²    (head-major, MLX y = x Wᵀ layout).
```
This is **identical** to F#742's `per_head_mass`. The 336 = 42·8 slabs are
ranked by μ. Top-20% = the ⌈0.20·336⌉ = **67** highest-μ slabs; bottom-20% =
the 67 lowest-μ slabs (same count). Per domain we use that domain's μ matrix
from F#742's `results.json["per_layer_head_mass"][domain]`.

## 4. Ablation operator (zero exactly what was ranked)

"Zero head slab `(l,h)`" = set the adapter's contribution along that head's
fan-out columns to zero. The forward delta is
```
Δy = scale · (x A^{(l)}) B^{(l)}.
```
Column block `[h·head_dim_l : (h+1)·head_dim_l]` of `B^{(l)}` is exactly the
slab whose mass is μ_{l,h}. Zeroing those columns of `B^{(l)}` zeroes that
head's entire ΔW_q contribution and nothing else — so we ablate precisely the
quantity that was ranked. The base (quantized) q_proj weights are untouched;
only the adapter delta is masked. Composition is the correct additive form
`base(x) + Σ scale·(xAᵢ)Bᵢ` (single adapter here, so no cross-product hazard);
LORA_SCALE = 6.0 ≤ 8. Implemented as an `nn.Module` subclass installed by
`setattr` on the parent attention block (never `__call__` monkey-patch — F#831).

## 5. Perplexity (target metric)

For a held-out domain corpus of chat samples, with assistant tokens only
contributing to loss (mask_prompt=True, matching training), PPL is
```
PPL = exp( (Σ_t NLL_t) / (Σ_t 1) )   over scored (assistant) tokens.
```
Held-out source: `exp_p1_t2_single_domain_training/data/<domain>/valid.jsonl`
(200 chat samples per domain; the model's own held-out validation split, never
seen in training-loss). We fix **50 samples** per domain (first 50, seed-free,
deterministic), identical across all four arms, max_seq_length=512.

Four PPL arms per domain ∈ {code, math, medical}:
- (a) `base` — no adapter.
- (b) `intact` — base + full q_proj adapter.
- (c) `top20_zeroed` — base + adapter with the 67 top-μ head slabs masked.
- (d) `bot20_zeroed` — base + adapter with the 67 bottom-μ head slabs masked.

Degradation `Δ = PPL_ablated − PPL_intact` (b is the reference; ablation can
only remove adapter signal). Ratio `R = Δ_top / Δ_bot`.

## 6. Kill criteria (pre-registered, target-gated per F#666)

Both KCs are on the **target metric PPL** (behavioral), so both are target-KCs.

- **K#1967 (TARGET — head-importance behavioral test).**
  Metric: degradation ratio `R̄ = mean over domains of Δ_top / Δ_bot`
  (per-domain ratio averaged; domains with Δ_bot ≤ 0 use the dominance check
  below). FIRE (= KILL "heads matter") iff `R̄ ≤ 2.0`. PASS iff `R̄ > 2.0`.
- **K#1968 (TARGET — adapter-meaningfulness sanity).**
  Metric: relative PPL improvement of intact adapter over base,
  `g = mean over domains of (PPL_base − PPL_intact) / PPL_base`.
  FIRE (= adapter inert, undermines test) iff `g < 0.05` (5%).
  PASS iff `g ≥ 0.05`.

Decision table (F#666 target-gated; both are target metrics):

| K#1968 (adapter meaningful) | K#1967 (top≫bottom) | Verdict |
|---|---|---|
| PASS (g≥5%) | PASS (R̄>2) | **SUPPORTED** — F#742 head ranking is behaviorally real; head-sparse serving viable (Success #114 if also ≥2/3 domains show R>2). |
| PASS (g≥5%) | FIRE (R̄≤2) | **KILLED** — adapter is behaviorally meaningful but head-importance ranking does NOT predict behavior; "heads matter" hypothesis killed (K#1967 reason). |
| FIRE (g<5%) | any | **KILLED** — adapter inert; the whole ablation test is undermined (K#1968 reason). The head ranking question cannot be answered with this adapter. |

`all_pass = (K#1967 PASS) AND (K#1968 PASS)`.
`verdict = "SUPPORTED" if all_pass else "KILLED"`.

## 7. Predictions

- **P1 (from F#742 J̄=0.349).** Functional specialization exists, so the
  top-μ heads should carry disproportionate behavioral signal: expect
  `R̄ > 2` and `R > 2` on ≥2/3 domains IF the ranking is behavioral.
- **P2 (from F#742 C_20=0.335, weak structural concentration).** Competing
  prediction: weak concentration could mean ablating any 20% removes similar
  energy, giving `R̄ ≈ 1` → KILL. F#742 left this genuinely open; this is the
  whole point of the experiment.
- **P3.** Adapter is a real trained rank-6 LoRA → expect `g ≥ 0.05` (K#1968
  PASS); if not, the adapter never learned domain behavior at q_proj.

## 8. Assumptions / limitations

- Base `mlx-community/gemma-4-e4b-it-4bit`; mlx 0.31.1, mlx-lm 0.31.2.
- q_proj rank-6 LoRA, scale 6.0; conclusion does not transfer to v_proj/o_proj
  (F#627) without re-running.
- 50 held-out chat samples/domain, assistant-token PPL, max_seq 512. Small but
  fixed and identical across arms — the ratio R is robust to corpus size
  because all arms share the corpus.
- Thresholds 2.0 (K#1967) and 0.05 (K#1968) are fixed before the run and MUST
  NOT be relaxed after seeing data.
