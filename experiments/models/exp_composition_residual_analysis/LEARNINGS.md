# LEARNINGS.md — exp_composition_residual_analysis

**Verdict:** SUPPORTED. F#302 and F#334 quantitatively replicated at Gemma 4 E4B 4-bit (F#752).

## Core Finding

Sum-of-deltas LoRA composition (N=3, r=6, q_proj, F#627 recipe, distinct seeds 42/1337/2718) at Gemma 4 E4B 4-bit is **structurally and behaviorally non-additive**:
- **τ = 0.482 joint** (0.474–0.557 per-domain) — residual is ~48% of Σ‖δh_i‖, first numeric magnitude at this platform (K1926 PASS, 4.8× threshold).
- **Δ_max = 2.19 / 0.21 / 0.37** (medical floor-amplified; code+math clear 10% threshold by ≥2× alone) — composed PPL on domain-i is materially worse than adapter-i alone (K1927 PASS).
- **Residual is systematic (27× iid-noise floor)** — per-dim mean bias has structure, not noise.

## Why

F#302 proved per-module algebraic linearity but full-model nonlinearity at prior base (logit MSE 53.9). F#334 called pre-sum a "unrouted mixture". Neither quantified τ at F#627-canonical Gemma 4 E4B 4-bit. MATH.md §3 theorem: LayerNorm σ(x+Σδ_i) couples deltas via 1/σ rescaling, producing O(⟨x,δ_i⟩⟨x,δ_j⟩/σ³) cross-terms; 42 layers × {attn, SwiGLU} compound the coupling. τ bounded away from 0 by structure — this experiment measured the magnitude.

## Reusable Building Block

**r-stacking for sum-of-deltas without retraining:** single LoRALinear at r=ΣR_i, `lora_a=hstack([A_i])`, `lora_b=vstack([B_i])`. Algebraically ΔW_stacked = Σ ΔW_i; rel-diff 6.28e-08. Lets one model produce {base, each individual, composed} via slot-zeroing.

## Implications for Next Experiment

- **L1 — τ-vs-depth profile (highest-leverage follow-up):** measure τ at each of 42 layers to verify MATH.md P4 monotonic compounding. Tests the structural theorem directly; cheap (one extra Phase C pass, no retraining with r-stacking). → `exp_composition_residual_layerwise`.
- **L2 — 3-seed-triplet variance on τ:** replicate with 3 distinct seed triplets to tighten τ point estimate and separate cross-term-structure vs seed-idiosyncratic signal. Addresses within-experiment consistency gap noted in REVIEW §L2.
- **L3 deferred:** τ-vs-N (composition count), τ at v_proj+o_proj, routing-from-residual. Lower leverage until L1 confirms compounding; L1 result will select whether L3/routing is worth pursuing.

**Bottom line:** pre-sum composition is not a drop-in routed composer at this scale. Either (a) route per-token (Pierre P1 direction) or (b) find structure in R that restores additivity — L1 is the minimal next test before either path.
