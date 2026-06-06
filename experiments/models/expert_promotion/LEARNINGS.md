# LEARNINGS.md: exp_expert_promotion

## Core Learning

**Single expert promotion at scale=5 into a pre-trained base works perfectly for
quality preservation (0pp MMLU, 13.4% medical PPL improvement) but the adapter
stacking implementation has a parameter unfreezing confound that needs fixing.**

## Positive Results

1. **MMLU preservation is exact:** 92% -> 92%, same 46 questions correct with
   identical per-subject breakdown. Davis-Kahan bound is conservative; actual
   rotation is negligible at scale=5.

2. **Promoted domain improves:** Medical PPL 6.058 -> 5.249 (13.4% improvement).
   The adapter knowledge is successfully "baked in."

3. **Cross-domain transfer:** Math PPL improved 6.2% from medical promotion.
   Medical and math share reasoning structure; promotion at scale=5 provides
   beneficial transfer without MMLU cost.

4. **New adapters train normally:** Both code and math adapters converge on the
   promoted base. Loss ratios 1.101x (code) and 1.076x (math) vs original base.
   Training speed 1.08x (8% overhead from extra LoRA layer).

5. **Memory footprint:** Promoted base with frozen LoRA overlay uses only 2.40 GB
   (vs 2.26 GB base) -- 0.14 GB overhead for permanent domain expertise.

## The Unfreeze Confound

**Problem:** `model.unfreeze(keys=["lora_b"])` unfreezes ALL lora_b parameters,
including the promoted (should-be-frozen) adapter's B-matrices.

**Impact:** 35M trainable params vs 17M. The promoted adapter's medical B-matrices
receive gradients from code/math training data, partially undoing the promotion.

**Fix options:**
1. Named parameter groups: unfreeze only `linear.lora_b` (new adapter) not
   `lora_b` (promoted adapter, which is at module root)
2. Custom freeze/unfreeze that tracks promoted vs new adapters
3. Use different key paths: e.g., `new_lora_b` for the trainable layer

**Assessment:** The confound makes K840 results WORSE than they would be with
proper freezing. The real performance gap would be smaller.

## Why Pre-trained Base Succeeds Where Random Init Failed

Finding #331 (self_growing_toy) was killed with 19.8% of joint training.
This experiment retains >90%. The structural explanation:

| Factor | Random init (#331) | Pre-trained (this) |
|--------|-------------------|-------------------|
| rank/d | 6.25% (4/64) | 0.625% (16/2560) |
| Promotions | 5 sequential | 1 single |
| Base knowledge | None | Trillions of tokens |
| Perturbation effect | Builds from scratch | Refines existing |

The key insight: on a pre-trained base, promotion adds a small refinement to
an already-rich representation space. The spectral gap is large, the perturbation
is small (5% per module), and one promotion cannot overwhelm the base's knowledge.

On random init, each promotion IS the base's knowledge. Five sequential promotions
compete for the same limited capacity (31% of weight space at rank/d=6.25%).

## Implications for Product Architecture

1. **Adapter flywheel is viable** (for single promotion): train -> promote -> train
   next. Quality preserved, MMLU intact.

2. **Scale=5 is the safe promotion scale.** At scale=20, MMLU catastrophe (-42pp
   for N=5 composition, Finding #330). At scale=5, 0pp.

3. **Promotion as "bake-in" for always-on adapters:** Instead of runtime LoRA
   for always-on adapters (instruction following, etc.), promote them into the base
   at scale=5. This eliminates the LoRA overhead for those adapters.

4. **Sequential promotion untested:** This experiment only proves 1 promotion.
   The adapter flywheel requires sequential promotions. #331 suggests caution,
   but the structural advantages (pre-trained base, low rank/d) may extend to 2-3
   promotions.

## Technical Notes

- QuantizedLinear prevents true weight modification. Promotion uses frozen LoRA
  overlay, which is mathematically equivalent but adds inference overhead.
- For non-quantized models, true weight modification (W' = W + delta) would be
  simpler and have zero inference overhead.
- The 252 modules = 36 layers x 7 target keys (q, k, v, o, gate, up, down).
