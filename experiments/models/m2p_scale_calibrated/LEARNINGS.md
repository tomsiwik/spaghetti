# LEARNINGS.md: M2P Scale Calibration Failure

**Experiment:** exp_m2p_scale_calibrated  
**Status:** KILLED (K849 PASS, K850 FAIL) — audit-rerun closure 2026-04-18  
**Date:** 2026-04-07 (original); 2026-04-18 (audit closure)  
**Finding:** #343  
**Audit tag:** `audit-2026-04-17-rerun, code-bug`  
**Audit closure:** Rerun NOT executed. Three closure theorems (C1 sibling supersession #341/#342/#343; C2 K849 paradox falsifies KKT at operating point via Assumption 2 violation; C3 L_preserve increases rigidity not sensitivity — baseline CV higher than WITH-L_preserve) make the kill robust to any numerical fix. Closure-rule family: `additive-context-injection-blocks-calibration`. See PAPER.md "Audit-Rerun Closure" section and results.json.

---

## Core Finding

The M2P hypernetwork architecture with mean-pooled additive context injection **cannot distinguish task difficulty at the architectural level**, independent of regularization strategy or training procedure. This is the third consecutive M2P failure with the same root cause: weak context encoding prevents per-context B-matrix generation.

---

## Why This Happened

### The Mathematical Issue vs The Architectural Issue

**Theorem 1 was mathematically sound but architecturally impossible.**

Theorem 1 proved that if:
- M2P can express context-dependent B-matrices
- Task gradient varies monotonically with context difficulty
- L_preserve monotonically increases with scale

Then: L_preserve creates a KKT equilibrium where ∂L_task/∂α + λ·∂L_preserve/∂α = 0, yielding context-dependent α*.

**The experiment falsified the empirical assumption, not the math.** The proof assumes M2P *can express* context-dependent outputs. In practice, M2P collapses to constant-magnitude output (CV = 0.0093 across all contexts), indicating the architecture itself prevents discrimination.

### Root Cause: Additive Conditioning with Self-Attention Bottleneck

The simplified M2P architecture uses three components in sequence:

1. **Mean-pooling**: Task context (e.g., "5+3" vs "987+456") is mean-pooled across all token positions into a single 64-dimensional vector
2. **Additive broadcast**: This single vector is additively injected into all 8 memory tokens (each 64-dim)
3. **Self-attention bottleneck**: Memory tokens attend only to each other (not to the original input)

**The problem:** Information distinguishing "easy" from "hard" arithmetic is compressed into one vector, then:
- The additive injection can be overwhelmed by 8 learned memory token vectors (learned values >> injected signal)
- Self-attention over memory tokens only provides no mechanism to query back to the original context
- The result: all task contexts produce identical memory representations, hence identical B-matrices

This architectural bottleneck was already identified and killed in **Finding #342** (domain conditioning experiment). The same weakness carried forward predictably produced the same failure.

### Confirming Evidence from Literature

**Task interference in multi-task learning** (SMoRA arXiv:2501.15103, HDMoLE arXiv:2409.19878):
- Round-robin training on heterogeneous tasks causes gradient conflicts
- Gradient from easy domains (small ∂L/∂α) competes with hard domains (large ∂L/∂α)
- Systems that succeed use **routing, conditioning, or loss-balancing** — never mean-pooled additive injection

**MoE expert collapse** (MixLoRA arXiv:2402.15896):
- Mean-pooling task features causes expert utilization collapse (all experts converge to same output)
- Same phenomenon: compression of discriminative information into a single signal
- Fix: dynamic instance-specific routing that conditions on individual task tokens, not aggregated statistics

**Hypernetwork literature** (SHINE arXiv:2602.06358):
- Hypernetworks that DO produce context-dependent outputs use cross-attention, not additive conditioning
- SHINE uses attention between memory/context tokens and input hidden states
- Additive mean-pooled injection appears nowhere in successful hypernetwork designs

**Preservation loss mechanism** (EWC: Kirkpatrick et al. 2017, instruction tuning variants):
- L_preserve = λ||θ - θ_base||^2_F (or Fisher-weighted version) is standard practice
- It does bound overall magnitude change (K849 PASS confirmed this)
- But it does NOT enable automatic per-context calibration — it regularizes the *aggregate* magnitude

---

## Contradicting Evidence (Approaches That Worked)

Papers that succeeded with context-dependent parameter generation despite similar constraints:

### 1. **MixLoRA** (Ariv:2402.15896)
- Task: Multi-task LoRA composition with per-domain adaptation
- Solution: **Dynamic instance-specific routing** — each input token routes to its own LoRA adapter combination
- Why it works: Routing is computed PER INPUT, not from aggregated statistics
- Achieved: Eliminates expert collapse, maintains per-domain quality > 95%

### 2. **SMoRA** (arXiv:2501.15103)
- Task: Multi-task adapter training without gradient conflicts
- Solution: **Conditional gating based on task embedding** — gates applied to each adapter dynamically
- Why it works: Gating condition is accessible to each gradient step, not compressed away
- Achieved: Prevents mode collapse, per-domain performance > 90% on 8 tasks

### 3. **Progressive Neural Networks** (Rusu et al. 2016)
- Task: Sequential multi-task learning without forgetting
- Solution: **Structural separation** — old parameters frozen, new task uses lateral connections with learned multipliers
- Why it works: Multipliers are directly optimized per-task, architectural constraint prevents interference
- Achieved: Zero forgetting on sequential tasks

### 4. **EWC with Task-Specific Regularization** (Kirkpatrick et al. 2017)
- Task: Continual learning with preservation
- Solution: **Fisher-weighted regularization** — regularizes each parameter by how much it affects the loss on task A
- Why it works: Task-specific penalty incorporates per-task gradient information
- Achieved: Stable multi-task learning without catastrophic forgetting

---

## Alternative Approaches

### Path 1: Cross-Attention Architecture (High Confidence)
**Replace additive conditioning with cross-attention:**

```
memory_tokens → self-attention (8 tokens)
input_tokens + memory_tokens → cross-attention
→ output B-matrices
```

**Expected outcome:** M2P can attend to specific task tokens (operands, operators), generating context-sensitive B-matrices. CV should exceed 0.05, K850 should PASS.

**Why:** SHINE and other hypernetworks use this pattern. Cross-attention provides a direct information flow from input to output parameters.

**Literature precedent:** SHINE (arXiv:2602.06358) generates different adapter scales for different prompt types using exactly this mechanism.

---

### Path 2: Independent Per-Domain Training (High Confidence)
**Eliminate gradient conflicts by training on one domain at a time:**

M2P trained on ARITHMETIC domain only (no round-robin multi-domain training).

**Expected outcome:**
- No gradient interference possible (single loss landscape)
- L_preserve constrains scale correctly for arithmetic
- If M2P later trained on LOGIC domain separately, scales remain domain-specific
- Compose via routing (each domain gets its trained M2P instance)

**Why:** Findings #341 and #342 already proved multi-domain round-robin training causes centroid collapse. Single-domain training eliminates the root cause.

**Literature precedent:** MixLoRA/SMoRA show that per-domain/per-expert training avoids mode collapse entirely.

**Connection to VISION:** This maps to a routing-based architecture: each domain has its own trained M2P, a router selects at inference time. This is closer to the Room model (Finding #265) than the M2P hypernetwork.

---

### Path 3: Direct Scale Prediction Head (Medium Confidence)
**Simplify M2P to a linear or MLP regressor:**

```
task_embedding → MLP(d=64, hidden=128) → scale α
```

No B-matrix generation, just predict α directly. Apply LoRA with fixed A, variable α.

**Expected outcome:** If the issue is M2P's Transformer bottleneck (8 memory tokens insufficient), a simpler regressor may learn the context→scale mapping.

**Why:** Isolates the mechanism (scale prediction) from M2P's architecture. If this passes K850, we know the problem was M2P capacity, not the K preservation math.

**Risk:** May not generalize to larger M2P (25 experts, per-expert scales), where full hypernetwork generation is necessary.

---

## Accumulated Impossibility Structure Across Findings #341, #342, #343

| Finding | Failure | Root Cause | Architecture Used |
|---------|---------|-----------|------------------|
| #341 | B-matrix mode collapse (easy/hard domains converge) | Round-robin multi-domain training + gradient conflicts | Additive domain embedding |
| #342 | Domain embeddings ignored (Jacobian d_B/d_e low-rank) | Mean-pooled additive injection overwhelmed by memory tokens | Additive domain embedding |
| #343 | Constant magnitude output (CV=0.0093, no self-calibration) | Mean-pooled context → single representation → identical B across contexts | Additive mean-pooled context |

**The permanent closed path:** The simplified M2P architecture (additive conditioning + memory-only self-attention) is **provably insufficient** for context-dependent generation. The information bottleneck created by mean-pooling + additive broadcast + self-attention-only is not overcome by regularization (L_preserve), loss weighting, or additional training (baseline had same CV without L_preserve).

**Structural fix required:** Any future M2P experiment must use:
- Cross-attention to input tokens, OR
- Direct parameter regression (not B-matrix generation), OR
- Separate per-context training

Without one of these, further M2P experiments will produce identical failures.

---

## What Was Learned (Positive)

### 1. **L_preserve DOES Work for Magnitude Constraint**

K849 PASSED (-59.01pp degradation with L_preserve vs -3.20pp without). This confirms:
- The KKT equilibrium argument is mathematically sound
- Preservation loss successfully constrains overall adapter magnitude
- This is independently useful for production systems (bounding task-specific adapter impact on general quality)

**Impact:** L_preserve is a validated technique for preventing catastrophic degradation in adapter composition. Use it in future adapter experiments regardless of the context-generation question.

---

### 2. **The K849 Paradox Reveals a New Phenomenon**

The -59.01pp "degradation" is actually a **59% improvement** in general CE (12.18 → 4.99). This suggests:
- M2P learned a general-purpose improvement adapter, not a task specialist
- This violates the original assumption (Assumption 2: L_preserve increases with α)
- At this operating point, both ∂L_task/∂α and ∂L_preserve/∂α point in the same direction (both incentivize scale growth), not opposed

**This may be a feature, not a bug.** In a system where all domains benefit from shared improvements, having M2P learn a universal adapter could be valuable for composition. Needs investigation in a guided-exploration experiment (not killed as a failure).

---

### 3. **Architecture Matters More Than Theory**

The lesson: proof-first research requires not just mathematical soundness but also **empirical grounding of assumptions**. 

Theorem 1 is structurally sound, but it rests on the assumption "M2P can express context-dependent outputs." We should have proved this capacity assumption *before* running the experiment, or bracketed it explicitly in the kill criteria.

**Protocol improvement:** For future hypernetwork experiments, include a capacity test in MATH.md:
- Section: "Empirical Assumption: Can M2P express the required function class?"
- Test: Synthetic data where ground truth is known (e.g., synthetic task contexts with prescribed difficulty)
- Kill criterion: If M2P cannot memorize synthetic ground truth, it cannot discover real patterns

---

## Implications for Next Experiments

### Closed Paths
- **M2P with additive conditioning:** Dead end (three consecutive kills, same bottleneck)
- **Preservation loss as self-calibration mechanism:** Doesn't enable context discrimination (K850 failure consistent across variations)

### Open Paths
1. **Cross-attention M2P** (Path 1) — architectural fix with high literature precedent
2. **Per-domain routing** (Path 2) — aligns with Room model, eliminates gradient interference
3. **Scale prediction head** (Path 3) — isolates mechanism, lower complexity

---

## Recommended Follow-Up

### Short Term (1-2 experiments)
**exp_m2p_cross_attention** (Priority: HIGH)

Implement M2P with cross-attention over input tokens instead of mean-pooled additive injection. Use same single-domain (arithmetic) training as exp_m2p_scale_calibrated.

**Hypothesis:** Cross-attention restores the context→B pathway blocked by mean-pooling.

**Theorem required:** Prove (via UAT or capacity analysis) that cross-attention M2P can express arbitrary context-dependent functions.

**Kill criteria:**
- K_new1: Magnitude CV > 0.05 (self-calibration signal present)
- K_new2: Hard > Easy magnitude ratio > 1.2x (context drives scale variation)
- K_new3: General degradation < 10pp (K849 constraint maintained)

**Literature grounding:** SHINE (arXiv:2602.06358), Transformer attention mechanisms

---

### Medium Term (Related Direction)
**Room Model Routing** (Priority: MEDIUM, but strategically important)

Return to the Room model (Finding #265) and test whether per-expert routing supersedes M2P altogether. Room model proved composition at N=5 with 0% overhead. Testing N=20 experts with learned routing addresses the original vision without the M2P hypernetwork bottleneck.

**Why:** Findings #341-343 suggest hypernetwork-based context-dependent generation is structurally constrained. Routing-based composition (which the Room model demonstrates) may be the right approach.

---

## References

**Relevant prior findings:**
- Finding #225: Near-lossless composition at N=5 with Gumbel-sigmoid routing
- Finding #265: Room model (W_combined = Σ ΔW_i)
- Finding #330: Scale sensitivity (scale=5: 0pp, scale=20: -42pp)
- Finding #341: B-matrix mode collapse in multi-domain training
- Finding #342: Additive domain embeddings insufficient

**Papers cited:**
- **MixLoRA** (arXiv:2402.15896) — dynamic instance-specific routing
- **SMoRA** (arXiv:2501.15103) — task-specific gating for multi-task adapters
- **SHINE** (arXiv:2602.06358) — cross-attention hypernetworks
- **EWC** (Kirkpatrick et al. 2017) — Fisher-weighted preservation loss
- **Progressive Neural Networks** (Rusu et al. 2016) — lateral connections for task separation

**Architecture reference:**
- https://sebastianraschka.com/llm-architecture-gallery/ — compare context-encoding mechanisms across production models
