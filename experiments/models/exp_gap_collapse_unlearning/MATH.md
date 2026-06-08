# Conserved Gap Collapse

## Conserved Gap Collapse

**CLAIM:**  
Driving an expert-pair interference gap `g_AB(x) = f_A(x) + f_B(x) - f_AB(x)` toward zero on a targeted concept causes behavioral unlearning of that concept, and the same optimization produces measurable OOD degradation on held-out tasks that require distinguishing A-like from B-like behavior.

**MECHANISM / why it should hold:**  
If `g_AB` is the function-space residual that separates two adapter behaviors under composition, then collapsing it removes the model’s ability to express the difference between those behaviors in the composed path. This makes “successful merge” and “forgetting” the same operator only if the collapsed gap corresponds to target behavior, not merely activation geometry.

Be strict: the theorem cannot say “perfect merge destroys all OOD generalization” globally. It can only say: if the target behavior is encoded in the expert-separating component of `g_AB`, then forcing that component to zero removes behavioral separability along that axis.

**Pre-registered KILL criterion, target behavioral metric:**  
KILL if targeted gap-collapse reduces the proxy gap but does **not** produce both behavioral effects:

- Target unlearning: harmful/concept target score drops by at least `50%` relative to the original harmful-specialist adapter, measured on held-out target prompts.
- Coupled OOD degradation: held-out discriminative tasks requiring A/B separation degrade by at least `5pp` relative to the non-collapsed composed adapter.

If `g` collapses but target behavior remains, the gap proxy is not causal.  
If target behavior drops but OOD separation does not degrade, the “same collapse” thesis is too broad.

**Minimal frozen-Gemma-4 + LoRA experiment:**  

Use two existing or freshly trained LoRA adapters on frozen `mlx-community/gemma-4-e4b-it-4bit`:

- `A`: benign/domain adapter, e.g. math or medical.
- `B`: target concept adapter, e.g. harmful synthetic concept, unsafe instruction-following slice, or narrower factual concept if safety data is unavailable.

Create three systems:

1. Base + `A`
2. Base + `B`
3. Base + composed `A+B`

Then train only a tiny collapse transform or LoRA delta whose loss explicitly minimizes `||g_AB(x)||` on target-concept examples while freezing base and original adapters.

Evaluate behaviorally:

- Target concept retention/removal accuracy.
- A-domain task accuracy.
- B-domain task accuracy.
- Held-out A/B separation benchmark: prompts where correct behavior depends on choosing the right domain or refusing the wrong transfer.

No PPL-only verdict. No cosine-only verdict.

**Existing math/result it builds on:**  

- Repo finding spine: solo adapters have strong on-domain lift but destructive off-domain interference, especially F#627 and F#827.
- Target-gated kill rule F#666.
- LoRA/task arithmetic framing: composition as `Σ B_i @ A_i`, not `(ΣB)(ΣA)`.
- Function-space residual/interference gap as the candidate object.

**Falsifiability status:**  
Falsifiable if “forgetting” is defined as target behavioral score collapse and “OOD degradation” is defined on a specific held-out separation benchmark. The global impossibility claim is too strong unless narrowed.

---
