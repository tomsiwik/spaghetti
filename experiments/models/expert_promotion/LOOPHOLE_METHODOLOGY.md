# Methodology Critique: Expert Promotion

## Verdict: INVALID (Theoretical Sleight of Hand & Unverified Axioms)

### 1. The Davis-Kahan Illusion & Vacuous Calibration
**The Claim:** Theorem 1 applies the Davis-Kahan $\sin(\theta)$ theorem to prove that a small perturbation ($\|E\|_{op} \le \alpha \|B\|_{op}$) results in bounded rotation of knowledge subspaces ($\sin(\theta) \le \|E\|_{op} / \Delta$).
**The Flaw:** The mathematical bound is entirely dependent on the spectral gap $\Delta$. The methodology conveniently assumes $\Delta \sim 2$ without any empirical measurement on the Qwen3-4B-4bit model. Worse, the "empirical calibration" of the safe operating point ($\alpha \le 5$) is based on a statistically vacuous 50-question MMLU evaluation. A mathematical proof calibrated on noise is not a proof; it is numerical theater. The threshold between "safe" and "catastrophic" rotation is derived from pure variance.

### 2. The Missing Requantization Error Term
**The Claim:** The experiment claims to theoretically model the perturbation of "promotion into a pre-trained base" via $W' = W + E$.
**The Flaw:** The base model is quantized (4-bit). The true mathematical operation of promotion into a quantized base is $W' = Quantize(Dequantize(W) + E)$. The methodology completely omits the requantization error term $E_{quant}$ from the perturbation model. The implementation dodged this by using a fake floating-point LoRA overlay. The mathematical framework proves properties for a continuous weight update that never actually occurred, willfully ignoring the catastrophic degradation that a true discrete quantized update would cause.

### 3. Untested Hessian Lipschitz Continuity
**The Claim:** Theorem 2 asserts that new adapters train well because the Hessian is Lipschitz continuous, bounding the change in the loss landscape: $\|H(W') - H(W)\|_{op} \le L_H \|E\|_F$.
**The Flaw:** This is a notoriously fragile assumption in deep neural networks, especially around sharp minima or in heavily quantized spaces. More importantly, the empirical validation of Theorem 2 (predictions P4 and P5) is fundamentally broken by the unfreeze capacity confound. The new adapters did not converge smoothly because the landscape was preserved; they converged because the "frozen" base adapter was secretly unfrozen, doubling the trainable capacity. The methodology relies on a theoretical property that was never empirically validated in this experiment.

### 4. Subspace Incompatibility
**The Flaw:** The methodology assumes that the perturbation $E = \alpha B^T A^T$ uniformly rotates the space. However, $A$ is initialized from a Grassmannian skeleton. While $\|A\|_{op} = 1$, the subspace spanned by $A$ might heavily intersect with the most critical knowledge singular vectors of $W$. The bound assumes a generic perturbation, but LoRA updates are notoriously low-rank and highly aligned with specific feature directions. The theoretical worst-case rotation could be localized precisely where it causes the most damage, meaning the global $\|E\|_F / \|W\|_F \ll 1$ argument is insufficient to guarantee non-interference with specific capabilities.