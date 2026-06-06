# Follow-up Design: Frechet Merge vs. Truncated SVD

## Verdict
**Invalid.** The original experiment relies on tautological metrics (subspace preservation exactly matched to the optimization objective) and violates the Eckart-Young-Mirsky theorem by discarding B-matrix magnitude information, resulting in catastrophic downstream performance compared to naive addition.

## Hypothesis for Follow-up
A baseline using Truncated SVD on the naive sum of the adapters ($\Delta_{naive}$) will strictly outperform the chordal Frechet mean on all downstream reconstruction and performance tasks (MSE, task accuracy) when compressed to the same rank $r$, demonstrating that preserving magnitude-weighted (A and B) information is superior to purely geometric (A-subspace) preservation.

## Math Sketch
Given $N$ LoRA adapters $\Delta_i = B_i A_i$ for $i=1 \dots N$.
1. Compute the exact naive sum: $\Delta_{naive} = \sum_{i=1}^N B_i A_i$
2. Compute the Truncated SVD of $\Delta_{naive}$ to rank $r$: $U_r \Sigma_r V_r^T \approx \Delta_{naive}$
3. Define $\Delta_{svd} = U_r \Sigma_r V_r^T$
4. Compare $\Delta_{svd}$ against $\Delta_{chordal}$ (from chordal Frechet mean of $A_i$) on:
   - Frobenius norm error: $||\Delta_{naive} - \Delta||_F$
   - Downstream task metric (e.g., perplexity, accuracy on a held-out set).

By the Eckart-Young-Mirsky theorem, $||\Delta_{naive} - \Delta_{svd}||_F \le ||\Delta_{naive} - \Delta_{chordal}||_F$ is guaranteed.

## Kill Criteria
The follow-up experiment is killed if Truncated SVD fails to strictly outperform the chordal Frechet mean on downstream reconstruction MSE or task accuracy for rank $r$, which would indicate an error in the Eckart-Young-Mirsky premise or code implementation.