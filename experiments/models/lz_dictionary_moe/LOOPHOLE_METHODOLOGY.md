# Loophole Methodology: LZ Dictionary MoE

## 1. Invalid Linearization of Non-Linear Capacity
`MATH.md` attempts to define an "Effective Rank per Expert" using the equation:
`W1_eff_i = sum_j alpha_{i,j} * [W^up_j; 0] @ [W^down_j; 0] + [Delta^up_i; 0] @ [Delta^down_i; 0]`
This mathematical derivation is fundamentally flawed. It implicitly commutes the summation with the ReLU non-linearity. The actual computation is `sum_j alpha_{i,j} * W^up_j * ReLU(W^down_j * x)`, which cannot be factored into a single effective weight matrix `W1_eff_i` operating on `x`. By improperly treating a non-linear composition as linear, the capacity analysis drastically misrepresents the theoretical behavior of the network and invalidates the parameter efficiency claims.

## 2. Theoretical Nullification of Routing (MoE Bypass)
The mathematical foundation of the MoE dictionary relies on experts learning distinct convex combinations of dictionary entries. However, the theoretical formulation does not enforce sparsity or distinctness on `alpha`. With uniform initialization and no sparsity regularization, the theoretical behavior mathematically collapses to:
`expert_i(x) = S(x) + delta_i(x)`
where `S(x) = (1/D) \sum_{j=1}^D dict_j(x)` is a constant shared sub-network.
When routed, `\sum w_i expert_i(x) = S(x) + \sum w_i delta_i(x)`.
This proves that the dictionary completely bypasses the MoE router. The architecture is mathematically equivalent to a Dense model with a tiny MoE residual, breaking the core premise of Dictionary MoE.

## 3. Flawed Kill Criterion and Baselines
The theoretical kill criterion requires the Dict MoE to be compared against a Standard MoE "at same total params." However, the methodology fails to align the mathematical parameters, comparing a 236K Dict MoE against a 596K Standard MoE. Furthermore, the Standard MoE baseline theoretically overfits at this micro scale, performing worse than a 202K Dense GPT. A valid mathematical methodology would require comparing the 236K Dict MoE against a parameter-matched Dense baseline, which would reveal that the dictionary mechanism provides no theoretical advantage over a standard dense network.

## Verdict
**Invalid.** The methodology relies on invalid algebraic manipulations that ignore non-linearities, suffers from theoretical collapse where the MoE routing is bypassed, and employs flawed baselines that obscure the model's true lack of efficacy.