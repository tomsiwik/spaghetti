# LEARNINGS — KAN Compositional Orthogonality

## Key Learning

**Dependency kills propagate cleanly.** When a parent experiment proves a representation
is fundamentally broken, all child experiments testing properties of that representation
are automatically invalidated.

## Applicable to Future Work

- The Stiefel-KAN hybrid experiment (`exp_pierre_stiefel_kan_hybrid`) should also be
  killed — it adds a Stiefel constraint to a KAN representation that doesn't work.
- Any future "alternative basis" experiments must first pass a single-adapter expressivity
  gate (within 5pp of standard PoLAR) before testing composition properties.
