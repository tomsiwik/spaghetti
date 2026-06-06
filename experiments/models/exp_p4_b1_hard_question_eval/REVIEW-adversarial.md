# REVIEW-adversarial.md — P4.B1: Gap-Targeted Evaluation

## V2 Audit Reconstruction Review (2026-04-18)

**V2 scope:** Audit tag `smoke-to-full` intended full N=15 rerun; rerun blocked by
deleted P1 T2 adapter weights (all 5 domains). V2 closes on structural grounds.

**Reviewer concerns against V2 closure:**

1. *"Smoke kill should not block a full rerun."* Reconciled via closure C1/C2:
   K1227 is N-independent (base 0.43-0.63, threshold 0.25 — cannot be moved by
   sampling 15 questions instead of 5). K1228 is training-data-limited (V_train
   disjoint from V_hard on advanced subdomain content), not rank- or N-limited.
2. *"K1229 PASS at N=5 — maybe gap hypothesis survives."* Marginal pass (r=-0.3026)
   does not rescue `all_pass=false` from K1227+K1228 FAIL. Even steel-manned to
   reliable PASS at N=15, verdict stays KILLED.
3. *"Reconstruction substitutes narrative for measurement."* Numbers (`base_scores`,
   `adapted_scores`, per-question arrays) are the original smoke measurements from
   2026-04-11 PAPER.md, preserved verbatim; V2 adds only derivation transparency
   and `_reconstruction_note` provenance.

**V2 verdict: PROCEED (KILLED, closure holds).** Finding #478 unchanged.

---

## Original Verdict: PROCEED (killed, structural analysis complete)

---

## Mathematical Critique

### Concern 1: N=5 is very noisy — is this a real kill?
**Response**: K1227 is structural, not statistical. Single base scores of 0.83 and 1.0 are observed on questions like "What is Zorn's lemma?" — Gemma 4 4B clearly knows this content. Even if we ran N=15, the base would remain > 0.25. The kill is not due to noise.

### Concern 2: Notation artifact claim — is this proven or speculative?
**Response**: Evidence is circumstantial but strong:
- P4.B0 math base = 0.307 with keywords: "a^2", "u dv", "f(g(x))", "0/0"
- P4.B1 math base = 0.633 with keywords: "Zorn", "maximal element", "eigenvalue"
- Same domain, same model, 2× different base scores depending on keyword type
- P4.B0 math adapter was trained on Q&A data that uses these notation patterns
- The notation-artifact hypothesis is the most parsimonious explanation

### Concern 3: What if a larger N run would show more improvement?
**Response**: The adapter improvement mechanism is clear: adapters shift token probabilities toward training vocabulary. If base already produces domain vocabulary (base=0.63), there's little room for the adapter to act. K1228 requires ≥15pp improvement; even with N=15 the mean would not reach +15pp for domains where base is already 0.60+.

---

## Experimental Critique

### Concern 1: Code adapter -13pp suggests domain mismatch, not structural impossibility
**Response**: This IS the structural impossibility — the adapter training distribution (basic algorithms: sorting, BST, hash tables) is mismatched with the test distribution (Byzantine faults, MESI protocol, Raft). This confirms condition (2) of the impossibility theorem: V_train ∩ V_hard = ∅.

### Concern 2: Should we test rank-16 adapters before declaring impossibility?
**Response**: Rank-16 only increases capacity, not training data coverage. The impossibility is not rank-limited but training-data-limited. However, if future work tests adapters TRAINED on advanced content, they might help.

---

## Forward Direction

The structural analysis reveals P4.C design principles:
1. Find tasks where Gemma 4 has FORMATTING gaps (not knowledge gaps)
2. LaTeX notation output, clinical note format (SOAP), legal document structure
3. Code in specific frameworks/styles (not general coding knowledge)

This is the only viable direction for adapter improvement on capable base models.

---

## Final Verdict: PROCEED (killed, structural analysis complete)

The experiment is correctly killed. MATH.md, PAPER.md, and Finding #478 are accurate.
The notation-artifact discovery is the key finding and should inform P4.C design.
