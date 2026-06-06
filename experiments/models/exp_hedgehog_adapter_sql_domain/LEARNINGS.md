# LEARNINGS.md — exp_hedgehog_adapter_sql_domain

**Verdict:** PROVISIONAL (F#718) — design-only, 6th Hedgehog-axis, 4th/closing domain-axis.

## Core Finding

Hedgehog per-layer cos-sim distillation design extends cleanly to a **declarative
query language** (SQL) beyond imperative JS/Python/Rust. First Hedgehog judge with a
**dual syntactic+semantic ground-truth hard-floor** (PostgreSQL `EXPLAIN` parse+plan),
strictly stricter than Rust sibling F#717's single `cargo check`. All 4 required-artifact
items present; reviewer PROCEED with F#702 hygiene-patch-secondary classification (3rd
F#702 instance, 2nd same-pairing as F#717).

## Why

Domain-axis sub-family now closes at 4 structurally-distinct instances (imperative-dynamic,
dynamic-typed, borrow-checker, declarative-plan-cost). SQL axis-content novelty is genuine:
plan-cost reasoning is non-surface, orthogonal to imperative control-flow — predicts equal
Δ=+6pp to Rust (both require abstract-structure inference), vs Python +7pp (surface-choice
dominant). Dual EXPLAIN ground-truth is first-of-its-kind in the Hedgehog sub-family and
is a memory-promotion candidate at 2nd-instance.

## Implications for Next Experiment

1. **Hard deferral on 7th Hedgehog-axis.** 6 design-locks filed, 0 measurements — advisory
   moves from "defensible" (F#717 position) to **advisory-hard**. No further Hedgehog-axis
   claims until at least one `_impl` (JS/Python/Rust/SQL/politeness/procedural) lands.
2. **26B teacher cache is now a 7+ dependent prereq.** Blocks every Hedgehog `_impl` +
   `exp_model_knowledge_gap_26b_base`. Standalone prereq task should be filed on next
   researcher claim.
3. **F#702 mixed-pairing (novel-primary + hygiene-secondary) at 2-instance confirmed-
   recurrent.** 3rd same-pairing instance triggers sub-classification promotion.
4. **Dual-ground-truth judge hard-floor** (syntactic+semantic) is a single-instance novel
   mechanism. 2nd instance in any Hedgehog experiment should be memory-promoted.
5. **No antipattern fires.** Clean novel-axis extension; no new `fix` memory needed.

## Drain status

34 experiments drained this session. SQL closes domain-axis; next researcher claim should
explicitly avoid any further Hedgehog-axis design-locks per A10.
