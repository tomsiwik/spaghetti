# LEARNINGS — exp_spark_conflict_subspace_deflation

## Core finding
Deflating the top-k SVD directions of the summed per-layer delta D=BA_math+BA_med recovers only
+1.25pp aggregate over uniform-1/N (best k=2/k=4), far below the +3pp threshold, killing the
hypothesis that a low-rank shared subspace drives composition interference.

## Why
The mechanism is falsified on two independent grounds: (1) D's singular spectrum decays smoothly
(σ_i/σ1 = [1.0, 0.72, 0.53, 0.40, ...], σ1/σ12 ≈ 35), with no σ1 ≫ σ2 cliff — no dominant
clash subspace exists to deflate; (2) deflation trades domains (math 52→57/58, medical 43→40/39 at
every k) rather than removing a shared mode, revealing that the top directions are
domain-discriminating, not domain-neutral interference.

## Implication for the next experiment
Future composition-repair experiments require a setup where uniform-1/N clearly falls below the
no-adapter base (demonstrable aggregate interference); the null here was underpowered because
uniform-1/N aggregate (0.5938) tied the base, leaving nothing to recover. Global low-rank subspace
deflation is not the lever; the sparker should avoid any approach premised on a dominant shared
conflict subspace in the summed delta spectrum.
