# LEARNINGS — exp_wildcat_dilution_norm_vs_structure

**Status:** PROVISIONAL (re-reviewed, confirmed)

**Core finding.** The F#881 random-mask dilution effect replicates (mask f=0.4335 on the
dense thinking delta: B mean 0.84 vs dense A 0.60, 3-seed spread 5pp), and the mask is NOT
reducible to a scalar alpha: the norm-matched dense control C (α=√f, same ‖Δ‖_F=2.081)
scores 0.79, a 5.0pp gap sitting exactly on the pre-registered structural/inconclusive
boundary at n=100.

**Why.** Knife-edge gap (exactly 1/20; per-arm SE ~4pp, per-seed McNemar n.s., pooled
paired p~0.02) can't cleanly cross a 5pp threshold at n=100. The secondary D/E "outlier"
probes are norm-confounded (D ‖Δ‖_F=3.080, E=0.431 vs C=2.081); across all arms EM is
monotone in ‖Δ‖_F, so D/E/C2 provide zero independent structural evidence.

**Implication for the next experiment.** Two pre-registered follow-ups before any
supported claim: (a) B vs C at n≥300 (or paired significance) to resolve the 5pp gap;
(b) norm-matched D'/E' rescaled to ‖C‖_F=2.081 so the structure probe stops being a norm
proxy. The pure-norm kill did NOT fire, so the mask arc stays alive — but only via the
norm-matched B-vs-C contrast.

**PIERRE-IMPACT:** shelved — provisional (knife-edge 5.0pp at n=100, secondary arms
norm-confounded); wildcat outside the bet ladders, and only supported/conclusive bet
findings can ship. No pierre branch change until the n≥300 B-vs-C + norm-matched D'/E'
follow-up lands.
