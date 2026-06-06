"""exp_followup_bce_with_logits_routing — preemptive KILL stub.

This experiment was preemptively killed. See MATH.md Theorem + three-lemma
proof. No code is executed: the BCE-with-logits fix + balanced-class
retraining are structurally invariant to both kill criteria at N=24
(K585 zero-headroom invariance; K584 false-positive cascade in
decentralized-without-calibration argmax).
"""

import sys

if __name__ == "__main__":
    print("KILLED — preemptive. See MATH.md.")
    sys.exit(0)
