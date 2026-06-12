## Core finding

Decode-step alpha scheduling (position-decayed gating) cannot rescue an adapter that is net-negative on its own home domain: the math adapter at scale 6.0 degraded GSM8K from 0.72 to 0.64 (on_lift_always = −0.08, net −4/50 answer flips) with parser-failure symmetric, confirming the adapter itself — not the schedule — is the root problem.

## Why

A time-axis gate can only suppress or blend adapter signal; it cannot manufacture a positive scaffold that is absent. The decay schedule actually worsened off-domain damage (degradation_recovery 2.0: DECAY 0.72 vs OFF 0.88), meaning partial activation of a harmful adapter is still harmful.

## Implication for the next experiment

Do not reuse the F#627 math adapter at scale 6.0 as a positive-transfer source in any downstream experiment — it is a net-negative on GSM8K and will contaminate any composition that assumes it contributes skill. Any gating or routing idea must first verify the base adapter is individually positive on its home domain before testing the gating mechanism.
