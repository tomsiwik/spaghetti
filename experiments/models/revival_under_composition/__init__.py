"""Revival Dynamics Under Composition (exp_revival_under_composition).

Measures whether composition suppresses or amplifies capsule revival
compared to single-domain fine-tuning, directly relevant to the
"prune after training" recommendation from Exp 18.

Kill criterion: composition changes revival rate by <5pp vs single-domain.
"""

from .revival_under_composition import RevivalUnderCompositionGPT  # noqa: F401
