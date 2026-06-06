"""Multi-Token Prediction effect on expert composition (exp_mtp_composition).

Tests whether MTP-trained capsule groups compose better than standard NTP groups.
Kill criteria:
  1. MTP-trained capsule groups compose >5% worse than standard NTP groups
  2. MTP provides <2% quality improvement over standard training for composed models
"""

from .mtp_composition import MTPCapsuleMoEGPT
