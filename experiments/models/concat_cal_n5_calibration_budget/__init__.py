"""Concat+Calibrate N=5 calibration budget sweep.

Parameter sweep of calibration steps {100, 200, 300, 500} at N=5 domains.
Builds directly on lora_merging_bakeoff -- no new model, just a sweep.
"""

# No model registration needed -- this is a parameter sweep experiment,
# not a new model. It reuses RoutedDeltaGPT from lora_procrustes.
