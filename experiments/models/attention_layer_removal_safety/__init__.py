"""Attention Layer Removal Safety: tests expert removal in the high-cosine
attention-layer regime (cos~0.85).

Parent experiment (expert_removal_graceful) proved naive subtraction works at
cos~0.001 (SOLE MLP layers). But attention layers for related domains have
cos=0.85, placing them firmly in the "recomputation required" regime.

This experiment verifies:
(1) Naive subtraction error at realistic attention cosines (cos=0.85)
(2) GS recompute cost for attention-only removal
(3) Partial removal: remove only attention component, keep MLP intact

Kill criteria:
  K1: naive subtraction error >3% for attention layers at cos=0.85
  K2: GS recompute for attention layers takes >10s at N=50
"""
