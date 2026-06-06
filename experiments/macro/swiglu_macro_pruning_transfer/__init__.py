"""Gate-product pruning transfer to macro-scale SwiGLU models.

Profiles gate products on pretrained Qwen2.5-0.5B to test whether the bimodal
distribution observed in micro experiments (with aux sparsity loss) exists in
production models trained with standard cross-entropy only.
"""
