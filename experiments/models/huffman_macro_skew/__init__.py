from ..huffman_tree.huffman_tree import HuffmanTreeGPT
from .. import register

# Re-register huffman_tree under a new name for this experiment's tracking
# The actual model is identical -- this experiment is about ANALYSIS of routing
# distributions, not a new architecture.

# We don't register a new model here because the experiment is analytical:
# it measures whether real-world expert utilization distributions produce
# enough skew for Huffman routing to help. The model code lives in huffman_tree/.
