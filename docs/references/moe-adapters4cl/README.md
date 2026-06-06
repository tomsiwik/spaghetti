# MoE-Adapters4CL: MoE Adapter-Based Continual Learning

**Source:** https://arxiv.org/abs/2404.09855 (2024)

**Key Insight:** Uses MoE adapters for continual learning without catastrophic
forgetting. ~87% final accuracy / ~2% forgetting on Split CIFAR-100. Combines
adapter isolation with MoE routing for task selection.

**Relevance to our work:**
- Most architecturally similar to our approach: MoE adapters on a frozen base
- Their routing mechanism for adapter selection maps to our router calibration
- Key question: does their approach handle independently-trained adapters
  (our contribution protocol) or only jointly-trained ones?
- Relevant to `exp5_macro_match` and `exp11_training_time_compat`

**What to use:**
- Their MoE adapter architecture
- Their forgetting measurement protocol
- Comparison against other CL methods (comprehensive benchmark)
