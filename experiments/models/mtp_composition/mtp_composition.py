"""MTP Capsule MoE -- Multi-Token Prediction training for capsule composition.

Architecture: CapsuleMoEGPT with D auxiliary MTP heads. Each MTP head k (k=1..D-1)
predicts token t+k+1 from the hidden state at position t. Following DeepSeek-V3:

  For depth k:
    h_k = RMSNorm(Linear_k(h_{k-1}) + emb(token_{t+k}))  -- sequential dependency
    logits_k = lm_head(h_k)  -- shared lm_head across all depths

The MTP loss is:
  L = L_ntp + lambda_mtp * (1/(D-1)) * sum_{k=1}^{D-1} L_k

where L_k is the cross-entropy for predicting token t+k+1 at each position.

Key design: MTP heads share the lm_head with the main model (DeepSeek-V3 style).
Each MTP module has: one Linear projection (d -> d) + one RMSNorm. The sequential
chaining means depth-k prediction depends on depths 1..k-1, forcing the model to
build coherent multi-step representations.

At inference: only the standard NTP output is used. MTP heads are discarded.
"""

import mlx.core as mx
import mlx.nn as nn

from .. import register
from ..gpt import RMSNorm, CausalSelfAttention
from ..capsule_moe.capsule_moe import CapsulePool, CapsuleBlock


class MTPModule(nn.Module):
    """Single MTP prediction module (one depth level).

    Takes hidden state from previous depth, combines with token embedding
    of the target position, projects, and normalizes.

    h_k = RMSNorm(proj(h_{k-1}) + emb(token_{t+k}))
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.norm = RMSNorm(n_embd)

    def __call__(self, h_prev, token_emb):
        """
        h_prev:    (B, T, d) -- hidden state from previous depth
        token_emb: (B, T, d) -- embedding of token at target position
        returns:   (B, T, d) -- hidden state for this depth
        """
        return self.norm(self.proj(h_prev) + token_emb)


@register("mtp_capsule_moe", parent="capsule_moe")
class MTPCapsuleMoEGPT(nn.Module):
    """CapsuleMoEGPT with Multi-Token Prediction training objective.

    Architecture:
    - Same as CapsuleMoEGPT: token/pos embeddings, capsule blocks, lm_head
    - Additional: D-1 MTP modules, each predicting one additional future token
    - MTP modules share the main model's lm_head (parameter-efficient)

    During training:
    - Forward pass produces main logits (NTP) + D-1 auxiliary logit sequences
    - MTP loss added as auxiliary objective
    - MTP forces the model to learn multi-step sequential dependencies

    During inference:
    - Only main NTP logits used; MTP heads discarded
    - No computational overhead at inference time

    Parameters vs CapsuleMoEGPT:
    - Additional: (D-1) * (d^2 + d) params for MTP projections + norms
    - At d=64, D=3: +8,320 params (~4% overhead)
    """

    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4,
                 n_groups: int = 4, n_capsules_per_group: int = 64,
                 top_k_groups: int = 2, mtp_depth: int = 2,
                 mtp_lambda: float = 0.3):
        super().__init__()
        self.mtp_depth = mtp_depth
        self.mtp_lambda = mtp_lambda
        self.n_embd = n_embd

        # Standard CapsuleMoE components
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [CapsuleBlock(n_embd, n_head, n_groups,
                                    n_capsules_per_group, top_k_groups)
                       for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # MTP modules: D-1 sequential prediction heads
        if mtp_depth > 1:
            self.mtp_modules = [MTPModule(n_embd) for _ in range(mtp_depth - 1)]
        else:
            self.mtp_modules = []

        # Storage for MTP loss computation
        self._mtp_logits = []  # list of (B, T', V) for each MTP depth
        self._mtp_ready = False

    def _encode(self, tokens):
        """Run the main transformer stack, return final hidden states and token embeddings."""
        B, T = tokens.shape
        pos = mx.arange(T)
        tok_emb = self.wte(tokens)       # (B, T, d)
        x = tok_emb + self.wpe(pos)      # (B, T, d)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return x, tok_emb  # hidden_states, token_embeddings

    def __call__(self, tokens):
        """Forward pass. Returns NTP logits (B, T, V).

        Side effect: computes and stores MTP logits for mtp_loss().
        """
        B, T = tokens.shape
        h, tok_emb = self._encode(tokens)

        # Main NTP logits
        logits = self.lm_head(h)  # (B, T, V)

        # Compute MTP logits if training (mtp_depth > 1)
        self._mtp_logits = []
        if self.mtp_depth > 1 and T > 1:
            h_prev = h
            for k, mtp_mod in enumerate(self.mtp_modules):
                # For depth k+1: predict token at position t+k+2
                # Input: h_prev at positions [0..T-2-k], tok_emb at positions [k+1..T-1]
                # This ensures we only predict where we have ground truth
                max_pos = T - 1 - k
                if max_pos < 1:
                    break

                h_slice = h_prev[:, :max_pos, :]          # (B, max_pos, d)
                emb_slice = tok_emb[:, k+1:k+1+max_pos, :]  # (B, max_pos, d)

                h_k = mtp_mod(h_slice, emb_slice)  # (B, max_pos, d)
                mtp_logits_k = self.lm_head(h_k)   # (B, max_pos, V)
                self._mtp_logits.append(mtp_logits_k)

                # Chain: next depth uses this depth's hidden state
                h_prev = mx.zeros_like(h)
                h_prev = h_prev.at[:, :max_pos, :].add(h_k)

            self._mtp_ready = True

        return logits

    def mtp_loss(self, targets) -> mx.array:
        """Compute MTP auxiliary loss from stored MTP logits.

        targets: (B, T) -- the standard NTP target sequence (token IDs)

        For MTP depth k (1-indexed), the target at position t is targets[t+k].
        We only compute loss where targets are available.
        """
        if not self._mtp_ready or len(self._mtp_logits) == 0:
            return mx.array(0.0)

        B, T = targets.shape
        total_mtp_loss = mx.array(0.0)
        n_valid = 0

        for k, mtp_logits in enumerate(self._mtp_logits):
            # mtp_logits: (B, max_pos, V) where max_pos = T - 1 - k
            max_pos = mtp_logits.shape[1]

            # Target for depth k+1: token at position t+k+2 in the original sequence
            # Since targets[t] = tokens[t+1], the MTP-k target at position t is targets[t+k+1]
            mtp_targets = targets[:, k+1:k+1+max_pos]  # (B, max_pos)

            if mtp_targets.shape[1] == 0:
                continue

            BM, TM, V = mtp_logits.shape
            loss_k = nn.losses.cross_entropy(
                mtp_logits.reshape(BM * TM, V),
                mtp_targets.reshape(BM * TM),
                reduction="mean",
            )
            total_mtp_loss = total_mtp_loss + loss_k
            n_valid += 1

        if n_valid == 0:
            return mx.array(0.0)

        return self.mtp_lambda * total_mtp_loss / n_valid

    def aux_loss(self) -> mx.array:
        """Balance loss from capsule routing (same as CapsuleMoEGPT)."""
        total = mx.array(0.0)
        for layer in self.layers:
            total = total + layer.capsule_pool.balance_loss()
        return 0.01 * total

    def on_domain_switch(self, domain: str):
        pass

    def generate(self, prompt_tokens, max_new=20, temperature=0.8):
        """Autoregressively generate tokens (NTP only, no MTP)."""
        tokens = list(prompt_tokens.tolist()) if hasattr(prompt_tokens, 'tolist') else list(prompt_tokens)
        block_size = self.wpe.weight.shape[0]
        bos = self.wte.weight.shape[0] - 1
        for _ in range(max_new):
            x = mx.array([tokens[-block_size:]])
            logits = self(x)[0, -1]
            if temperature == 0:
                next_tok = mx.argmax(logits).item()
            else:
                probs = mx.softmax(logits / temperature)
                next_tok = mx.random.categorical(mx.log(probs)).item()
            tokens.append(next_tok)
            if next_tok == bos:
                break
        return tokens
