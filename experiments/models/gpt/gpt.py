"""Dense GPT — MLX port of microgpt.py."""

import mlx.core as mx
import mlx.nn as nn

from .. import register


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def __call__(self, x):
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(ms + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@register("gpt")
class GPT(nn.Module):
    def __init__(self, vocab_size: int = 28, block_size: int = 32,
                 n_embd: int = 64, n_head: int = 4, n_layer: int = 4):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def generate(self, prompt_tokens, max_new=20, temperature=0.8):
        """Autoregressively generate tokens from a prompt."""
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

    def aux_loss(self) -> mx.array:
        return mx.array(0.0)

    def on_domain_switch(self, domain: str):
        pass
