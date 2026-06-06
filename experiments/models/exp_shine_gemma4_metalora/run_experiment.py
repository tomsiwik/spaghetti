"""
SHINE S3: Meta LoRA Encoding + Multi-Projection Generation

Three structural fixes for the centroid trap (Finding #484, cos=0.998):
1. Meta LoRA (rank 128) during memory extraction → richer memory states
2. Multi-projection M2P (q+v+o) → 3x output capacity
3. Diversity regularizer → makes centroid a saddle point

Grounded: arXiv:2602.06358 S3.1, Finding #484 (S2), Finding #345 (centroid trap),
          Finding #480 (v+o format priors).

Kill criteria:
  K1258: S3 test CE ratio < S2 test CE ratio (0.134)
  K1259: q+v+o CE < q-only CE
  K1260: Grassmannian cos(meta LoRA, generated LoRA) < 1e-4
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np

# Memory safety
_dev = mx.device_info()
mx.set_memory_limit(_dev["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(4 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
SEED = 42

# Gemma 4 constants
N_LAYERS = 42
HIDDEN_DIM = 2560
N_MEM_TOKENS = 32

# M2P hyperparams
M2P_DIM = 128
M2P_BLOCKS = 2
M2P_HEADS = 4
LORA_RANK = 2

# Meta LoRA
META_LORA_RANK = 128

# Training
N_STEPS = 1000
LR = 3e-4
CTX_LEN = 128

# Diversity loss
DIVERSITY_LAMBDA = 0.1
DIVERSITY_WARMUP = 100
DIVERSITY_CACHE_SIZE = 16

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
if SMOKE_TEST:
    N_STEPS = 50
    print("[SMOKE] Reduced to 50 steps")


def log(m):
    print(m, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


# ── Diverse Text Passages (same as S2) ──────────────────────────────────────

PASSAGES = [
    """The mitochondria is the powerhouse of the cell. It generates ATP through
    oxidative phosphorylation. The electron transport chain consists of four
    protein complexes embedded in the inner mitochondrial membrane. Complex I
    accepts electrons from NADH, Complex II from succinate. Protons are pumped
    across the membrane creating a gradient that drives ATP synthase. This
    process produces approximately thirty ATP molecules per glucose molecule.
    The Krebs cycle occurs in the mitochondrial matrix and produces NADH and
    FADH2 electron carriers essential for the chain. Mitochondria have their
    own circular DNA encoding thirteen proteins critical to respiration.""",

    """The Roman Republic transitioned to the Roman Empire through a series of
    civil wars and political crises. Julius Caesar crossed the Rubicon in 49 BC,
    triggering civil war. After his assassination in 44 BC, his adopted heir
    Octavian defeated Mark Antony at the Battle of Actium in 31 BC. The Senate
    granted Octavian the title Augustus in 27 BC, marking the beginning of the
    Principate. Augustus reformed the military, established the Praetorian Guard,
    and created a system of provinces governed by imperial legates. The Pax
    Romana that followed lasted nearly two centuries of relative peace.""",

    """Penicillin was discovered by Alexander Fleming in 1928 when he noticed
    that a mold called Penicillium notatum inhibited bacterial growth. Howard
    Florey and Ernst Boris Chain later developed methods to mass-produce the
    antibiotic during World War II. Penicillin works by inhibiting the synthesis
    of peptidoglycan, a critical component of bacterial cell walls. Beta-lactam
    antibiotics bind to penicillin-binding proteins and prevent cross-linking
    of the cell wall. Antibiotic resistance has emerged through beta-lactamase
    enzymes that break down the beta-lactam ring structure.""",

    """The doctrine of stare decisis requires courts to follow precedent
    established by higher courts in the same jurisdiction. This principle
    promotes predictability and consistency in the legal system. However,
    courts can distinguish cases on their facts or overrule prior decisions
    when compelling reasons exist. The Supreme Court has overruled its own
    precedents numerous times, including Brown v Board of Education which
    overturned Plessy v Ferguson. Horizontal stare decisis binds courts of
    the same level while vertical stare decisis binds lower courts.""",

    """Transformer architectures revolutionized natural language processing
    after the publication of Attention Is All You Need in 2017. The self-
    attention mechanism allows each token to attend to all other tokens in
    the sequence with computational complexity quadratic in sequence length.
    Multi-head attention projects queries keys and values into multiple
    subspaces enabling the model to capture different types of relationships.
    Positional encodings provide sequence order information since attention
    is permutation invariant. Layer normalization and residual connections
    stabilize training of deep transformer models.""",

    """The Black-Scholes model provides a theoretical framework for pricing
    European options. The model assumes geometric Brownian motion for the
    underlying asset price with constant volatility and risk-free rate. The
    Black-Scholes partial differential equation is derived by constructing
    a riskless portfolio of the option and the underlying asset. The Greeks
    measure sensitivity of the option price to various parameters: delta
    measures price sensitivity, gamma measures delta sensitivity, theta
    measures time decay, and vega measures volatility sensitivity. The
    implied volatility surface reveals market deviations from the model.""",

    """Quantum entanglement occurs when particles become correlated such that
    the quantum state of each particle cannot be described independently. When
    two particles are entangled, measuring the spin of one instantaneously
    determines the spin of the other regardless of distance. Einstein called
    this spooky action at a distance. Bell's theorem proved that no local
    hidden variable theory can reproduce all predictions of quantum mechanics.
    Experiments by Alain Aspect confirmed violations of Bell inequalities.
    Entanglement is now used in quantum computing and quantum cryptography
    for secure key distribution protocols.""",

    """Hash tables provide average-case constant time lookup by mapping keys
    to array indices through a hash function. Collision resolution strategies
    include chaining where each bucket contains a linked list and open
    addressing where collisions probe for empty slots. The load factor ratio
    of elements to buckets determines performance degradation. Robin Hood
    hashing reduces variance in probe lengths by displacing elements with
    shorter probe distances. Cuckoo hashing guarantees worst-case constant
    lookup by using multiple hash functions and relocating elements on
    collision. Bloom filters provide probabilistic set membership testing.""",

    """CRISPR-Cas9 is a gene editing technology derived from bacterial immune
    systems. Bacteria use CRISPR arrays to store fragments of viral DNA and
    Cas proteins to cut matching sequences in future infections. Jennifer
    Doudna and Emmanuelle Charpentier adapted this system for programmable
    gene editing by designing guide RNA sequences. The Cas9 nuclease creates
    double-strand breaks at targeted genomic locations. Cells repair these
    breaks through non-homologous end joining which introduces insertions or
    deletions or through homology-directed repair which allows precise edits
    using a donor template.""",

    """The fundamental theorem of calculus connects differentiation and
    integration. The first part states that if F is an antiderivative of f
    on an interval then the definite integral of f from a to b equals F(b)
    minus F(a). The second part states that the derivative of the integral
    of f from a to x equals f(x). Lebesgue integration generalizes Riemann
    integration by partitioning the range instead of the domain. Measure
    theory provides the rigorous foundation for integration and probability.
    The dominated convergence theorem allows interchange of limits and
    integrals under appropriate conditions.""",
]


def prepare_data(tokenizer, n_train=40, n_test=10):
    all_tokens = []
    for passage in PASSAGES:
        toks = tokenizer.encode(passage)
        all_tokens.extend(toks)
    while len(all_tokens) < (n_train + n_test) * CTX_LEN:
        all_tokens = all_tokens + all_tokens
    chunks = []
    for i in range(0, len(all_tokens) - CTX_LEN, CTX_LEN):
        chunks.append(all_tokens[i : i + CTX_LEN])
        if len(chunks) >= n_train + n_test:
            break
    train_chunks = [mx.array([c]) for c in chunks[:n_train]]
    test_chunks = [mx.array([c]) for c in chunks[n_train : n_train + n_test]]
    log(f"  Data: {len(train_chunks)} train, {len(test_chunks)} test chunks of {CTX_LEN} tokens")
    return train_chunks, test_chunks


# ── Memory Extractor (from S1/S2, Finding #482) ────────────────────────────

class MemoryExtractor(nn.Module):
    def __init__(self, text_model, num_mem_tokens: int = N_MEM_TOKENS):
        super().__init__()
        self.text_model = text_model
        self.num_mem_tokens = num_mem_tokens
        hidden_size = text_model.config.hidden_size
        mx.random.seed(SEED)
        self.mem_tokens = mx.random.normal(shape=(1, num_mem_tokens, hidden_size)) * 0.02

    def extract(self, input_ids: mx.array):
        tm = self.text_model
        config = tm.config
        M = self.num_mem_tokens
        L = config.num_hidden_layers

        ctx_embeds = tm.embed_tokens(input_ids)
        mem_embeds = mx.broadcast_to(self.mem_tokens, (1, M, config.hidden_size))
        h = mx.concatenate([ctx_embeds, mem_embeds], axis=1)
        h = h * config.hidden_size**0.5

        if tm.hidden_size_per_layer_input:
            ctx_pli = tm._get_per_layer_inputs(input_ids)
            mem_pli = mx.zeros((1, M, L, tm.hidden_size_per_layer_input))
            full_pli = mx.concatenate([ctx_pli, mem_pli], axis=1)
            full_pli = tm._project_per_layer_inputs(h, full_pli)
            per_layer_inputs = [full_pli[:, :, i, :] for i in range(L)]
        else:
            per_layer_inputs = [None] * L

        from mlx_lm.models.base import create_attention_mask
        mask = {}
        masks = []
        for layer in tm.layers:
            if layer.layer_type not in mask:
                if layer.layer_type == "full_attention":
                    mask["full_attention"] = create_attention_mask(h, None)
                elif layer.layer_type == "sliding_attention":
                    mask["sliding_attention"] = create_attention_mask(
                        h, None, window_size=tm.window_size
                    )
            masks.append(mask[layer.layer_type])

        memory_states = []
        intermediates = [(None, None)] * L
        cache = [None] * L
        for idx in range(L):
            layer = tm.layers[idx]
            prev_idx = tm.previous_kvs[idx]
            kvs, offset = intermediates[prev_idx]
            h, kvs, offset = layer(
                h, masks[idx], cache[idx],
                per_layer_input=per_layer_inputs[idx],
                shared_kv=kvs, offset=offset,
            )
            intermediates[idx] = (kvs, offset)
            memory_states.append(h[:, -M:, :])

        result = mx.stack(memory_states, axis=0).squeeze(1)  # (L, M, d)
        return result.astype(mx.float16)


# ── M2P Components ──────────────────────────────────────────────────────────

class M2PAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        hd = self.head_dim
        q = self.wq(x).reshape(B, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        scale = hd**-0.5
        attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
        return self.wo((attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C))


class M2PBlock(nn.Module):
    def __init__(self, dim, n_heads=4, is_column=True):
        super().__init__()
        self.attn = M2PAttention(dim, n_heads)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp_w1 = nn.Linear(dim, 4 * dim, bias=False)
        self.mlp_w2 = nn.Linear(4 * dim, dim, bias=False)
        self.is_column = is_column

    def __call__(self, x):
        L, M, H = x.shape
        if self.is_column:
            x_t = x.transpose(1, 0, 2)
            x_t = x_t + self.attn(self.norm1(x_t))
            x_t = x_t + self.mlp_w2(nn.gelu(self.mlp_w1(self.norm2(x_t))))
            return x_t.transpose(1, 0, 2)
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp_w2(nn.gelu(self.mlp_w1(self.norm2(x))))
            return x


# ── M2P Multi-Projection ──────────────────────────────────────────────────

class M2PMultiProjection(nn.Module):
    """Maps memory states (L, M, d) to LoRA for q, v, o projections.

    Handles Gemma 4's mixed dims: sliding vs full attention layers.
    """

    def __init__(self, n_layers, n_mem_tokens, input_dim,
                 layer_qproj_dims, layer_vproj_dims, layer_oproj_input_dims,
                 m2p_dim, lora_rank, n_blocks, n_heads):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.lora_rank = lora_rank
        self.m2p_dim = m2p_dim

        self.layer_qproj_dims = layer_qproj_dims
        self.layer_vproj_dims = layer_vproj_dims
        self.layer_oproj_input_dims = layer_oproj_input_dims

        self.input_proj = nn.Linear(input_dim, m2p_dim, bias=False)

        scale = math.sqrt(2.0 / (1 + m2p_dim))
        self.p_layer = mx.random.normal((n_layers, 1, m2p_dim)) * scale
        self.p_token = mx.random.normal((1, n_mem_tokens, m2p_dim)) * scale

        self.blocks = [
            M2PBlock(m2p_dim, n_heads, is_column=(i % 2 == 0))
            for i in range(n_blocks)
        ]
        self.final_norm = nn.RMSNorm(m2p_dim)

        r = lora_rank
        # Q: A(input_dim, r) + B(r, d_q)
        self.q_projs, self._q_idx, self._q_dims = self._build_proj_group(
            [input_dim] * n_layers, layer_qproj_dims, r, m2p_dim
        )
        # V: A(input_dim, r) + B(r, d_v)
        self.v_projs, self._v_idx, self._v_dims = self._build_proj_group(
            [input_dim] * n_layers, layer_vproj_dims, r, m2p_dim
        )
        # O: A(d_attn, r) + B(r, input_dim)
        self.o_projs, self._o_idx, self._o_dims = self._build_proj_group(
            layer_oproj_input_dims, [input_dim] * n_layers, r, m2p_dim
        )

    def _build_proj_group(self, a_dims, b_dims, r, m2p_dim):
        unique = sorted(set(zip(a_dims, b_dims)))
        projs = []
        pair_to_idx = {}
        for i, (a_d, b_d) in enumerate(unique):
            size = a_d * r + r * b_d
            p = nn.Linear(m2p_dim, size, bias=False)
            p.weight = p.weight * 0.01
            projs.append(p)
            pair_to_idx[(a_d, b_d)] = i
        layer_idx = [pair_to_idx[(a, b)] for a, b in zip(a_dims, b_dims)]
        layer_dims = list(zip(a_dims, b_dims))
        return projs, layer_idx, layer_dims

    def __call__(self, memory_states):
        x = self.input_proj(memory_states.astype(mx.float32))
        x = x + self.p_layer + self.p_token
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = mx.mean(x, axis=1)  # (L, m2p_dim)

        r = self.lora_rank
        q_lora = self._gen_lora(x, self.q_projs, self._q_idx, self._q_dims, r)
        v_lora = self._gen_lora(x, self.v_projs, self._v_idx, self._v_dims, r)
        o_lora = self._gen_lora(x, self.o_projs, self._o_idx, self._o_dims, r)
        return q_lora, v_lora, o_lora

    def _gen_lora(self, x, projs, layer_idx, layer_dims, r):
        pairs = []
        for i in range(self.n_layers):
            a_d, b_d = layer_dims[i]
            pidx = layer_idx[i]
            flat = projs[pidx](x[i : i + 1]).squeeze(0)
            A = flat[: a_d * r].reshape(a_d, r)
            B = flat[a_d * r :].reshape(r, b_d) * (1.0 / math.sqrt(r))
            pairs.append((A, B))
        return pairs


# ── S3 Model ──────────────────────────────────────────────────────────────

class S3Model(nn.Module):
    """Joint meta LoRA + M2P for end-to-end optimization."""

    def __init__(self, m2p, n_layers, hidden_dim, layer_qproj_dims, meta_rank):
        super().__init__()
        self.m2p = m2p
        self.n_layers = n_layers
        self.meta_rank = meta_rank
        # Meta LoRA: A random, B zero (no effect at init)
        scale = 1.0 / math.sqrt(meta_rank)
        self.meta_A = [
            mx.random.normal((hidden_dim, meta_rank)) * scale
            for _ in range(n_layers)
        ]
        self.meta_B = [mx.zeros((meta_rank, d)) for d in layer_qproj_dims]


# ── LoRA Injection ──────────────────────────────────────────────────────────

class LoRAProxy:
    """Drop-in replacement that adds LoRA delta to output.
    Works with quantized base layers since we modify output, not weights."""

    def __init__(self, base_linear, A, B):
        self.base = base_linear
        self.A = A
        self.B = B
        if hasattr(base_linear, "weight"):
            self.weight = base_linear.weight

    def __call__(self, x):
        return self.base(x) + (x @ self.A @ self.B)


def inject_lora_qproj(text_model, lora_pairs):
    originals = []
    for idx in range(len(text_model.layers)):
        originals.append(text_model.layers[idx].self_attn.q_proj)
        A, B = lora_pairs[idx]
        text_model.layers[idx].self_attn.q_proj = LoRAProxy(originals[idx], A, B)
    return originals


def restore_qproj(text_model, originals):
    for idx in range(len(text_model.layers)):
        text_model.layers[idx].self_attn.q_proj = originals[idx]


def inject_multi_lora(text_model, q_lora, v_lora, o_lora):
    originals = []
    for idx in range(len(text_model.layers)):
        attn = text_model.layers[idx].self_attn
        orig = {"q": attn.q_proj, "v": attn.v_proj, "o": attn.o_proj}
        originals.append(orig)
        attn.q_proj = LoRAProxy(orig["q"], *q_lora[idx])
        attn.v_proj = LoRAProxy(orig["v"], *v_lora[idx])
        attn.o_proj = LoRAProxy(orig["o"], *o_lora[idx])
    return originals


def restore_multi_lora(text_model, originals):
    for idx in range(len(text_model.layers)):
        attn = text_model.layers[idx].self_attn
        attn.q_proj = originals[idx]["q"]
        attn.v_proj = originals[idx]["v"]
        attn.o_proj = originals[idx]["o"]


# ── Loss ──────────────────────────────────────────────────────────────────

def compute_ce(model, input_ids):
    logits = model(input_ids)
    targets = input_ids[:, 1:]
    logits = logits[:, :-1]
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
    )


def make_div_vec(q_lora, v_lora):
    """Flatten q and v A matrices into a single vector for diversity."""
    q_flat = mx.concatenate([a.reshape(-1) for a, _ in q_lora])
    v_flat = mx.concatenate([a.reshape(-1) for a, _ in v_lora])
    return mx.concatenate([q_flat, v_flat])


def compute_diversity_loss(current_vec, cache):
    if len(cache) == 0:
        return mx.array(0.0)
    cur_norm = mx.sqrt(mx.sum(current_vec * current_vec)) + 1e-8
    losses = []
    for cached in cache[-DIVERSITY_CACHE_SIZE:]:
        cached_norm = mx.sqrt(mx.sum(cached * cached)) + 1e-8
        cos = mx.sum(current_vec * cached) / (cur_norm * cached_norm)
        losses.append(cos * cos)
    return mx.mean(mx.stack(losses))


# ── Training ──────────────────────────────────────────────────────────────

def train_s3(s3, model, text_model, extractor, train_chunks, n_layers):
    log(f"\n=== Training S3 ({N_STEPS} steps) ===")

    optimizer = opt.Adam(learning_rate=LR)
    lora_cache = []
    losses = []
    ntp_losses_list = []
    _div_vec = [None]
    _ntp_val = [None]
    step_ref = [0]

    def loss_fn(s3, input_ids):
        # 1. Inject meta LoRA for memory extraction
        meta_pairs = [(s3.meta_A[i], s3.meta_B[i]) for i in range(n_layers)]
        meta_orig = inject_lora_qproj(text_model, meta_pairs)

        # 2. Extract memory states (through meta-LoRA-augmented model)
        mem = extractor.extract(input_ids)
        restore_qproj(text_model, meta_orig)

        # 3. M2P generates multi-projection LoRA
        q_lora, v_lora, o_lora = s3.m2p(mem)

        # 4. Inject generated LoRA, compute NTP loss
        gen_orig = inject_multi_lora(text_model, q_lora, v_lora, o_lora)
        ntp = compute_ce(model, input_ids)
        restore_multi_lora(text_model, gen_orig)

        # 5. Diversity loss
        div_vec = make_div_vec(q_lora, v_lora)
        _div_vec[0] = mx.stop_gradient(div_vec)
        _ntp_val[0] = mx.stop_gradient(ntp)

        div = compute_diversity_loss(div_vec, lora_cache)
        warmup = min(1.0, step_ref[0] / DIVERSITY_WARMUP) if DIVERSITY_WARMUP > 0 else 1.0
        total = ntp + warmup * DIVERSITY_LAMBDA * div
        return total

    loss_and_grad = nn.value_and_grad(s3, loss_fn)

    t0 = time.time()
    gc.disable()
    for step in range(N_STEPS):
        step_ref[0] = step
        idx = np.random.randint(0, len(train_chunks))
        input_ids = train_chunks[idx]

        loss, grads = loss_and_grad(s3, input_ids)
        optimizer.update(s3, grads)

        # Eval all: loss, parameters, optimizer state, and cached div_vec
        to_eval = [loss, s3.parameters(), optimizer.state]
        if _div_vec[0] is not None:
            to_eval.append(_div_vec[0])
        if _ntp_val[0] is not None:
            to_eval.append(_ntp_val[0])
        mx.eval(*to_eval)

        losses.append(loss.item())
        if _ntp_val[0] is not None:
            ntp_losses_list.append(_ntp_val[0].item())

        # Update diversity cache
        if _div_vec[0] is not None:
            lora_cache.append(_div_vec[0])
            if len(lora_cache) > DIVERSITY_CACHE_SIZE * 2:
                lora_cache = lora_cache[-DIVERSITY_CACHE_SIZE:]

        if (step + 1) % 100 == 0 or step == 0:
            elapsed = time.time() - t0
            ntp_str = f", ntp={ntp_losses_list[-1]:.4f}" if ntp_losses_list else ""
            log(
                f"  Step {step+1}/{N_STEPS}: loss={losses[-1]:.4f}{ntp_str}, "
                f"elapsed={elapsed:.1f}s"
            )
            log_memory(f"step-{step+1}")

        del loss, grads

    gc.enable()
    total_time = time.time() - t0
    log(f"  Training done in {total_time:.1f}s ({total_time/N_STEPS*1000:.1f}ms/step)")
    return losses, ntp_losses_list, total_time


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_ce(s3, model, text_model, extractor, chunks, n_layers,
                label="", projections="qvo"):
    """Evaluate CE with generated LoRA. projections='qvo' or 'q' for ablation."""
    log(f"\n=== Evaluate: {label} (proj={projections}, {len(chunks)} chunks) ===")
    base_ces = []
    adapted_ces = []

    for i in range(len(chunks)):
        input_ids = chunks[i]

        # Base CE (no LoRA)
        base_logits = model(input_ids)
        targets = input_ids[:, 1:]
        B, T, V = base_logits[:, :-1].shape
        base_ce = nn.losses.cross_entropy(
            base_logits[:, :-1].reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        )
        mx.eval(base_ce)
        base_ces.append(base_ce.item())

        # Extract memory with meta LoRA
        meta_pairs = [(s3.meta_A[j], s3.meta_B[j]) for j in range(n_layers)]
        meta_orig = inject_lora_qproj(text_model, meta_pairs)
        mem = extractor.extract(input_ids)
        mx.eval(mem)
        restore_qproj(text_model, meta_orig)

        # Generate LoRA
        q_lora, v_lora, o_lora = s3.m2p(mem)

        # Inject based on projection mode
        if projections == "qvo":
            gen_orig = inject_multi_lora(text_model, q_lora, v_lora, o_lora)
        else:
            gen_orig = inject_lora_qproj(text_model, q_lora)

        adapted_logits = model(input_ids)
        adapted_ce = nn.losses.cross_entropy(
            adapted_logits[:, :-1].reshape(B * T, V),
            targets.reshape(B * T),
            reduction="mean",
        )
        mx.eval(adapted_ce)
        adapted_ces.append(adapted_ce.item())

        # Restore
        if projections == "qvo":
            restore_multi_lora(text_model, gen_orig)
        else:
            restore_qproj(text_model, gen_orig)

        del base_logits, adapted_logits
        mx.clear_cache()

    mean_base = sum(base_ces) / len(base_ces)
    mean_adapted = sum(adapted_ces) / len(adapted_ces)
    ratio = mean_adapted / mean_base if mean_base > 0 else float("inf")
    log(f"  Base CE: {mean_base:.4f}, Adapted CE: {mean_adapted:.4f}, Ratio: {ratio:.4f}")
    return {"base_ce": mean_base, "adapted_ce": mean_adapted, "ratio": ratio}


def evaluate_context_specificity(s3, extractor, text_model, chunks, n_layers):
    log("\n=== Context Specificity ===")
    n = min(10, len(chunks))
    lora_vecs = []
    for i in range(n):
        meta_pairs = [(s3.meta_A[j], s3.meta_B[j]) for j in range(n_layers)]
        meta_orig = inject_lora_qproj(text_model, meta_pairs)
        mem = extractor.extract(chunks[i])
        mx.eval(mem)
        restore_qproj(text_model, meta_orig)

        q_lora, v_lora, o_lora = s3.m2p(mem)
        flat = mx.concatenate(
            [mx.concatenate([a.reshape(-1), b.reshape(-1)]) for a, b in q_lora + v_lora + o_lora]
        )
        mx.eval(flat)
        lora_vecs.append(flat)

    cos_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = lora_vecs[i], lora_vecs[j]
            cos = float(
                (vi * vj).sum()
                / (mx.sqrt((vi * vi).sum()) * mx.sqrt((vj * vj).sum()) + 1e-8)
            )
            cos_sims.append(cos)

    mean_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 1.0
    max_cos = max(cos_sims) if cos_sims else 1.0
    log(f"  Pairwise LoRA cosine: mean={mean_cos:.4f}, max={max_cos:.4f}")
    log(f"  Context-specific (mean < 0.9): {'YES' if mean_cos < 0.9 else 'NO'}")
    return {"mean_cosine": mean_cos, "max_cosine": max_cos, "n_pairs": len(cos_sims)}


def evaluate_grassmannian(s3, extractor, text_model, chunks, n_layers):
    """Max column cosine between meta LoRA A and generated LoRA A subspaces."""
    log("\n=== Grassmannian Orthogonality ===")

    meta_pairs = [(s3.meta_A[j], s3.meta_B[j]) for j in range(n_layers)]
    meta_orig = inject_lora_qproj(text_model, meta_pairs)
    mem = extractor.extract(chunks[0])
    mx.eval(mem)
    restore_qproj(text_model, meta_orig)

    q_lora, _, _ = s3.m2p(mem)

    max_cos_per_layer = []
    for i in range(n_layers):
        meta_A = s3.meta_A[i]  # (hidden_dim, meta_rank)
        gen_A = q_lora[i][0]  # (hidden_dim, lora_rank)
        mx.eval(meta_A, gen_A)
        # Normalize columns
        meta_norm = meta_A / (mx.sqrt(mx.sum(meta_A * meta_A, axis=0, keepdims=True)) + 1e-8)
        gen_norm = gen_A / (mx.sqrt(mx.sum(gen_A * gen_A, axis=0, keepdims=True)) + 1e-8)
        cos_matrix = mx.abs(meta_norm.T @ gen_norm)  # (meta_rank, lora_rank)
        mx.eval(cos_matrix)
        max_cos_per_layer.append(float(cos_matrix.max().item()))

    mean_max = sum(max_cos_per_layer) / len(max_cos_per_layer)
    overall_max = max(max_cos_per_layer)
    log(f"  Mean max column cos: {mean_max:.6f}")
    log(f"  Overall max: {overall_max:.6f}")
    log(f"  K1260 (cos < 1e-4): {'PASS' if overall_max < 1e-4 else 'FAIL'}")
    log(f"  Note: random subspaces rank-128 vs rank-2 in R^2560 expect cos ~0.25")
    return {"mean_max_cos": mean_max, "max_cos": overall_max,
            "per_layer": max_cos_per_layer}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    mx.random.seed(SEED)
    np.random.seed(SEED)
    log("SHINE S3: Meta LoRA Encoding + Multi-Projection Generation")
    log("=" * 70)

    # --- Phase 0: Load model ---
    log("\n=== Phase 0: Load Gemma 4 E4B 4-bit ===")
    from mlx_lm import load

    model, tokenizer = load("mlx-community/gemma-4-e4b-it-4bit")
    text_model = model.language_model.model
    model.freeze()
    n_layers = len(text_model.layers)
    hidden_dim = text_model.config.hidden_size
    log(f"  Layers: {n_layers}, hidden_dim: {hidden_dim}")
    log_memory("after-load")

    # --- Phase 1: Prepare data ---
    log("\n=== Phase 1: Prepare Data ===")
    n_train = 40 if not SMOKE_TEST else 10
    n_test = 10 if not SMOKE_TEST else 5
    train_chunks, test_chunks = prepare_data(tokenizer, n_train, n_test)

    # --- Phase 2: Probe projection dimensions ---
    log("\n=== Phase 2: Probe Dimensions ===")
    probe = mx.zeros((1, 1, hidden_dim))
    layer_qproj_dims = []
    layer_vproj_dims = []
    layer_oproj_input_dims = []
    for i in range(n_layers):
        attn = text_model.layers[i].self_attn
        q_out = attn.q_proj(probe)
        v_out = attn.v_proj(probe)
        mx.eval(q_out, v_out)
        layer_qproj_dims.append(q_out.shape[-1])
        layer_vproj_dims.append(v_out.shape[-1])
        layer_oproj_input_dims.append(q_out.shape[-1])
    del probe, q_out, v_out
    q_counts = {d: layer_qproj_dims.count(d) for d in sorted(set(layer_qproj_dims))}
    v_counts = {d: layer_vproj_dims.count(d) for d in sorted(set(layer_vproj_dims))}
    log(f"  q_proj dims: {q_counts}")
    log(f"  v_proj dims: {v_counts}")
    log(f"  o_proj input dims: same as q_proj, output always {hidden_dim}")

    # --- Phase 3: Build S3 model ---
    log("\n=== Phase 3: Build S3 Model ===")
    extractor = MemoryExtractor(text_model)

    m2p = M2PMultiProjection(
        n_layers=n_layers,
        n_mem_tokens=N_MEM_TOKENS,
        input_dim=hidden_dim,
        layer_qproj_dims=layer_qproj_dims,
        layer_vproj_dims=layer_vproj_dims,
        layer_oproj_input_dims=layer_oproj_input_dims,
        m2p_dim=M2P_DIM,
        lora_rank=LORA_RANK,
        n_blocks=M2P_BLOCKS,
        n_heads=M2P_HEADS,
    )

    s3 = S3Model(m2p, n_layers, hidden_dim, layer_qproj_dims, META_LORA_RANK)
    mx.eval(s3.parameters())

    s3_params = sum(p.size for _, p in nn.utils.tree_flatten(s3.parameters()))
    m2p_params = sum(p.size for _, p in nn.utils.tree_flatten(m2p.parameters()))
    meta_params = s3_params - m2p_params
    log(f"  M2P params: {m2p_params:,}")
    log(f"  Meta LoRA params: {meta_params:,}")
    log(f"  Total S3 params: {s3_params:,}")
    log(f"  Config: m2p_dim={M2P_DIM}, blocks={M2P_BLOCKS}, heads={M2P_HEADS}")
    log(f"          lora_rank={LORA_RANK}, meta_rank={META_LORA_RANK}")
    log(f"          div_lambda={DIVERSITY_LAMBDA}, div_warmup={DIVERSITY_WARMUP}")
    log_memory("after-build")

    # --- Smoke check ---
    log("\n=== Smoke Check ===")
    meta_pairs = [(s3.meta_A[i], s3.meta_B[i]) for i in range(n_layers)]
    meta_orig = inject_lora_qproj(text_model, meta_pairs)
    test_mem = extractor.extract(train_chunks[0])
    mx.eval(test_mem)
    restore_qproj(text_model, meta_orig)
    log(f"  Memory state shape: {test_mem.shape}, dtype: {test_mem.dtype}")

    q_lora, v_lora, o_lora = s3.m2p(test_mem)
    mx.eval(*[p for pair in q_lora + v_lora + o_lora for p in pair])
    log(f"  q_lora: {len(q_lora)} layers, A={q_lora[0][0].shape}, B={q_lora[0][1].shape}")
    log(f"  v_lora: {len(v_lora)} layers, A={v_lora[0][0].shape}, B={v_lora[0][1].shape}")
    log(f"  o_lora: {len(o_lora)} layers, A={o_lora[0][0].shape}, B={o_lora[0][1].shape}")

    gen_orig = inject_multi_lora(text_model, q_lora, v_lora, o_lora)
    test_logits = model(train_chunks[0])
    mx.eval(test_logits)
    log(f"  Logits shape: {test_logits.shape}")
    restore_multi_lora(text_model, gen_orig)
    del test_logits, test_mem, q_lora, v_lora, o_lora
    log_memory("after-smoke")

    # --- Phase 4: Train ---
    losses, ntp_losses, train_time = train_s3(
        s3, model, text_model, extractor, train_chunks, n_layers
    )
    log_memory("after-train")

    # --- Phase 5: Evaluate ---
    train_eval_full = evaluate_ce(
        s3, model, text_model, extractor, train_chunks, n_layers, "Train q+v+o", "qvo"
    )
    test_eval_full = evaluate_ce(
        s3, model, text_model, extractor, test_chunks, n_layers, "Test q+v+o", "qvo"
    )
    test_eval_q = evaluate_ce(
        s3, model, text_model, extractor, test_chunks, n_layers, "Test q-only", "q"
    )

    specificity = evaluate_context_specificity(
        s3, extractor, text_model, train_chunks, n_layers
    )
    grassmannian = evaluate_grassmannian(
        s3, extractor, text_model, train_chunks, n_layers
    )
    log_memory("after-eval")

    # --- Phase 6: Kill Criteria ---
    initial_loss = float(np.mean(losses[:10]))
    final_loss = float(np.mean(losses[-10:]))
    loss_decrease_pct = (initial_loss - final_loss) / initial_loss * 100

    s2_test_ratio = 0.134
    k1258_pass = test_eval_full["ratio"] < s2_test_ratio
    k1259_pass = test_eval_full["ratio"] < test_eval_q["ratio"]
    k1260_pass = grassmannian["max_cos"] < 1e-4

    total_time = time.time() - t_start

    results = {
        "experiment": "exp_shine_gemma4_metalora",
        "type": "guided_exploration",
        "total_time_s": round(total_time, 1),
        "model": "gemma-4-e4b-it-4bit",
        "s3_config": {
            "m2p_params": m2p_params,
            "meta_lora_params": meta_params,
            "total_params": s3_params,
            "meta_lora_rank": META_LORA_RANK,
            "lora_rank": LORA_RANK,
            "m2p_dim": M2P_DIM,
            "m2p_blocks": M2P_BLOCKS,
            "diversity_lambda": DIVERSITY_LAMBDA,
            "diversity_warmup": DIVERSITY_WARMUP,
        },
        "data": {
            "n_train": len(train_chunks),
            "n_test": len(test_chunks),
            "ctx_len": CTX_LEN,
        },
        "training": {
            "n_steps": N_STEPS,
            "lr": LR,
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(final_loss, 4),
            "loss_decrease_pct": round(loss_decrease_pct, 2),
            "train_time_s": round(train_time, 1),
            "ms_per_step": round(train_time / N_STEPS * 1000, 1),
        },
        "evaluation": {
            "train_full": train_eval_full,
            "test_full": test_eval_full,
            "test_q_only": test_eval_q,
            "context_specificity": specificity,
            "grassmannian": grassmannian,
        },
        "s2_comparison": {
            "s2_test_ratio": s2_test_ratio,
            "s3_test_ratio": round(test_eval_full["ratio"], 4),
        },
        "kill_criteria": {
            "K1258": {
                "criterion": "S3 CE ratio < S2 CE ratio (0.134)",
                "s3_ratio": round(test_eval_full["ratio"], 4),
                "s2_ratio": s2_test_ratio,
                "pass": bool(k1258_pass),
            },
            "K1259": {
                "criterion": "q+v+o CE < q-only CE",
                "full_ratio": round(test_eval_full["ratio"], 4),
                "q_only_ratio": round(test_eval_q["ratio"], 4),
                "pass": bool(k1259_pass),
            },
            "K1260": {
                "criterion": "Grassmannian cos < 1e-4",
                "max_cos": round(grassmannian["max_cos"], 6),
                "note": "Random rank-128 vs rank-2 in R^2560 expects cos ~0.25",
                "pass": bool(k1260_pass),
            },
        },
        "predictions": {
            "P1_better_than_s2": bool(k1258_pass),
            "P2_multiproj_helps": bool(k1259_pass),
            "P3_diversity_cos_lt_0.9": specificity["mean_cosine"] < 0.9,
            "P4_grassmannian_cos_lt_0.3": grassmannian["max_cos"] < 0.3,
        },
        "all_kill_pass": bool(k1258_pass and k1259_pass and k1260_pass),
        "status": "supported" if (k1258_pass and k1259_pass) else "killed",
    }

    log(f"\n{'='*70}")
    log("RESULTS")
    log(f"{'='*70}")
    log(f"S3: {s3_params:,} params (M2P={m2p_params:,}, meta={meta_params:,})")
    log(f"Training: {N_STEPS} steps, {train_time:.1f}s ({train_time/N_STEPS*1000:.1f}ms/step)")
    log(f"Loss: {initial_loss:.4f} -> {final_loss:.4f} ({loss_decrease_pct:.1f}%)")
    log("")
    log(f"K1258 (CE < S2 0.134): {'PASS' if k1258_pass else 'FAIL'} "
        f"(S3={test_eval_full['ratio']:.4f})")
    log(f"K1259 (q+v+o > q-only): {'PASS' if k1259_pass else 'FAIL'} "
        f"(full={test_eval_full['ratio']:.4f} vs q={test_eval_q['ratio']:.4f})")
    log(f"K1260 (grass cos < 1e-4): {'PASS' if k1260_pass else 'FAIL'} "
        f"(cos={grassmannian['max_cos']:.6f})")
    log("")
    log(f"Context specificity: mean_cos={specificity['mean_cosine']:.4f} (S2 was 0.998)")
    log(f"Status: {results['status'].upper()}")
    log(f"Total time: {total_time:.1f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_FILE}")

    cleanup(s3, extractor)
    return results


if __name__ == "__main__":
    main()
