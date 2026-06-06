"""
SHINE S2: Context Reconstruction via M2P-Generated LoRA on Gemma 4

Trains M2P transformer to map Gemma 4 memory states (42, 32, 2560) into
LoRA weights for q_proj. Generated LoRA should enable context reconstruction.

Grounded: arXiv:2602.06358 (SHINE) S3.3, Finding #482 (S1), Finding #339.

Kill criteria:
  K1255: M2P training loss decreases > 20% over 1000 steps
  K1256: CE with generated LoRA < 2x base CE on same context
  K1257: Completion accuracy > random baseline with generated LoRA
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

# Training
N_STEPS = 1000
LR = 3e-4
CTX_LEN = 128  # tokens per paragraph chunk

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


# ── Diverse Text Passages ────────────────────────────────────────────────────

PASSAGES = [
    # Science
    """The mitochondria is the powerhouse of the cell. It generates ATP through
    oxidative phosphorylation. The electron transport chain consists of four
    protein complexes embedded in the inner mitochondrial membrane. Complex I
    accepts electrons from NADH, Complex II from succinate. Protons are pumped
    across the membrane creating a gradient that drives ATP synthase. This
    process produces approximately thirty ATP molecules per glucose molecule.
    The Krebs cycle occurs in the mitochondrial matrix and produces NADH and
    FADH2 electron carriers essential for the chain. Mitochondria have their
    own circular DNA encoding thirteen proteins critical to respiration.""",

    # History
    """The Roman Republic transitioned to the Roman Empire through a series of
    civil wars and political crises. Julius Caesar crossed the Rubicon in 49 BC,
    triggering civil war. After his assassination in 44 BC, his adopted heir
    Octavian defeated Mark Antony at the Battle of Actium in 31 BC. The Senate
    granted Octavian the title Augustus in 27 BC, marking the beginning of the
    Principate. Augustus reformed the military, established the Praetorian Guard,
    and created a system of provinces governed by imperial legates. The Pax
    Romana that followed lasted nearly two centuries of relative peace.""",

    # Medicine
    """Penicillin was discovered by Alexander Fleming in 1928 when he noticed
    that a mold called Penicillium notatum inhibited bacterial growth. Howard
    Florey and Ernst Boris Chain later developed methods to mass-produce the
    antibiotic during World War II. Penicillin works by inhibiting the synthesis
    of peptidoglycan, a critical component of bacterial cell walls. Beta-lactam
    antibiotics bind to penicillin-binding proteins and prevent cross-linking
    of the cell wall. Antibiotic resistance has emerged through beta-lactamase
    enzymes that break down the beta-lactam ring structure.""",

    # Law
    """The doctrine of stare decisis requires courts to follow precedent
    established by higher courts in the same jurisdiction. This principle
    promotes predictability and consistency in the legal system. However,
    courts can distinguish cases on their facts or overrule prior decisions
    when compelling reasons exist. The Supreme Court has overruled its own
    precedents numerous times, including Brown v Board of Education which
    overturned Plessy v Ferguson. Horizontal stare decisis binds courts of
    the same level while vertical stare decisis binds lower courts.""",

    # Technology
    """Transformer architectures revolutionized natural language processing
    after the publication of Attention Is All You Need in 2017. The self-
    attention mechanism allows each token to attend to all other tokens in
    the sequence with computational complexity quadratic in sequence length.
    Multi-head attention projects queries keys and values into multiple
    subspaces enabling the model to capture different types of relationships.
    Positional encodings provide sequence order information since attention
    is permutation invariant. Layer normalization and residual connections
    stabilize training of deep transformer models.""",

    # Finance
    """The Black-Scholes model provides a theoretical framework for pricing
    European options. The model assumes geometric Brownian motion for the
    underlying asset price with constant volatility and risk-free rate. The
    Black-Scholes partial differential equation is derived by constructing
    a riskless portfolio of the option and the underlying asset. The Greeks
    measure sensitivity of the option price to various parameters: delta
    measures price sensitivity, gamma measures delta sensitivity, theta
    measures time decay, and vega measures volatility sensitivity. The
    implied volatility surface reveals market deviations from the model.""",

    # Physics
    """Quantum entanglement occurs when particles become correlated such that
    the quantum state of each particle cannot be described independently. When
    two particles are entangled, measuring the spin of one instantaneously
    determines the spin of the other regardless of distance. Einstein called
    this spooky action at a distance. Bell's theorem proved that no local
    hidden variable theory can reproduce all predictions of quantum mechanics.
    Experiments by Alain Aspect confirmed violations of Bell inequalities.
    Entanglement is now used in quantum computing and quantum cryptography
    for secure key distribution protocols.""",

    # Computer Science
    """Hash tables provide average-case constant time lookup by mapping keys
    to array indices through a hash function. Collision resolution strategies
    include chaining where each bucket contains a linked list and open
    addressing where collisions probe for empty slots. The load factor ratio
    of elements to buckets determines performance degradation. Robin Hood
    hashing reduces variance in probe lengths by displacing elements with
    shorter probe distances. Cuckoo hashing guarantees worst-case constant
    lookup by using multiple hash functions and relocating elements on
    collision. Bloom filters provide probabilistic set membership testing.""",

    # Biology
    """CRISPR-Cas9 is a gene editing technology derived from bacterial immune
    systems. Bacteria use CRISPR arrays to store fragments of viral DNA and
    Cas proteins to cut matching sequences in future infections. Jennifer
    Doudna and Emmanuelle Charpentier adapted this system for programmable
    gene editing by designing guide RNA sequences. The Cas9 nuclease creates
    double-strand breaks at targeted genomic locations. Cells repair these
    breaks through non-homologous end joining which introduces insertions or
    deletions or through homology-directed repair which allows precise edits
    using a donor template.""",

    # Mathematics
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
    """Tokenize passages and chunk into CTX_LEN segments."""
    all_tokens = []
    for passage in PASSAGES:
        toks = tokenizer.encode(passage)
        all_tokens.extend(toks)

    # Repeat to get enough chunks
    while len(all_tokens) < (n_train + n_test) * CTX_LEN:
        all_tokens = all_tokens + all_tokens

    chunks = []
    for i in range(0, len(all_tokens) - CTX_LEN, CTX_LEN):
        chunks.append(all_tokens[i:i + CTX_LEN])
        if len(chunks) >= n_train + n_test:
            break

    train_chunks = [mx.array([c]) for c in chunks[:n_train]]  # each (1, CTX_LEN)
    test_chunks = [mx.array([c]) for c in chunks[n_train:n_train + n_test]]
    log(f"  Data: {len(train_chunks)} train, {len(test_chunks)} test chunks of {CTX_LEN} tokens")
    return train_chunks, test_chunks


# ── Memory Extractor (from S1, Finding #482) ─────────────────────────────────

class MemoryExtractor(nn.Module):
    """Appends learnable memory tokens, extracts per-layer hidden states."""

    def __init__(self, text_model, num_mem_tokens: int = N_MEM_TOKENS):
        super().__init__()
        self.text_model = text_model
        self.num_mem_tokens = num_mem_tokens
        hidden_size = text_model.config.hidden_size
        mx.random.seed(SEED)
        self.mem_tokens = mx.random.normal(shape=(1, num_mem_tokens, hidden_size)) * 0.02

    def extract(self, input_ids: mx.array):
        """Returns memory_states: (L, M, d) as float16 for memory efficiency."""
        tm = self.text_model
        config = tm.config
        M = self.num_mem_tokens
        L = config.num_hidden_layers

        # Embed and concatenate memory tokens
        ctx_embeds = tm.embed_tokens(input_ids)
        mem_embeds = mx.broadcast_to(self.mem_tokens, (1, M, config.hidden_size))
        h = mx.concatenate([ctx_embeds, mem_embeds], axis=1)
        h = h * config.hidden_size**0.5

        # Per-layer inputs
        if tm.hidden_size_per_layer_input:
            ctx_pli = tm._get_per_layer_inputs(input_ids)
            mem_pli = mx.zeros((1, M, L, tm.hidden_size_per_layer_input))
            full_pli = mx.concatenate([ctx_pli, mem_pli], axis=1)
            full_pli = tm._project_per_layer_inputs(h, full_pli)
            per_layer_inputs = [full_pli[:, :, i, :] for i in range(L)]
        else:
            per_layer_inputs = [None] * L

        # Attention masks
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

        # Forward through layers, collecting memory states
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
        return result.astype(mx.float16)  # save memory


# ── M2P Transformer ──────────────────────────────────────────────────────────

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
        scale = hd ** -0.5
        attn = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)
        return self.wo((attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C))


class M2PBlock(nn.Module):
    """Alternating row/column attention (SHINE S3.4)."""
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
            # Attention across layers (for each memory token position)
            x_t = x.transpose(1, 0, 2)  # (M, L, H)
            x_t = x_t + self.attn(self.norm1(x_t))
            x_t = x_t + self.mlp_w2(nn.gelu(self.mlp_w1(self.norm2(x_t))))
            return x_t.transpose(1, 0, 2)
        else:
            # Attention across memory tokens (for each layer)
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp_w2(nn.gelu(self.mlp_w1(self.norm2(x))))
            return x


class M2PTransformer(nn.Module):
    """Maps memory states (L, M, d) to LoRA pairs for each layer's q_proj.

    Handles Gemma 4's mixed attention: sliding layers have head_dim=256
    (q_proj out=2048), full layers have global_head_dim=512 (q_proj out=4096).
    Two output projections handle the different LoRA shapes.
    """

    def __init__(self, n_layers=N_LAYERS, n_mem_tokens=N_MEM_TOKENS,
                 input_dim=HIDDEN_DIM, layer_qproj_dims=None, m2p_dim=M2P_DIM,
                 lora_rank=LORA_RANK, n_blocks=M2P_BLOCKS, n_heads=M2P_HEADS):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.lora_rank = lora_rank
        self.m2p_dim = m2p_dim
        # Per-layer q_proj output dims (e.g. [2048, 2048, ..., 4096, ...])
        self.layer_qproj_dims = layer_qproj_dims or [input_dim] * n_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, m2p_dim, bias=False)

        # Positional embeddings
        scale = math.sqrt(2.0 / (1 + m2p_dim))
        self.p_layer = mx.random.normal((n_layers, 1, m2p_dim)) * scale
        self.p_token = mx.random.normal((1, n_mem_tokens, m2p_dim)) * scale

        # M2P blocks
        self.blocks = [
            M2PBlock(m2p_dim, n_heads, is_column=(i % 2 == 0))
            for i in range(n_blocks)
        ]
        self.final_norm = nn.RMSNorm(m2p_dim)

        # Output projections: one per unique q_proj output dim
        unique_dims = sorted(set(self.layer_qproj_dims))
        self.output_projs = []
        self._dim_to_proj_idx = {}
        for i, d_out in enumerate(unique_dims):
            adapter_size = input_dim * lora_rank + lora_rank * d_out
            proj = nn.Linear(m2p_dim, adapter_size, bias=False)
            proj.weight = proj.weight * 0.01  # small init
            self.output_projs.append(proj)
            self._dim_to_proj_idx[d_out] = i

        # Pre-compute which projection to use per layer
        self._layer_proj_idx = [self._dim_to_proj_idx[d] for d in self.layer_qproj_dims]

    def __call__(self, memory_states):
        """memory_states: (L, M, d) float16 -> list of (A, B) per layer."""
        x = self.input_proj(memory_states.astype(mx.float32))  # (L, M, m2p_dim)
        x = x + self.p_layer + self.p_token

        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        # Mean pool across memory tokens
        x = mx.mean(x, axis=1)  # (L, m2p_dim)

        d_in = self.input_dim
        r = self.lora_rank
        lora_pairs = []
        for i in range(self.n_layers):
            d_out = self.layer_qproj_dims[i]
            proj_idx = self._layer_proj_idx[i]
            p = self.output_projs[proj_idx](x[i:i+1]).squeeze(0)
            A = p[:d_in * r].reshape(d_in, r)
            B = p[d_in * r:].reshape(r, d_out)
            lora_pairs.append((A, B * (1.0 / math.sqrt(r))))
        return lora_pairs


# ── LoRA Injection ────────────────────────────────────────────────────────────

class LoRAProxy:
    """Drop-in replacement for nn.Linear that adds LoRA delta to output.
    Works with quantized base layers since we modify output, not weights."""

    def __init__(self, base_linear, A, B):
        self.base = base_linear
        self.A = A  # (input_dim, rank)
        self.B = B  # (rank, output_dim)
        # Preserve attributes the attention module might read
        if hasattr(base_linear, 'weight'):
            self.weight = base_linear.weight

    def __call__(self, x):
        return self.base(x) + (x @ self.A @ self.B)


def inject_lora(text_model, lora_pairs):
    """Replace q_proj in each layer with LoRA proxy. Returns originals."""
    originals = []
    for idx in range(len(text_model.layers)):
        layer = text_model.layers[idx]
        originals.append(layer.self_attn.q_proj)
        A, B = lora_pairs[idx]
        layer.self_attn.q_proj = LoRAProxy(originals[idx], A, B)
    return originals


def restore_qproj(text_model, originals):
    """Restore original q_proj layers."""
    for idx in range(len(text_model.layers)):
        text_model.layers[idx].self_attn.q_proj = originals[idx]


# ── Training ──────────────────────────────────────────────────────────────────

def compute_ce(model, input_ids):
    """Compute CE loss for next-token prediction."""
    logits = model(input_ids)
    targets = input_ids[:, 1:]
    logits = logits[:, :-1]
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
    )


def train_m2p(m2p, model, text_model, train_memory_states, train_chunks):
    """Train M2P to generate LoRA that reduces reconstruction CE."""
    log(f"\n=== Training M2P ({N_STEPS} steps) ===")

    optimizer = opt.Adam(learning_rate=LR)
    n_train = len(train_chunks)
    losses = []
    lora_norms = []

    def loss_fn(m2p, memory_states, input_ids):
        lora_pairs = m2p(memory_states)
        originals = inject_lora(text_model, lora_pairs)
        loss = compute_ce(model, input_ids)
        restore_qproj(text_model, originals)
        return loss

    loss_and_grad = nn.value_and_grad(m2p, loss_fn)

    t0 = time.time()
    gc.disable()
    for step in range(N_STEPS):
        idx = np.random.randint(0, n_train)
        memory_states = train_memory_states[idx]
        input_ids = train_chunks[idx]

        loss, grads = loss_and_grad(m2p, memory_states, input_ids)
        optimizer.update(m2p, grads)
        mx.eval(loss, m2p.parameters(), optimizer.state)
        losses.append(loss.item())

        if (step + 1) % 100 == 0 or step == 0:
            elapsed = time.time() - t0
            # Check LoRA norm for one example
            test_lora = m2p(train_memory_states[0])
            norms = [float(mx.sqrt((a @ b).square().sum()).item())
                     for a, b in test_lora]
            mean_norm = sum(norms) / len(norms)
            lora_norms.append(mean_norm)
            log(f"  Step {step+1}/{N_STEPS}: loss={losses[-1]:.4f}, "
                f"mean_lora_norm={mean_norm:.6f}, elapsed={elapsed:.1f}s")
            log_memory(f"step-{step+1}")

        del loss, grads
    gc.enable()

    total_time = time.time() - t0
    log(f"  Training done in {total_time:.1f}s ({total_time/N_STEPS*1000:.1f}ms/step)")
    return losses, lora_norms, total_time


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(m2p, model, text_model, memory_states_list, chunks_list, label=""):
    """Evaluate CE with and without M2P-generated LoRA."""
    log(f"\n=== Evaluate: {label} ({len(chunks_list)} chunks) ===")
    base_ces = []
    adapted_ces = []

    for i in range(len(chunks_list)):
        input_ids = chunks_list[i]

        # Base CE
        base_logits = model(input_ids)
        targets = input_ids[:, 1:]
        B, T, V = base_logits[:, :-1].shape
        base_ce = nn.losses.cross_entropy(
            base_logits[:, :-1].reshape(B * T, V),
            targets.reshape(B * T), reduction="mean"
        )
        mx.eval(base_ce)
        base_ces.append(base_ce.item())

        # Adapted CE
        lora_pairs = m2p(memory_states_list[i])
        originals = inject_lora(text_model, lora_pairs)
        adapted_logits = model(input_ids)
        adapted_ce = nn.losses.cross_entropy(
            adapted_logits[:, :-1].reshape(B * T, V),
            targets.reshape(B * T), reduction="mean"
        )
        mx.eval(adapted_ce)
        adapted_ces.append(adapted_ce.item())
        restore_qproj(text_model, originals)

        del base_logits, adapted_logits
        mx.clear_cache()

    mean_base = sum(base_ces) / len(base_ces)
    mean_adapted = sum(adapted_ces) / len(adapted_ces)
    ratio = mean_adapted / mean_base if mean_base > 0 else float('inf')
    log(f"  Base CE: {mean_base:.4f}")
    log(f"  Adapted CE: {mean_adapted:.4f}")
    log(f"  Ratio (adapted/base): {ratio:.4f}")
    return {"base_ce": mean_base, "adapted_ce": mean_adapted, "ratio": ratio,
            "per_example_base": base_ces, "per_example_adapted": adapted_ces}


def evaluate_context_specificity(m2p, memory_states_list):
    """Check that M2P generates different LoRA for different contexts."""
    log("\n=== Context Specificity ===")
    n = min(10, len(memory_states_list))
    # Get LoRA for each context (flatten to vector)
    lora_vecs = []
    for i in range(n):
        pairs = m2p(memory_states_list[i])
        # Flatten all A, B into one vector
        flat = mx.concatenate([mx.concatenate([a.reshape(-1), b.reshape(-1)])
                               for a, b in pairs])
        mx.eval(flat)
        lora_vecs.append(flat)

    # Pairwise cosine similarity
    cos_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = lora_vecs[i], lora_vecs[j]
            cos = float((vi * vj).sum() / (mx.sqrt((vi * vi).sum()) * mx.sqrt((vj * vj).sum()) + 1e-8))
            cos_sims.append(cos)

    mean_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 1.0
    max_cos = max(cos_sims) if cos_sims else 1.0
    log(f"  Pairwise LoRA cosine: mean={mean_cos:.4f}, max={max_cos:.4f}")
    log(f"  Context-specific (mean < 0.9): {'YES' if mean_cos < 0.9 else 'NO'}")
    return {"mean_cosine": mean_cos, "max_cosine": max_cos, "n_pairs": len(cos_sims)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    mx.random.seed(SEED)
    np.random.seed(SEED)
    log("SHINE S2: Context Reconstruction via M2P-Generated LoRA on Gemma 4")
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
    log(f"  q_proj shape: {text_model.layers[0].self_attn.q_proj.weight.shape}")
    log_memory("after-load")

    # --- Phase 1: Prepare data ---
    log("\n=== Phase 1: Prepare Data ===")
    n_train = 40 if not SMOKE_TEST else 10
    n_test = 10 if not SMOKE_TEST else 5
    train_chunks, test_chunks = prepare_data(tokenizer, n_train, n_test)

    # --- Phase 2: Pre-extract memory states ---
    log("\n=== Phase 2: Pre-extract Memory States ===")
    extractor = MemoryExtractor(text_model)
    extractor.text_model.freeze()  # redundant but explicit

    t_extract = time.time()
    train_memory = []
    for i, chunk in enumerate(train_chunks):
        ms = extractor.extract(chunk)
        mx.eval(ms)
        train_memory.append(ms)
        if (i + 1) % 10 == 0:
            log(f"  Extracted {i+1}/{len(train_chunks)}...")

    test_memory = []
    for chunk in test_chunks:
        ms = extractor.extract(chunk)
        mx.eval(ms)
        test_memory.append(ms)

    extract_time = time.time() - t_extract
    log(f"  Extraction: {extract_time:.1f}s for {len(train_memory)+len(test_memory)} chunks")
    log(f"  Memory state shape: {train_memory[0].shape}, dtype: {train_memory[0].dtype}")
    log_memory("after-extract")

    # Verify non-degeneracy (spot check)
    ms0 = train_memory[0].astype(mx.float32)
    flat = ms0.reshape(n_layers, -1)
    norms = mx.sqrt((flat * flat).sum(axis=-1, keepdims=True))
    normed = flat / (norms + 1e-8)
    cos_matrix = normed @ normed.T
    mx.eval(cos_matrix)
    # Mean of upper triangle (exclude diagonal)
    cos_vals = []
    cos_list = cos_matrix.tolist()
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            cos_vals.append(cos_list[i][j])
    mean_cos = sum(cos_vals) / len(cos_vals)
    log(f"  Cross-layer cosine (spot check): {mean_cos:.4f} (S1 reported 0.182)")
    del ms0, flat, norms, normed, cos_matrix

    # --- Phase 3: Build M2P ---
    log("\n=== Phase 3: Build M2P ===")
    # Detect per-layer q_proj output dims (Gemma 4 has mixed head_dim)
    probe = mx.zeros((1, 1, hidden_dim))
    layer_qproj_dims = []
    for i in range(n_layers):
        out = text_model.layers[i].self_attn.q_proj(probe)
        mx.eval(out)
        layer_qproj_dims.append(out.shape[-1])
    del probe, out
    unique_dims = sorted(set(layer_qproj_dims))
    dim_counts = {d: layer_qproj_dims.count(d) for d in unique_dims}
    log(f"  q_proj dims: {dim_counts}")

    m2p = M2PTransformer(
        n_layers=n_layers, n_mem_tokens=N_MEM_TOKENS,
        input_dim=hidden_dim, layer_qproj_dims=layer_qproj_dims, m2p_dim=M2P_DIM,
        lora_rank=LORA_RANK, n_blocks=M2P_BLOCKS, n_heads=M2P_HEADS,
    )
    mx.eval(m2p.parameters())
    m2p_params = sum(p.size for _, p in nn.utils.tree_flatten(m2p.parameters()))
    log(f"  M2P parameters: {m2p_params:,}")
    log(f"  Config: dim={M2P_DIM}, blocks={M2P_BLOCKS}, heads={M2P_HEADS}, rank={LORA_RANK}")
    log_memory("after-m2p-build")

    # Smoke check: M2P forward
    test_lora = m2p(train_memory[0])
    mx.eval(*[p for pair in test_lora for p in pair])
    log(f"  Smoke: M2P output = {len(test_lora)} layers, A={test_lora[0][0].shape}, B={test_lora[0][1].shape}")
    init_norm = float(mx.sqrt((test_lora[0][0] @ test_lora[0][1]).square().sum()).item())
    log(f"  Initial LoRA norm (layer 0): {init_norm:.6f}")

    # Smoke check: inject and forward
    originals = inject_lora(text_model, test_lora)
    test_logits = model(train_chunks[0])
    mx.eval(test_logits)
    log(f"  Smoke: logits shape = {test_logits.shape}")
    restore_qproj(text_model, originals)
    del test_logits, test_lora

    # --- Phase 4: Train ---
    losses, lora_norms, train_time = train_m2p(
        m2p, model, text_model, train_memory, train_chunks
    )
    log_memory("after-train")

    # --- Phase 5: Evaluate ---
    train_eval = evaluate(m2p, model, text_model, train_memory, train_chunks, "Train")
    test_eval = evaluate(m2p, model, text_model, test_memory, test_chunks, "Test")
    specificity = evaluate_context_specificity(m2p, train_memory)
    log_memory("after-eval")

    # --- Kill Criteria ---
    # K1255: Training loss decreases > 20%
    initial_loss = np.mean(losses[:10])  # avg first 10 steps
    final_loss = np.mean(losses[-10:])    # avg last 10 steps
    loss_decrease_pct = (initial_loss - final_loss) / initial_loss * 100
    k1255_pass = loss_decrease_pct > 20

    # K1256: CE < 2x base CE
    k1256_pass = test_eval["ratio"] < 2.0

    # K1257: Completion accuracy > random
    # Proxy: adapted CE < base CE on test set (better predictions = better completion)
    k1257_pass = test_eval["adapted_ce"] < test_eval["base_ce"]

    total_time = time.time() - t_start

    results = {
        "experiment": "exp_shine_gemma4_reconstruction",
        "type": "frontier_extension",
        "total_time_s": round(total_time, 1),
        "model": "gemma-4-e4b-it-4bit",
        "m2p": {
            "params": m2p_params,
            "dim": M2P_DIM,
            "blocks": M2P_BLOCKS,
            "heads": M2P_HEADS,
            "lora_rank": LORA_RANK,
        },
        "data": {
            "n_train": len(train_chunks),
            "n_test": len(test_chunks),
            "ctx_len": CTX_LEN,
            "n_passages": len(PASSAGES),
        },
        "training": {
            "n_steps": N_STEPS,
            "lr": LR,
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(final_loss, 4),
            "loss_decrease_pct": round(loss_decrease_pct, 2),
            "train_time_s": round(train_time, 1),
            "ms_per_step": round(train_time / N_STEPS * 1000, 1),
            "lora_norms": lora_norms,
        },
        "evaluation": {
            "train": train_eval,
            "test": test_eval,
            "context_specificity": specificity,
        },
        "kill_criteria": {
            "K1255": {
                "criterion": "M2P training loss decreases > 20%",
                "measured": round(loss_decrease_pct, 2),
                "pass": bool(k1255_pass),
            },
            "K1256": {
                "criterion": "CE with LoRA < 2x base CE",
                "ratio": round(test_eval["ratio"], 4),
                "base_ce": round(test_eval["base_ce"], 4),
                "adapted_ce": round(test_eval["adapted_ce"], 4),
                "pass": bool(k1256_pass),
            },
            "K1257": {
                "criterion": "Adapted CE < base CE (better completion)",
                "adapted_ce": round(test_eval["adapted_ce"], 4),
                "base_ce": round(test_eval["base_ce"], 4),
                "pass": bool(k1257_pass),
            },
        },
        "predictions": {
            "D1_loss_decrease": bool(k1255_pass),
            "D2_ce_ratio": round(test_eval["ratio"], 4),
            "D3_context_specific": specificity["mean_cosine"] < 0.9,
        },
        "all_pass": bool(k1255_pass and k1256_pass and k1257_pass),
        "status": "supported" if (k1255_pass and k1256_pass and k1257_pass) else "killed",
    }

    log(f"\n{'='*70}")
    log(f"RESULTS")
    log(f"{'='*70}")
    log(f"M2P: {m2p_params:,} params, {M2P_DIM}d, {M2P_BLOCKS} blocks, rank={LORA_RANK}")
    log(f"Training: {N_STEPS} steps, {train_time:.1f}s ({train_time/N_STEPS*1000:.1f}ms/step)")
    log(f"")
    log(f"K1255 (loss decrease > 20%): {'PASS' if k1255_pass else 'FAIL'} ({loss_decrease_pct:.1f}%)")
    log(f"  initial={initial_loss:.4f}, final={final_loss:.4f}")
    log(f"K1256 (CE ratio < 2.0): {'PASS' if k1256_pass else 'FAIL'} ({test_eval['ratio']:.4f})")
    log(f"  base_ce={test_eval['base_ce']:.4f}, adapted_ce={test_eval['adapted_ce']:.4f}")
    log(f"K1257 (adapted < base CE): {'PASS' if k1257_pass else 'FAIL'}")
    log(f"  adapted={test_eval['adapted_ce']:.4f} vs base={test_eval['base_ce']:.4f}")
    log(f"")
    log(f"Context specificity: mean_cos={specificity['mean_cosine']:.4f}")
    log(f"Status: {results['status'].upper()}")
    log(f"Total time: {total_time:.1f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_FILE}")

    cleanup(m2p, extractor)
    return results


if __name__ == "__main__":
    main()
