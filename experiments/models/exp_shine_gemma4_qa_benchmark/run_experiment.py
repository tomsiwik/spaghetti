"""
SHINE S4: Document QA Without Context in Prompt (vs ICL)

End-to-end behavioral test: encode document -> generate session LoRA ->
answer questions WITHOUT document in prompt. Compare to ICL baseline.

Architecture: S2 + multi-projection (no meta LoRA).
  - Pre-extract memory states (Finding #482)
  - M2P with q+v+o output (S3: 7.7x validated)
  - Train on reconstruction, evaluate on QA

Grounded: arXiv:2602.06358 (SHINE), Finding #484 (S2), S3 PAPER.md,
          Finding #480 (v+o format priors).

Kill criteria:
  K1261: QA F1 > 30% without document in prompt
  K1262: QA F1 >= 50% of in-context learning baseline
  K1263: Session adapter generation < 5s
"""

import gc
import json
import math
import os
import re
import time
from collections import Counter
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
CTX_LEN = 128

# QA generation
MAX_GEN_TOKENS = 64

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


# ── QA Data ─────────────────────────────────────────────────────────────────

# Documents for training M2P (reconstruction task)
TRAIN_PASSAGES = [
    """The mitochondria is the powerhouse of the cell. It generates ATP through
    oxidative phosphorylation. The electron transport chain consists of four
    protein complexes embedded in the inner mitochondrial membrane. Complex I
    accepts electrons from NADH, Complex II from succinate. Protons are pumped
    across the membrane creating a gradient that drives ATP synthase. This
    process produces approximately thirty ATP molecules per glucose molecule.""",

    """The Roman Republic transitioned to the Roman Empire through a series of
    civil wars and political crises. Julius Caesar crossed the Rubicon in 49 BC,
    triggering civil war. After his assassination in 44 BC, his adopted heir
    Octavian defeated Mark Antony at the Battle of Actium in 31 BC. The Senate
    granted Octavian the title Augustus in 27 BC.""",

    """Penicillin was discovered by Alexander Fleming in 1928 when he noticed
    that a mold called Penicillium notatum inhibited bacterial growth. Howard
    Florey and Ernst Boris Chain later developed methods to mass-produce the
    antibiotic during World War II. Penicillin works by inhibiting the synthesis
    of peptidoglycan, a critical component of bacterial cell walls.""",

    """Transformer architectures revolutionized natural language processing
    after the publication of Attention Is All You Need in 2017. The self-
    attention mechanism allows each token to attend to all other tokens in
    the sequence with computational complexity quadratic in sequence length.
    Multi-head attention projects queries keys and values into multiple
    subspaces enabling the model to capture different types of relationships.""",

    """The Black-Scholes model provides a theoretical framework for pricing
    European options. The model assumes geometric Brownian motion for the
    underlying asset price with constant volatility and risk-free rate. The
    Greeks measure sensitivity of the option price to various parameters: delta
    measures price sensitivity, gamma measures delta sensitivity, theta
    measures time decay, and vega measures volatility sensitivity.""",

    """Quantum entanglement occurs when particles become correlated such that
    the quantum state of each particle cannot be described independently. When
    two particles are entangled, measuring the spin of one instantaneously
    determines the spin of the other regardless of distance. Einstein called
    this spooky action at a distance. Bell's theorem proved that no local
    hidden variable theory can reproduce all predictions of quantum mechanics.""",

    """Hash tables provide average-case constant time lookup by mapping keys
    to array indices through a hash function. Collision resolution strategies
    include chaining where each bucket contains a linked list and open
    addressing where collisions probe for empty slots. The load factor ratio
    of elements to buckets determines performance degradation.""",

    """CRISPR-Cas9 is a gene editing technology derived from bacterial immune
    systems. Bacteria use CRISPR arrays to store fragments of viral DNA and
    Cas proteins to cut matching sequences in future infections. Jennifer
    Doudna and Emmanuelle Charpentier adapted this system for programmable
    gene editing by designing guide RNA sequences.""",
]

# QA documents (separate from training) with questions and gold answers.
# Mix of base-model-knowable facts and document-specific details.
QA_DOCUMENTS = [
    {
        "id": "photosynthesis",
        "text": (
            "Photosynthesis occurs in the chloroplasts of plant cells. The light-dependent "
            "reactions take place in the thylakoid membranes where chlorophyll absorbs photons "
            "and splits water molecules into oxygen, protons, and electrons. The Calvin cycle "
            "occurs in the stroma and uses carbon dioxide along with ATP and NADPH from the "
            "light reactions to produce glyceraldehyde-3-phosphate, a three-carbon sugar. "
            "A single chloroplast contains between ten and one hundred grana stacks. "
            "The enzyme RuBisCO catalyzes the first step of carbon fixation in the Calvin "
            "cycle and is considered the most abundant protein on Earth."
        ),
        "questions": [
            {
                "q": "Where do the light-dependent reactions of photosynthesis take place?",
                "a": "thylakoid membranes",
                "type": "factual",
            },
            {
                "q": "What enzyme catalyzes the first step of carbon fixation?",
                "a": "RuBisCO",
                "type": "factual",
            },
            {
                "q": "How many grana stacks does a single chloroplast contain?",
                "a": "between ten and one hundred",
                "type": "specific_detail",
            },
        ],
    },
    {
        "id": "napoleon",
        "text": (
            "Napoleon Bonaparte was born in Corsica in 1769 and rose to prominence during "
            "the French Revolution. He crowned himself Emperor in 1804 at Notre-Dame Cathedral "
            "in Paris. His military campaigns reshaped the map of Europe through victories at "
            "Austerlitz in 1805, Jena in 1806, and Wagram in 1809. The disastrous invasion "
            "of Russia in 1812 began with an army of over six hundred thousand soldiers. Only "
            "about twenty-seven thousand effective soldiers returned. He was exiled to Elba "
            "in 1814, escaped and ruled for the Hundred Days before final defeat at Waterloo "
            "in June 1815. He died on Saint Helena in 1821."
        ),
        "questions": [
            {
                "q": "Where was Napoleon born?",
                "a": "Corsica",
                "type": "factual",
            },
            {
                "q": "How many soldiers did Napoleon begin the Russian invasion with?",
                "a": "over six hundred thousand",
                "type": "specific_detail",
            },
            {
                "q": "Where was Napoleon exiled after his first abdication?",
                "a": "Elba",
                "type": "factual",
            },
        ],
    },
    {
        "id": "dna_replication",
        "text": (
            "DNA replication is a semiconservative process where each strand of the double "
            "helix serves as a template for a new complementary strand. Helicase unwinds the "
            "double helix at the replication fork. Primase synthesizes short RNA primers that "
            "provide the three-prime hydroxyl group needed by DNA polymerase III to begin "
            "synthesis. The leading strand is synthesized continuously in the five-prime to "
            "three-prime direction. The lagging strand is synthesized discontinuously as "
            "Okazaki fragments of approximately one thousand to two thousand nucleotides each. "
            "DNA ligase joins these fragments. In E. coli, the replication fork moves at "
            "approximately one thousand nucleotides per second."
        ),
        "questions": [
            {
                "q": "What enzyme unwinds the double helix at the replication fork?",
                "a": "helicase",
                "type": "factual",
            },
            {
                "q": "How long are Okazaki fragments?",
                "a": "one thousand to two thousand nucleotides",
                "type": "specific_detail",
            },
            {
                "q": "How fast does the replication fork move in E. coli?",
                "a": "one thousand nucleotides per second",
                "type": "specific_detail",
            },
        ],
    },
    {
        "id": "mars_rover",
        "text": (
            "The Perseverance rover landed on Mars on February 18, 2021 in Jezero Crater, "
            "a 45-kilometer-wide impact basin that once contained a lake fed by a river delta. "
            "Perseverance carries seven scientific instruments including SHERLOC, a deep "
            "ultraviolet Raman spectrometer for detecting organic compounds, and PIXL, an "
            "X-ray fluorescence spectrometer for elemental analysis. The rover weighs "
            "approximately 1025 kilograms and is powered by a multi-mission radioisotope "
            "thermoelectric generator containing 4.8 kilograms of plutonium-238. "
            "Ingenuity, the 1.8-kilogram helicopter, flew alongside Perseverance and "
            "completed 72 flights before its mission ended in January 2024."
        ),
        "questions": [
            {
                "q": "Where did the Perseverance rover land on Mars?",
                "a": "Jezero Crater",
                "type": "factual",
            },
            {
                "q": "How many flights did the Ingenuity helicopter complete?",
                "a": "72",
                "type": "specific_detail",
            },
            {
                "q": "What instrument does Perseverance use for detecting organic compounds?",
                "a": "SHERLOC",
                "type": "specific_detail",
            },
        ],
    },
    {
        "id": "coffee_chemistry",
        "text": (
            "Coffee contains over one thousand chemical compounds that contribute to its "
            "flavor and aroma. Caffeine, a purine alkaloid, blocks adenosine receptors in "
            "the brain and typically has a half-life of five to six hours in healthy adults. "
            "Chlorogenic acids make up six to twelve percent of dry weight in green coffee "
            "beans and decompose during roasting to form quinic acid and caffeic acid. The "
            "Maillard reaction between amino acids and reducing sugars during roasting above "
            "150 degrees Celsius produces melanoidins that give coffee its brown color. "
            "Arabica beans contain approximately 1.2 percent caffeine while Robusta beans "
            "contain about 2.2 percent caffeine."
        ),
        "questions": [
            {
                "q": "What is the half-life of caffeine in healthy adults?",
                "a": "five to six hours",
                "type": "specific_detail",
            },
            {
                "q": "What temperature does the Maillard reaction occur above during roasting?",
                "a": "150 degrees Celsius",
                "type": "specific_detail",
            },
            {
                "q": "How much caffeine do Robusta beans contain?",
                "a": "about 2.2 percent",
                "type": "specific_detail",
            },
        ],
    },
    {
        "id": "plate_tectonics",
        "text": (
            "The theory of plate tectonics describes the large-scale motion of Earth's "
            "lithosphere. The lithosphere is divided into seven major plates and several "
            "minor plates. The Pacific Plate moves northwest at approximately seven centimeters "
            "per year. At divergent boundaries, magma rises to create new oceanic crust at "
            "mid-ocean ridges. The Mid-Atlantic Ridge extends approximately sixteen thousand "
            "kilometers from the Arctic to the Southern Ocean. At convergent boundaries, "
            "oceanic crust subducts beneath continental crust at angles between thirty and "
            "seventy degrees. The Mariana Trench, the deepest point in the ocean at approximately "
            "eleven kilometers, was formed by the Pacific Plate subducting beneath the "
            "Philippine Sea Plate."
        ),
        "questions": [
            {
                "q": "How fast does the Pacific Plate move?",
                "a": "approximately seven centimeters per year",
                "type": "specific_detail",
            },
            {
                "q": "How long is the Mid-Atlantic Ridge?",
                "a": "approximately sixteen thousand kilometers",
                "type": "specific_detail",
            },
            {
                "q": "What formed the Mariana Trench?",
                "a": "the Pacific Plate subducting beneath the Philippine Sea Plate",
                "type": "factual",
            },
        ],
    },
    {
        "id": "silk_road",
        "text": (
            "The Silk Road was a network of trade routes connecting China to the Mediterranean "
            "spanning approximately six thousand four hundred kilometers. It was established "
            "during the Han Dynasty around 130 BC when Zhang Qian was sent as an envoy to "
            "Central Asia. Beyond silk, traders exchanged spices, precious metals, glassware, "
            "and paper. The city of Samarkand in modern Uzbekistan was a major trading hub. "
            "Caravanserais, roadside inns spaced approximately thirty kilometers apart, "
            "provided shelter for merchants and their animals. The Silk Road also facilitated "
            "the spread of Buddhism from India to China and the transmission of the Black "
            "Death from Central Asia to Europe in the fourteenth century."
        ),
        "questions": [
            {
                "q": "How long was the Silk Road?",
                "a": "approximately six thousand four hundred kilometers",
                "type": "specific_detail",
            },
            {
                "q": "Who was sent as an envoy to Central Asia during the Han Dynasty?",
                "a": "Zhang Qian",
                "type": "factual",
            },
            {
                "q": "How far apart were caravanserais spaced?",
                "a": "approximately thirty kilometers",
                "type": "specific_detail",
            },
        ],
    },
]


# ── F1 Scoring ──────────────────────────────────────────────────────────────

def normalize_text(text):
    """Lower, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction, gold):
    """SQuAD-style token-level F1."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Diverse Text Passages for Training ──────────────────────────────────────

def prepare_train_data(tokenizer, n_train=40, n_test=10):
    """Tokenize training passages and chunk into CTX_LEN segments."""
    all_tokens = []
    for passage in TRAIN_PASSAGES:
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


# ── Memory Extractor (from S1, Finding #482) ───────────────────────────────

class MemoryExtractor(nn.Module):
    def __init__(self, text_model, num_mem_tokens: int = N_MEM_TOKENS):
        super().__init__()
        self.text_model = text_model
        self.num_mem_tokens = num_mem_tokens
        hidden_size = text_model.config.hidden_size
        mx.random.seed(SEED)
        self.mem_tokens = mx.random.normal(shape=(1, num_mem_tokens, hidden_size)) * 0.02

    def extract(self, input_ids: mx.array):
        """Returns memory_states: (L, M, d) as float16."""
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

        result = mx.stack(memory_states, axis=0).squeeze(1)
        return result.astype(mx.float16)


# ── M2P with Multi-Projection (q+v+o) ──────────────────────────────────────

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


class M2PMultiProjection(nn.Module):
    """Maps memory states (L, M, d) to LoRA for q, v, o projections."""

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
        self.q_projs, self._q_idx, self._q_dims = self._build_proj_group(
            [input_dim] * n_layers, layer_qproj_dims, r, m2p_dim
        )
        self.v_projs, self._v_idx, self._v_dims = self._build_proj_group(
            [input_dim] * n_layers, layer_vproj_dims, r, m2p_dim
        )
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


# ── LoRA Injection ──────────────────────────────────────────────────────────

class LoRAProxy:
    def __init__(self, base_linear, A, B):
        self.base = base_linear
        self.A = A
        self.B = B
        if hasattr(base_linear, "weight"):
            self.weight = base_linear.weight

    def __call__(self, x):
        return self.base(x) + (x @ self.A @ self.B)


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


# ── Training ────────────────────────────────────────────────────────────────

def compute_ce(model, input_ids):
    logits = model(input_ids)
    targets = input_ids[:, 1:]
    logits = logits[:, :-1]
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
    )


def train_m2p(m2p, model, text_model, train_memory, train_chunks):
    """Train M2P on reconstruction task (same as S2 but with q+v+o)."""
    log(f"\n=== Training M2P ({N_STEPS} steps, q+v+o multi-projection) ===")

    optimizer = opt.Adam(learning_rate=LR)
    n_train = len(train_chunks)
    losses = []

    def loss_fn(m2p, memory_states, input_ids):
        q_lora, v_lora, o_lora = m2p(memory_states)
        originals = inject_multi_lora(text_model, q_lora, v_lora, o_lora)
        loss = compute_ce(model, input_ids)
        restore_multi_lora(text_model, originals)
        return loss

    loss_and_grad = nn.value_and_grad(m2p, loss_fn)

    t0 = time.time()
    gc.disable()
    for step in range(N_STEPS):
        idx = np.random.randint(0, n_train)
        loss, grads = loss_and_grad(m2p, train_memory[idx], train_chunks[idx])
        optimizer.update(m2p, grads)
        mx.eval(loss, m2p.parameters(), optimizer.state)
        losses.append(loss.item())

        if (step + 1) % 100 == 0 or step == 0:
            elapsed = time.time() - t0
            log(f"  Step {step+1}/{N_STEPS}: loss={losses[-1]:.4f}, "
                f"elapsed={elapsed:.1f}s, {elapsed/(step+1)*1000:.0f}ms/step")
            log_memory(f"step-{step+1}")

        del loss, grads

    gc.enable()
    train_time = time.time() - t0
    log(f"  Training done in {train_time:.1f}s ({train_time/N_STEPS*1000:.1f}ms/step)")
    return losses, train_time


# ── CE Evaluation ───────────────────────────────────────────────────────────

def evaluate_ce(m2p, model, text_model, memory_list, chunks, label=""):
    """Evaluate CE with and without generated LoRA."""
    log(f"\n=== CE Evaluate: {label} ({len(chunks)} chunks) ===")
    base_ces, adapted_ces = [], []

    for i in range(len(chunks)):
        input_ids = chunks[i]

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
        q_lora, v_lora, o_lora = m2p(memory_list[i])
        originals = inject_multi_lora(text_model, q_lora, v_lora, o_lora)
        adapted_logits = model(input_ids)
        adapted_ce = nn.losses.cross_entropy(
            adapted_logits[:, :-1].reshape(B * T, V),
            targets.reshape(B * T), reduction="mean"
        )
        mx.eval(adapted_ce)
        adapted_ces.append(adapted_ce.item())
        restore_multi_lora(text_model, originals)

        del base_logits, adapted_logits
        mx.clear_cache()

    mean_base = sum(base_ces) / len(base_ces)
    mean_adapted = sum(adapted_ces) / len(adapted_ces)
    ratio = mean_adapted / mean_base if mean_base > 0 else float("inf")
    log(f"  Base CE: {mean_base:.4f}, Adapted CE: {mean_adapted:.4f}, Ratio: {ratio:.4f}")
    return {"base_ce": mean_base, "adapted_ce": mean_adapted, "ratio": ratio}


# ── Text Generation ─────────────────────────────────────────────────────────

def generate_text(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS):
    """Greedy decode with stop at newline or EOS."""
    input_ids = mx.array([tokenizer.encode(prompt)])
    generated = []

    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)
        tok_id = next_token.item()

        if tok_id == tokenizer.eos_token_id:
            break
        text = tokenizer.decode([tok_id])
        if "\n" in text and len(generated) > 3:
            break
        generated.append(tok_id)
        input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

    del logits
    mx.clear_cache()
    return tokenizer.decode(generated)


# ── QA Evaluation ───────────────────────────────────────────────────────────

def make_qa_prompt(question, with_document=None):
    """Format QA prompt. Gemma 4 instruct format."""
    if with_document:
        return (
            f"<start_of_turn>user\n"
            f"Read the following passage and answer the question briefly.\n\n"
            f"Passage: {with_document}\n\n"
            f"Question: {question}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"Answer: "
        )
    else:
        return (
            f"<start_of_turn>user\n"
            f"Answer the following question briefly.\n\n"
            f"Question: {question}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"Answer: "
        )


def evaluate_qa(m2p, model, text_model, extractor, tokenizer):
    """Run QA evaluation across all documents and conditions."""
    log("\n=== QA Evaluation ===")

    results_per_doc = []
    all_no_adapter = []
    all_shine = []
    all_icl = []
    gen_times = []

    n_docs = len(QA_DOCUMENTS) if not SMOKE_TEST else 3

    for doc_idx in range(n_docs):
        doc = QA_DOCUMENTS[doc_idx]
        log(f"\n  --- Document: {doc['id']} ---")

        # Encode document and generate session LoRA
        doc_tokens = mx.array([tokenizer.encode(doc["text"])])
        t_gen = time.time()
        memory_states = extractor.extract(doc_tokens)
        mx.eval(memory_states)
        q_lora, v_lora, o_lora = m2p(memory_states)
        # Force eval of all LoRA weights
        for pairs in [q_lora, v_lora, o_lora]:
            for a, b in pairs:
                mx.eval(a, b)
        gen_time = time.time() - t_gen
        gen_times.append(gen_time)
        log(f"    Adapter generation: {gen_time:.3f}s")

        del doc_tokens
        mx.clear_cache()

        doc_results = {"doc_id": doc["id"], "gen_time_s": gen_time, "questions": []}

        for qa in doc["questions"]:
            q_text = qa["q"]
            gold = qa["a"]

            # Condition 1: No adapter (base model, no document)
            prompt_no_doc = make_qa_prompt(q_text)
            ans_no_adapter = generate_text(model, tokenizer, prompt_no_doc)
            f1_no_adapter = token_f1(ans_no_adapter, gold)

            # Condition 2: SHINE (LoRA applied, no document)
            originals = inject_multi_lora(text_model, q_lora, v_lora, o_lora)
            ans_shine = generate_text(model, tokenizer, prompt_no_doc)
            restore_multi_lora(text_model, originals)
            f1_shine = token_f1(ans_shine, gold)

            # Condition 3: ICL (document in prompt, no LoRA)
            prompt_icl = make_qa_prompt(q_text, with_document=doc["text"])
            ans_icl = generate_text(model, tokenizer, prompt_icl)
            f1_icl = token_f1(ans_icl, gold)

            log(f"    Q: {q_text[:60]}...")
            log(f"      Gold: {gold}")
            log(f"      No-adapter: F1={f1_no_adapter:.3f} | {ans_no_adapter[:80]}")
            log(f"      SHINE:      F1={f1_shine:.3f} | {ans_shine[:80]}")
            log(f"      ICL:        F1={f1_icl:.3f} | {ans_icl[:80]}")

            all_no_adapter.append(f1_no_adapter)
            all_shine.append(f1_shine)
            all_icl.append(f1_icl)

            doc_results["questions"].append({
                "question": q_text,
                "gold": gold,
                "type": qa["type"],
                "no_adapter": {"answer": ans_no_adapter, "f1": f1_no_adapter},
                "shine": {"answer": ans_shine, "f1": f1_shine},
                "icl": {"answer": ans_icl, "f1": f1_icl},
            })

            mx.clear_cache()

        results_per_doc.append(doc_results)

    # Aggregate
    mean_f1_no_adapter = sum(all_no_adapter) / len(all_no_adapter)
    mean_f1_shine = sum(all_shine) / len(all_shine)
    mean_f1_icl = sum(all_icl) / len(all_icl)
    mean_gen_time = sum(gen_times) / len(gen_times)

    # Split by question type
    factual_shine = [f for f, qa in zip(all_shine,
        [q for d in QA_DOCUMENTS[:n_docs] for q in d["questions"]])
        if qa["type"] == "factual"]
    detail_shine = [f for f, qa in zip(all_shine,
        [q for d in QA_DOCUMENTS[:n_docs] for q in d["questions"]])
        if qa["type"] == "specific_detail"]

    log(f"\n=== QA Results Summary ===")
    log(f"  No-adapter F1: {mean_f1_no_adapter:.3f}")
    log(f"  SHINE F1:      {mean_f1_shine:.3f}")
    log(f"  ICL F1:        {mean_f1_icl:.3f}")
    log(f"  SHINE/ICL:     {mean_f1_shine/mean_f1_icl:.3f}" if mean_f1_icl > 0 else "  ICL=0")
    log(f"  Mean gen time: {mean_gen_time:.3f}s")
    if factual_shine:
        log(f"  Factual F1:    {sum(factual_shine)/len(factual_shine):.3f}")
    if detail_shine:
        log(f"  Detail F1:     {sum(detail_shine)/len(detail_shine):.3f}")

    return {
        "mean_f1_no_adapter": mean_f1_no_adapter,
        "mean_f1_shine": mean_f1_shine,
        "mean_f1_icl": mean_f1_icl,
        "shine_over_icl": mean_f1_shine / mean_f1_icl if mean_f1_icl > 0 else 0,
        "mean_gen_time_s": mean_gen_time,
        "n_questions": len(all_shine),
        "factual_f1": sum(factual_shine) / len(factual_shine) if factual_shine else 0,
        "detail_f1": sum(detail_shine) / len(detail_shine) if detail_shine else 0,
        "per_document": results_per_doc,
        "all_f1_no_adapter": all_no_adapter,
        "all_f1_shine": all_shine,
        "all_f1_icl": all_icl,
        "gen_times": gen_times,
    }


# ── Context Specificity ────────────────────────────────────────────────────

def evaluate_specificity(m2p, extractor, tokenizer):
    """Check LoRA diversity across QA documents."""
    log("\n=== Context Specificity (QA documents) ===")
    lora_vecs = []
    n = min(7, len(QA_DOCUMENTS))
    for i in range(n):
        doc_tokens = mx.array([tokenizer.encode(QA_DOCUMENTS[i]["text"])])
        mem = extractor.extract(doc_tokens)
        mx.eval(mem)
        q_lora, v_lora, o_lora = m2p(mem)
        flat = mx.concatenate(
            [mx.concatenate([a.reshape(-1), b.reshape(-1)])
             for a, b in q_lora + v_lora + o_lora]
        )
        mx.eval(flat)
        lora_vecs.append(flat)
        del doc_tokens, mem
        mx.clear_cache()

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
    return {"mean_cosine": mean_cos, "max_cosine": max_cos}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    mx.random.seed(SEED)
    np.random.seed(SEED)
    log("SHINE S4: Document QA Without Context in Prompt (vs ICL)")
    log("Architecture: S2 + multi-projection (no meta LoRA)")
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

    # --- Phase 1: Prepare training data ---
    log("\n=== Phase 1: Prepare Training Data ===")
    n_train = 40 if not SMOKE_TEST else 10
    n_test = 10 if not SMOKE_TEST else 5
    train_chunks, test_chunks = prepare_train_data(tokenizer, n_train, n_test)

    # --- Phase 2: Pre-extract memory states ---
    log("\n=== Phase 2: Pre-extract Memory States ===")
    extractor = MemoryExtractor(text_model)

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
    log_memory("after-extract")

    # --- Phase 3: Probe dimensions and build M2P ---
    log("\n=== Phase 3: Build M2P (multi-projection) ===")
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

    m2p = M2PMultiProjection(
        n_layers=n_layers, n_mem_tokens=N_MEM_TOKENS,
        input_dim=hidden_dim,
        layer_qproj_dims=layer_qproj_dims,
        layer_vproj_dims=layer_vproj_dims,
        layer_oproj_input_dims=layer_oproj_input_dims,
        m2p_dim=M2P_DIM, lora_rank=LORA_RANK,
        n_blocks=M2P_BLOCKS, n_heads=M2P_HEADS,
    )
    mx.eval(m2p.parameters())
    m2p_params = sum(p.size for _, p in nn.utils.tree_flatten(m2p.parameters()))
    log(f"  M2P parameters: {m2p_params:,}")
    log_memory("after-m2p-build")

    # --- Phase 4: Train M2P ---
    losses, train_time = train_m2p(m2p, model, text_model, train_memory, train_chunks)
    log_memory("after-train")

    # --- Phase 5: CE evaluation (sanity check) ---
    test_ce = evaluate_ce(m2p, model, text_model, test_memory, test_chunks, "Test")

    # --- Phase 6: QA evaluation ---
    qa_results = evaluate_qa(m2p, model, text_model, extractor, tokenizer)

    # --- Phase 7: Context specificity ---
    specificity = evaluate_specificity(m2p, extractor, tokenizer)
    log_memory("after-eval")

    # --- Kill Criteria ---
    k1261_pass = qa_results["mean_f1_shine"] > 0.30
    k1262_pass = qa_results["shine_over_icl"] >= 0.50
    k1263_pass = qa_results["mean_gen_time_s"] < 5.0

    initial_loss = float(np.mean(losses[:10]))
    final_loss = float(np.mean(losses[-10:]))
    loss_decrease = (initial_loss - final_loss) / initial_loss * 100

    total_time = time.time() - t_start

    results = {
        "experiment": "exp_shine_gemma4_qa_benchmark",
        "type": "frontier_extension",
        "total_time_s": round(total_time, 1),
        "model": "gemma-4-e4b-it-4bit",
        "architecture": "S2 + multi-projection (no meta LoRA)",
        "m2p": {
            "params": m2p_params,
            "dim": M2P_DIM,
            "blocks": M2P_BLOCKS,
            "heads": M2P_HEADS,
            "lora_rank": LORA_RANK,
            "projections": "q+v+o",
        },
        "training": {
            "n_steps": N_STEPS,
            "lr": LR,
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(final_loss, 4),
            "loss_decrease_pct": round(loss_decrease, 2),
            "train_time_s": round(train_time, 1),
            "ms_per_step": round(train_time / N_STEPS * 1000, 1),
        },
        "ce_evaluation": test_ce,
        "qa_evaluation": {
            "mean_f1_no_adapter": qa_results["mean_f1_no_adapter"],
            "mean_f1_shine": qa_results["mean_f1_shine"],
            "mean_f1_icl": qa_results["mean_f1_icl"],
            "shine_over_icl": qa_results["shine_over_icl"],
            "mean_gen_time_s": qa_results["mean_gen_time_s"],
            "n_questions": qa_results["n_questions"],
            "factual_f1": qa_results["factual_f1"],
            "detail_f1": qa_results["detail_f1"],
            "per_document": qa_results["per_document"],
        },
        "context_specificity": specificity,
        "kill_criteria": {
            "K1261": {
                "criterion": "QA F1 > 30% without document in prompt",
                "measured": round(qa_results["mean_f1_shine"], 4),
                "pass": bool(k1261_pass),
            },
            "K1262": {
                "criterion": "QA F1 >= 50% of ICL baseline",
                "measured": round(qa_results["shine_over_icl"], 4),
                "pass": bool(k1262_pass),
            },
            "K1263": {
                "criterion": "Session adapter generation < 5s",
                "measured": round(qa_results["mean_gen_time_s"], 4),
                "pass": bool(k1263_pass),
            },
        },
        "predictions": {
            "P1_f1_below_10pct": qa_results["mean_f1_shine"] < 0.10,
            "P2_factual_near_icl": (
                abs(qa_results["factual_f1"] - qa_results["mean_f1_icl"]) < 0.2
                if qa_results["mean_f1_icl"] > 0 else False
            ),
            "P3_detail_below_icl": qa_results["detail_f1"] < qa_results["mean_f1_icl"] * 0.5,
            "P4_gen_under_5s": qa_results["mean_gen_time_s"] < 5.0,
            "P5_ce_ratio_under_020": test_ce["ratio"] < 0.20,
        },
        "all_pass": bool(k1261_pass and k1262_pass and k1263_pass),
        "status": "supported" if (k1261_pass and k1262_pass and k1263_pass) else "killed",
    }

    log(f"\n{'='*70}")
    log(f"RESULTS: SHINE S4 QA Benchmark")
    log(f"{'='*70}")
    log(f"Architecture: S2 + multi-projection q+v+o, {m2p_params:,} params")
    log(f"Training: {N_STEPS} steps, {train_time:.1f}s, loss {initial_loss:.2f} -> {final_loss:.2f}")
    log(f"CE ratio (test): {test_ce['ratio']:.4f}")
    log(f"")
    log(f"QA F1 Results:")
    log(f"  No-adapter: {qa_results['mean_f1_no_adapter']:.3f}")
    log(f"  SHINE:      {qa_results['mean_f1_shine']:.3f}")
    log(f"  ICL:        {qa_results['mean_f1_icl']:.3f}")
    log(f"  SHINE/ICL:  {qa_results['shine_over_icl']:.3f}")
    log(f"")
    log(f"Context specificity: mean_cos={specificity['mean_cosine']:.4f}")
    log(f"Mean adapter gen time: {qa_results['mean_gen_time_s']:.3f}s")
    log(f"")
    log(f"K1261 (F1 > 30%): {'PASS' if k1261_pass else 'FAIL'} ({qa_results['mean_f1_shine']:.3f})")
    log(f"K1262 (>= 50% ICL): {'PASS' if k1262_pass else 'FAIL'} ({qa_results['shine_over_icl']:.3f})")
    log(f"K1263 (gen < 5s): {'PASS' if k1263_pass else 'FAIL'} ({qa_results['mean_gen_time_s']:.3f}s)")
    log(f"")
    log(f"Status: {results['status'].upper()}")
    log(f"Total time: {total_time:.1f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nResults saved to {RESULTS_FILE}")

    cleanup(m2p, extractor)
    return results


if __name__ == "__main__":
    main()
