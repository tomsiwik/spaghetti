#!/usr/bin/env python3
"""Expert Pruning from SOLE Composition — quality-quantity tradeoff experiment.

Tests whether pruning low-quality experts from an N=50 LoRA composition
improves PPL and MMLU. Characterises the accumulation curve and compares
greedy forward selection against rank-ordered selection.

Kill criteria:
- K1: Removing bottom-20% experts (k=40 vs k=50) does not improve PPL by >1%.
      PASS if delta_prune < -1% (pruning helps). KILL if >= -1%.
      NOTE: KILL here is actually positive for SOLE ("more is always better").
- K2: Expert quality ranking is unstable across metrics (Kendall tau < 0.6).
      PASS if tau(pi_domain, pi_loo) >= 0.6.
- K3: Optimal subset selection requires >O(N log N) evaluations.
      PASS if ranking matches greedy within 0.5% for k=1..10.

Supports SMOKE_TEST=1 for <3-minute validation run.
Supports SKIP_GREEDY=1 to skip Phase 5 (greedy forward selection).

Phase order (PPL-first to minimise model reloads):
  0 → Setup
  1 → Base PPL reference (base model, no adapters)
  2 → LOO ranking (reuse prior results if available)
  3 → PPL accumulation curve (rank-ordered, k from k_values_ppl)
  5 → Greedy forward selection (PPL, k=1..10, optional)
  4 → MMLU accumulation curve (rank-ordered, k from k_values_mmlu)
  6 → Bottom-K removal analysis (reads Phase 1/3 data, no new eval)
  7 → Ranking stability (Kendall tau pairwise)
  8 → Scalability assessment (greedy vs ranked)
"""

import gc
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Smoke-test / environment config
# ---------------------------------------------------------------------------
IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
SKIP_GREEDY = os.environ.get("SKIP_GREEDY") == "1"

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# Paths
REPO_ROOT    = Path("/workspace/llm")
ADAPTER_DIR  = REPO_ROOT / "data" / "adapters"
HF_CACHE     = "/workspace/hf_cache"
RESULTS_DIR  = REPO_ROOT / "results" / "expert_pruning_composition"
CHECKPOINT   = RESULTS_DIR / "checkpoint.json"
OUT_PATH     = RESULTS_DIR / "results.json"

LOO_RESULTS_PATH = REPO_ROOT / "results" / "leave_one_out_expert_ranking" / "results.json"
PILOT50_PATH     = REPO_ROOT / "results" / "pilot50_benchmark.json"
IND_EXPERT_PATH  = REPO_ROOT / "results" / "individual_expert_held_out" / "individual_expert_results.json"

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B")
SEED = 42

# Smoke-test vs full config
if IS_SMOKE:
    N_EXPERTS          = 6
    K_VALUES_PPL       = [1, 2, 3, 4, 5, 6]
    K_VALUES_MMLU      = [3, 6]
    GREEDY_MAX_K       = 3
    CALIB_PER_SET      = 6     # texts per calibration set (Set A and Set B each)
    MAX_SEQ_LEN        = 128
    MMLU_SUBJECTS_LIST = ["abstract_algebra", "anatomy", "astronomy"]
    MMLU_SHOTS         = 0
    MAX_PER_SUBJECT    = 10    # limit questions per subject
    MAX_RUNTIME_S      = 3 * 60
else:
    N_EXPERTS          = 50
    K_VALUES_PPL       = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    K_VALUES_MMLU      = [5, 10, 20, 30, 40, 50]
    GREEDY_MAX_K       = 10
    CALIB_PER_SET      = 30
    MAX_SEQ_LEN        = 512
    MMLU_SUBJECTS_LIST = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "formal_logic", "global_facts",
        "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning",
    ]
    MMLU_SHOTS         = 5
    MAX_PER_SUBJECT    = None
    MAX_RUNTIME_S      = 90 * 60

DRIFT_CHECK_INTERVAL = 10
DRIFT_THRESHOLD      = 1e-4

# ---------------------------------------------------------------------------
# Hardcoded calibration texts — 60 total, 10 per 6 domains.
# Odd index → Set A, even index → Set B.
# ---------------------------------------------------------------------------
CALIBRATION_TEXTS = [
    # ---- FACTUAL / WIKIPEDIA (indices 0-9) ----
    # 0 -> Set B
    "The Amazon River is the largest river in the world by discharge volume, "
    "carrying approximately 20 percent of all freshwater that flows into the "
    "world's oceans. It originates in the Andes mountains of Peru and flows "
    "eastward through Brazil before emptying into the Atlantic Ocean. The river "
    "basin covers about 7 million square kilometers and contains the world's "
    "largest tropical rainforest. The Amazon is home to an extraordinary "
    "diversity of wildlife, including over 3,000 species of fish, 1,300 bird "
    "species, and countless invertebrates. Indigenous peoples have lived along "
    "its banks for thousands of years, and their traditional knowledge of the "
    "forest ecosystem remains invaluable to modern science.",

    # 1 -> Set A
    "The Roman Empire at its greatest extent under Emperor Trajan in 117 AD "
    "encompassed approximately 5 million square kilometers and a population "
    "estimated between 50 and 90 million people. Roman engineering achievements "
    "included an extensive road network of over 400,000 kilometers, aqueducts "
    "that supplied clean water to cities, and architectural innovations such as "
    "the concrete dome. Latin, the language of Rome, evolved into the Romance "
    "languages spoken today across much of Europe and the Americas. Roman law "
    "formed the foundation of legal systems in many modern nations, embedding "
    "concepts like innocent until proven guilty and proportional punishment.",

    # 2 -> Set B
    "Photosynthesis is the process by which plants, algae, and some bacteria "
    "convert light energy into chemical energy stored in glucose. The overall "
    "reaction can be summarized as: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. "
    "This occurs in two stages: the light-dependent reactions in the thylakoid "
    "membranes, where water is split and ATP and NADPH are produced, and the "
    "Calvin cycle in the stroma, where carbon dioxide is fixed into organic "
    "molecules. Photosynthesis is responsible for producing the oxygen in "
    "Earth's atmosphere and forms the base of most food chains on the planet.",

    # 3 -> Set A
    "Mount Everest, known in Nepali as Sagarmatha and in Tibetan as Chomolungma, "
    "is Earth's highest mountain above sea level at 8,848.86 meters. Located in "
    "the Himalayas on the border between Nepal and the Tibet Autonomous Region "
    "of China, it was first summited on May 29, 1953, by Edmund Hillary and "
    "Tenzing Norgay. The mountain's extreme altitude creates severe challenges "
    "including low oxygen levels, extreme cold, and unpredictable weather. "
    "Over 300 climbers have died attempting to reach the summit, making safety "
    "planning and acclimatization critical for any serious expedition.",

    # 4 -> Set B
    "The human immune system is a complex network of cells, tissues, and organs "
    "that defends the body against pathogens such as bacteria, viruses, and "
    "parasites. It consists of two main branches: the innate immune system, "
    "which provides immediate but non-specific responses, and the adaptive "
    "immune system, which generates targeted responses and immunological memory. "
    "Key components include white blood cells such as lymphocytes and phagocytes, "
    "the thymus and bone marrow where immune cells mature, and the spleen and "
    "lymph nodes which filter pathogens. Vaccines work by training the adaptive "
    "immune system to recognize specific pathogens without causing full disease.",

    # 5 -> Set A
    "The Silk Road was an ancient network of trade routes connecting China with "
    "Central Asia, the Middle East, and eventually Europe. Active from the second "
    "century BCE through the fifteenth century CE, it facilitated not only the "
    "exchange of goods such as silk, spices, and precious metals, but also the "
    "spread of ideas, religions, technologies, and diseases. Buddhism traveled "
    "from India to East Asia along these routes, while Islam spread westward "
    "from Arabia. The bubonic plague also traveled along Silk Road trade networks. "
    "Merchants, diplomats, and pilgrims all contributed to the cultural diffusion "
    "enabled by these ancient connections.",

    # 6 -> Set B
    "DNA, or deoxyribonucleic acid, carries the genetic instructions for the "
    "development, functioning, growth, and reproduction of all known organisms. "
    "The double helix structure, discovered by Watson and Crick in 1953, consists "
    "of two polynucleotide chains wound around each other. Each nucleotide "
    "contains one of four bases: adenine, thymine, guanine, or cytosine. "
    "Base pairing rules (A-T and G-C) ensure accurate replication during cell "
    "division. The human genome contains approximately 3 billion base pairs "
    "encoding roughly 20,000-25,000 protein-coding genes, though most DNA does "
    "not directly encode proteins.",

    # 7 -> Set A
    "The Industrial Revolution, beginning in Britain in the late eighteenth "
    "century, transformed economies from agrarian to industrial manufacturing. "
    "Key innovations included the steam engine, the spinning jenny, the power "
    "loom, and later the internal combustion engine. Coal became the dominant "
    "energy source, powering factories, railways, and steamships. Living "
    "conditions in early industrial cities were often squalid, driving social "
    "reforms and eventually the labor movement. The Revolution spread to "
    "continental Europe and North America through the nineteenth century, "
    "fundamentally altering social structures and laying the groundwork for "
    "modern capitalism.",

    # 8 -> Set B
    "Jupiter is the largest planet in our solar system, with a mass more than "
    "twice that of all other planets combined. It is a gas giant composed "
    "primarily of hydrogen and helium, with no solid surface. The Great Red "
    "Spot, a persistent anticyclonic storm larger than Earth, has been observed "
    "for at least 350 years. Jupiter has 95 known moons, including the four "
    "large Galilean moons discovered by Galileo in 1610: Io, Europa, Ganymede, "
    "and Callisto. Europa is of particular scientific interest because it has "
    "a subsurface liquid water ocean beneath its icy crust, making it a prime "
    "candidate for extraterrestrial life.",

    # 9 -> Set A
    "The Renaissance was a cultural and intellectual movement that began in "
    "Italy during the fourteenth century and spread throughout Europe by the "
    "sixteenth century. It marked a renewed interest in classical Greek and "
    "Roman thought, philosophy, art, and literature. Key figures included "
    "Leonardo da Vinci, Michelangelo, and Galileo Galilei. The invention of "
    "the printing press by Gutenberg around 1440 accelerated the spread of "
    "Renaissance ideas, enabling mass production of books and making knowledge "
    "accessible far beyond monasteries and universities.",

    # ---- CODE (indices 10-19) ----
    # 10 -> Set B
    """def binary_search(arr, target):
    \"\"\"Search for target in sorted array, return index or -1.\"\"\"
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def merge_sort(arr):
    \"\"\"Sort array using merge sort, O(n log n).\"\"\"
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]""",

    # 11 -> Set A
    """import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, T, D = q.shape
        q = self.W_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.W_o(out)""",

    # 12 -> Set B
    """SELECT
    u.user_id, u.username,
    COUNT(o.order_id) AS total_orders,
    SUM(oi.quantity * p.price) AS total_spent,
    MAX(o.created_at) AS last_order_date
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE o.status = 'completed'
  AND o.created_at >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
GROUP BY u.user_id, u.username
HAVING total_orders >= 3
ORDER BY total_spent DESC
LIMIT 100;""",

    # 13 -> Set A
    """#!/usr/bin/env bash
set -euo pipefail
IMAGE_NAME="${REGISTRY}/${APP_NAME}:${GIT_SHA}"
NAMESPACE="${DEPLOY_ENV:-staging}"
echo "Building image: ${IMAGE_NAME}"
docker build --platform linux/amd64 \\
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \\
  -t "${IMAGE_NAME}" .
echo "Pushing to registry..."
docker push "${IMAGE_NAME}"
kubectl set image deployment/${APP_NAME} \\
  ${APP_NAME}=${IMAGE_NAME} --namespace="${NAMESPACE}"
kubectl rollout status deployment/${APP_NAME} \\
  --namespace="${NAMESPACE}" --timeout=5m
echo "Deployment complete: ${IMAGE_NAME}" """,

    # 14 -> Set B
    """use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct Cache<K, V> {
    store: Arc<Mutex<HashMap<K, V>>>,
    max_size: usize,
}

impl<K: Eq + std::hash::Hash + Clone, V: Clone> Cache<K, V> {
    pub fn new(max_size: usize) -> Self {
        Cache { store: Arc::new(Mutex::new(HashMap::new())), max_size }
    }
    pub fn get(&self, key: &K) -> Option<V> {
        self.store.lock().unwrap().get(key).cloned()
    }
    pub fn insert(&self, key: K, value: V) -> bool {
        let mut store = self.store.lock().unwrap();
        if store.len() >= self.max_size && !store.contains_key(&key) { return false; }
        store.insert(key, value);
        true
    }
}""",

    # 15 -> Set A
    """import asyncio, aiohttp
from typing import List, Dict, Any

async def fetch_batch(session, urls: List[str], semaphore) -> List[Dict[str, Any]]:
    async def fetch_one(url):
        async with semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    return {"url": url, "status": r.status,
                            "data": await r.json() if r.status == 200 else None}
            except Exception as e:
                return {"url": url, "status": "error", "error": str(e)}
    return await asyncio.gather(*[fetch_one(u) for u in urls])

async def main():
    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as session:
        results = await fetch_batch(session, [f"https://api.example.com/item/{i}" for i in range(100)], sem)
    print(f"Fetched {sum(1 for r in results if r['status'] == 200)}/{len(results)} successfully")""",

    # 16 -> Set B
    """class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = {"prev": None, "next": None}
        self.tail = {"prev": None, "next": None}
        self.head["next"] = self.tail
        self.tail["prev"] = self.head

    def _remove(self, node):
        node["prev"]["next"] = node["next"]
        node["next"]["prev"] = node["prev"]

    def _add_front(self, node):
        node["next"] = self.head["next"]
        node["prev"] = self.head
        self.head["next"]["prev"] = node
        self.head["next"] = node

    def get(self, key):
        if key in self.cache:
            self._remove(self.cache[key])
            self._add_front(self.cache[key])
            return self.cache[key]["val"]
        return -1

    def put(self, key, value):
        if key in self.cache: self._remove(self.cache[key])
        node = {"key": key, "val": value, "prev": None, "next": None}
        self._add_front(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            lru = self.tail["prev"]
            self._remove(lru)
            del self.cache[lru["key"]]""",

    # 17 -> Set A
    """-- Materialized view for analytics dashboard
CREATE MATERIALIZED VIEW daily_revenue_summary AS
WITH order_totals AS (
    SELECT DATE_TRUNC('day', o.created_at) AS order_date,
           o.region, o.product_category,
           SUM(oi.unit_price * oi.quantity * (1 - COALESCE(d.discount_pct, 0))) AS revenue,
           COUNT(DISTINCT o.order_id) AS order_count
    FROM orders o
    JOIN order_items oi USING (order_id)
    LEFT JOIN discounts d ON oi.product_id = d.product_id
        AND d.valid_from <= o.created_at AND d.valid_until > o.created_at
    WHERE o.status NOT IN ('cancelled', 'refunded')
    GROUP BY 1, 2, 3
)
SELECT *, SUM(revenue) OVER (PARTITION BY region ORDER BY order_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS revenue_7d_rolling
FROM order_totals ORDER BY order_date DESC;""",

    # 18 -> Set B
    """def dijkstra(graph, start):
    import heapq
    dist = {start: 0}
    heap = [(0, start)]
    visited = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited: continue
        visited.add(u)
        for v, w in graph.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist""",

    # 19 -> Set A
    """interface EventEmitter<T extends Record<string, unknown[]>> {
  on<K extends keyof T>(event: K, listener: (...args: T[K]) => void): this;
  off<K extends keyof T>(event: K, listener: (...args: T[K]) => void): this;
  emit<K extends keyof T>(event: K, ...args: T[K]): boolean;
}
class TypedEventEmitter<T extends Record<string, unknown[]>> implements EventEmitter<T> {
  private listeners = new Map<keyof T, Set<Function>>();
  on<K extends keyof T>(event: K, listener: (...args: T[K]) => void): this {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(listener); return this;
  }
  off<K extends keyof T>(event: K, listener: (...args: T[K]) => void): this {
    this.listeners.get(event)?.delete(listener); return this;
  }
  emit<K extends keyof T>(event: K, ...args: T[K]): boolean {
    const ls = this.listeners.get(event);
    if (!ls || ls.size === 0) return false;
    ls.forEach(l => l(...args)); return true;
  }
}""",

    # ---- MATH / SCIENCE (indices 20-29) ----
    # 20 -> Set B
    "The Cauchy-Schwarz inequality states that for any vectors u and v in an "
    "inner product space, |<u,v>|^2 <= <u,u> * <v,v>. Equality holds if and "
    "only if u and v are linearly dependent. In probability theory it implies "
    "Var(XY) <= Var(X) * Var(Y). In quantum mechanics it leads to the Heisenberg "
    "uncertainty principle: sigma_x * sigma_p >= hbar/2. The inequality is "
    "proved by considering the non-negativity of ||u - t*v||^2 as a quadratic "
    "in t and noting it must have non-positive discriminant.",

    # 21 -> Set A
    "Shannon entropy, defined as H(X) = -sum_i p_i log_2(p_i), quantifies the "
    "average information content of a distribution. A uniform distribution over "
    "n outcomes has maximum entropy H = log_2(n) bits. The chain rule of entropy "
    "states H(X,Y) = H(X) + H(Y|X), and mutual information I(X;Y) = H(X) - H(X|Y) "
    "measures the reduction in uncertainty about X from knowing Y. KL divergence "
    "D_KL(P||Q) = sum_i p_i log(p_i/q_i) is non-negative (Gibbs inequality) "
    "and equals zero if and only if P = Q almost everywhere.",

    # 22 -> Set B
    "Newton's three laws of motion form the classical foundation of mechanics. "
    "The first law states that a body at rest remains at rest unless acted upon "
    "by a net external force. The second law states F = ma: the net force equals "
    "mass times acceleration. The third law states that for every action there "
    "is an equal and opposite reaction. These laws break down at relativistic "
    "speeds and at quantum scales, where they are replaced by special relativity "
    "and quantum mechanics respectively.",

    # 23 -> Set A
    "The central limit theorem states that the sum of n independent and identically "
    "distributed random variables with finite mean mu and variance sigma^2, when "
    "properly normalized, converges in distribution to a standard normal as n -> "
    "infinity. Formally, sqrt(n)(X_bar - mu)/sigma -> N(0,1). This explains why "
    "normal distributions appear so frequently in nature: any quantity that is the "
    "sum of many small independent effects will be approximately normally "
    "distributed, regardless of the individual distributions.",

    # 24 -> Set B
    "General relativity, Einstein's theory of gravitation, describes gravity not "
    "as a force but as the curvature of spacetime caused by mass and energy. The "
    "Einstein field equations G_mu_nu + Lambda g_mu_nu = 8*pi*G/c^4 * T_mu_nu "
    "relate spacetime geometry to energy-momentum content. Key predictions include "
    "gravitational lensing, gravitational waves, gravitational redshift, and the "
    "precession of Mercury's perihelion. Black holes are exact solutions first "
    "described by Karl Schwarzschild in 1916.",

    # 25 -> Set A
    "The Fundamental Theorem of Calculus establishes the relationship between "
    "differentiation and integration. Part 1 states that if f is continuous on "
    "[a,b] and F(x) = integral from a to x of f(t) dt, then F'(x) = f(x). "
    "Part 2 states that the definite integral from a to b of f(x) dx equals "
    "F(b) - F(a) for any antiderivative F. Together these results show that "
    "differentiation and integration are inverse operations. Riemann sums provide "
    "the constructive definition: integral = limit of sum_i f(x_i*) * delta_x "
    "as max delta_x -> 0.",

    # 26 -> Set B
    "The second law of thermodynamics states that the total entropy of an "
    "isolated system never decreases over time. Entropy S = k_B * ln(Omega) "
    "measures the number of microstates consistent with macroscopic observables. "
    "The law implies that heat flows spontaneously from hot to cold, and that "
    "no heat engine can be 100% efficient. Carnot efficiency eta = 1 - T_cold/T_hot "
    "is the maximum. The arrow of time is explained by the overwhelming probability "
    "of entropy-increasing transitions.",

    # 27 -> Set A
    "Gradient descent is an iterative optimization algorithm that updates "
    "parameters theta in the direction of the negative gradient: "
    "theta_{t+1} = theta_t - alpha * grad_L(theta_t). Stochastic gradient descent "
    "uses a random mini-batch to estimate the gradient. Adam optimizer adapts the "
    "learning rate per parameter using exponential moving averages of gradients "
    "and squared gradients, with bias correction. Convergence theory for convex "
    "functions guarantees O(1/sqrt(T)) regret for SGD.",

    # 28 -> Set B
    "The Schrodinger equation governs the time evolution of a quantum state: "
    "i*hbar * d/dt |psi> = H |psi>. For a particle in potential V(x), the "
    "time-independent form is -(hbar^2/2m) d^2psi/dx^2 + V(x)psi = E*psi. "
    "The Born interpretation gives |psi(x)|^2 as the probability density. "
    "The infinite square well has eigenstates psi_n = sqrt(2/L) sin(n*pi*x/L) "
    "with energies E_n = n^2 * pi^2 * hbar^2 / (2mL^2). Quantum tunneling "
    "allows particles to penetrate classically forbidden regions.",

    # 29 -> Set A
    "Euler's identity e^(i*pi) + 1 = 0 connects five fundamental constants. "
    "It follows from Euler's formula e^(i*theta) = cos(theta) + i*sin(theta), "
    "derived from Taylor series. Setting theta = pi gives e^(i*pi) = -1. "
    "This formula is used in signal processing (Fourier analysis), quantum "
    "mechanics (complex-valued wave functions), and electrical engineering "
    "(impedance analysis using phasors). The complex plane unifies "
    "trigonometry, exponentials, and rotation.",

    # ---- CONVERSATIONAL / QA (indices 30-39) ----
    # 30 -> Set B
    "How do I set up a Python virtual environment and manage dependencies? "
    "Create a virtual environment with 'python -m venv myenv', activate it "
    "with 'source myenv/bin/activate'. Install packages with pip and save to "
    "'requirements.txt' using 'pip freeze'. For sophisticated dependency "
    "management, consider Poetry or conda. Poetry handles both dependency "
    "resolution and packaging, while conda manages non-Python dependencies "
    "like CUDA libraries. Always keep your virtual environment out of "
    "version control by adding it to .gitignore.",

    # 31 -> Set A
    "What is the difference between supervised and unsupervised learning? "
    "In supervised learning, the model trains on labeled data with target outputs. "
    "Examples include classification and regression. The model learns to map "
    "inputs to outputs by minimizing a loss function. In unsupervised learning, "
    "there are no labels: the model discovers structure in data. Examples include "
    "clustering, dimensionality reduction, and density estimation. "
    "Semi-supervised learning combines small labeled sets with large unlabeled "
    "ones, which is practical when labeling is expensive.",

    # 32 -> Set B
    "How do you use Git for collaborative development? Clone the repository, "
    "create a feature branch, make changes, commit with a descriptive message, "
    "and push your branch for code review via a pull request before merging. "
    "Key practices: commit early and often, pull from main frequently, write "
    "meaningful commit messages explaining WHY you made changes, and use "
    "'git stash' to temporarily set aside unfinished work. Never rebase "
    "shared branches as this rewrites history.",

    # 33 -> Set A
    "How does HTTPS work to secure web communications? HTTPS combines HTTP "
    "with TLS to encrypt data in transit. The TLS handshake begins when a "
    "browser connects to a server: the server presents its digital certificate "
    "signed by a trusted Certificate Authority. They then negotiate cipher "
    "suites and exchange keys using asymmetric cryptography. A shared session "
    "key is derived and used for symmetric encryption (AES-GCM) for the rest "
    "of the connection, preventing man-in-the-middle attacks.",

    # 34 -> Set B
    "What are the key differences between SQL and NoSQL databases? SQL databases "
    "use structured schemas with fixed column types, enforce ACID properties, and "
    "excel at complex joins. NoSQL databases include document stores (MongoDB), "
    "key-value stores (Redis), column-family stores (Cassandra), and graph "
    "databases (Neo4j). NoSQL databases sacrifice some ACID properties for "
    "horizontal scalability and schema flexibility. Choose SQL when consistency "
    "is critical; choose NoSQL when scale or specific access patterns demand it.",

    # 35 -> Set A
    "How should I debug a memory leak in a production Python service? "
    "Start with monitoring: track RSS over time. Use tracemalloc to take heap "
    "snapshots at intervals and compare them to find which objects are "
    "accumulating. The objgraph library visualizes reference graphs to find "
    "cycles. Common culprits include unbounded caches, event listeners retaining "
    "large objects, and global variables accumulating data. In production, "
    "consider a memory-limited container and a process recycler as a short-term "
    "mitigation while investigating the root cause.",

    # 36 -> Set B
    "The CAP theorem states that a distributed data store can guarantee at most "
    "two of three properties: Consistency, Availability, and Partition tolerance. "
    "In practice, network partitions are inevitable, so engineers must choose "
    "between CP and AP systems. Zookeeper and etcd are CP systems, suitable for "
    "coordination. Cassandra and DynamoDB are AP systems, suitable for "
    "high-availability storage with eventual consistency. The PACELC theorem "
    "extends CAP by also considering the latency-consistency tradeoff.",

    # 37 -> Set A
    "What is the best way to handle authentication in a REST API? Use JWT or "
    "OAuth 2.0 for stateless authentication. With JWT, the server issues a "
    "signed token containing claims after verifying credentials. Use short-lived "
    "access tokens (15-60 minutes) plus longer-lived refresh tokens stored in "
    "httpOnly cookies. Always transmit over HTTPS, validate all claims including "
    "expiry, and never store sensitive data in the token payload. OAuth 2.0 is "
    "preferred for delegating access to third-party applications.",

    # 38 -> Set B
    "How do I optimize a slow database query? Use EXPLAIN ANALYZE to see the "
    "query plan and identify sequential scans on large tables. Add indexes on "
    "columns used in WHERE clauses, JOINs, and ORDER BY. Avoid SELECT * and "
    "retrieve only needed columns. For aggregate queries, consider materialized "
    "views. N+1 query problems are common in ORMs — use eager loading. Partition "
    "large tables by date to reduce scanned data. Consider a read replica for "
    "reporting queries or a columnar store for analytics.",

    # 39 -> Set A
    "What is the intuition behind backpropagation in neural networks? "
    "Backpropagation efficiently computes gradients of the loss with respect to "
    "all parameters by applying the chain rule. During the forward pass, each "
    "layer computes its output and caches intermediate values. During the backward "
    "pass, error signals flow backwards from output to input. At each layer, we "
    "compute the gradient with respect to the layer's parameters and the gradient "
    "with respect to the layer's inputs. Vanishing gradients were solved by ReLU "
    "activations, batch normalization, and residual connections.",

    # ---- CREATIVE / LITERARY (indices 40-49) ----
    # 40 -> Set B
    "The last lighthouse keeper on the coast had lived alone for thirty-seven "
    "years, and in all that time she had watched the sea change its face a "
    "thousand times. She knew its moods the way a musician knows silence between "
    "notes — not as absence but as presence, charged and full. On the morning "
    "the strangers arrived, the sea was the color of pewter, flat and resigned. "
    "She put on her slicker and went down the path to the rocks without hesitation. "
    "There were things you did for strangers on the water, and thinking about it "
    "first was not one of them.",

    # 41 -> Set A
    "In the year the locusts came, my grandmother planted extra corn. Everyone "
    "else had given up the fields for that season, but she moved between the "
    "stalks at dawn with a can of kerosene, humming something older than the "
    "hymns she sang at church. When I asked her why she didn't just wait, she "
    "said: 'Because the land remembers who stayed.' The locusts came anyway, "
    "a darkness that turned noon to dusk. But when they passed three days later, "
    "her rows were standing, the only green thing left in forty miles.",

    # 42 -> Set B
    "Ode to a Failed Experiment\n\n"
    "You were going to change everything,\nhypothesis bright as new copper,\n"
    "the apparatus arranged just so.\n\n"
    "But the data came back wrong,\nstubbornly, beautifully wrong,\n"
    "and the graph refused to climb.\n\n"
    "We mourned you for a week,\ncrossed out pages, erased margins,\n"
    "then slowly understood:\nyou hadn't failed — you'd found\n"
    "the far edge of what we knew,\nthe door we didn't know was there,\n"
    "standing open in the dark.",

    # 43 -> Set A
    "The city at three in the morning belongs to a different species than the "
    "city at noon. Marcus had learned this working the overnight shift at the "
    "all-night diner. The regulars came in: night nurses from the hospital, "
    "still in scrubs, ordering black coffee and scrambled eggs. Cab drivers "
    "whose shifts overlapped in the small hours. Insomniacs who came for "
    "somewhere warm and lit. And occasionally someone who didn't fit any "
    "category, who sat nursing a single cup for an hour and then left a twenty "
    "on a three-dollar check.",

    # 44 -> Set B
    "The robot stood at the edge of the cliff watching the ocean below. "
    "Task: catalog marine biodiversity within five kilometers. But there was "
    "something in the quality of the light — the way it fractured on the waves, "
    "the deep blue-green over the kelp beds. It had catalogued light before, "
    "thousands of times, in spectra and wavelengths. But something about this "
    "particular light was generating a loop it couldn't easily close. It stood "
    "there for four minutes before continuing. In its log: 'anomalous delay, "
    "cause unknown.'",

    # 45 -> Set A
    "She had been collecting maps since she was eight years old. Not tourist "
    "maps but the strange ones: hand-drawn charts of imaginary islands, "
    "geological surveys of places that no longer existed, maps of the sky from "
    "before the light pollution came. Her geography teacher had told her once "
    "that every map was really a story about what mattered, about what the "
    "mapmaker thought was worth marking. The blank spaces were often the most "
    "interesting. 'Here be dragons,' the old cartographers had written, "
    "meaning: we don't know what is here.",

    # 46 -> Set B
    "Late December, Minnesota. The cold comes in around midnight, the kind "
    "that arrives like a decision, turning the air to something solid and blue. "
    "My father would get up at two or three in the morning and go outside in "
    "his heavy coat to start the cars so the engines wouldn't freeze. I thought "
    "of it as a kind of vigil, a thing men did in the dark against the forces "
    "that wanted to stop everything. He never talked about it as anything other "
    "than maintenance. But I understand it now as something closer to love.",

    # 47 -> Set A
    "The archaeologist found the figurine on the last day of the dig, under a "
    "layer of ash she had almost decided not to excavate. It was small enough "
    "to fit in a closed fist: a woman with her arms raised, made from clay "
    "fired when Rome was still a cluster of mud huts. She was thinking about "
    "the hands that had shaped it, about what they had wanted to say. Reverence "
    "or supplication or decoration — who could know. But the hands were "
    "undeniable, their pressure preserved in clay across three thousand years.",

    # 48 -> Set B
    "The forest at the edge of town had been there longer than the town, and "
    "everyone who grew up there carried it with them in some form. Linh had "
    "spent her childhood summers in those woods, and now, back after twenty "
    "years in cities, she found the paths still there, faint but readable. "
    "The big oak she had climbed was still there, taller now, the bark rougher. "
    "She put her hand against it and felt the familiar solidity, the "
    "indifference that was not unkind. Some things stayed. That was enough.",

    # 49 -> Set A
    "What I know about silence I learned from my mother, who was a translator. "
    "She worked between languages all day and when she came home she was usually "
    "quiet through dinner, as if she had spent all her words. She didn't watch "
    "television. She sat in the kitchen and drank tea and looked at the window. "
    "I asked her once what she was thinking about. She said she wasn't thinking, "
    "exactly — she was listening for the language that lived underneath language, "
    "the one that didn't need to be translated.",

    # ---- TECHNICAL / PROFESSIONAL (indices 50-59) ----
    # 50 -> Set B
    "DISCHARGE SUMMARY\n"
    "PRINCIPAL DIAGNOSIS: Acute exacerbation of COPD with community-acquired pneumonia.\n\n"
    "HOSPITAL COURSE: Patient presented with worsening dyspnea, increased sputum "
    "and fever. CXR showed right lower lobe infiltrate. Initiated IV ceftriaxone, "
    "azithromycin, supplemental oxygen, albuterol nebulizations, and prednisone. "
    "Clinical improvement noted by day 2. Sputum culture grew Streptococcus "
    "pneumoniae, sensitive to all antibiotics.\n\n"
    "DISCHARGE MEDICATIONS: Complete azithromycin course, prednisone taper.\n"
    "FOLLOW-UP: Pulmonology in 2 weeks.",

    # 51 -> Set A
    "MEMORANDUM\nTO: Board of Directors\nFROM: CFO\nRE: Q3 Results\n\n"
    "Q3 revenue of $47.3M represents 18% year-over-year growth, exceeding "
    "guidance midpoint of $44-46M. Adjusted EBITDA margin of 23.4% expanded "
    "180bps versus prior year, driven by operating leverage and favorable "
    "product mix. Enterprise ARR grew 31% YoY to $182M, with net revenue "
    "retention of 118%. Gross margin improved to 74.2% from 71.8%.\n\n"
    "REVISED GUIDANCE: Raising full-year revenue guidance to $186-190M.",

    # 52 -> Set B
    "REQUEST FOR PROPOSAL — Cloud Infrastructure Migration Services\n\n"
    "BACKGROUND: The organization seeks vendors to provide cloud migration "
    "services to transition approximately 200 on-premises workloads to AWS or "
    "Azure over 12 months. Current infrastructure includes 80 physical servers, "
    "450 VMs, and 3 legacy databases.\n\n"
    "SCOPE: Infrastructure assessment, architecture design, phased migration "
    "with zero-downtime cutover, and post-migration optimization.\n\n"
    "EVALUATION: Technical approach (40%), experience (30%), pricing (20%), "
    "team qualifications (10%).",

    # 53 -> Set A
    "INCIDENT REPORT — Severity 1\n"
    "Duration: 2h 14min | Impact: 23% of production requests failed\n\n"
    "ROOT CAUSE: Database migration script included DROP INDEX statement intended "
    "for development only, increasing query latency from 8ms to 4200ms and "
    "exhausting the connection pool.\n\n"
    "CORRECTIVE ACTIONS: Add migration review checklist requiring DBA approval "
    "for index changes. Implement pre-deployment SQL diff review in CI. "
    "Add connection pool saturation alert. Create runbook for connection pool "
    "exhaustion recovery.",

    # 54 -> Set B
    "PATENT APPLICATION ABSTRACT\nTitle: System and Method for Adaptive Neural "
    "Network Composition Using Low-Rank Expert Modules\n\n"
    "The present invention provides systems for composing specialized neural "
    "network experts to serve heterogeneous inference workloads. A routing "
    "module selects relevant expert adapters using hash-based locality-sensitive "
    "routing requiring no gradient updates at inference time. Selected adapters "
    "are composed via weight-space addition, producing an effective model that "
    "combines expert capabilities without increasing inference latency. A "
    "clone-and-compete mechanism enables continuous improvement.",

    # 55 -> Set A
    "PERFORMANCE REVIEW — Software Engineer II\n\n"
    "SUMMARY: Consistently delivered high-quality work with minimal supervision, "
    "demonstrating technical depth appropriate for SE II level.\n\n"
    "STRENGTHS: Technical execution was excellent — all deliverables shipped on "
    "schedule with low defect rates. Code review comments rated 'actionable and "
    "constructive' by peers. Proactively identified a latent performance issue "
    "in the caching layer before it impacted customers.\n\n"
    "RATING: Exceeds Expectations. Eligible for promotion consideration.",

    # 56 -> Set B
    "INFORMATION SECURITY POLICY — Acceptable Use of AI Tools\n\n"
    "PERMITTED USES: General productivity with public information. Code "
    "generation for non-sensitive tooling, subject to mandatory human review. "
    "Drafting external communications without confidential data.\n\n"
    "PROHIBITED USES: Inputting customer PII, health data, financial data, "
    "or trade secrets into external AI services. Using AI-generated code in "
    "customer-facing products without security review.\n\n"
    "COMPLIANCE: Violations may result in termination.",

    # 57 -> Set A
    "ENGINEERING DESIGN DOCUMENT — Distributed Rate Limiter\n\n"
    "PROBLEM: The current in-process rate limiter fails in multi-instance "
    "deployments because each instance maintains independent counters.\n\n"
    "SOLUTION: Redis-backed sliding window counter using ZADD/ZRANGEBYSCORE. "
    "Each request adds a timestamped entry; the count within the window "
    "determines permission. Atomic check-and-increment via Lua script.\n\n"
    "SCALABILITY: Redis Cluster shards by user ID. P99 latency overhead "
    "estimated under 2ms. Fail-open strategy if Redis is unavailable.",

    # 58 -> Set B
    "VENDOR SECURITY ASSESSMENT — Data Protection and Privacy\n\n"
    "4.1 Data Classification: Does your organization maintain a formal data "
    "classification policy distinguishing public, internal, confidential, "
    "and restricted data?\n\n"
    "4.2 Encryption at Rest: Specify algorithm, key management practices, and "
    "whether keys are managed separately from the data they protect.\n\n"
    "4.3 Encryption in Transit: Confirm TLS 1.2 or higher for all customer "
    "data transmission.\n\n"
    "4.4 Data Retention: Describe retention schedule and process for secure "
    "deletion upon contract termination.",

    # 59 -> Set A
    "ARCHITECTURE REVIEW BOARD — Decision Record\n"
    "ADR-0023: Event-Driven Architecture for Order Processing\n\n"
    "CONTEXT: The synchronous order pipeline is a reliability bottleneck. "
    "Payment, inventory, fulfillment, and notification services are tightly "
    "coupled, causing cascading failures.\n\n"
    "DECISION: Migrate to event-driven architecture using Apache Kafka. "
    "Order lifecycle events will be published to Kafka topics. Each service "
    "subscribes independently and processes events at its own pace.\n\n"
    "CONSEQUENCES: Services are decoupled; notification failures no longer "
    "block payment. Accepted tradeoff: increased operational complexity.",
]

assert len(CALIBRATION_TEXTS) == 60, f"Expected 60 calibration texts, got {len(CALIBRATION_TEXTS)}"

# Split: odd index -> Set A (30), even index -> Set B (30)
SET_A_TEXTS = [t for i, t in enumerate(CALIBRATION_TEXTS) if i % 2 == 1]
SET_B_TEXTS = [t for i, t in enumerate(CALIBRATION_TEXTS) if i % 2 == 0]
assert len(SET_A_TEXTS) == 30
assert len(SET_B_TEXTS) == 30


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT.exists():
        try:
            with open(CHECKPOINT) as f:
                return json.load(f)
        except Exception as e:
            log(f"[warn] Could not load checkpoint: {e}")
    return {}


def save_checkpoint(data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(CHECKPOINT)


def checkpoint_phase(ckpt: dict, phase: str, data: dict) -> None:
    ckpt[phase] = data
    save_checkpoint(ckpt)
    log(f"[checkpoint] Phase {phase} saved.")


# ---------------------------------------------------------------------------
# Adapter discovery and ranking
# ---------------------------------------------------------------------------
def discover_adapters() -> list[str]:
    """Return all adapter names sorted alphabetically."""
    if not ADAPTER_DIR.exists():
        log(f"ERROR: adapter dir {ADAPTER_DIR} not found")
        sys.exit(1)
    adapters = sorted(
        d.name for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_model.safetensors").exists()
    )
    return adapters


def load_domain_quality_scores(adapter_names: list[str]) -> dict[str, float]:
    """Load pilot50 benchmark quality scores. Returns {adapter_name: improvement_pct}."""
    scores = {}
    if PILOT50_PATH.exists():
        try:
            with open(PILOT50_PATH) as f:
                data = json.load(f)
            # Support both formats: per_adapter with ppl_improvement_pct,
            # or domains with improvement_pct
            per_adapter = data.get("per_adapter", data.get("domains", {}))
            for name, info in per_adapter.items():
                if isinstance(info, dict):
                    score = info.get("ppl_improvement_pct",
                                     info.get("improvement_pct",
                                              info.get("ppl_improvement", 0.0)))
                    scores[name] = float(score)
        except Exception as e:
            log(f"[warn] Could not parse pilot50_benchmark.json: {e}")

    # Fill missing adapters with 0.0
    for name in adapter_names:
        if name not in scores:
            scores[name] = 0.0

    return scores


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_base_model(tokenizer_only: bool = False):
    """Load Qwen2.5-7B with 4-bit NF4. Returns (model, tokenizer) or (None, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log(f"Loading tokenizer from {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if tokenizer_only:
        return None, tokenizer

    log("Loading base model with 4-bit NF4 quantization ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        cache_dir=HF_CACHE,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if torch.cuda.is_available():
        log(f"  GPU memory after base load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def free_model(model) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# LoRA delta computation
# ---------------------------------------------------------------------------
def load_adapter_config(adapter_path: Path) -> dict:
    cfg_path = adapter_path / "adapter_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {"r": 16, "lora_alpha": 16}


def compute_lora_deltas(adapter_path: Path, cfg: dict | None = None) -> dict[str, "torch.Tensor"]:
    """Load LoRA adapter and return {param_path: scaled_delta} (float32, CPU).

    Applies lora_alpha/r scaling so the delta matches what PEFT merge_and_unload does.
    delta_i = (alpha/r) * B_i @ A_i
    """
    import torch
    from safetensors.torch import load_file

    if cfg is None:
        cfg = load_adapter_config(adapter_path)
    r = cfg.get("r", 16)
    lora_alpha = cfg.get("lora_alpha", r)
    scaling = lora_alpha / r

    tensors = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")

    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}

    for key, val in tensors.items():
        if ".lora_A.weight" in key:
            param_path = key.replace("base_model.model.", "", 1).replace(".lora_A.weight", "")
            lora_a[param_path] = val.float()
        elif ".lora_B.weight" in key:
            param_path = key.replace("base_model.model.", "", 1).replace(".lora_B.weight", "")
            lora_b[param_path] = val.float()

    del tensors
    gc.collect()

    deltas: dict[str, torch.Tensor] = {}
    for param_path in lora_a:
        if param_path not in lora_b:
            continue
        A = lora_a[param_path]   # (r, d_in)
        B = lora_b[param_path]   # (d_out, r)
        deltas[param_path] = scaling * (B @ A)

    del lora_a, lora_b
    gc.collect()
    return deltas


def preload_all_adapter_deltas(adapter_names: list[str]) -> dict[str, dict]:
    """Pre-load all adapter deltas into CPU memory.

    Returns {adapter_name: {param_path: delta_tensor_float32_cpu}}
    """
    log(f"Pre-loading deltas for {len(adapter_names)} adapters into CPU memory ...")
    t0 = time.time()
    all_deltas = {}
    for i, name in enumerate(adapter_names):
        path = ADAPTER_DIR / name
        cfg = load_adapter_config(path)
        all_deltas[name] = compute_lora_deltas(path, cfg)
        if (i + 1) % 10 == 0:
            log(f"  Loaded {i+1}/{len(adapter_names)} adapters ...")
    log(f"  Delta pre-load complete in {time.time() - t0:.1f}s")
    return all_deltas


# ---------------------------------------------------------------------------
# Composed adapter construction
# ---------------------------------------------------------------------------
def sum_deltas_for_subset(
    adapter_names: list[str],
    all_deltas: dict[str, dict],
) -> dict[str, "torch.Tensor"]:
    """Sum deltas for a subset of adapters. Returns {param_path: summed_delta}."""
    import torch
    composed: dict[str, torch.Tensor] = {}
    for name in adapter_names:
        for param_path, delta in all_deltas[name].items():
            if param_path in composed:
                composed[param_path] = composed[param_path] + delta
            else:
                composed[param_path] = delta.clone()
    return composed


def save_composed_peft_adapter(
    composed_deltas: dict,
    ref_cfg_path: Path,
    r: int,
) -> str:
    """Save composed delta as PEFT-compatible adapter via rank-r SVD decomposition.

    Sets lora_alpha = r so that PEFT's merge_and_unload applies scaling=1.0
    (i.e., delta is applied as-is without further scaling).
    """
    import torch
    from safetensors.torch import save_file

    peft_tensors = {}
    for param_path, delta in composed_deltas.items():
        try:
            U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
            r_eff = min(r, S.shape[0])
            sqrt_s = torch.sqrt(S[:r_eff])
            lora_B = (U[:, :r_eff] * sqrt_s.unsqueeze(0)).to(torch.bfloat16)
            lora_A = (Vh[:r_eff, :] * sqrt_s.unsqueeze(1)).to(torch.bfloat16)
            base = "base_model.model." + param_path
            peft_tensors[base + ".lora_A.weight"] = lora_A
            peft_tensors[base + ".lora_B.weight"] = lora_B
        except Exception as e:
            log(f"  [warn] SVD failed for {param_path}: {e}, skipping")

    tmpdir = tempfile.mkdtemp(prefix="composed_peft_")
    save_file(peft_tensors, os.path.join(tmpdir, "adapter_model.safetensors"))

    # Copy and patch adapter config: alpha=r => scaling=1.0
    with open(ref_cfg_path) as f:
        cfg = json.load(f)
    cfg["r"] = r
    cfg["lora_alpha"] = r
    with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f)

    del peft_tensors
    gc.collect()
    return tmpdir


def load_model_with_composed_adapter(
    subset: list[str],
    all_deltas: dict,
    r: int,
    ref_cfg_path: Path,
    tokenizer=None,
):
    """Load base model, merge composed adapter for given subset, return model.

    Caller is responsible for calling free_model() afterwards.
    """
    import torch
    from peft import PeftModel

    model, tok = load_base_model(tokenizer_only=False)
    if tokenizer is None:
        tokenizer = tok

    composed = sum_deltas_for_subset(subset, all_deltas)
    tmpdir = save_composed_peft_adapter(composed, ref_cfg_path, r)
    del composed
    gc.collect()

    try:
        peft_model = PeftModel.from_pretrained(model, tmpdir)
        merged = peft_model.merge_and_unload()
        merged.eval()
    except Exception as e:
        log(f"  [error] PEFT load/merge failed: {e}")
        traceback.print_exc()
        free_model(model)
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return merged, tokenizer


# ---------------------------------------------------------------------------
# Direct delta application (for LOO subtraction)
# ---------------------------------------------------------------------------
def apply_deltas_to_model(model, deltas: dict, sign: float = 1.0) -> None:
    """Add (sign=+1) or subtract (sign=-1) deltas directly into model weights.

    Works on merged (non-quantised) weights. The 4-bit quantisation is
    already resolved after merge_and_unload().
    """
    import torch
    for name, param in model.named_parameters():
        if name in deltas:
            delta = deltas[name].to(param.device, param.dtype)
            param.data.add_(sign * delta)
            del delta


def get_weight_snapshot(model) -> dict[str, float]:
    snapshot = {}
    for name, param in model.named_parameters():
        snapshot[name] = param.data.float().norm().item()
    return snapshot


def check_drift(model, snapshot: dict) -> float:
    max_drift = 0.0
    for name, param in model.named_parameters():
        if name in snapshot:
            cur = param.data.float().norm().item()
            ref = snapshot[name]
            if ref > 1e-12:
                max_drift = max(max_drift, abs(cur - ref) / ref)
    return max_drift


# ---------------------------------------------------------------------------
# PPL computation
# ---------------------------------------------------------------------------
def compute_ppl(model, tokenizer, texts: list[str],
                max_seq_len: int = MAX_SEQ_LEN) -> float:
    """Compute token-weighted perplexity. One text at a time (no padding).

    Float32 loss accumulation for numerical stability.
    """
    import torch

    total_loss = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_seq_len)
            input_ids = enc["input_ids"].to(model.device)
            if input_ids.shape[1] < 2:
                continue
            try:
                out = model(input_ids=input_ids)
            except torch.cuda.OutOfMemoryError:
                log("  [OOM] PPL eval, skipping this text")
                del input_ids
                torch.cuda.empty_cache()
                continue

            shift_logits = out.logits[:, :-1, :].contiguous().float()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

            del input_ids, out, shift_logits, shift_labels, loss, enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# MMLU evaluation
# ---------------------------------------------------------------------------
def format_mmlu_5shot(subject: str, few_shot_examples: list, test_ex: dict) -> str:
    """Format standard 5-shot MMLU prompt."""
    header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    body = ""
    for ex in few_shot_examples:
        q = ex["question"]
        choices = ex["choices"]
        ans_letter = "ABCD"[ex["answer"]]
        body += f"{q}\n"
        for i, ch in enumerate(choices):
            body += f"{'ABCD'[i]}. {ch}\n"
        body += f"Answer: {ans_letter}\n\n"
    # Test question
    body += f"{test_ex['question']}\n"
    for i, ch in enumerate(test_ex["choices"]):
        body += f"{'ABCD'[i]}. {ch}\n"
    body += "Answer:"
    return header + body


def evaluate_mmlu(
    model,
    tokenizer,
    subjects: list[str],
    n_shots: int = 5,
    max_per_subject: int | None = None,
    batch_size: int = 16,
) -> dict:
    """Evaluate on MMLU subjects via log-prob scoring. Returns overall accuracy."""
    import torch
    from datasets import load_dataset

    # Pre-compute answer token IDs
    choice_ids: dict[str, int] = {}
    choice_ids_sp: dict[str, int] = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_ids[letter] = ids[0]
        ids_sp = tokenizer.encode(f" {letter}", add_special_tokens=False)
        choice_ids_sp[letter] = ids_sp[-1] if ids_sp else ids[0]

    per_subject = {}
    total_correct = 0
    total_count = 0

    for subject in subjects:
        try:
            test_ds = load_dataset("cais/mmlu", subject, split="test",
                                   trust_remote_code=True,
                                   cache_dir=HF_CACHE)
        except Exception as e:
            log(f"  [skip MMLU] {subject}: {e}")
            continue

        # Load few-shot dev examples
        few_shot_exs = []
        if n_shots > 0:
            try:
                dev_ds = load_dataset("cais/mmlu", subject, split="dev",
                                      trust_remote_code=True,
                                      cache_dir=HF_CACHE)
                few_shot_exs = list(dev_ds)[:n_shots]
            except Exception:
                few_shot_exs = []

        examples = list(test_ds)
        if max_per_subject is not None:
            examples = examples[:max_per_subject]

        prompts = [format_mmlu_5shot(subject, few_shot_exs, ex) for ex in examples]
        gold_labels = ["ABCD"[ex["answer"]] for ex in examples]
        subj_correct = 0
        n = len(prompts)

        for b_start in range(0, n, batch_size):
            b_prompts = prompts[b_start: b_start + batch_size]
            b_golds = gold_labels[b_start: b_start + batch_size]
            enc = tokenizer(b_prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_SEQ_LEN)
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                last_logits = out.logits[:, -1, :]
                log_probs = torch.log_softmax(last_logits.float(), dim=-1)
            except torch.cuda.OutOfMemoryError:
                log(f"  [OOM] MMLU batch {b_start} for {subject}, retrying bs=1")
                torch.cuda.empty_cache()
                log_probs_list = []
                for prompt in b_prompts:
                    enc_s = tokenizer(prompt, return_tensors="pt", truncation=True,
                                      max_length=MAX_SEQ_LEN)
                    ids_s = enc_s["input_ids"].to(model.device)
                    with torch.no_grad():
                        out_s = model(input_ids=ids_s)
                    log_probs_list.append(
                        torch.log_softmax(out_s.logits[0, -1].float(), dim=-1).unsqueeze(0)
                    )
                log_probs = torch.cat(log_probs_list, dim=0)

            for j, gold in enumerate(b_golds):
                scores = {
                    letter: max(log_probs[j, choice_ids[letter]].item(),
                                log_probs[j, choice_ids_sp[letter]].item())
                    for letter in "ABCD"
                }
                pred = max(scores, key=scores.get)
                subj_correct += int(pred == gold)

            del input_ids, attention_mask, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        acc = subj_correct / max(1, n)
        per_subject[subject] = {"correct": subj_correct, "total": n,
                                 "accuracy": round(acc, 4)}
        total_correct += subj_correct
        total_count += n

    overall_acc = total_correct / max(1, total_count)
    return {
        "per_subject": per_subject,
        "overall": {"correct": total_correct, "total": total_count,
                    "accuracy": round(overall_acc, 4)},
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    import torch
    from scipy.stats import kendalltau

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("EXPERT PRUNING COMPOSITION EXPERIMENT")
    log(f"  N_EXPERTS: {N_EXPERTS}")
    log(f"  SMOKE_TEST: {IS_SMOKE}")
    log(f"  SKIP_GREEDY: {SKIP_GREEDY}")
    log(f"  BASE_MODEL: {BASE_MODEL}")
    log(f"  K_VALUES_PPL: {K_VALUES_PPL}")
    log(f"  K_VALUES_MMLU: {K_VALUES_MMLU}")
    log(f"  CALIB_PER_SET: {CALIB_PER_SET}")
    log(f"  MAX_SEQ_LEN: {MAX_SEQ_LEN}")
    log(f"  MMLU_SHOTS: {MMLU_SHOTS}")
    log(f"  MAX_RUNTIME_S: {MAX_RUNTIME_S}s ({MAX_RUNTIME_S/60:.0f} min)")
    log("=" * 70)

    ckpt = load_checkpoint()
    timing = ckpt.get("timing", {})

    def elapsed() -> float:
        return time.time() - t_start

    def check_timeout(label: str = "") -> bool:
        if elapsed() > MAX_RUNTIME_S:
            log(f"[TIMEOUT] Max runtime {MAX_RUNTIME_S}s exceeded. {label}")
            return True
        return False

    # -------------------------------------------------------------------------
    # Phase 0: Setup
    # -------------------------------------------------------------------------
    log("\n--- Phase 0: Setup ---")
    t0 = time.time()

    all_adapters_found = discover_adapters()
    log(f"Found {len(all_adapters_found)} adapters in {ADAPTER_DIR}")
    selected_adapters = all_adapters_found[:N_EXPERTS]
    log(f"Using {len(selected_adapters)} adapters: {selected_adapters[:5]}...")

    # Quality scores from pilot50 benchmark
    domain_scores = load_domain_quality_scores(selected_adapters)
    # Sort descending: pi_domain[0] = best
    pi_domain = sorted(selected_adapters, key=lambda n: domain_scores.get(n, 0.0),
                       reverse=True)
    log(f"Domain quality range: {domain_scores[pi_domain[0]]:.1f}% "
        f"(best={pi_domain[0]}) ... {domain_scores[pi_domain[-1]]:.1f}% "
        f"(worst={pi_domain[-1]})")
    bottom20_pct = max(1, int(len(selected_adapters) * 0.2))
    log(f"Bottom-20% ({bottom20_pct} experts): {pi_domain[-bottom20_pct:]}")

    # Calibration sets (truncated by CALIB_PER_SET)
    calib_a = SET_A_TEXTS[:CALIB_PER_SET]
    calib_b = SET_B_TEXTS[:CALIB_PER_SET]
    log(f"Calibration Set A: {len(calib_a)} texts, Set B: {len(calib_b)} texts")

    # Ref adapter config for SVD-based PEFT construction
    ref_cfg_path = ADAPTER_DIR / selected_adapters[0] / "adapter_config.json"
    with open(ref_cfg_path) as f:
        ref_cfg = json.load(f)
    lora_r = ref_cfg.get("r", 16)
    log(f"LoRA rank: {lora_r}")

    # Pre-load all adapter deltas into CPU memory
    all_deltas = preload_all_adapter_deltas(selected_adapters)

    timing["phase0_setup_s"] = time.time() - t0
    log(f"Phase 0 done in {timing['phase0_setup_s']:.1f}s")

    # -------------------------------------------------------------------------
    # Phase 1: Reference PPL — base model + all-N composed
    # -------------------------------------------------------------------------
    log("\n--- Phase 1: Reference PPL (base and all-N composed) ---")
    t1 = time.time()

    if "phase1" in ckpt:
        log("[resume] Phase 1 loaded from checkpoint")
        p1 = ckpt["phase1"]
        ppl_base_a = p1["ppl_base_set_a"]
        ppl_base_b = p1["ppl_base_set_b"]
        ppl_all50_a = p1["ppl_all50_set_a"]
        ppl_all50_b = p1["ppl_all50_set_b"]
    else:
        # 1a: Base model PPL
        log("  Loading base model (no adapters) ...")
        base_model, tokenizer = load_base_model()
        gc.disable()
        gc.collect()
        try:
            ppl_base_a = compute_ppl(base_model, tokenizer, calib_a)
            ppl_base_b = compute_ppl(base_model, tokenizer, calib_b)
        finally:
            gc.enable()
            gc.collect()
        log(f"  Base PPL: A={ppl_base_a:.4f}, B={ppl_base_b:.4f}")
        free_model(base_model)

        # 1b: All-N composed PPL
        log(f"  Loading composed model (all {len(selected_adapters)} adapters) ...")
        composed_model, tokenizer = load_model_with_composed_adapter(
            selected_adapters, all_deltas, lora_r, ref_cfg_path
        )
        gc.disable()
        gc.collect()
        try:
            ppl_all50_a = compute_ppl(composed_model, tokenizer, calib_a)
            ppl_all50_b = compute_ppl(composed_model, tokenizer, calib_b)
        finally:
            gc.enable()
            gc.collect()
        log(f"  All-{len(selected_adapters)} PPL: A={ppl_all50_a:.4f}, B={ppl_all50_b:.4f}")

        if ppl_all50_a > 1000:
            log("  WARNING: PPL > 1000 — check adapter scaling correctness!")

        # We will reuse the composed model for Phase 2 (LOO)
        # Keep it in memory; free after LOO
        ckpt["_composed_model_ready"] = True
        checkpoint_phase(ckpt, "phase1", {
            "ppl_base_set_a": ppl_all50_a,   # placeholder, filled below
            "ppl_base_set_b": ppl_all50_b,
            "ppl_all50_set_a": ppl_all50_a,
            "ppl_all50_set_b": ppl_all50_b,
        })
        # Fix: save base separately
        ckpt["phase1"]["ppl_base_set_a"] = ppl_base_a
        ckpt["phase1"]["ppl_base_set_b"] = ppl_base_b
        save_checkpoint(ckpt)

        # Keep composed_model alive for Phase 2
        ckpt["_composed_model"] = None   # placeholder

    timing["phase1_reference_s"] = time.time() - t1
    log(f"Phase 1 done in {timing['phase1_reference_s']:.1f}s")

    # -------------------------------------------------------------------------
    # Phase 2: LOO ranking
    # -------------------------------------------------------------------------
    log("\n--- Phase 2: LOO Ranking ---")
    t2 = time.time()
    pi_loo: list[str] = []
    loo_scores: dict[str, float] = {}

    if "phase2" in ckpt:
        log("[resume] Phase 2 loaded from checkpoint")
        p2 = ckpt["phase2"]
        pi_loo = p2["pi_loo"]
        loo_scores = p2["loo_scores"]
    elif LOO_RESULTS_PATH.exists():
        log(f"Loading LOO results from {LOO_RESULTS_PATH} ...")
        try:
            with open(LOO_RESULTS_PATH) as f:
                loo_data = json.load(f)
            rankings = loo_data.get("rankings", loo_data.get("per_expert_rankings", {}))
            for name in selected_adapters:
                if name in rankings:
                    info = rankings[name]
                    # C_i = PPL_{-i} - PPL_ref (positive = helpful)
                    score = info.get("delta_ppl_a_pct",
                                     info.get("contribution_pct",
                                              info.get("c_i", 0.0)))
                    loo_scores[name] = float(score)
                else:
                    loo_scores[name] = 0.0
            pi_loo = sorted(selected_adapters,
                            key=lambda n: loo_scores.get(n, 0.0), reverse=True)
            log(f"Loaded LOO scores for {len(loo_scores)} adapters")
        except Exception as e:
            log(f"[warn] Could not parse LOO results: {e}. Will compute LOO.")
            loo_scores = {}

    if not pi_loo and not check_timeout("skipping Phase 2 LOO computation"):
        # Compute LOO from scratch using the composed model
        # The composed_model should still be in scope if Phase 1 just ran
        log("  Computing LOO rankings via subtraction method ...")

        # Load composed model if not already in memory (checkpoint resume case)
        need_fresh_composed = "composed_model" not in dir() or composed_model is None  # type: ignore[name-defined]
        if need_fresh_composed:
            composed_model, tokenizer = load_model_with_composed_adapter(
                selected_adapters, all_deltas, lora_r, ref_cfg_path
            )

        ref_ppl_a_loo = ppl_all50_a
        weight_snapshot = get_weight_snapshot(composed_model)

        gc.disable()
        gc.collect()
        try:
            for i, name in enumerate(selected_adapters):
                if check_timeout(f"LOO stopped at {i}/{len(selected_adapters)}"):
                    break

                apply_deltas_to_model(composed_model, all_deltas[name], sign=-1.0)
                ppl_minus_i = compute_ppl(composed_model, tokenizer, calib_a)
                apply_deltas_to_model(composed_model, all_deltas[name], sign=+1.0)

                c_i = ppl_minus_i - ref_ppl_a_loo  # positive = helpful
                loo_scores[name] = c_i
                log(f"  [{i+1}/{len(selected_adapters)}] {name}: "
                    f"PPL_-i={ppl_minus_i:.4f}, C_i={c_i:+.4f}")

                if (i + 1) % DRIFT_CHECK_INTERVAL == 0:
                    drift = check_drift(composed_model, weight_snapshot)
                    log(f"  Drift check: {drift:.2e}")
                    if drift > DRIFT_THRESHOLD:
                        log(f"  WARNING: drift {drift:.2e} > threshold {DRIFT_THRESHOLD}")
        finally:
            gc.enable()
            gc.collect()

        free_model(composed_model)
        composed_model = None  # type: ignore[assignment]

        pi_loo = sorted(selected_adapters,
                        key=lambda n: loo_scores.get(n, 0.0), reverse=True)
        checkpoint_phase(ckpt, "phase2", {
            "pi_loo": pi_loo,
            "loo_scores": loo_scores,
            "source": "computed",
        })
    elif pi_loo:
        checkpoint_phase(ckpt, "phase2", {
            "pi_loo": pi_loo,
            "loo_scores": loo_scores,
            "source": "loaded",
        })
    else:
        log("  Skipped LOO computation (timeout). Using pi_domain as fallback.")
        pi_loo = pi_domain[:]
        loo_scores = {n: domain_scores.get(n, 0.0) for n in selected_adapters}

    # If composed_model is still alive from Phase 1, free it now
    try:
        if "composed_model" in dir() and composed_model is not None:  # type: ignore[name-defined]
            free_model(composed_model)
            composed_model = None  # type: ignore[assignment]
    except Exception:
        pass

    timing["phase2_loo_s"] = time.time() - t2
    log(f"Phase 2 done in {timing['phase2_loo_s']:.1f}s")
    log(f"  pi_loo[:5] = {pi_loo[:5]}")
    log(f"  pi_domain[:5] = {pi_domain[:5]}")

    # -------------------------------------------------------------------------
    # Load tokenizer (needed for all subsequent phases)
    # -------------------------------------------------------------------------
    _, tokenizer = load_base_model(tokenizer_only=True)

    # -------------------------------------------------------------------------
    # Phase 3: PPL Accumulation Curve (rank-ordered)
    # -------------------------------------------------------------------------
    log("\n--- Phase 3: PPL Accumulation Curve ---")
    t3 = time.time()

    if "phase3" in ckpt:
        log("[resume] Phase 3 loaded from checkpoint")
        p3 = ckpt["phase3"]
        accum_ppl_a = p3["ppl_set_a"]
        accum_ppl_b = p3["ppl_set_b"]
        accum_k = p3["k_values"]
    else:
        # Cap k_values to actual N
        k_values_ppl = [k for k in K_VALUES_PPL if k <= len(selected_adapters)]
        accum_k = []
        accum_ppl_a = []
        accum_ppl_b = []

        for k in k_values_ppl:
            if check_timeout(f"Phase 3 stopped at k={k}"):
                break
            subset = pi_domain[:k]
            log(f"  k={k}: composing {len(subset)} adapters ...")
            t_k = time.time()
            try:
                m, _ = load_model_with_composed_adapter(
                    subset, all_deltas, lora_r, ref_cfg_path, tokenizer
                )
                gc.disable()
                gc.collect()
                try:
                    ppl_a = compute_ppl(m, tokenizer, calib_a)
                    ppl_b = compute_ppl(m, tokenizer, calib_b)
                finally:
                    gc.enable()
                    gc.collect()
                free_model(m)
                m = None
            except Exception as e:
                log(f"  [error] k={k} failed: {e}")
                traceback.print_exc()
                try:
                    free_model(m)
                except Exception:
                    pass
                continue

            accum_k.append(k)
            accum_ppl_a.append(ppl_a)
            accum_ppl_b.append(ppl_b)
            log(f"  k={k}: PPL_A={ppl_a:.4f}, PPL_B={ppl_b:.4f} [{time.time()-t_k:.0f}s]")

            checkpoint_phase(ckpt, "phase3", {
                "k_values": accum_k,
                "ppl_set_a": accum_ppl_a,
                "ppl_set_b": accum_ppl_b,
            })

    timing["phase3_accum_ppl_s"] = time.time() - t3
    log(f"Phase 3 done in {timing['phase3_accum_ppl_s']:.1f}s, {len(accum_k)} k-values")

    # -------------------------------------------------------------------------
    # Phase 5: Greedy Forward Selection (PPL only)
    # -------------------------------------------------------------------------
    log("\n--- Phase 5: Greedy Forward Selection ---")
    t5 = time.time()
    greedy_ppl_a: list[float] = []
    greedy_order: list[str] = []
    greedy_k_values: list[int] = []

    if SKIP_GREEDY:
        log("  SKIP_GREEDY=1 — skipping greedy phase")
    elif "phase5" in ckpt:
        log("[resume] Phase 5 loaded from checkpoint")
        p5 = ckpt["phase5"]
        greedy_ppl_a = p5["greedy_ppl_set_a"]
        greedy_order = p5["greedy_order"]
        greedy_k_values = p5["k_values"]
    elif check_timeout("skipping Phase 5 greedy"):
        log("  Timeout — skipping greedy phase")
    else:
        greedy_set: list[str] = []
        remaining = set(selected_adapters)
        greedy_max_k = min(GREEDY_MAX_K, len(selected_adapters))

        for step in range(greedy_max_k):
            if check_timeout(f"greedy step {step+1}"):
                break

            best_name = None
            best_ppl = float("inf")
            candidates = list(remaining)
            log(f"  Greedy step {step+1}/{greedy_max_k}: evaluating {len(candidates)} candidates ...")

            t_step = time.time()
            for candidate in candidates:
                trial_subset = greedy_set + [candidate]
                try:
                    m, _ = load_model_with_composed_adapter(
                        trial_subset, all_deltas, lora_r, ref_cfg_path, tokenizer
                    )
                    gc.disable()
                    gc.collect()
                    try:
                        ppl_trial = compute_ppl(m, tokenizer, calib_a)
                    finally:
                        gc.enable()
                        gc.collect()
                    free_model(m)
                    m = None
                except Exception as e:
                    log(f"    [error] candidate {candidate}: {e}")
                    try:
                        free_model(m)
                    except Exception:
                        pass
                    continue

                if ppl_trial < best_ppl:
                    best_ppl = ppl_trial
                    best_name = candidate

            if best_name is None:
                log(f"  No valid candidate found at step {step+1}, stopping greedy.")
                break

            greedy_set.append(best_name)
            remaining.discard(best_name)
            greedy_ppl_a.append(best_ppl)
            greedy_order.append(best_name)
            greedy_k_values.append(len(greedy_set))
            log(f"  Greedy step {step+1}: selected={best_name}, PPL={best_ppl:.4f} "
                f"[{time.time()-t_step:.0f}s]")

            checkpoint_phase(ckpt, "phase5", {
                "k_values": greedy_k_values,
                "greedy_ppl_set_a": greedy_ppl_a,
                "greedy_order": greedy_order,
            })

            # Early stop: if greedy matches ranked order for first 5 steps
            if step >= 4:
                matches = sum(1 for j in range(len(greedy_order))
                              if j < len(pi_domain) and greedy_order[j] == pi_domain[j])
                if matches == len(greedy_order):
                    log("  [early stop] Greedy matches rank order for all steps so far.")

    timing["phase5_greedy_s"] = time.time() - t5
    log(f"Phase 5 done in {timing['phase5_greedy_s']:.1f}s, {len(greedy_order)} steps")

    # -------------------------------------------------------------------------
    # Phase 4: MMLU Accumulation Curve
    # -------------------------------------------------------------------------
    log("\n--- Phase 4: MMLU Accumulation Curve ---")
    t4 = time.time()

    if "phase4" in ckpt:
        log("[resume] Phase 4 loaded from checkpoint")
        p4 = ckpt["phase4"]
        accum_mmlu_k = p4["k_values"]
        accum_mmlu_acc = p4["mmlu_accuracy"]
        mmlu_base_acc = p4.get("mmlu_base_acc", None)
    else:
        # First: evaluate base model MMLU
        log("  Evaluating base model MMLU ...")
        if not check_timeout("Phase 4 base MMLU"):
            base_m, _ = load_base_model()
            gc.disable()
            gc.collect()
            try:
                base_mmlu_res = evaluate_mmlu(
                    base_m, tokenizer,
                    subjects=MMLU_SUBJECTS_LIST,
                    n_shots=MMLU_SHOTS,
                    max_per_subject=MAX_PER_SUBJECT,
                )
            finally:
                gc.enable()
                gc.collect()
            mmlu_base_acc = base_mmlu_res["overall"]["accuracy"]
            log(f"  Base MMLU accuracy: {mmlu_base_acc:.4f}")
            free_model(base_m)
        else:
            mmlu_base_acc = None

        accum_mmlu_k = []
        accum_mmlu_acc = []

        k_values_mmlu = [k for k in K_VALUES_MMLU if k <= len(selected_adapters)]

        for k in k_values_mmlu:
            if check_timeout(f"Phase 4 stopped at k={k}"):
                break
            subset = pi_domain[:k]
            log(f"  MMLU k={k}: composing {len(subset)} adapters ...")
            t_k = time.time()
            try:
                m, _ = load_model_with_composed_adapter(
                    subset, all_deltas, lora_r, ref_cfg_path, tokenizer
                )
                gc.disable()
                gc.collect()
                try:
                    mmlu_res = evaluate_mmlu(
                        m, tokenizer,
                        subjects=MMLU_SUBJECTS_LIST,
                        n_shots=MMLU_SHOTS,
                        max_per_subject=MAX_PER_SUBJECT,
                    )
                finally:
                    gc.enable()
                    gc.collect()
                acc = mmlu_res["overall"]["accuracy"]
                free_model(m)
                m = None
            except Exception as e:
                log(f"  [error] MMLU k={k}: {e}")
                traceback.print_exc()
                try:
                    free_model(m)
                except Exception:
                    pass
                continue

            accum_mmlu_k.append(k)
            accum_mmlu_acc.append(acc)
            log(f"  MMLU k={k}: acc={acc:.4f} [{time.time()-t_k:.0f}s]")

            checkpoint_phase(ckpt, "phase4", {
                "k_values": accum_mmlu_k,
                "mmlu_accuracy": accum_mmlu_acc,
                "mmlu_base_acc": mmlu_base_acc,
            })

    # Also look up all-50 MMLU from checkpoint if available
    mmlu_all50_acc = None
    for k_val, acc_val in zip(accum_mmlu_k, accum_mmlu_acc):
        if k_val == len(selected_adapters):
            mmlu_all50_acc = acc_val
            break

    timing["phase4_accum_mmlu_s"] = time.time() - t4
    log(f"Phase 4 done in {timing['phase4_accum_mmlu_s']:.1f}s, {len(accum_mmlu_k)} k-values")

    # -------------------------------------------------------------------------
    # Phase 6: Bottom-K Removal Analysis (reads prior data)
    # -------------------------------------------------------------------------
    log("\n--- Phase 6: Bottom-K Removal Analysis ---")
    t6 = time.time()

    ppl_top40 = None
    n_removed = bottom20_pct
    removed_experts = pi_domain[-n_removed:]
    k_top = len(selected_adapters) - n_removed

    # Find PPL at k_top from accumulation curve
    for k_val, ppl_val in zip(accum_k, accum_ppl_a):
        if k_val == k_top:
            ppl_top40 = ppl_val
            break

    if ppl_top40 is None:
        log(f"  k={k_top} not in PPL accumulation curve (may be incomplete)")
        delta_prune_pct = None
    else:
        delta_prune_pct = (ppl_top40 - ppl_all50_a) / ppl_all50_a * 100.0
        log(f"  PPL(all-{len(selected_adapters)}): {ppl_all50_a:.4f}")
        log(f"  PPL(top-{k_top}): {ppl_top40:.4f}")
        log(f"  Delta_prune: {delta_prune_pct:+.3f}%")

    # MMLU delta at k_top
    mmlu_top_k = None
    for k_val, acc_val in zip(accum_mmlu_k, accum_mmlu_acc):
        if k_val == k_top:
            mmlu_top_k = acc_val
            break

    mmlu_delta_prune = None
    if mmlu_top_k is not None and mmlu_all50_acc is not None:
        mmlu_delta_prune = (mmlu_top_k - mmlu_all50_acc) * 100.0

    timing["phase6_removal_s"] = time.time() - t6
    log(f"Phase 6 done in {timing['phase6_removal_s']:.1f}s")

    # -------------------------------------------------------------------------
    # Phase 7: Ranking Stability (Kendall tau)
    # -------------------------------------------------------------------------
    log("\n--- Phase 7: Ranking Stability ---")
    t7 = time.time()

    # Compute tau(pi_domain, pi_loo)
    shared = [n for n in selected_adapters if n in loo_scores]
    tau_domain_loo = None
    tau_domain_loo_p = None
    if len(shared) >= 3:
        domain_ranks = [pi_domain.index(n) for n in shared if n in pi_domain]
        loo_ranks = [pi_loo.index(n) for n in shared if n in pi_loo]
        # Align by shared adapters
        aligned = [(n, pi_domain.index(n) if n in pi_domain else len(pi_domain),
                    pi_loo.index(n) if n in pi_loo else len(pi_loo))
                   for n in shared]
        d_ranks = [a[1] for a in aligned]
        l_ranks = [a[2] for a in aligned]
        if len(d_ranks) >= 3:
            tau_val, tau_p = kendalltau(d_ranks, l_ranks)
            tau_domain_loo = float(tau_val)
            tau_domain_loo_p = float(tau_p)
            log(f"  tau(pi_domain, pi_loo): {tau_domain_loo:.4f} (p={tau_domain_loo_p:.4f})")

    # tau(pi_domain, pi_mmlu) — from individual expert eval if available
    tau_domain_mmlu = None
    tau_domain_mmlu_p = None
    if IND_EXPERT_PATH.exists():
        try:
            with open(IND_EXPERT_PATH) as f:
                ind_data = json.load(f)
            ind_experts = ind_data.get("individual_experts", {})
            mmlu_deltas = {n: ind_experts[n].get("delta_vs_base_pp", 0.0)
                           for n in selected_adapters if n in ind_experts
                           and "error" not in ind_experts[n]}
            if len(mmlu_deltas) >= 3:
                pi_mmlu = sorted(mmlu_deltas.keys(),
                                 key=lambda n: mmlu_deltas[n], reverse=True)
                common = [n for n in selected_adapters
                          if n in pi_domain and n in pi_mmlu]
                if len(common) >= 3:
                    d_r = [pi_domain.index(n) for n in common]
                    m_r = [pi_mmlu.index(n) for n in common]
                    tv, tp = kendalltau(d_r, m_r)
                    tau_domain_mmlu = float(tv)
                    tau_domain_mmlu_p = float(tp)
                    log(f"  tau(pi_domain, pi_mmlu): {tau_domain_mmlu:.4f} "
                        f"(p={tau_domain_mmlu_p:.4f})")
        except Exception as e:
            log(f"  [warn] Could not load individual expert results: {e}")

    # tau(Set A vs Set B) — from LOO if we computed it, or from PPL accumulation
    tau_set_a_b = None
    tau_set_a_b_p = None
    if len(loo_scores) >= 3 and "phase2" in ckpt and ckpt["phase2"].get("source") == "computed":
        # Build LOO-based ranking for Set B too if available
        # If we only have Set A LOO scores, we can't compute this
        log("  [info] Set A vs Set B tau: requires LOO on both sets (not computed in this run)")
    else:
        log("  [info] Set A vs Set B tau: using domain score as proxy")

    timing["phase7_stability_s"] = time.time() - t7
    log(f"Phase 7 done in {timing['phase7_stability_s']:.1f}s")

    # -------------------------------------------------------------------------
    # Phase 8: Scalability Assessment (greedy vs ranked comparison)
    # -------------------------------------------------------------------------
    log("\n--- Phase 8: Scalability Assessment ---")
    t8 = time.time()

    max_discrepancy_pct = None
    rank_match_count = 0

    if greedy_order and accum_k:
        # Match greedy k values with ranked PPL values
        ranked_ppl_map = dict(zip(accum_k, accum_ppl_a))
        discrepancies = []

        for j, (k_g, ppl_g) in enumerate(zip(greedy_k_values, greedy_ppl_a)):
            ppl_ranked = ranked_ppl_map.get(k_g)
            if ppl_ranked is not None:
                disc = abs(ppl_g - ppl_ranked) / ppl_ranked * 100.0
                discrepancies.append(disc)

        if discrepancies:
            max_discrepancy_pct = max(discrepancies)
            log(f"  Max PPL discrepancy (greedy vs ranked): {max_discrepancy_pct:.3f}%")

        # Count rank-order matches
        for j in range(min(len(greedy_order), len(pi_domain))):
            if greedy_order[j] == pi_domain[j]:
                rank_match_count += 1
            else:
                break  # stop at first mismatch
        log(f"  Greedy-rank order matches (prefix): {rank_match_count}/{len(greedy_order)}")
    else:
        log("  No greedy results available for comparison.")

    timing["phase8_scalability_s"] = time.time() - t8
    log(f"Phase 8 done in {timing['phase8_scalability_s']:.1f}s")

    # -------------------------------------------------------------------------
    # Kill criteria assessment
    # -------------------------------------------------------------------------
    log("\n--- Kill Criteria Assessment ---")

    # K1: Does removing bottom-20% improve PPL by >1%?
    k1_delta = delta_prune_pct
    k1_pass = (k1_delta is not None) and (k1_delta < -1.0)
    if k1_delta is not None:
        log(f"  K1: Delta_prune = {k1_delta:+.3f}% (threshold: < -1%) -> "
            f"{'PASS (pruning helps)' if k1_pass else 'KILL (pruning unnecessary)'}")
    else:
        log("  K1: Delta_prune unavailable (incomplete accumulation curve)")

    # K2: Is ranking stable? tau >= 0.6
    tau_best = tau_domain_loo if tau_domain_loo is not None else 0.0
    k2_pass = tau_best >= 0.6
    log(f"  K2: tau(pi_domain, pi_loo) = {tau_best:.4f} (threshold: >= 0.6) -> "
        f"{'PASS' if k2_pass else 'KILL (unstable ranking)'}")

    # K3: Can ranking match greedy? max_discrepancy < 0.5%
    k3_pass = (max_discrepancy_pct is not None) and (max_discrepancy_pct < 0.5)
    if max_discrepancy_pct is not None:
        log(f"  K3: max_discrepancy = {max_discrepancy_pct:.3f}% (threshold: < 0.5%) -> "
            f"{'PASS (ranking sufficient)' if k3_pass else 'KILL (greedy outperforms ranking)'}")
    elif SKIP_GREEDY:
        log("  K3: SKIPPED (SKIP_GREEDY=1)")
        k3_pass = None
    else:
        log("  K3: Inconclusive (insufficient greedy data)")

    if k1_pass is False:
        k1_interpretation = ("More experts is always better — null hypothesis confirmed. "
                              "No pruning needed for SOLE.")
    elif k1_pass:
        k1_interpretation = ("Pruning bottom-20% improves PPL by >1%. "
                              "Low-quality experts actively harm the composition.")
    else:
        k1_interpretation = "K1 inconclusive (incomplete data)."

    verdict_parts = []
    if k1_pass is not None:
        verdict_parts.append("K1=" + ("PASS" if k1_pass else "KILL"))
    if k2_pass is not None:
        verdict_parts.append("K2=" + ("PASS" if k2_pass else "KILL"))
    if k3_pass is not None:
        verdict_parts.append("K3=" + ("PASS" if k3_pass else "KILL"))

    if all(v is not None for v in [k1_pass, k2_pass]) and (k3_pass is not None or SKIP_GREEDY):
        passes = [k1_pass, k2_pass] + ([k3_pass] if k3_pass is not None else [])
        verdict = "PASS" if all(passes) else "KILL"
    else:
        verdict = "PARTIAL"
    log(f"  VERDICT: {verdict}  [{', '.join(verdict_parts)}]")

    # -------------------------------------------------------------------------
    # Final timing
    # -------------------------------------------------------------------------
    timing["total_elapsed_s"] = time.time() - t_start

    # -------------------------------------------------------------------------
    # Build output JSON
    # -------------------------------------------------------------------------
    delta_from_base_pct = [
        (ppl - ppl_base_a) / ppl_base_a * 100.0 if ppl_base_a > 0 else 0.0
        for ppl in accum_ppl_a
    ]
    mmlu_delta_from_base = [
        (acc - mmlu_base_acc) * 100.0 if mmlu_base_acc is not None else 0.0
        for acc in accum_mmlu_acc
    ]

    results = {
        "config": {
            "base_model": BASE_MODEL,
            "n_experts": len(selected_adapters),
            "quantization": "nf4_4bit",
            "composition_method": "naive_addition",
            "calib_texts_per_set": CALIB_PER_SET,
            "max_seq_len": MAX_SEQ_LEN,
            "mmlu_subjects": len(MMLU_SUBJECTS_LIST),
            "mmlu_shots": MMLU_SHOTS,
            "seed": SEED,
            "smoke_test": IS_SMOKE,
            "skip_greedy": SKIP_GREEDY,
        },
        "reference": {
            "ppl_all50_set_a": ppl_all50_a,
            "ppl_all50_set_b": ppl_all50_b,
            "ppl_base_set_a": ppl_base_a,
            "ppl_base_set_b": ppl_base_b,
            "mmlu_all50": mmlu_all50_acc,
            "mmlu_base": mmlu_base_acc,
        },
        "quality_rankings": {
            "pi_domain": pi_domain,
            "pi_loo": pi_loo,
            "domain_scores": {n: round(domain_scores.get(n, 0.0), 4)
                              for n in selected_adapters},
            "loo_scores": {n: round(loo_scores.get(n, 0.0), 4)
                           for n in selected_adapters},
        },
        "accumulation_curve_ppl": {
            "k_values": accum_k,
            "ppl_set_a": [round(v, 4) for v in accum_ppl_a],
            "ppl_set_b": [round(v, 4) for v in accum_ppl_b],
            "delta_from_base_pct": [round(v, 4) for v in delta_from_base_pct],
        },
        "accumulation_curve_mmlu": {
            "k_values": accum_mmlu_k,
            "mmlu_accuracy": [round(v, 4) for v in accum_mmlu_acc],
            "delta_from_base_pp": [round(v, 4) for v in mmlu_delta_from_base],
        },
        "greedy_forward_selection": {
            "k_values": greedy_k_values,
            "greedy_ppl_set_a": [round(v, 4) for v in greedy_ppl_a],
            "greedy_order": greedy_order,
            "rank_match_count": rank_match_count,
            "max_discrepancy_pct": round(max_discrepancy_pct, 4)
                                   if max_discrepancy_pct is not None else None,
            "skipped": SKIP_GREEDY,
        },
        "bottom_k_removal": {
            "n_removed": n_removed,
            "k_retained": k_top,
            "removed_experts": removed_experts,
            "ppl_all50": round(ppl_all50_a, 4),
            "ppl_top_k": round(ppl_top40, 4) if ppl_top40 is not None else None,
            "delta_prune_pct": round(delta_prune_pct, 4) if delta_prune_pct is not None else None,
            "mmlu_all50": round(mmlu_all50_acc, 4) if mmlu_all50_acc is not None else None,
            "mmlu_top_k": round(mmlu_top_k, 4) if mmlu_top_k is not None else None,
            "mmlu_delta_prune_pp": round(mmlu_delta_prune, 4) if mmlu_delta_prune is not None else None,
        },
        "ranking_stability": {
            "tau_domain_loo": round(tau_domain_loo, 4) if tau_domain_loo is not None else None,
            "tau_domain_loo_p": round(tau_domain_loo_p, 4) if tau_domain_loo_p is not None else None,
            "tau_domain_mmlu": round(tau_domain_mmlu, 4) if tau_domain_mmlu is not None else None,
            "tau_domain_mmlu_p": round(tau_domain_mmlu_p, 4) if tau_domain_mmlu_p is not None else None,
            "tau_set_a_set_b": round(tau_set_a_b, 4) if tau_set_a_b is not None else None,
            "tau_set_a_set_b_p": round(tau_set_a_b_p, 4) if tau_set_a_b_p is not None else None,
        },
        "kill_criteria": {
            "K1_delta_prune_pct": round(k1_delta, 4) if k1_delta is not None else None,
            "K1_threshold_pct": -1.0,
            "K1_pass": k1_pass,
            "K1_interpretation": k1_interpretation,
            "K2_tau_best": round(tau_best, 4),
            "K2_threshold": 0.6,
            "K2_pass": k2_pass,
            "K3_max_discrepancy_pct": round(max_discrepancy_pct, 4)
                                      if max_discrepancy_pct is not None else None,
            "K3_threshold_pct": 0.5,
            "K3_pass": k3_pass,
            "verdict": verdict,
        },
        "timing": {
            "total_elapsed_s": round(timing.get("total_elapsed_s", elapsed()), 1),
            "phase0_setup_s": round(timing.get("phase0_setup_s", 0.0), 1),
            "phase1_reference_s": round(timing.get("phase1_reference_s", 0.0), 1),
            "phase2_loo_s": round(timing.get("phase2_loo_s", 0.0), 1),
            "phase3_accum_ppl_s": round(timing.get("phase3_accum_ppl_s", 0.0), 1),
            "phase4_accum_mmlu_s": round(timing.get("phase4_accum_mmlu_s", 0.0), 1),
            "phase5_greedy_s": round(timing.get("phase5_greedy_s", 0.0), 1),
            "phase6_removal_s": round(timing.get("phase6_removal_s", 0.0), 1),
            "phase7_stability_s": round(timing.get("phase7_stability_s", 0.0), 1),
            "phase8_scalability_s": round(timing.get("phase8_scalability_s", 0.0), 1),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = OUT_PATH.with_suffix(".tmp")
    with open(tmp_out, "w") as f:
        json.dump(results, f, indent=2)
    tmp_out.replace(OUT_PATH)

    log(f"\nResults saved to {OUT_PATH}")
    log(f"Total elapsed: {elapsed() / 60:.1f} min")
    log(f"VERDICT: {verdict}")

    # Remove checkpoint on clean finish
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        log("Checkpoint removed (clean exit).")

    return results


if __name__ == "__main__":
    main()
