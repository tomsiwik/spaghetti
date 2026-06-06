"""Experiment: Hash function load balance on consistent hash ring.

Hypothesis: xxHash/MurmurHash3 reduces hash ring load imbalance vs FNV1a.

The hash ring removal experiment (experiments/models/hash_ring_remove_expert/) found
FNV1a produces 1.8x displacement_ratio at N=8 when removing expert 0, meaning
that expert handles 22.5% of tokens vs the theoretical 12.5%. This experiment
isolates the hash function as the variable: does switching from FNV1a to a
better hash function reduce the max/min load ratio?

Kill criteria:
  K1: xxHash/MurmurHash3 load imbalance >= FNV1a 1.8x at N=8
      (hash function is not the problem)
  K2: imbalance with any hash function > 1.3x at N>=16 with V>=200 virtual nodes

Protocol:
  1. For each hash function (FNV1a, xxHash32, MurmurHash3, SHA-256, Python hash):
     a. Build a consistent hash ring with N experts and V virtual nodes
     b. Generate T=1M uniformly random query keys
     c. Route each key to its primary expert
     d. Measure load per expert
     e. Compute max/min ratio and Jain's fairness index
  2. Sweep: N in {4, 8, 16, 32, 64, 128}, V in {100, 200, 500, 1000}
  3. Repeat for 3 seeds

Metric definitions:
  - max_min_ratio = max(load) / min(load)  -- 1.0 is perfect
  - jain_index = (sum(x))^2 / (N * sum(x^2))  -- 1.0 is perfect
  - cv = std(load) / mean(load)  -- 0.0 is perfect

This is a PURE HASH UNIFORMITY experiment. No model, no training, no inference.
"""

import json
import struct
import hashlib
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import xxhash
import mmh3


# ========================================================================== #
# Hash functions: all take bytes, return uint32
# ========================================================================== #

def hash_fnv1a(data: bytes) -> int:
    """FNV-1a 32-bit (baseline from hash_ring_remove_expert)."""
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def hash_xxhash32(data: bytes) -> int:
    """xxHash 32-bit -- fast non-crypto hash with excellent distribution."""
    return xxhash.xxh32(data).intdigest()


def hash_murmurhash3(data: bytes) -> int:
    """MurmurHash3 32-bit -- widely used non-crypto hash."""
    return mmh3.hash(data, signed=False)


def hash_sha256(data: bytes) -> int:
    """SHA-256 truncated to 32 bits -- gold standard uniformity."""
    return int.from_bytes(hashlib.sha256(data).digest()[:4], 'big')


def hash_python(data: bytes) -> int:
    """Python built-in hash (PYTHONHASHSEED-dependent, 64-bit truncated)."""
    return hash(data) & 0xFFFFFFFF


HASH_FUNCTIONS = {
    "FNV1a": hash_fnv1a,
    "xxHash32": hash_xxhash32,
    "MurmurHash3": hash_murmurhash3,
    "SHA-256": hash_sha256,
    "Python_hash": hash_python,
}


# ========================================================================== #
# Hash ring with pluggable hash function
# ========================================================================== #

class HashRing:
    """Consistent hash ring with virtual nodes and pluggable hash function."""

    RING_MAX = 0xFFFFFFFF

    def __init__(self, n_experts: int, n_virtual: int, hash_fn, seed: int = 0):
        self.n_experts = n_experts
        self.n_virtual = n_virtual
        self.hash_fn = hash_fn
        self.seed = seed

        # Build ring: place virtual nodes
        self.ring = []  # sorted list of (position, expert_id)
        for expert_id in range(n_experts):
            for v in range(n_virtual):
                # Use expert_id + virtual_node_index + seed as key
                key = struct.pack(">III", expert_id, v, seed)
                pos = hash_fn(key)
                self.ring.append((pos, expert_id))
        self.ring.sort(key=lambda x: x[0])

        # Pre-compute sorted positions for fast binary search
        self._positions = np.array([p for p, _ in self.ring], dtype=np.int64)
        self._experts = np.array([e for _, e in self.ring], dtype=np.int32)

    def find_primary_batch(self, query_hashes: np.ndarray) -> np.ndarray:
        """Route a batch of query hashes to their primary experts.

        Uses numpy searchsorted for vectorized binary search.
        """
        # searchsorted finds insertion points
        indices = np.searchsorted(self._positions, query_hashes, side='left')
        # Wrap around: if past the end, go to first node
        indices = indices % len(self.ring)
        return self._experts[indices]


def measure_load_balance(n_experts: int, n_virtual: int, hash_fn,
                          n_queries: int, seed: int) -> dict:
    """Measure load balance for a given configuration.

    Returns metrics: max_min_ratio, jain_index, cv, per_expert_load.
    """
    ring = HashRing(n_experts, n_virtual, hash_fn, seed=seed)

    # Generate uniform random query hashes
    rng = np.random.RandomState(seed)
    query_hashes = rng.randint(0, 0xFFFFFFFF, size=n_queries, dtype=np.int64)

    # Route all queries
    assignments = ring.find_primary_batch(query_hashes)

    # Count per-expert load
    unique, counts = np.unique(assignments, return_counts=True)
    load = np.zeros(n_experts, dtype=np.int64)
    for u, c in zip(unique, counts):
        load[u] = c

    # Metrics
    load_f = load.astype(np.float64)
    max_load = load_f.max()
    min_load = load_f.min()
    mean_load = load_f.mean()
    std_load = load_f.std()

    max_min_ratio = max_load / min_load if min_load > 0 else float('inf')
    cv = std_load / mean_load if mean_load > 0 else float('inf')

    # Jain's fairness index: J = (sum x_i)^2 / (N * sum x_i^2)
    jain = (load_f.sum() ** 2) / (n_experts * (load_f ** 2).sum()) if n_experts > 0 else 0.0

    # Theoretical perfect load
    theoretical = n_queries / n_experts

    return {
        "max_min_ratio": float(max_min_ratio),
        "jain_index": float(jain),
        "cv": float(cv),
        "max_load_pct": float(max_load / n_queries * 100),
        "min_load_pct": float(min_load / n_queries * 100),
        "theoretical_pct": float(theoretical / n_queries * 100),
        "max_expert": int(load_f.argmax()),
        "min_expert": int(load_f.argmin()),
        "per_expert_pct": [float(l / n_queries * 100) for l in load],
    }


def run_full_experiment():
    """Run the complete hash function load balance experiment."""
    print("=" * 75)
    print("  EXPERIMENT: Hash Function Load Balance on Consistent Hash Ring")
    print("  K1: xxHash/MurmurHash3 imbalance >= FNV1a 1.8x at N=8 -> KILL")
    print("  K2: any hash > 1.3x at N>=16 with V>=200 -> KILL")
    print("=" * 75)

    N_VALUES = [4, 8, 16, 32, 64, 128]
    V_VALUES = [100, 200, 500, 1000]
    SEEDS = [42, 123, 777]
    N_QUERIES = 1_000_000  # 1M queries for statistical stability

    all_results = []

    # ================================================================
    # Test 1: Full sweep across hash functions, N, V
    # ================================================================
    print("\n" + "=" * 75)
    print("  TEST 1: Full parameter sweep")
    print("=" * 75)

    for V in V_VALUES:
        print(f"\n--- V={V} virtual nodes ---")
        print(f"{'Hash':>14} {'N':>4} {'MaxMin':>8} {'Jain':>8} {'CV':>8} "
              f"{'Max%':>7} {'Min%':>7} {'Theory%':>8}")
        print("-" * 75)

        for hash_name, hash_fn in HASH_FUNCTIONS.items():
            for N in N_VALUES:
                ratios = []
                jains = []
                cvs = []
                maxes = []
                mins = []

                for seed in SEEDS:
                    r = measure_load_balance(N, V, hash_fn, N_QUERIES, seed)
                    ratios.append(r["max_min_ratio"])
                    jains.append(r["jain_index"])
                    cvs.append(r["cv"])
                    maxes.append(r["max_load_pct"])
                    mins.append(r["min_load_pct"])

                    all_results.append({
                        "hash_function": hash_name,
                        "N": N,
                        "V": V,
                        "seed": seed,
                        **r,
                    })

                mean_ratio = np.mean(ratios)
                mean_jain = np.mean(jains)
                mean_cv = np.mean(cvs)
                mean_max = np.mean(maxes)
                mean_min = np.mean(mins)
                theory = 100.0 / N

                print(f"{hash_name:>14} {N:>4} {mean_ratio:>8.3f} "
                      f"{mean_jain:>8.6f} {mean_cv:>8.4f} "
                      f"{mean_max:>7.2f} {mean_min:>7.2f} {theory:>8.2f}")

    # ================================================================
    # Test 2: Focused N=8 comparison (matches original experiment)
    # ================================================================
    print("\n" + "=" * 75)
    print("  TEST 2: Focused N=8 comparison at V=150 (original config)")
    print("=" * 75)

    V = 150
    N = 8
    print(f"\n{'Hash':>14} {'Seed':>6} {'MaxMin':>8} {'Jain':>8} "
          f"{'Max%':>7} {'Min%':>7} {'MaxExp':>7} {'MinExp':>7}")
    print("-" * 75)

    for hash_name, hash_fn in HASH_FUNCTIONS.items():
        for seed in SEEDS:
            r = measure_load_balance(N, V, hash_fn, N_QUERIES, seed)
            all_results.append({
                "hash_function": hash_name,
                "N": N,
                "V": V,
                "seed": seed,
                "test": "focused_n8",
                **r,
            })
            print(f"{hash_name:>14} {seed:>6} {r['max_min_ratio']:>8.3f} "
                  f"{r['jain_index']:>8.6f} "
                  f"{r['max_load_pct']:>7.2f} {r['min_load_pct']:>7.2f} "
                  f"{r['max_expert']:>7} {r['min_expert']:>7}")

    # ================================================================
    # Test 3: Ring segment distribution analysis
    # ================================================================
    print("\n" + "=" * 75)
    print("  TEST 3: Ring arc analysis (what fraction of ring each expert owns)")
    print("=" * 75)

    N = 8
    V = 150
    seed = 42

    for hash_name, hash_fn in HASH_FUNCTIONS.items():
        ring = HashRing(N, V, hash_fn, seed=seed)
        positions = ring._positions
        experts = ring._experts

        # Compute arc lengths for each expert
        # Each virtual node owns the arc from the previous node to itself
        arc_by_expert = defaultdict(float)
        total_ring = 0xFFFFFFFF

        for i in range(len(positions)):
            prev_pos = positions[i - 1] if i > 0 else positions[-1] - total_ring
            arc_len = (positions[i] - prev_pos) % total_ring
            # This arc is "owned" by expert at position i
            # Actually, in consistent hashing, a node owns the arc BEFORE it
            # (tokens in that arc route to this node)
            arc_by_expert[experts[i]] += arc_len

        total_arc = sum(arc_by_expert.values())
        pcts = {e: arc / total_arc * 100 for e, arc in sorted(arc_by_expert.items())}

        print(f"\n  {hash_name} (N={N}, V={V}):")
        for e in range(N):
            bar_len = int(pcts.get(e, 0) / 100 * 50)
            print(f"    Expert {e}: {pcts.get(e, 0):>6.2f}%  {'#' * bar_len}")

        pct_vals = list(pcts.values())
        ratio = max(pct_vals) / min(pct_vals) if min(pct_vals) > 0 else float('inf')
        print(f"    Max/Min ratio: {ratio:.3f}")

    # ================================================================
    # Aggregate analysis
    # ================================================================
    print("\n" + "=" * 75)
    print("  AGGREGATE: Mean max/min ratio by hash function and N")
    print("=" * 75)

    # Group by (hash, N, V) and average over seeds
    from collections import defaultdict as dd
    grouped = dd(list)
    for r in all_results:
        if "test" not in r:  # Only from Test 1
            key = (r["hash_function"], r["N"], r["V"])
            grouped[key].append(r["max_min_ratio"])

    print(f"\n{'':>14}", end="")
    for N in N_VALUES:
        print(f"  N={N:>3}", end="")
    print()
    print("-" * (14 + len(N_VALUES) * 8))

    for V in V_VALUES:
        print(f"\n  V={V}:")
        for hash_name in HASH_FUNCTIONS:
            print(f"  {hash_name:>12}", end="")
            for N in N_VALUES:
                key = (hash_name, N, V)
                vals = grouped.get(key, [])
                if vals:
                    print(f"  {np.mean(vals):>6.3f}", end="")
                else:
                    print(f"  {'---':>6}", end="")
            print()

    # ================================================================
    # Kill criteria assessment
    # ================================================================
    print("\n" + "=" * 75)
    print("  KILL CRITERIA ASSESSMENT")
    print("=" * 75)

    # K1: xxHash/MurmurHash3 imbalance >= FNV1a 1.8x at N=8
    fnv_n8 = [r["max_min_ratio"] for r in all_results
              if r["hash_function"] == "FNV1a" and r["N"] == 8 and "test" not in r]
    xxh_n8 = [r["max_min_ratio"] for r in all_results
              if r["hash_function"] == "xxHash32" and r["N"] == 8 and "test" not in r]
    mmh_n8 = [r["max_min_ratio"] for r in all_results
              if r["hash_function"] == "MurmurHash3" and r["N"] == 8 and "test" not in r]
    sha_n8 = [r["max_min_ratio"] for r in all_results
              if r["hash_function"] == "SHA-256" and r["N"] == 8 and "test" not in r]

    print(f"\n  K1: xxHash/Murmur imbalance at N=8 vs FNV1a 1.8x threshold")
    print(f"    FNV1a mean max/min:      {np.mean(fnv_n8):.3f} (across all V, 3 seeds)")
    print(f"    xxHash32 mean max/min:   {np.mean(xxh_n8):.3f}")
    print(f"    MurmurHash3 mean max/min: {np.mean(mmh_n8):.3f}")
    print(f"    SHA-256 mean max/min:    {np.mean(sha_n8):.3f}")

    # Best alternative hash at N=8
    best_alt = min(np.mean(xxh_n8), np.mean(mmh_n8))
    fnv_mean = np.mean(fnv_n8)
    k1_pass = best_alt < fnv_mean  # alternative must be BETTER (lower) than FNV1a
    # But kill if best_alt >= 1.8x
    k1_killed = best_alt >= 1.8

    print(f"\n    Best alternative: {best_alt:.3f}")
    print(f"    FNV1a: {fnv_mean:.3f}")
    if k1_killed:
        print(f"    K1 VERDICT: KILL (best alternative {best_alt:.3f} >= 1.8x)")
    elif k1_pass:
        print(f"    K1 VERDICT: PASS (alternative {best_alt:.3f} < FNV1a {fnv_mean:.3f})")
    else:
        print(f"    K1 VERDICT: KILL (alternative {best_alt:.3f} >= FNV1a {fnv_mean:.3f})")

    # K2: any hash > 1.3x at N>=16 with V>=200
    k2_violations = []
    for r in all_results:
        if r["N"] >= 16 and r["V"] >= 200 and "test" not in r:
            if r["max_min_ratio"] > 1.3:
                k2_violations.append(r)

    n_k2_configs = len([r for r in all_results
                        if r["N"] >= 16 and r["V"] >= 200 and "test" not in r])

    print(f"\n  K2: any hash > 1.3x at N>=16, V>=200")
    print(f"    Configs tested: {n_k2_configs}")
    print(f"    Violations: {len(k2_violations)}")

    if k2_violations:
        print("    Worst violations:")
        k2_violations.sort(key=lambda r: -r["max_min_ratio"])
        for v in k2_violations[:5]:
            print(f"      {v['hash_function']:>14} N={v['N']:>3} V={v['V']:>4} "
                  f"seed={v['seed']} ratio={v['max_min_ratio']:.3f}")

    k2_pass = len(k2_violations) == 0
    print(f"    K2 VERDICT: {'PASS' if k2_pass else 'KILL'}")

    # ================================================================
    # Summary statistics table
    # ================================================================
    print("\n" + "=" * 75)
    print("  SUMMARY: Best hash function per N (averaged over V and seeds)")
    print("=" * 75)

    for N in N_VALUES:
        best_hash = None
        best_ratio = float('inf')
        for hash_name in HASH_FUNCTIONS:
            vals = [r["max_min_ratio"] for r in all_results
                    if r["hash_function"] == hash_name and r["N"] == N
                    and "test" not in r]
            if vals:
                m = np.mean(vals)
                if m < best_ratio:
                    best_ratio = m
                    best_hash = hash_name
        print(f"  N={N:>3}: best={best_hash:>14} ratio={best_ratio:.3f}")

    # ================================================================
    # Scaling analysis: how imbalance scales with V
    # ================================================================
    print("\n" + "=" * 75)
    print("  SCALING: Imbalance vs virtual node count (N=8, SHA-256)")
    print("=" * 75)

    print(f"\n  {'V':>6} {'MaxMin':>8} {'Jain':>10} {'CV':>8}")
    print("  " + "-" * 38)

    for V in V_VALUES:
        vals = [r for r in all_results
                if r["hash_function"] == "SHA-256" and r["N"] == 8
                and r["V"] == V and "test" not in r]
        if vals:
            mr = np.mean([v["max_min_ratio"] for v in vals])
            ji = np.mean([v["jain_index"] for v in vals])
            cv = np.mean([v["cv"] for v in vals])
            print(f"  {V:>6} {mr:>8.3f} {ji:>10.7f} {cv:>8.5f}")

    # Theory: with V virtual nodes per expert, the expected max/min ratio
    # follows from the maximum of N multinomials with NV bins.
    # For large V: max load ~ (1 + sqrt(2*ln(N)/(NV))) * (1/N)
    print(f"\n  Theoretical (Poisson balls-into-bins):")
    print(f"  {'V':>6} {'Expected':>10}")
    print("  " + "-" * 20)
    for V in V_VALUES:
        N = 8
        # max/min ratio approximation: (1 + c*sqrt(ln(N)/(N*V))) / (1 - c*sqrt(ln(N)/(N*V)))
        c = np.sqrt(2 * np.log(N) / (N * V))
        approx_ratio = (1 + c) / (1 - c) if c < 1 else float('inf')
        print(f"  {V:>6} {approx_ratio:>10.4f}")

    # Save results
    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n  Results saved to {results_path}")

    # Save aggregate summary
    summary = {
        "k1_pass": k1_pass and not k1_killed,
        "k2_pass": k2_pass,
        "overall": "PASS" if (k1_pass and not k1_killed and k2_pass) else "KILL",
        "fnv1a_n8_mean_ratio": float(fnv_mean),
        "xxhash_n8_mean_ratio": float(np.mean(xxh_n8)),
        "murmur_n8_mean_ratio": float(np.mean(mmh_n8)),
        "sha256_n8_mean_ratio": float(np.mean(sha_n8)),
        "best_alternative_n8": float(best_alt),
        "k2_violations": len(k2_violations),
        "k2_configs_tested": n_k2_configs,
    }

    summary_path = Path(__file__).parent / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")

    return all_results, summary


if __name__ == "__main__":
    t0 = time.time()
    results, summary = run_full_experiment()
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Overall verdict: {summary['overall']}")
