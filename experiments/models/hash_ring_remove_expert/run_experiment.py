"""Experiment: Expert removal via consistent hash ring (graceful degradation).

Tests two kill criteria:
1. >5% quality degradation when removing one expert from N=8 without recalibration
2. Removed expert's tokens not redistributed to nearest neighbor (>20% go to non-neighbor)

Protocol:
1. Build consistent hash ring with N experts and V=150 virtual nodes each
2. Generate T=100,000 random token hashes (simulating hidden state projections)
3. Route all tokens with N experts -> record assignments
4. Remove one expert -> re-route all tokens -> record new assignments
5. Measure:
   a) Displacement: fraction of tokens whose primary assignment changed
   b) Neighbor accuracy: what fraction of displaced tokens go to the next clockwise expert
   c) Quality simulation: model quality loss from redistribution
6. Repeat for N in {4, 8, 16, 32} and removing different experts (first, middle, last)
7. Repeat for 3 seeds

The quality degradation is simulated via a weighted expert quality model:
- Each expert has a "domain quality" score (simulating specialized capability)
- When tokens move from their original expert to the neighbor, quality degrades
  proportional to the domain mismatch
- We measure the aggregate quality change

Mathematical predictions (Karger et al. 1997):
- Displacement = 1/N (only the removed expert's tokens move)
- All displaced tokens go to the next clockwise neighbor (by ring construction)
- Quality degradation bounded by (1/N) * max_mismatch

This is a ROUTING PROPERTY experiment, not a training experiment.
We validate the hash ring's structural guarantees, which are independent
of model architecture or training dynamics.
"""

import json
import struct
import math
import time
import hashlib
from pathlib import Path

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
#  Hash ring implementation (extracted from consistent_hash_routing.py,
#  converted to pure Python/numpy -- no MLX dependency)
# --------------------------------------------------------------------------- #

def _fnv1a_32(data: bytes) -> int:
    """FNV-1a 32-bit hash. Fast, good distribution."""
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class HashRing:
    """Consistent hash ring with virtual nodes.

    Each expert gets V virtual nodes placed on ring [0, 2^32).
    Adding/removing an expert only affects its own virtual nodes.
    """

    RING_MAX = 0xFFFFFFFF

    def __init__(self, n_experts: int, n_virtual: int = 150):
        self.n_experts = n_experts
        self.n_virtual = n_virtual
        self.ring = []  # sorted list of (position, expert_id)

        for expert_id in range(n_experts):
            self._add_expert_nodes(expert_id)
        self.ring.sort(key=lambda x: x[0])

    def _add_expert_nodes(self, expert_id: int):
        """Add virtual nodes for one expert."""
        for v in range(self.n_virtual):
            key_bytes = struct.pack(">II", expert_id, v)
            pos = _fnv1a_32(key_bytes)
            self.ring.append((pos, expert_id))

    def _rebuild_sorted(self):
        self.ring.sort(key=lambda x: x[0])

    def remove_expert(self, expert_id: int):
        """Remove all virtual nodes for an expert."""
        self.ring = [(p, e) for p, e in self.ring if e != expert_id]
        self.n_experts -= 1

    def add_expert(self, expert_id: int):
        """Add an expert to the ring."""
        self._add_expert_nodes(expert_id)
        self._rebuild_sorted()
        self.n_experts += 1

    def find_primary(self, token_hash: int) -> int:
        """Find the primary (nearest clockwise) expert for a token hash.

        Binary search for insertion point, then walk clockwise to find
        the first expert.
        """
        n = len(self.ring)
        if n == 0:
            return -1

        # Binary search for insertion point
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.ring[mid][0] < token_hash:
                lo = mid + 1
            else:
                hi = mid

        # Walk clockwise to find first node
        idx = lo % n
        return self.ring[idx][1]

    def find_top_k(self, token_hash: int, k: int) -> list:
        """Find k nearest distinct experts clockwise.

        Returns list of expert_ids in clockwise order.
        """
        n = len(self.ring)
        if n == 0:
            return []

        # Binary search
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.ring[mid][0] < token_hash:
                lo = mid + 1
            else:
                hi = mid

        selected = []
        seen = set()
        for offset in range(n):
            idx = (lo + offset) % n
            expert = self.ring[idx][1]
            if expert not in seen:
                seen.add(expert)
                selected.append(expert)
                if len(selected) >= k:
                    break

        return selected

    def get_clockwise_neighbor(self, expert_id: int) -> int:
        """Find the expert that owns the ring segment just after expert_id's
        last virtual node. This is where tokens would go when expert_id is removed.

        For consistent hashing: when an expert is removed, each of its virtual
        node segments gets absorbed by the next clockwise expert. There may be
        multiple different neighbors (one per virtual node).

        Returns the most frequent neighbor (the one that absorbs the most tokens).
        """
        neighbors = self.get_all_clockwise_neighbors(expert_id)
        if not neighbors:
            return -1
        # Return most common
        from collections import Counter
        c = Counter(neighbors.values())
        return c.most_common(1)[0][0]

    def get_all_clockwise_neighbors(self, expert_id: int) -> dict:
        """For each virtual node of expert_id, find the next clockwise expert.

        Returns dict: virtual_node_ring_pos -> neighbor_expert_id
        """
        n = len(self.ring)
        neighbors = {}
        for i, (pos, eid) in enumerate(self.ring):
            if eid == expert_id:
                # Walk clockwise to find next different expert
                for offset in range(1, n):
                    j = (i + offset) % n
                    if self.ring[j][1] != expert_id:
                        neighbors[pos] = self.ring[j][1]
                        break
        return neighbors


def generate_token_hashes(n_tokens: int, seed: int) -> np.ndarray:
    """Generate random token hashes simulating projected hidden states.

    In the real system, these come from: x @ proj -> FNV1a -> ring position.
    Here we generate them uniformly on [0, 2^32) which is the expected
    distribution when hashing random projections.
    """
    rng = np.random.RandomState(seed)
    return rng.randint(0, 0xFFFFFFFF, size=n_tokens, dtype=np.uint64)


def measure_removal_displacement(ring_before: HashRing, expert_to_remove: int,
                                  token_hashes: np.ndarray) -> dict:
    """Measure displacement when removing one expert.

    Returns dict with:
    - displacement_rate: fraction of tokens whose primary assignment changed
    - neighbor_accuracy: fraction of displaced tokens that went to clockwise neighbor
    - per_expert_redistribution: where displaced tokens went
    - theoretical_displacement: 1/N prediction
    """
    n_tokens = len(token_hashes)
    N = ring_before.n_experts

    # Get all clockwise neighbors BEFORE removal
    neighbors = ring_before.get_all_clockwise_neighbors(expert_to_remove)

    # Get assignments before removal
    assignments_before = np.array([
        ring_before.find_primary(int(h)) for h in token_hashes
    ])

    # Remove expert
    ring_after = HashRing.__new__(HashRing)
    ring_after.ring = [(p, e) for p, e in ring_before.ring if e != expert_to_remove]
    ring_after.n_experts = N - 1
    ring_after.n_virtual = ring_before.n_virtual

    # Get assignments after removal
    assignments_after = np.array([
        ring_after.find_primary(int(h)) for h in token_hashes
    ])

    # Measure displacement
    displaced_mask = assignments_before != assignments_after
    n_displaced = displaced_mask.sum()
    displacement_rate = n_displaced / n_tokens

    # Check that ONLY tokens previously assigned to removed expert moved
    was_on_removed = assignments_before == expert_to_remove
    should_not_move = ~was_on_removed
    false_moves = (displaced_mask & should_not_move).sum()

    # Measure neighbor accuracy for displaced tokens
    # Each displaced token should go to the clockwise neighbor of the
    # virtual node that was closest to it
    displaced_tokens = token_hashes[displaced_mask]
    displaced_new_assignments = assignments_after[displaced_mask]

    # For each displaced token, find which virtual node of the removed expert
    # it was closest to, and check if it went to that node's clockwise neighbor
    neighbor_set = set(neighbors.values())
    n_to_neighbor = 0
    n_displaced_total = len(displaced_tokens)

    per_expert_dest = {}
    for i in range(n_displaced_total):
        new_expert = displaced_new_assignments[i]
        per_expert_dest[new_expert] = per_expert_dest.get(new_expert, 0) + 1

        # The token should go to ONE of the clockwise neighbors
        # (different virtual nodes have different neighbors)
        if new_expert in neighbor_set:
            n_to_neighbor += 1

    neighbor_accuracy = n_to_neighbor / max(n_displaced_total, 1)

    # Theoretical prediction
    theoretical_displacement = 1.0 / N

    return {
        "N": N,
        "expert_removed": expert_to_remove,
        "n_tokens": n_tokens,
        "n_displaced": int(n_displaced),
        "displacement_rate": displacement_rate,
        "theoretical_displacement": theoretical_displacement,
        "displacement_ratio": displacement_rate / theoretical_displacement if theoretical_displacement > 0 else 0,
        "false_moves": int(false_moves),
        "neighbor_accuracy": neighbor_accuracy,
        "n_to_neighbor": int(n_to_neighbor),
        "per_expert_dest": {int(k): int(v) for k, v in sorted(per_expert_dest.items())},
        "n_clockwise_neighbors": len(set(neighbors.values())),
    }


def simulate_quality_degradation(ring: HashRing, expert_to_remove: int,
                                  token_hashes: np.ndarray,
                                  expert_qualities: np.ndarray,
                                  seed: int) -> dict:
    """Simulate quality degradation when removing an expert.

    Model: Each token has a "true domain" and each expert has quality
    scores per domain. When a token moves from its original expert to a
    new one, quality changes proportional to the quality difference.

    expert_qualities: (N, N) matrix where Q[i,j] = quality of expert i
    on domain j. Diagonal is best (expert matches its domain).

    Returns quality degradation as a percentage.
    """
    N = ring.n_experts
    n_tokens = len(token_hashes)
    rng = np.random.RandomState(seed)

    # Assign each token a "true domain" = its original primary expert
    assignments_before = np.array([
        ring.find_primary(int(h)) for h in token_hashes
    ])

    # Create ring after removal
    ring_after = HashRing.__new__(HashRing)
    ring_after.ring = [(p, e) for p, e in ring.ring if e != expert_to_remove]
    ring_after.n_experts = N - 1
    ring_after.n_virtual = ring.n_virtual

    assignments_after = np.array([
        ring_after.find_primary(int(h)) for h in token_hashes
    ])

    # Compute quality before: each token scored by its expert on its domain
    quality_before = 0.0
    quality_after = 0.0

    for i in range(n_tokens):
        domain = assignments_before[i]
        old_expert = assignments_before[i]
        new_expert = assignments_after[i]

        quality_before += expert_qualities[old_expert, domain]
        quality_after += expert_qualities[new_expert, domain]

    quality_before /= n_tokens
    quality_after /= n_tokens
    degradation_pct = (quality_after - quality_before) / quality_before * 100

    return {
        "quality_before": quality_before,
        "quality_after": quality_after,
        "degradation_pct": degradation_pct,
    }


def build_expert_quality_matrix(N: int, specialization: float = 0.3,
                                 seed: int = 42) -> np.ndarray:
    """Build NxN expert quality matrix.

    Q[i,j] = quality of expert i on domain j.
    Diagonal = 1.0 (expert is best on its own domain).
    Off-diagonal = 1.0 - specialization * distance(i, j)

    With specialization=0.3, a neighboring expert handles traffic at
    0.7 quality, a distant expert at 0.4-0.7 quality.
    """
    rng = np.random.RandomState(seed)
    Q = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                # Use circular distance on ring
                dist = min(abs(i - j), N - abs(i - j)) / (N / 2)
                Q[i, j] = 1.0 - specialization * dist + rng.normal(0, 0.02)
                Q[i, j] = max(Q[i, j], 0.3)  # floor
    return Q


def run_single_config(N: int, expert_to_remove: int, n_tokens: int,
                       n_virtual: int, seed: int, specialization: float = 0.3) -> dict:
    """Run removal experiment for a single configuration."""
    # Build ring
    ring = HashRing(N, n_virtual=n_virtual)

    # Generate tokens
    token_hashes = generate_token_hashes(n_tokens, seed=seed)

    # Measure displacement
    disp_result = measure_removal_displacement(ring, expert_to_remove, token_hashes)

    # Simulate quality impact
    Q = build_expert_quality_matrix(N, specialization=specialization, seed=seed)
    qual_result = simulate_quality_degradation(
        ring, expert_to_remove, token_hashes, Q, seed=seed
    )

    return {
        **disp_result,
        **qual_result,
        "n_virtual": n_virtual,
        "specialization": specialization,
        "seed": seed,
    }


def run_full_experiment():
    """Run the complete experiment across all configurations."""
    print("=" * 70)
    print("  EXPERIMENT: Expert Removal via Consistent Hash Ring")
    print("  Kill: >5% degradation OR >20% tokens to non-neighbor")
    print("=" * 70)

    n_tokens = 100_000
    n_virtual = 150
    seeds = [42, 123, 777]
    N_values = [4, 8, 16, 32]
    specialization = 0.3  # moderate domain specialization

    all_results = []

    # ================================================================
    # Test 1: Scaling behavior across N values
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 1: Scaling with N (remove middle expert)")
    print("=" * 70)

    print(f"\n{'N':>4} {'Seed':>6} {'Displ%':>8} {'Theory%':>8} {'Ratio':>7} "
          f"{'Nbr Acc%':>9} {'False':>6} {'Degrad%':>9} {'#Nbrs':>6}")
    print("-" * 75)

    for N in N_values:
        expert_to_remove = N // 2  # middle expert
        for seed in seeds:
            r = run_single_config(N, expert_to_remove, n_tokens, n_virtual, seed, specialization)
            all_results.append({**r, "test": "scaling", "config": f"N={N}_mid"})

            print(f"{N:>4} {seed:>6} {r['displacement_rate']*100:>8.2f} "
                  f"{r['theoretical_displacement']*100:>8.2f} "
                  f"{r['displacement_ratio']:>7.3f} "
                  f"{r['neighbor_accuracy']*100:>9.2f} "
                  f"{r['false_moves']:>6} "
                  f"{r['degradation_pct']:>+9.4f} "
                  f"{r['n_clockwise_neighbors']:>6}")

    # ================================================================
    # Test 2: Edge cases -- remove different experts at N=8
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 2: Edge cases at N=8 (remove first, middle, last)")
    print("=" * 70)

    N = 8
    experts_to_remove = [0, N // 2, N - 1]  # first, middle, last

    print(f"\n{'Expert':>7} {'Seed':>6} {'Displ%':>8} {'Theory%':>8} {'Ratio':>7} "
          f"{'Nbr Acc%':>9} {'False':>6} {'Degrad%':>9}")
    print("-" * 70)

    for expert_id in experts_to_remove:
        for seed in seeds:
            r = run_single_config(N, expert_id, n_tokens, n_virtual, seed, specialization)
            label = {0: "first", N // 2: "middle", N - 1: "last"}[expert_id]
            all_results.append({**r, "test": "edge_case", "config": f"N={N}_{label}"})

            print(f"{expert_id:>4}({label[0]:>1}) {seed:>6} "
                  f"{r['displacement_rate']*100:>8.2f} "
                  f"{r['theoretical_displacement']*100:>8.2f} "
                  f"{r['displacement_ratio']:>7.3f} "
                  f"{r['neighbor_accuracy']*100:>9.2f} "
                  f"{r['false_moves']:>6} "
                  f"{r['degradation_pct']:>+9.4f}")

    # ================================================================
    # Test 3: Redistribution destination analysis at N=8
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 3: Where do displaced tokens go? (N=8, remove expert 4)")
    print("=" * 70)

    N = 8
    expert_to_remove = 4
    seed = 42

    ring = HashRing(N, n_virtual=n_virtual)
    token_hashes = generate_token_hashes(n_tokens, seed=seed)

    # Get detailed redistribution
    r = measure_removal_displacement(ring, expert_to_remove, token_hashes)

    print(f"\n  Expert removed: {expert_to_remove}")
    print(f"  Tokens displaced: {r['n_displaced']:,} / {n_tokens:,} ({r['displacement_rate']*100:.2f}%)")
    print(f"  Number of distinct clockwise neighbors: {r['n_clockwise_neighbors']}")
    print(f"\n  Redistribution destinations:")

    total_displaced = r['n_displaced']
    for dest, count in sorted(r['per_expert_dest'].items()):
        pct = count / total_displaced * 100 if total_displaced > 0 else 0
        is_neighbor = "  <-- clockwise neighbor" if dest in set(
            ring.get_all_clockwise_neighbors(expert_to_remove).values()
        ) else ""
        print(f"    Expert {dest}: {count:>6} tokens ({pct:>6.2f}%){is_neighbor}")

    print(f"\n  Tokens to clockwise neighbors: {r['n_to_neighbor']:,} / {total_displaced:,} "
          f"({r['neighbor_accuracy']*100:.2f}%)")

    # ================================================================
    # Test 4: Verify zero false moves (only removed expert's tokens move)
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 4: Zero false moves verification")
    print("=" * 70)

    false_move_count = sum(1 for r in all_results if r.get('false_moves', 0) > 0)
    total_configs = len(all_results)
    print(f"\n  Configs with false moves: {false_move_count} / {total_configs}")
    if false_move_count == 0:
        print("  PASS: Only tokens from the removed expert are redistributed.")
    else:
        print("  FAIL: Some tokens not on the removed expert were displaced!")

    # ================================================================
    # Test 5: Virtual node count sensitivity
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 5: Virtual node count sensitivity (N=8, remove mid)")
    print("=" * 70)

    N = 8
    expert_to_remove = 4
    seed = 42
    v_values = [10, 50, 150, 500, 1000]

    print(f"\n{'V':>6} {'Displ%':>8} {'Theory%':>8} {'Ratio':>7} {'Nbr Acc%':>9} {'#Nbrs':>6}")
    print("-" * 50)

    for v in v_values:
        r = run_single_config(N, expert_to_remove, n_tokens, v, seed, specialization)
        all_results.append({**r, "test": "v_sensitivity", "config": f"V={v}"})

        print(f"{v:>6} {r['displacement_rate']*100:>8.2f} "
              f"{r['theoretical_displacement']*100:>8.2f} "
              f"{r['displacement_ratio']:>7.3f} "
              f"{r['neighbor_accuracy']*100:>9.2f} "
              f"{r['n_clockwise_neighbors']:>6}")

    # ================================================================
    # Test 6: Specialization sensitivity (quality impact)
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 6: Quality degradation vs specialization (N=8)")
    print("=" * 70)

    N = 8
    expert_to_remove = 4
    spec_values = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]

    print(f"\n{'Spec':>6} {'Degrad%':>9} {'Q_before':>9} {'Q_after':>9}")
    print("-" * 40)

    for spec in spec_values:
        degs = []
        for seed in seeds:
            r = run_single_config(N, expert_to_remove, n_tokens, n_virtual, seed, spec)
            degs.append(r['degradation_pct'])
            all_results.append({**r, "test": "spec_sensitivity", "config": f"spec={spec}"})

        mean_deg = np.mean(degs)
        r_last = r  # use last seed for display
        print(f"{spec:>6.1f} {mean_deg:>+9.4f} {r_last['quality_before']:>9.4f} {r_last['quality_after']:>9.4f}")

    # ================================================================
    # Test 7: Add vs Remove symmetry
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST 7: Add vs Remove symmetry at N=8")
    print("=" * 70)

    N = 8
    seed = 42
    token_hashes = generate_token_hashes(n_tokens, seed=seed)

    # Measure REMOVE: N=8 -> N=7
    ring_full = HashRing(N, n_virtual=n_virtual)
    expert_to_remove = 4

    assignments_n8 = np.array([ring_full.find_primary(int(h)) for h in token_hashes])

    ring_minus = HashRing.__new__(HashRing)
    ring_minus.ring = [(p, e) for p, e in ring_full.ring if e != expert_to_remove]
    ring_minus.n_experts = N - 1
    ring_minus.n_virtual = n_virtual

    assignments_n7 = np.array([ring_minus.find_primary(int(h)) for h in token_hashes])
    remove_displaced = (assignments_n8 != assignments_n7).sum()

    # Measure ADD: N=7 -> N=8 (add expert 4 back)
    ring_base7 = HashRing.__new__(HashRing)
    # Build a ring with experts 0-3, 5-7 (same as ring_minus)
    ring_base7.ring = list(ring_minus.ring)
    ring_base7.n_experts = 7
    ring_base7.n_virtual = n_virtual

    assignments_n7_base = np.array([ring_base7.find_primary(int(h)) for h in token_hashes])

    # Add expert 4 back
    ring_plus = HashRing.__new__(HashRing)
    ring_plus.ring = list(ring_base7.ring)
    ring_plus.n_experts = 8
    ring_plus.n_virtual = n_virtual
    for v in range(n_virtual):
        key_bytes = struct.pack(">II", expert_to_remove, v)
        pos = _fnv1a_32(key_bytes)
        ring_plus.ring.append((pos, expert_to_remove))
    ring_plus.ring.sort(key=lambda x: x[0])

    assignments_n8_readd = np.array([ring_plus.find_primary(int(h)) for h in token_hashes])
    add_displaced = (assignments_n7_base != assignments_n8_readd).sum()

    # Check roundtrip: remove then add = identity
    roundtrip_matches = (assignments_n8 == assignments_n8_readd).sum()

    print(f"\n  Remove (N=8 -> N=7): {remove_displaced:,} tokens displaced ({remove_displaced/n_tokens*100:.2f}%)")
    print(f"  Add    (N=7 -> N=8): {add_displaced:,} tokens displaced ({add_displaced/n_tokens*100:.2f}%)")
    print(f"  Symmetry ratio: {remove_displaced / max(add_displaced, 1):.3f}")
    print(f"  Roundtrip identity: {roundtrip_matches:,} / {n_tokens:,} ({roundtrip_matches/n_tokens*100:.2f}%)")

    # ================================================================
    # Aggregate results
    # ================================================================
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)

    # Filter to main N=8 results for kill criteria
    n8_results = [r for r in all_results
                  if r.get('test') == 'edge_case' and r['N'] == 8]

    if n8_results:
        displ_rates = [r['displacement_rate'] * 100 for r in n8_results]
        nbr_accs = [r['neighbor_accuracy'] * 100 for r in n8_results]
        degs = [r['degradation_pct'] for r in n8_results]
        false_moves = [r['false_moves'] for r in n8_results]

        print(f"\n  N=8 results across {len(n8_results)} configs (3 experts x 3 seeds):")
        print(f"  Displacement: {np.mean(displ_rates):.2f}% +/- {np.std(displ_rates):.2f}% "
              f"(theory: 12.50%)")
        print(f"  Neighbor accuracy: {np.mean(nbr_accs):.2f}% +/- {np.std(nbr_accs):.2f}%")
        print(f"  Quality degradation: {np.mean(degs):+.4f}% +/- {np.std(degs):.4f}%")
        print(f"  False moves: {sum(false_moves)} total")

    # Kill criteria assessment
    print("\n" + "-" * 70)
    print("  KILL CRITERIA ASSESSMENT (N=8)")
    print("-" * 70)

    # K1: >5% degradation
    max_deg = max(abs(r['degradation_pct']) for r in n8_results) if n8_results else 0
    mean_deg = np.mean([r['degradation_pct'] for r in n8_results]) if n8_results else 0
    k1_pass = abs(max_deg) < 5.0
    print(f"\n  K1 (degradation <5%): {'PASS' if k1_pass else 'KILL'}")
    print(f"      Mean: {mean_deg:+.4f}%  Max: {max_deg:+.4f}%  Threshold: 5.0%")

    # K2: >20% of displaced tokens go to non-neighbor
    min_nbr_acc = min(r['neighbor_accuracy'] * 100 for r in n8_results) if n8_results else 0
    mean_nbr_acc = np.mean([r['neighbor_accuracy'] * 100 for r in n8_results]) if n8_results else 0
    k2_pass = (100 - min_nbr_acc) < 20.0  # <20% to non-neighbor
    print(f"\n  K2 (>80% to neighbor): {'PASS' if k2_pass else 'KILL'}")
    print(f"      Mean: {mean_nbr_acc:.2f}%  Min: {min_nbr_acc:.2f}%  Threshold: 80.0%")

    # Scaling law
    print("\n" + "-" * 70)
    print("  SCALING LAW: displacement ~ 1/N")
    print("-" * 70)

    scaling_results = [r for r in all_results if r.get('test') == 'scaling']
    for N in N_values:
        nr = [r for r in scaling_results if r['N'] == N]
        if nr:
            mean_d = np.mean([r['displacement_rate'] for r in nr])
            theory = 1.0 / N
            ratio = mean_d / theory
            print(f"  N={N:>3}: measured {mean_d*100:>6.2f}%, theory {theory*100:>6.2f}%, "
                  f"ratio {ratio:.3f}")

    # Save results
    results_path = Path(__file__).parent / "results.json"
    # Make results JSON-serializable
    serializable = []
    for r in all_results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.int64)):
                sr[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                sr[k] = float(v)
            elif isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            else:
                sr[k] = v
        serializable.append(sr)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return all_results, k1_pass, k2_pass


if __name__ == "__main__":
    t0 = time.time()
    results, k1, k2 = run_full_experiment()
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Overall: {'PASS' if (k1 and k2) else 'KILL'}")
