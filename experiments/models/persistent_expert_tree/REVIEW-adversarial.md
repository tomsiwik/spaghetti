# Peer Review: Persistent Expert Tree

## NotebookLM Findings

Skipped (local review sufficient -- the issues are structural, not subtle).

## Mathematical Soundness

### Path-copying math (Sections 1-2): CORRECT

The persistent tree formalization is standard Okasaki (1998) applied to a binary tree of expert modules. The space complexity analysis (O(D) new nodes per single-leaf update) is correct. The worked example in Section 5 is internally consistent.

### Cross-version composition (Section 3): CORRECT but TRIVIALLY SO

The argument that cross-version composition works relies on three claims:
1. LoRA deltas are orthogonal (validated elsewhere at cos~0.0002)
2. Leaves are independent experts (true by construction)
3. Gate recalibration fixes routing (established in prior experiments)

These are all true, but the conclusion is tautological: if experts are independent modules and you can swap any module for any other trained-from-same-base module, of course the output is comparable. This is just "loading different LoRA adapters works." The persistent tree framing adds no mathematical insight beyond what was already known from the composition experiments.

### Memory analysis: PAPER CLAIMS CONTRADICT ACTUAL DATA

**Critical discrepancy.** PAPER.md Section "Memory Overhead" states KC2 fails at micro with "100% overhead." The HYPOTHESES.yml evidence says "100% overhead per version." But `results.json` shows:

```
v1_delta_params: 203932  (100% of base)
v2_delta_params: 203932  (100% of base)
overhead_vs_mutable_pct: 200.0
savings_vs_full_copy_pct: 0.0
```

The overhead is **200%**, not 100%. And savings vs full copy are **0%**, not 33.3% as the MATH.md Section 2.3 projects. This means every parameter changed during fine-tuning -- there is zero structural sharing in practice. The "persistent" structure provided no memory benefit whatsoever.

The MATH.md Section 2.2 projects 100.2% overhead per layer with 50.1% delta/base, but the actual delta/base is 100% -- every single parameter in the snapshot changed after 200 fine-tuning steps. The path-copying analysis is mathematically correct but empirically irrelevant: when the entire model is fine-tuned (not just specific leaves), ALL parameters diverge from base, making structural sharing impossible.

### Rollback (Section 4): CORRECT but TRIVIAL

Restoring a parameter snapshot and getting identical outputs is not a property of persistent data structures -- it is a property of deterministic computation on identical inputs. Any `torch.save()`/`torch.load()` achieves this.

## Novelty Assessment

### Prior art

**The persistent tree API is never called in the experiment.** The functions `compose_versions()`, `update_leaf()`, and `update_leaves()` -- the entire novel contribution -- appear only in unit tests. The actual experiment (`run_experiment.py`) uses:
- `snapshot_tree_params()` -- saves all parameters to a dict
- `restore_tree_params()` -- loads all parameters from a dict
- Manual parameter surgery (lines 193-206) -- copies specific leaf parameters from v1/v2 dicts

This is standard parameter checkpoint management. The persistent tree data structure is implemented but never exercised under the experimental conditions that produce the reported numbers.

**Cherry-picking vs averaging comparison is apples-to-oranges.** The "same-version" baseline averages ALL parameters from v1 and v2. The "cross-version" baseline cherry-picks domain-A leaves from v1 and domain-B leaves from v2. These are fundamentally different operations. Finding that cherry-picking domain-specialized parameters slightly outperforms averaging specialized with unspecialized ones is expected and uninformative about versioning.

**Delta over existing work: near zero.** The cross-version result reduces to: "LoRA adapters trained from the same base can be loaded independently and composed." This is already established in the LoRA merging literature and in this project's own `lora_merging_bakeoff` and `attn_lora_composition` experiments.

## Experimental Design

### The experiment does not test its stated hypothesis

The hypothesis is about *persistent data structures for version-aware expert composition*. The experiment tests *parameter snapshot/restore with manual cherry-picking*. These are different things.

Specifically:
1. **No path copying is exercised during training.** Fine-tuning updates ALL model parameters (full fine-tuning, not leaf-only), so the structural sharing that path copying provides is destroyed. The v1_delta is 100% of base, confirming this.
2. **No structural sharing is measured.** The memory overhead calculation uses `count_delta_params()` on full parameter snapshots, not the tree's `memory_report()` which would track actual structural sharing.
3. **No version graph operations are used.** The experiment creates versions by training separate models, not by calling the tree API.

### Controls are inadequate

There is no control for "just load different LoRA adapters without any tree structure." The persistent tree adds 128 parameters of overhead and significant implementation complexity, but the experiment never demonstrates that the tree structure provides any benefit over a flat dict of parameter snapshots.

### The right experiment would be

To test the persistent tree mechanism:
1. Fine-tune only specific leaves (freeze everything else), so structural sharing is meaningful
2. Use the `update_leaf()` / `compose_versions()` API during the experiment
3. Measure `memory_report()` to verify actual structural sharing
4. Compare persistent-tree storage against naive full-copy storage
5. Show that path copying actually saves memory in practice, not just in theory

## Hypothesis Graph Consistency

The HYPOTHESES.yml kill criteria are:
- KC1: cross-version composition >5% worse than same-version -- tested, passes
- KC2: persistent structure overhead >15% -- tested, FAILS (200%, not 100% as reported)

The evidence string claims "100% overhead" but actual data shows 200%. The evidence should be corrected.

The "proven" status is questionable: KC1 passes but tests something trivial (loading different checkpoints works), and KC2 fails by 13x the threshold. The macro projection (1.2%/version) is plausible but untested.

## Macro-Scale Risks (advisory)

1. **The macro projection is likely correct.** At macro scale with LoRA adapters (0.3% of base), path copying would indeed provide meaningful savings. The mechanism is sound in principle; it just cannot be validated at micro scale where full fine-tuning changes everything.
2. **Leaf-only fine-tuning is the prerequisite.** The persistent tree only works if fine-tuning is restricted to specific leaves/adapters. Full fine-tuning destroys structural sharing.
3. **Version graph complexity.** The experiment tests a single branch point (v0 -> v1, v0 -> v2). Real version graphs with depth >2 and merge conflicts are untested.

## Verdict

**REVISE**

The persistent tree data structure is well-implemented and the idea is sound for macro scale. But the experiment does not actually test the mechanism it claims to validate. The reported numbers come from parameter snapshot/restore, not from the persistent tree API. The memory overhead is misreported (200%, not 100%). Specific fixes:

1. **Correct the memory overhead reporting.** PAPER.md and HYPOTHESES.yml claim 100% overhead; results.json shows 200%. Update all documents to reflect the actual data, including 0% savings vs full copy.

2. **Run the experiment using the persistent tree API.** Replace `snapshot_tree_params()` / `restore_tree_params()` / manual parameter surgery with `update_leaves()` / `compose_versions()` / `set_active_version()`. The API exists and is unit-tested; the experiment should use it.

3. **Fine-tune leaves only, not the full model.** Freeze all non-tree parameters (embeddings, attention, norms, lm_head) during domain fine-tuning. This is the only regime where path copying provides structural sharing. Without this, delta/base = 100% and the persistent structure is pointless.

4. **Report `memory_report()` from the tree API** instead of computing overhead from full parameter snapshots. This measures actual structural sharing.

5. **Add a flat-dict baseline.** Compare the persistent tree against a simple `{version_id: {param_name: tensor}}` dict to show the tree structure adds value beyond naive checkpointing.

6. **Rename the "same-version" baseline or add a proper control.** The current comparison (weight averaging vs cherry-picking) conflates two orthogonal factors: (a) version mixing and (b) aggregation method. Add a control that cherry-picks from the SAME version to isolate the version-crossing factor.

Without fixes 1-4, the experiment validates "parameter checkpointing works" rather than "persistent expert trees work." The mechanism is likely sound at macro scale, but this micro experiment does not provide evidence for that claim.
