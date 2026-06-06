"""Tests for zero-shot base transfer experiment."""

import torch
import pytest
from experiments.models.base_free_composition.base_free_composition import (
    GPT, LoRALinear, LoRAGPT,
    CharTokenizer, CharDataset, load_names, domain_split,
    compute_delta, svd_truncate, reconstruct_with_delta,
    train_gpt, evaluate_model, train_lora_expert,
)
from .zero_shot_base_transfer import (
    apply_lora_deltas_to_base,
    evaluate_expert_zero_shot,
    run_experiment,
)


class TestApplyLoraDeltas:
    def test_zero_delta_is_identity(self):
        """Applying zero LoRA deltas should not change the model."""
        torch.manual_seed(42)
        model = GPT(28, 16, 32, 4, 2)
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Zero deltas
        deltas = []
        for i, layer in enumerate(model.layers):
            for name in ["fc1", "fc2"]:
                linear = getattr(layer.mlp, name)
                delta = torch.zeros(linear.in_features, linear.out_features)
                deltas.append((i, name, delta))

        result = apply_lora_deltas_to_base(model, deltas)
        for k in original_state:
            assert torch.allclose(result.state_dict()[k], original_state[k], atol=1e-6), \
                f"Mismatch at {k}"

    def test_nonzero_delta_changes_weights(self):
        """Applying nonzero deltas should change the model weights."""
        torch.manual_seed(42)
        model = GPT(28, 16, 32, 4, 2)
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        deltas = [(0, "fc1", torch.ones(32, 128) * 0.01)]
        result = apply_lora_deltas_to_base(model, deltas)

        # fc1 weight should be different
        fc1_key = "layers.0.mlp.fc1.weight"
        assert not torch.allclose(result.state_dict()[fc1_key],
                                  original_state[fc1_key], atol=1e-6)

    def test_delta_applied_correctly(self):
        """Verify delta is added as transpose to weight."""
        torch.manual_seed(42)
        model = GPT(28, 16, 32, 4, 2)

        delta = torch.randn(32, 128)  # (in_features, out_features)
        deltas = [(0, "fc1", delta)]

        original_weight = model.layers[0].mlp.fc1.weight.clone()
        result = apply_lora_deltas_to_base(model, deltas)
        new_weight = result.layers[0].mlp.fc1.weight

        # Weight is (out, in), delta is (in, out), so we add delta.T
        expected = original_weight + delta.T
        assert torch.allclose(new_weight, expected, atol=1e-6)

    def test_deepcopy_isolation(self):
        """apply_lora_deltas should not modify the original model."""
        torch.manual_seed(42)
        model = GPT(28, 16, 32, 4, 2)
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        deltas = [(0, "fc1", torch.ones(32, 128) * 0.01)]
        _ = apply_lora_deltas_to_base(model, deltas)

        # Original model should be unchanged
        for k in original_state:
            assert torch.allclose(model.state_dict()[k], original_state[k], atol=1e-6)


class TestZeroShotEvaluation:
    def test_same_base_matches_lora_eval(self):
        """Zero-shot eval on same base should match LoRA training eval."""
        docs = load_names()[:200]
        tok = CharTokenizer(docs)
        train_ds = CharDataset(docs[:160], tok, block_size=16)
        val_ds = CharDataset(docs[160:], tok, block_size=16)

        torch.manual_seed(42)
        model = GPT(tok.vocab_size, 16, 32, 4, 2)
        train_gpt(model, train_ds, steps=30, batch_size=16, log_every=100)

        # Train expert
        deltas, val_loss_lora = train_lora_expert(
            model, train_ds, val_ds, rank=4, steps=30, batch_size=16, seed=42
        )

        # Zero-shot eval on same base
        base_state = {k: v.clone() for k, v in model.state_dict().items()}
        val_loss_zs = evaluate_expert_zero_shot(
            base_state, deltas, val_ds,
            vocab_size=tok.vocab_size, block_size=16,
            n_embd=32, n_head=4, n_layer=2, batch_size=16,
        )

        # Should be very close (not exact due to eval randomness in batching)
        assert abs(val_loss_lora - val_loss_zs) < 0.01, \
            f"Mismatch: LoRA={val_loss_lora:.4f} vs ZS={val_loss_zs:.4f}"


class TestIntegration:
    def test_run_experiment_small(self):
        """Run a minimal experiment to verify the pipeline works."""
        r = run_experiment(
            n_embd=32, n_head=2, n_layer=2, block_size=16,
            lora_rank=4, total_pretrain_steps=50,
            expert_train_steps=20, n_experts=2,
            batch_size=16, delta_ranks=[16, 8],
            seed=42,
        )
        # Should have zero-shot conditions
        assert len(r.zero_shot_conditions) >= 4  # pretrained_zeroshot, delta_full, delta_r16, delta_r8, skeleton
        # pretrained_zeroshot should closely match reference
        pz = next(c for c in r.zero_shot_conditions if c["name"] == "pretrained_zeroshot")
        ratio = pz["mean_expert_loss"] / (r.reference_condition["mean_expert_loss"] + 1e-12)
        assert 0.95 < ratio < 1.05, f"Pretrained zero-shot should match reference, got ratio {ratio}"

    def test_full_delta_matches_pretrained(self):
        """Full delta zero-shot should be identical to pretrained."""
        r = run_experiment(
            n_embd=32, n_head=2, n_layer=2, block_size=16,
            lora_rank=4, total_pretrain_steps=50,
            expert_train_steps=20, n_experts=2,
            batch_size=16, delta_ranks=[16],
            seed=42,
        )
        pz = next(c for c in r.zero_shot_conditions if c["name"] == "pretrained_zeroshot")
        df = next(c for c in r.zero_shot_conditions if c["name"] == "delta_full")
        # Full delta reconstruction should match pretrained exactly
        assert abs(pz["mean_expert_loss"] - df["mean_expert_loss"]) < 0.01

    def test_skeleton_is_worst(self):
        """Skeleton-only should be the worst condition."""
        r = run_experiment(
            n_embd=32, n_head=2, n_layer=2, block_size=16,
            lora_rank=4, total_pretrain_steps=50,
            expert_train_steps=20, n_experts=2,
            batch_size=16, delta_ranks=[16, 8],
            seed=42,
        )
        sk = next(c for c in r.zero_shot_conditions if c["name"] == "skeleton_only")
        pz = next(c for c in r.zero_shot_conditions if c["name"] == "pretrained_zeroshot")
        assert sk["mean_expert_loss"] > pz["mean_expert_loss"]
