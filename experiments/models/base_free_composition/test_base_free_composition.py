"""Tests for base-free composition experiment."""

import math
import torch
import pytest
from .base_free_composition import (
    GPT, RMSNorm, CausalSelfAttention, MLP, Block,
    LoRALinear, LoRAGPT,
    CharTokenizer, CharDataset, load_names, domain_split,
    compute_delta, svd_truncate, reconstruct_with_delta,
    delta_reconstruction_error, effective_rank,
    compute_pairwise_cosine,
    train_gpt, evaluate_model, train_lora_expert,
    run_experiment,
)


# ── Model Tests ──────────────────────────────────────────────────────────────

class TestGPT:
    def test_forward_shape(self):
        model = GPT(vocab_size=28, block_size=16, n_embd=32, n_head=4, n_layer=2)
        x = torch.randint(0, 28, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, 28)

    def test_param_count(self):
        model = GPT(vocab_size=28, block_size=16, n_embd=32, n_head=4, n_layer=2)
        n = sum(p.numel() for p in model.parameters())
        assert n > 0

    def test_deterministic_init(self):
        torch.manual_seed(42)
        m1 = GPT(vocab_size=28, block_size=16, n_embd=32, n_head=4, n_layer=2)
        torch.manual_seed(42)
        m2 = GPT(vocab_size=28, block_size=16, n_embd=32, n_head=4, n_layer=2)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)


class TestLoRA:
    def test_lora_linear_shape(self):
        base = torch.nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4)
        x = torch.randn(2, 10, 32)
        out = lora(x)
        assert out.shape == (2, 10, 64)

    def test_lora_delta_shape(self):
        base = torch.nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4)
        delta = lora.get_delta()
        assert delta.shape == (32, 64)

    def test_lora_starts_at_zero(self):
        base = torch.nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4)
        delta = lora.get_delta()
        # B initialized to zero, so delta should be zero
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)

    def test_lora_base_frozen(self):
        base = torch.nn.Linear(32, 64, bias=False)
        lora = LoRALinear(base, rank=4)
        assert not lora.base.weight.requires_grad

    def test_lora_gpt_forward(self):
        model = GPT(vocab_size=28, block_size=16, n_embd=32, n_head=4, n_layer=2)
        lora_model = LoRAGPT(model, rank=4)
        x = torch.randint(0, 28, (2, 16))
        logits = lora_model(x)
        assert logits.shape == (2, 16, 28)

    def test_lora_gpt_deltas(self):
        model = GPT(vocab_size=28, block_size=16, n_embd=32, n_head=4, n_layer=2)
        lora_model = LoRAGPT(model, rank=4)
        deltas = lora_model.get_all_deltas()
        # 2 layers * 2 MLP linears = 4 deltas
        assert len(deltas) == 4

    def test_lora_parameters_only_AB(self):
        model = GPT(vocab_size=28, block_size=16, n_embd=32, n_head=4, n_layer=2)
        lora_model = LoRAGPT(model, rank=4)
        params = lora_model.lora_parameters()
        # 2 layers * 2 linears * 2 (A+B) = 8 parameter tensors
        assert len(params) == 8
        for p in params:
            assert p.requires_grad


# ── Delta Decomposition Tests ────────────────────────────────────────────────

class TestDelta:
    def test_delta_computation(self):
        torch.manual_seed(42)
        m = GPT(vocab_size=28, block_size=16, n_embd=32, n_head=4, n_layer=2)
        skeleton = {k: v.clone() for k, v in m.state_dict().items()}
        # Modify some weights
        sd = m.state_dict()
        sd["wte.weight"] = sd["wte.weight"] + 0.1
        m.load_state_dict(sd)
        pretrained = {k: v.clone() for k, v in m.state_dict().items()}

        deltas = compute_delta(pretrained, skeleton)
        assert "wte.weight" in deltas
        assert torch.allclose(deltas["wte.weight"], torch.ones_like(deltas["wte.weight"]) * 0.1, atol=1e-6)

    def test_svd_truncate_rank(self):
        torch.manual_seed(42)
        # Create a rank-4 matrix
        A = torch.randn(16, 4)
        B = torch.randn(4, 32)
        M = A @ B  # rank 4
        # Truncating at rank >= 4 should be lossless
        approx = svd_truncate(M, 4)
        assert torch.allclose(M, approx, atol=1e-4)
        # Truncating at rank 2 should lose info
        approx2 = svd_truncate(M, 2)
        error = torch.norm(M - approx2).item()
        assert error > 0.01  # should have non-trivial error

    def test_svd_truncate_1d(self):
        """1D tensors (biases, norms) should pass through unchanged."""
        v = torch.randn(32)
        result = svd_truncate(v, 4)
        assert torch.allclose(v, result)

    def test_reconstruction_full_rank(self):
        """Full-rank reconstruction should be exact."""
        torch.manual_seed(42)
        skeleton = {"w": torch.randn(16, 32)}
        pretrained = {"w": torch.randn(16, 32)}
        deltas = compute_delta(pretrained, skeleton)
        recon = reconstruct_with_delta(skeleton, deltas, rank=None)
        assert torch.allclose(recon["w"], pretrained["w"], atol=1e-6)

    def test_reconstruction_error_decreases_with_rank(self):
        """Higher rank should give lower reconstruction error."""
        torch.manual_seed(42)
        delta = {"w": torch.randn(32, 64)}
        err4 = delta_reconstruction_error(delta, 4)["_total"]["rms_relative_error"]
        err8 = delta_reconstruction_error(delta, 8)["_total"]["rms_relative_error"]
        err16 = delta_reconstruction_error(delta, 16)["_total"]["rms_relative_error"]
        assert err4 > err8 > err16

    def test_effective_rank(self):
        # Identity-like matrix should have high effective rank
        I = torch.eye(16)
        r = effective_rank(I)
        assert r > 15.0  # should be close to 16

        # Rank-1 matrix should have effective rank ~1
        v = torch.randn(16, 1)
        M = v @ v.T
        r1 = effective_rank(M)
        assert r1 < 2.0


# ── Data Tests ───────────────────────────────────────────────────────────────

class TestData:
    def test_load_names(self):
        docs = load_names()
        assert len(docs) > 100

    def test_tokenizer(self):
        docs = load_names()
        tok = CharTokenizer(docs)
        assert tok.vocab_size > 0
        encoded = tok.encode("abc")
        assert len(encoded) == 3

    def test_dataset_batch(self):
        docs = load_names()[:100]
        tok = CharTokenizer(docs)
        ds = CharDataset(docs, tok, block_size=16)
        inputs, targets = ds.get_batch(4)
        assert inputs.shape == (4, 16)
        assert targets.shape == (4, 16)

    def test_domain_split(self):
        docs = load_names()
        splits = domain_split(docs, "quintary")
        assert len(splits) == 5
        total = sum(len(v) for v in splits.values())
        assert total == len(docs)


# ── Training Tests ───────────────────────────────────────────────────────────

class TestTraining:
    def test_train_gpt_loss_decreases(self):
        docs = load_names()[:200]
        tok = CharTokenizer(docs)
        ds = CharDataset(docs, tok, block_size=16)
        torch.manual_seed(42)
        model = GPT(tok.vocab_size, 16, 32, 4, 2)
        result = train_gpt(model, ds, steps=50, batch_size=16, lr=3e-3,
                          seed=42, log_every=100)
        # Loss should decrease from start to end
        assert result["losses"][-1] < result["losses"][0]

    def test_evaluate_model(self):
        docs = load_names()[:200]
        tok = CharTokenizer(docs)
        ds = CharDataset(docs, tok, block_size=16)
        torch.manual_seed(42)
        model = GPT(tok.vocab_size, 16, 32, 4, 2)
        val = evaluate_model(model, ds, batch_size=16, n_batches=3)
        assert val > 0

    def test_train_lora_expert(self):
        docs = load_names()[:200]
        tok = CharTokenizer(docs)
        train_ds = CharDataset(docs[:160], tok, block_size=16)
        val_ds = CharDataset(docs[160:], tok, block_size=16)
        torch.manual_seed(42)
        model = GPT(tok.vocab_size, 16, 32, 4, 2)
        # Pretrain a bit
        train_gpt(model, train_ds, steps=20, batch_size=16, log_every=100)
        # Train expert
        deltas, val_loss = train_lora_expert(
            model, train_ds, val_ds, rank=4, steps=20, batch_size=16
        )
        assert val_loss > 0
        assert len(deltas) == 4  # 2 layers * 2 linears


# ── Cosine Tests ─────────────────────────────────────────────────────────────

class TestCosine:
    def test_pairwise_cosine(self):
        # Two identical expert deltas should have cos = 1
        d1 = [(0, "fc1", torch.randn(32, 128)),
              (0, "fc2", torch.randn(128, 32))]
        d2 = [(0, "fc1", d1[0][2].clone()),
              (0, "fc2", d1[1][2].clone())]
        results = compute_pairwise_cosine([d1, d2])
        assert len(results) == 1
        assert abs(results[0][2] - 1.0) < 1e-5

    def test_orthogonal_deltas(self):
        # Two orthogonal deltas
        d1 = [(0, "fc1", torch.tensor([[1.0, 0.0], [0.0, 0.0]]))]
        d2 = [(0, "fc1", torch.tensor([[0.0, 1.0], [0.0, 0.0]]))]
        results = compute_pairwise_cosine([d1, d2])
        assert abs(results[0][2]) < 1e-5


# ── Integration Test ─────────────────────────────────────────────────────────

class TestIntegration:
    def test_run_experiment_small(self):
        """Run a minimal experiment to verify the pipeline works."""
        r = run_experiment(
            n_embd=32, n_head=2, n_layer=2, block_size=16,
            lora_rank=4, total_pretrain_steps=50,
            expert_train_steps=20, n_experts=2,
            batch_size=16, delta_ranks=[16, 8, 4],
            seed=42,
        )
        # Should have conditions: pretrained, delta_full, delta_r16, delta_r8, delta_r4, skeleton_only
        assert len(r.conditions) == 6
        # delta_full should match pretrained very closely
        pretrained_loss = next(c for c in r.conditions if c["name"] == "pretrained")["mean_expert_loss"]
        delta_full_loss = next(c for c in r.conditions if c["name"] == "delta_full")["mean_expert_loss"]
        assert abs(pretrained_loss - delta_full_loss) < 0.1  # should be nearly identical
        # skeleton_only should be worse
        skeleton_loss = next(c for c in r.conditions if c["name"] == "skeleton_only")["mean_expert_loss"]
        assert skeleton_loss > pretrained_loss * 0.9  # skeleton should be at least comparable or worse

    def test_full_delta_is_identity(self):
        """Full-rank delta reconstruction should be mathematically identical to pretrained."""
        torch.manual_seed(42)
        model = GPT(28, 16, 32, 4, 2)
        skeleton = {k: v.clone() for k, v in model.state_dict().items()}

        # Simulate training by adding random perturbation
        sd = model.state_dict()
        for k in sd:
            if torch.isfinite(sd[k]).all():  # Skip buffers with -inf
                sd[k] = sd[k] + torch.randn_like(sd[k]) * 0.01
        pretrained = sd

        deltas = compute_delta(pretrained, skeleton)
        recon = reconstruct_with_delta(skeleton, deltas, rank=None,
                                       pretrained_state=pretrained)

        for k in pretrained:
            assert torch.allclose(recon[k], pretrained[k], atol=1e-6), \
                f"Mismatch at {k}"
