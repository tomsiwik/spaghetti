"""Unit tests for LR schedule death experiment mechanism.

Validates that:
1. LRScheduleDeathGPT is registered and instantiable
2. LR schedule creation produces correct schedules
3. Training with schedules runs without error
4. Schedule produces expected LR values at key steps
"""

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.models import get_model, MODEL_REGISTRY
from .test_lr_schedule_death import make_lr_schedule, SCHEDULES


def test_registration():
    """LRScheduleDeathGPT is registered with correct parent."""
    assert "lr_schedule_death" in MODEL_REGISTRY
    assert MODEL_REGISTRY["lr_schedule_death"]["parent"] == "training_duration"


def test_instantiation():
    """Model instantiates and produces correct output shape."""
    model = get_model("lr_schedule_death", vocab_size=28, block_size=32,
                      n_embd=64, n_head=4, n_layer=4, n_capsules=128)
    mx.eval(model.parameters())
    tokens = mx.zeros((2, 16), dtype=mx.int32)
    logits = model(tokens)
    assert logits.shape == (2, 16, 28)


def test_constant_schedule():
    """Constant schedule returns fixed LR."""
    sched = make_lr_schedule("constant", peak_lr=3e-3, total_steps=100)
    assert isinstance(sched, float)
    assert sched == 3e-3


def test_warmup_schedule():
    """Warmup schedule starts at 0 and reaches peak."""
    sched = make_lr_schedule("warmup", peak_lr=1e-1, total_steps=100,
                             warmup_frac=0.10)
    # Create optimizer to step through schedule
    opt = optim.Adam(learning_rate=sched)
    dummy = {"w": mx.zeros((2, 2))}
    dummy_grads = {"w": mx.ones((2, 2))}

    # At step 0, LR should be near 0
    lr_0 = opt.learning_rate.item()
    assert lr_0 < 0.01, f"Initial LR should be near 0, got {lr_0}"

    # Step through warmup (10 steps for 10% of 100)
    for _ in range(10):
        opt.update(dummy, dummy_grads)

    # After warmup, should be at peak
    lr_10 = opt.learning_rate.item()
    assert abs(lr_10 - 0.1) < 0.02, f"Post-warmup LR should be ~0.1, got {lr_10}"

    # After many more steps, should remain at peak
    for _ in range(50):
        opt.update(dummy, dummy_grads)

    lr_60 = opt.learning_rate.item()
    assert abs(lr_60 - 0.1) < 0.02, f"Constant phase LR should be ~0.1, got {lr_60}"


def test_cosine_schedule():
    """Cosine schedule decays from peak to 0."""
    sched = make_lr_schedule("cosine", peak_lr=1e-1, total_steps=100)
    opt = optim.Adam(learning_rate=sched)
    dummy = {"w": mx.zeros((2, 2))}
    dummy_grads = {"w": mx.ones((2, 2))}

    # At step 0, should be at peak
    lr_0 = opt.learning_rate.item()
    assert abs(lr_0 - 0.1) < 0.01, f"Initial LR should be ~0.1, got {lr_0}"

    # At step 50 (midpoint), should be ~0.05
    for _ in range(50):
        opt.update(dummy, dummy_grads)

    lr_50 = opt.learning_rate.item()
    assert 0.02 < lr_50 < 0.08, f"Midpoint LR should be ~0.05, got {lr_50}"

    # At step 100, should be near 0
    for _ in range(50):
        opt.update(dummy, dummy_grads)

    lr_100 = opt.learning_rate.item()
    assert lr_100 < 0.02, f"Final LR should be near 0, got {lr_100}"


def test_warmup_cosine_schedule():
    """Warmup+cosine schedule: ramps up then decays."""
    sched = make_lr_schedule("warmup_cosine", peak_lr=1e-1, total_steps=100,
                             warmup_frac=0.10)
    opt = optim.Adam(learning_rate=sched)
    dummy = {"w": mx.zeros((2, 2))}
    dummy_grads = {"w": mx.ones((2, 2))}

    # At step 0, should be near 0
    lr_0 = opt.learning_rate.item()
    assert lr_0 < 0.01, f"Initial LR should be near 0, got {lr_0}"

    # After warmup (10 steps), should be near peak
    for _ in range(10):
        opt.update(dummy, dummy_grads)

    lr_10 = opt.learning_rate.item()
    assert lr_10 > 0.05, f"Post-warmup LR should be near peak, got {lr_10}"

    # At end (step 100), should be near 0
    for _ in range(90):
        opt.update(dummy, dummy_grads)

    lr_100 = opt.learning_rate.item()
    assert lr_100 < 0.02, f"Final LR should be near 0, got {lr_100}"


def test_all_schedules_create():
    """All schedule names produce valid schedules."""
    for name in SCHEDULES:
        sched = make_lr_schedule(name, peak_lr=3e-3, total_steps=3200)
        assert sched is not None, f"Schedule {name} returned None"


if __name__ == "__main__":
    test_registration()
    print("PASS: test_registration")
    test_instantiation()
    print("PASS: test_instantiation")
    test_constant_schedule()
    print("PASS: test_constant_schedule")
    test_warmup_schedule()
    print("PASS: test_warmup_schedule")
    test_cosine_schedule()
    print("PASS: test_cosine_schedule")
    test_warmup_cosine_schedule()
    print("PASS: test_warmup_cosine_schedule")
    test_all_schedules_create()
    print("PASS: test_all_schedules_create")
    print("\nAll mechanism tests passed.")
