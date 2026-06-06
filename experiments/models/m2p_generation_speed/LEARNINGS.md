# LEARNINGS.md — exp_m2p_generation_speed

## Core Finding

M2P generation overhead is production-viable: 5.31ms forward pass (K947 PASS, 19× margin),
11.34ms full pipeline — equivalent to <2 tokens delay at 165 tok/s on M5 Pro.

## Why

M2P is purely bandwidth-bound (FLOPs << BW cost). At 67.2% BW utilization (268.7 GB/s),
the 357M-param fp32 dispatch runs at 1.49× the theoretical BW lower bound — near-optimal
for unified memory architecture. One-shot per prompt amortizes to <1% overhead for T_gen>100.

## Implications for Next Experiment

VeRA bottleneck (exp_m2p_vera_bottleneck) will reduce M2P from 357M → 4.7M params (76×),
predicted to cut M2P forward from 5.31ms → 0.07ms, making pipeline entirely dominated by
hidden extraction (6.02ms). Speed is no longer a concern — bottleneck is parameter count.
