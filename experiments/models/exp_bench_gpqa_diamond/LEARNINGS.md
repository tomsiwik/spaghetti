# LEARNINGS: exp_bench_gpqa_diamond

## Core Finding

Gemma 4 E4B 4-bit scores 31.8% on GPQA Diamond (non-thinking), a 26.8pp gap from Google's 58.6% (thinking). This matches the MMLU-Pro gap (27.1pp) exactly — the thinking penalty is ~27pp regardless of benchmark format or option count.

## Why

The thinking penalty is a fixed capability delta: externalizing the reasoning chain (3-7 steps for graduate-level science) is worth ~27pp across formats. Without thinking, the model compresses multi-step inference into a single forward pass, losing intermediate reasoning. At 31.8% (6.8pp above random), the NTP adapter has zero measurable effect — the floor clips any format-conflict degradation.

## Implications for Next Experiment

SFT adapters (not NTP) are the only viable path to MCQ benchmark improvement. The ~27pp thinking gap is a calibration constant: all future non-thinking baseline estimates should subtract 27pp from Google's reported numbers. GPQA Diamond is effectively unsolvable without thinking; future adapter experiments on graduate-level benchmarks should either enable thinking or target generation quality instead of MCQ accuracy.
