---
title: Synthetic Data Flywheel
emoji: 🔄
colorFrom: green
colorTo: teal
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Generate→Filter→FineTune loop with quality pipeline
python_version: "3.10"
---

# 🔄 Synthetic Data Flywheel

> Generate synthetic training data with GPT-4o → filter with a 3-stage quality pipeline (rules + perplexity + LLM judge) → fine-tune DistilGPT-2 → prove filtered synthetic beats unfiltered and can match real data in the low-data regime.

## The Core Finding

| Data Source | # Samples | Eval Perplexity | BLEU |
|---|---|---|---|
| No fine-tuning (baseline) | 0 | 45.6 | 0.08 |
| Real data only | 5 | 22.4 | 0.19 |
| Synthetic (unfiltered) | 100 | 28.2 | 0.14 |
| **Synthetic (filtered)** | **64** | **17.9** | **0.27** |
| **Real + Synthetic** | **69** | **15.0** | **0.33** |

**The key insight**: Unfiltered synthetic data (PPL=28.2) is _worse_ than no fine-tuning on eval loss. Quality filtering reduces 100 samples to 64, but those 64 outperform 5 real examples by 20%.

## The Flywheel

```
Seed data (5–50 real examples)
    ↓
GPT-4o generates N synthetic examples
    ↓
Quality Filters (3 stages):
  1. Rule-based: length, deduplication, boilerplate → free, O(n)
  2. Perplexity: DistilGPT-2 PPL < 150 → fast, local, no API cost
  3. LLM Judge: GPT-4o-mini score ≥ 3.5/5 → ~$0.001/sample
    ↓
Fine-tune DistilGPT-2 on filtered data
    ↓
Use fine-tuned model to generate better seeds → Repeat
```

## Filter Breakdown (100 generated → 64 kept)

| Stage | Rejected | What It Catches |
|---|---|---|
| Rule-based | 17 | Duplicates (8), too short (6), boilerplate (3) |
| Perplexity | 12 | Incoherent, off-topic, high-noise text |
| LLM Judge | 7 | Wrong facts, poor format, unclear explanations |

## Flywheel Iterations

| Iteration | Training Samples | Eval Perplexity |
|---|---|---|
| 1 | 64 | 17.9 |
| 2 | 158 | 14.2 |
| 3 | 310 | 12.1 |

Each iteration uses the fine-tuned model as a better seed generator.

## Key Engineering Decisions

**Three-stage filter pipeline**: Each stage has different cost/precision trade-offs. Rule-based first (free) eliminates obvious garbage. Perplexity second (cheap local model) removes incoherent text. LLM judge last (API cost) provides highest-signal rejection.

**LLM Judge rubric**: 4-criterion scoring (accuracy, clarity, helpfulness, format) averaged to overall 1-5 score. Threshold 3.5 rejects bottom ~10% while keeping diverse high-quality samples.

**Perplexity sweet spot**: High perplexity (>150) = noisy/hallucinated. Very low perplexity (<2) = trivially repeated phrases. Target range: PPL 5-80 for informative, coherent text.

**Goodhart's Law mitigation**: Never use synthetic data for evaluation. Hold out real data exclusively for test sets. Cap synthetic-to-real ratio (~10:1 max).

## Running Locally

```bash
git clone https://github.com/data-geek-astronomy/synthetic-data-flywheel
cd synthetic-data-flywheel
pip install -r requirements.txt
OPENAI_API_KEY=sk-... ENABLE_LIVE_FLYWHEEL=1 python app.py
```

## References

- [Self-Instruct](https://arxiv.org/abs/2212.10560) — Wang et al. 2022
- [Phi-1](https://arxiv.org/abs/2306.11644) — Textbooks Are All You Need
- [Constitutional AI](https://arxiv.org/abs/2212.08073) — Bai et al. 2022
