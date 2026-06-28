"""
Synthetic Data Flywheel — Interactive Demo
==========================================
Generate → Filter → Fine-Tune → Evaluate → Repeat.
Shows why quality filtering makes synthetic data work.

Author: Aravind Kumar Nalukurthi
"""

import gradio as gr
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import List

from flywheel import (
    SyntheticDataGenerator, QualityFilterPipeline,
    SEED_DATA, get_precomputed_filter_results, get_precomputed_training_results,
)

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
ENABLE_LIVE = os.getenv("ENABLE_LIVE_FLYWHEEL", "0") == "1"

CSS = """
body, .gradio-container { background: #0a0d14 !important; }
.card { background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.3); border-radius: 12px; padding: 18px; margin: 8px 0; }
.pass { color: #22c55e; }
.fail { color: #ef4444; }
footer { display: none !important; }
"""

FILTER_DATA = get_precomputed_filter_results()
TRAINING_DATA = get_precomputed_training_results()


# ── Chart builders ────────────────────────────────────────────────────────────

def build_filter_funnel_chart():
    stages = list(FILTER_DATA["stages"].keys())
    passed = [100] + [FILTER_DATA["stages"][s]["passed"] for s in stages]
    stage_labels = ["Generated"] + stages

    fig = go.Figure(go.Funnel(
        y=stage_labels,
        x=passed,
        textinfo="value+percent initial",
        marker=dict(
            color=["#475569", "#6366f1", "#8b5cf6", "#a78bfa"],
        ),
        connector=dict(line=dict(color="#334155", width=2)),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
        title="Quality Filtering Funnel (100 generated → 64 kept)",
        height=360, margin=dict(t=50, b=10),
    )
    return fig


def build_quality_distribution_chart():
    bins = FILTER_DATA["quality_distribution"]["score_bins"]
    raw = FILTER_DATA["quality_distribution"]["counts_raw"]
    filtered = FILTER_DATA["quality_distribution"]["counts_filtered"]

    fig = go.Figure([
        go.Bar(name="Before Filtering", x=bins, y=raw, marker_color="#475569"),
        go.Bar(name="After Filtering", x=bins, y=filtered, marker_color="#6366f1"),
    ])
    fig.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title="Quality Score Distribution Before/After Filtering",
        xaxis_title="Quality Score Range", yaxis_title="# Samples",
        height=340, margin=dict(t=50, b=10),
    )
    return fig


def build_training_comparison_chart():
    conds = TRAINING_DATA["conditions"]
    names = [c["name"] for c in conds]
    ppl = [c["perplexity"] for c in conds]
    colors = ["#22c55e" if c.get("highlight") else "#475569" for c in conds]

    fig = go.Figure([
        go.Bar(
            x=names, y=ppl,
            marker_color=colors,
            text=[f"PPL={p:.1f}" for p in ppl],
            textposition="outside",
        )
    ])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
        title="Eval Perplexity by Training Data Source (lower = better)",
        yaxis_title="Perplexity (↓ better)", height=400,
        margin=dict(t=50, b=80),
        xaxis_tickangle=-15,
    )
    return fig


def build_flywheel_iterations_chart():
    fw = TRAINING_DATA["flywheel_iterations"]
    fig = go.Figure([
        go.Scatter(
            x=fw["iter"], y=fw["eval_perplexity"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=3),
            marker=dict(size=10, color="#a78bfa"),
            name="Eval Perplexity",
        ),
        go.Scatter(
            x=fw["iter"], y=fw["train_samples"],
            mode="lines+markers", yaxis="y2",
            line=dict(color="#22c55e", width=2, dash="dot"),
            marker=dict(size=8, color="#22c55e"),
            name="Training Samples",
        )
    ])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
        title="Flywheel Iterations: Each Round Generates Better Data",
        xaxis_title="Flywheel Iteration",
        yaxis=dict(title="Eval Perplexity (↓)", color="#a78bfa"),
        yaxis2=dict(title="Training Samples (→)", overlaying="y", side="right", color="#22c55e"),
        height=340, legend=dict(x=0.01, y=0.99),
        margin=dict(t=50, b=10),
    )
    return fig


# ── Live pipeline ──────────────────────────────────────────────────────────────

def run_live_generation(api_key: str, domain: str, n: int):
    if not api_key:
        return "❌ Enter API key", "", ""

    generator = SyntheticDataGenerator(api_key)
    samples = generator.generate_batch(domain, n=n)

    if not samples:
        return "❌ Generation failed — check API key", "", ""

    samples_html = "".join(
        f"""<div class='card' style='margin:6px 0'>
            <div style='color:#64748b;font-size:0.78em'>#{i+1} · {s.domain}</div>
            <div style='color:#a5b4fc;font-size:0.88em;margin:4px 0'><strong>Prompt:</strong> {s.prompt[:200]}</div>
            <div style='color:#e2e8f0;font-size:0.85em'><strong>Completion:</strong> {s.completion[:300]}</div>
        </div>"""
        for i, s in enumerate(samples)
    )

    pipeline = QualityFilterPipeline(
        openai_api_key=api_key,
        use_perplexity=False,
        use_llm_judge=True,
        min_llm_score=3.5,
    )
    report = pipeline.filter_batch(samples)

    passed = [s for s in report.samples if s.passed_filter]
    failed = [s for s in report.samples if not s.passed_filter]

    filter_html = f"""
    <div class='card'>
        <div style='display:flex;justify-content:space-between;margin-bottom:12px'>
            <div><span style='color:#22c55e;font-size:1.4em;font-weight:700'>{report.passed}</span>
                 <span style='color:#64748b;font-size:0.85em'> passed</span></div>
            <div><span style='color:#ef4444;font-size:1.4em;font-weight:700'>{report.total_in - report.passed}</span>
                 <span style='color:#64748b;font-size:0.85em'> rejected</span></div>
            <div><span style='color:#a78bfa;font-size:1.4em;font-weight:700'>{report.pass_rate:.0%}</span>
                 <span style='color:#64748b;font-size:0.85em'> pass rate</span></div>
        </div>
        {''.join(f"<div style='padding:4px 0'><span class='pass'>✓</span> <span style='color:#94a3b8;font-size:0.82em'>{s.prompt[:80]}...</span></div>" for s in passed[:3])}
        {''.join(f"<div style='padding:4px 0'><span class='fail'>✗</span> <span style='color:#64748b;font-size:0.78em'>{s.filter_reason}</span></div>" for s in failed[:3])}
    </div>
    """

    stats_html = f"""
    <div class='card'>
        <div style='color:#94a3b8;font-size:0.85em'>
            Generated: {report.total_in} · Passed: {report.passed} ·
            Rejected length: {report.rejected_length} ·
            Rejected duplicate: {report.rejected_duplicate} ·
            Rejected LLM judge: {report.rejected_llm_judge}
        </div>
    </div>
    """

    return samples_html, filter_html, stats_html


# ── Gradio App ─────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="violet"), title="Synthetic Data Flywheel") as demo:

    gr.HTML("""
    <div style='text-align:center;padding:28px 0 18px'>
        <div style='font-size:2.8em'>🔄</div>
        <h1 style='color:#e2e8f0;margin:10px 0 6px;font-size:1.9em;font-weight:700'>
            Synthetic Data Flywheel
        </h1>
        <p style='color:#64748b;max-width:720px;margin:0 auto;line-height:1.6'>
            Generate synthetic training data with GPT-4o → quality filter with 3-stage pipeline →
            fine-tune DistilGPT-2 → prove filtered synthetic beats unfiltered and matches real data.
        </p>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("🔄 Live Pipeline"):
            gr.HTML("""<div class='card'><p style='color:#94a3b8;margin:0'>
            Generate synthetic training samples live, then run quality filters on them.</p></div>""")
            with gr.Row():
                api_key = gr.Textbox(label="OpenAI API Key", type="password", value=OPENAI_KEY, scale=3)
                domain_select = gr.Dropdown(
                    choices=["instruction_following", "code_generation", "summarization"],
                    value="instruction_following", label="Domain", scale=1,
                )
                n_samples = gr.Slider(5, 20, value=10, step=5, label="Generate N samples", scale=1)

            gen_btn = gr.Button("🚀 Generate + Filter", variant="primary", size="lg")
            gen_display = gr.HTML(value="<div class='card'>Click Generate to begin.</div>")
            filter_display = gr.HTML()
            stats_display = gr.HTML()

            gen_btn.click(
                fn=run_live_generation,
                inputs=[api_key, domain_select, n_samples],
                outputs=[gen_display, filter_display, stats_display],
            )

        with gr.Tab("🔬 Filter Analysis"):
            gr.HTML(f"""
            <div class='card'>
                <div style='display:flex;gap:30px;flex-wrap:wrap'>
                    <div style='text-align:center'>
                        <div style='color:#6366f1;font-size:2em;font-weight:700'>100</div>
                        <div style='color:#64748b;font-size:0.82em'>Generated</div>
                    </div>
                    <div style='text-align:center'>
                        <div style='color:#22c55e;font-size:2em;font-weight:700'>64</div>
                        <div style='color:#64748b;font-size:0.82em'>Passed All Filters</div>
                    </div>
                    <div style='text-align:center'>
                        <div style='color:#ef4444;font-size:2em;font-weight:700'>36</div>
                        <div style='color:#64748b;font-size:0.82em'>Rejected</div>
                    </div>
                    <div style='text-align:center'>
                        <div style='color:#a78bfa;font-size:2em;font-weight:700'>{FILTER_DATA["final_pass_rate"]:.0%}</div>
                        <div style='color:#64748b;font-size:0.82em'>Pass Rate</div>
                    </div>
                </div>
            </div>
            """)
            with gr.Row():
                gr.Plot(build_filter_funnel_chart())
                gr.Plot(build_quality_distribution_chart())

            # Breakdown table
            breakdown_html = "<div class='card'><h4 style='color:#a5b4fc;margin:0 0 12px'>Rejection Breakdown by Stage</h4>"
            for stage, data in FILTER_DATA["stages"].items():
                breakdown_html += f"""
                <div style='background:#111827;border-radius:8px;padding:10px;margin:6px 0;display:flex;justify-content:space-between'>
                    <div style='color:#e2e8f0;font-weight:600'>{stage}</div>
                    <div style='text-align:right'>
                        <span style='color:#ef4444'>{data["rejected"]} rejected</span>
                        <span style='color:#64748b;margin:0 8px'>·</span>
                        <span style='color:#22c55e'>{data["passed"]} passed</span>
                        <span style='color:#64748b;margin:0 8px'>·</span>
                        <span style='color:#64748b;font-size:0.82em'>{data["cost"]}</span>
                    </div>
                </div>"""
            breakdown_html += "</div>"
            gr.HTML(breakdown_html)

        with gr.Tab("📊 Training Comparison"):
            gr.HTML(f"""
            <div class='card'>
                <h4 style='color:#a5b4fc;margin:0 0 12px'>Experiment: {TRAINING_DATA["task"]}</h4>
                <div style='color:#94a3b8;font-size:0.85em'>
                    Model: {TRAINING_DATA["model"]} · Eval set: {TRAINING_DATA["eval_set"]}
                </div>
            </div>
            """)
            with gr.Row():
                gr.Plot(build_training_comparison_chart())
                gr.Plot(build_flywheel_iterations_chart())

            # Key findings
            findings_html = "<div class='card'><h4 style='color:#a5b4fc;margin:0 0 12px'>Key Findings</h4>"
            for finding in TRAINING_DATA["key_findings"]:
                emoji = "⚠️" if "WORSE" in finding or "hurts" in finding else "✅"
                findings_html += f"<div style='color:#94a3b8;font-size:0.88em;padding:6px 0'>{emoji} {finding}</div>"
            findings_html += "</div>"
            gr.HTML(findings_html)

        with gr.Tab("🏗️ Architecture"):
            gr.Markdown("""
## The Synthetic Data Flywheel

```
Iteration 0:
  Seed data (5-50 real examples)
      ↓
  GPT-4o generates N synthetic examples (diverse variations)
      ↓
  Quality Filters:
    1. Rule-based (length, dedup, boilerplate) — free
    2. Perplexity filter (DistilGPT-2) — fast, local
    3. LLM Judge (GPT-4o-mini) — accurate, ~$0.001/sample
      ↓
  Fine-tune DistilGPT-2 on filtered data
      ↓
Iteration 1:
  Use fine-tuned model to generate better seeds
      ↓
  Repeat → each iteration improves quality
```

## Why Quality Filtering is the Secret Sauce

Without filtering, synthetic data contains:
- **Repetitions**: model regenerates variations of the same fact
- **Hallucinations**: plausible-sounding but wrong facts
- **Format errors**: completions that ignore the expected format
- **Boilerplate**: "As an AI language model..." style refusals

**Filter effectiveness on 100 generated samples:**

| Stage | Rejected | What It Catches |
|---|---|---|
| Rule-based | 17 | Duplicates (8), too short (6), boilerplate (3) |
| Perplexity | 12 | High-noise, off-topic, or incoherent text |
| LLM Judge | 7 | Wrong facts, poor format, unclear explanations |

## Perplexity as a Quality Signal

DistilGPT-2 perplexity on the completion text:
- **PPL < 2**: Suspiciously low — likely trivial or repeated phrase
- **PPL 5–80**: Normal range — coherent, informative text
- **PPL > 150**: High noise — garbled, off-topic, or hallucinated

We compute perplexity on the **completion only**, not the prompt.
We care about output quality, not question difficulty.

## Goodhart's Law Warning

If you over-rely on synthetic data:
1. You're training on GPT-4o's distribution, not the real world
2. The model learns GPT-4o's failure modes
3. Reinforcement of errors — the flywheel amplifies mistakes

Mitigation: always hold out real evaluation data, use diverse seeds,
and cap the synthetic-to-real ratio (~10:1 is typical).

## Practical Results

| Data | # Samples | Perplexity | BLEU |
|---|---|---|---|
| No fine-tuning | 0 | 45.6 | 0.08 |
| Real only | 5 | 22.4 | 0.19 |
| Synthetic (unfiltered) | 100 | 28.2 | 0.14 |
| Synthetic (filtered) | 64 | **17.9** | **0.27** |
| Real + Synthetic | 69 | **15.0** | **0.33** |

**The key result**: Filtered synthetic (64 samples) beats real-only (5 samples)
by 20% on perplexity, despite having 12.8x more training examples — because
filtering matters more than quantity.
            """)

demo.launch()
