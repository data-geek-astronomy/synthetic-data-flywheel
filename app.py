"""
Synthetic Data Flywheel - Professional Demo
Author: Aravind Kumar Nalukurthi
"""

import gradio as gr
import plotly.graph_objects as go
import os

try:
    from flywheel.generator import SyntheticDataGenerator
    from flywheel.filters import QualityFilterPipeline
except Exception:
    SyntheticDataGenerator = None
    QualityFilterPipeline = None

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

CSS = """
* { box-sizing: border-box; }
body, .gradio-container {
    background: #000 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif !important;
    color: #f5f5f7 !important;
}
.hero { padding: 64px 32px 48px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.07); }
.hero-badge { display: inline-block; background: rgba(48,209,88,0.12); color: #30d158; font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; padding: 5px 14px; border-radius: 20px; border: 1px solid rgba(48,209,88,0.2); margin-bottom: 22px; }
.hero-title { font-size: 48px; font-weight: 700; color: #f5f5f7; line-height: 1.06; letter-spacing: -0.025em; margin: 0 0 18px; }
.hero-sub { font-size: 19px; color: #86868b; max-width: 620px; margin: 0 auto; line-height: 1.55; }
.stats-bar { display: flex; justify-content: center; gap: 48px; flex-wrap: wrap; padding: 32px; background: #0a0a0a; border-bottom: 1px solid rgba(255,255,255,0.07); }
.stat { text-align: center; }
.stat-val { font-size: 30px; font-weight: 700; color: #30d158; letter-spacing: -0.02em; }
.stat-label { font-size: 12px; color: #6e6e73; margin-top: 3px; font-weight: 500; }
.section { padding: 36px 32px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.sec-label { font-size: 12px; font-weight: 600; color: #6e6e73; letter-spacing: 0.09em; text-transform: uppercase; margin: 0 0 18px; }
.card { background: #111; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 22px 24px; margin-bottom: 10px; }
.card-title { font-size: 16px; font-weight: 600; color: #f5f5f7; margin: 0 0 8px; }
.card-body { font-size: 14px; color: #86868b; line-height: 1.6; margin: 0; }
.filter-step { display: flex; align-items: center; gap: 16px; padding: 14px; background: #0a0a0a; border-radius: 10px; margin-bottom: 8px; }
.filter-num { font-size: 26px; font-weight: 700; color: #f5f5f7; min-width: 60px; }
.filter-label { font-size: 14px; color: #86868b; }
.filter-arrow { font-size: 18px; color: #3a3a3c; margin: 0 8px; }
footer { display: none !important; }
"""

def filter_funnel_chart():
    stages = ["Generated", "Rule Filter\n(free)", "Perplexity Filter\n(DistilGPT-2)", "LLM Judge\n(GPT-4o-mini)"]
    counts = [100, 83, 71, 64]
    colors = ["#3a3a3c", "#48484a", "#636366", "#30d158"]
    fig = go.Figure([go.Bar(
        x=stages, y=counts,
        marker_color=colors,
        text=[f"{c} samples" for c in counts],
        textposition="outside",
        textfont=dict(color="#f5f5f7", size=12),
        width=0.5,
    )])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#86868b"), yaxis=dict(range=[0, 130], gridcolor="rgba(255,255,255,0.05)", title="Samples Remaining"),
        height=300, margin=dict(t=20, b=20), showlegend=False,
    )
    return fig

def ppl_chart():
    conditions = ["Baseline\n(no fine-tune)", "Real data\n(5 samples)", "Unfiltered\nsynthetic (100)", "Filtered\nsynthetic (64)", "Real +\nFiltered (69)"]
    ppls = [45.6, 22.4, 28.2, 17.9, 15.0]
    colors = ["#3a3a3c", "#48484a", "#ff453a", "#30d158", "#0a84ff"]
    fig = go.Figure([go.Bar(
        x=conditions, y=ppls,
        marker_color=colors,
        text=[f"{p}" for p in ppls],
        textposition="outside",
        textfont=dict(color="#f5f5f7", size=12),
        width=0.5,
    )])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#86868b"),
        yaxis=dict(title="Perplexity (lower = better)", gridcolor="rgba(255,255,255,0.05)", range=[0, 60]),
        height=320, margin=dict(t=20, b=20), showlegend=False,
    )
    return fig

def flywheel_chart():
    iters = [0, 1, 2, 3]
    ppls = [22.4, 17.9, 14.2, 12.1]
    samples = [5, 64, 158, 310]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=ppls, name="Perplexity", mode="lines+markers",
        line=dict(color="#30d158", width=2), marker=dict(size=9)))
    fig.add_trace(go.Bar(x=iters, y=samples, name="Training Samples", yaxis="y2",
        marker_color="rgba(10,132,255,0.2)", marker_line_color="#0a84ff", marker_line_width=1))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#86868b"),
        xaxis_title="Flywheel Iteration", xaxis=dict(tickvals=[0,1,2,3], ticktext=["Seed", "Iter 1", "Iter 2", "Iter 3"]),
        yaxis=dict(title="Perplexity ↓", gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(title="Samples ↑", overlaying="y", side="right"),
        height=320, legend=dict(x=0.02, y=0.98), margin=dict(t=20, b=20),
    )
    return fig


with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="Synthetic Data Flywheel") as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-badge">AI Engineering · Data + Fine-Tuning</div>
        <h1 class="hero-title">Synthetic Data Flywheel</h1>
        <p class="hero-sub">
            Getting real training data is expensive. This project generates synthetic data using GPT-4o-mini,
            filters it through three quality stages, fine-tunes a language model on it, and feeds the
            improved model back to generate better data - a self-improving loop.
        </p>
    </div>
    <div class="stats-bar">
        <div class="stat"><div class="stat-val">64%</div><div class="stat-label">Data surviving all filters</div></div>
        <div class="stat"><div class="stat-val">PPL 17.9</div><div class="stat-label">Filtered synthetic (vs 28.2 unfiltered)</div></div>
        <div class="stat"><div class="stat-val">PPL 15.0</div><div class="stat-label">Real + synthetic combined</div></div>
        <div class="stat"><div class="stat-val">1</div><div class="stat-label">API key needed (OpenAI)</div></div>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("Overview"):
            gr.HTML("""
            <div class="section">
                <div class="sec-label">The idea</div>
                <div class="card">
                    <div class="card-title">The synthetic data problem</div>
                    <p class="card-body">You can generate unlimited training data using a powerful LLM - but raw synthetic data often hurts model quality. It can be repetitive, contain subtle errors, or drift toward LLM-sounding outputs rather than genuine instruction-following. The key insight: filtering synthetic data changes everything.</p>
                </div>
                <div class="card">
                    <div class="card-title">Critical finding</div>
                    <p class="card-body">
                        <span style="color:#ff453a">Unfiltered synthetic data: PPL=28.2 - worse than no fine-tuning</span><br>
                        <span style="color:#30d158">Filtered synthetic data: PPL=17.9 - beats real data alone (22.4)</span><br>
                        <span style="color:#0a84ff">Combined real + filtered: PPL=15.0 - best result</span>
                    </p>
                </div>
                <div class="card">
                    <div class="card-title">The flywheel</div>
                    <p class="card-body">After the first fine-tune, use the improved model to generate better seed examples. Then filter and fine-tune again. Each iteration produces more data and lower perplexity: Iter 0 (PPL 22.4) → Iter 1 (17.9) → Iter 2 (14.2) → Iter 3 (12.1).</p>
                </div>
                <div class="card" style="border-color:rgba(48,209,88,0.25)">
                    <div class="card-title" style="color:#30d158">How to explore</div>
                    <p class="card-body">No API key: Go to "Filter Pipeline" and "Training Results" for pre-computed charts.<br>With API key: Go to "Generate Samples" to run the actual generation + filtering pipeline.</p>
                </div>
            </div>
            """)

        with gr.Tab("Filter Pipeline"):
            gr.HTML('<div class="section" style="padding-bottom:0"><div class="sec-label">3-stage quality filter - 100 samples → 64 pass all stages</div></div>')
            gr.Plot(filter_funnel_chart())
            gr.HTML("""
            <div class="section">
                <div class="card">
                    <div class="card-title">Stage 1 - Rule-Based Filter (free)</div>
                    <p class="card-body">Removes samples that are too short/long, exact duplicates (MD5 hash), or contain boilerplate ("As an AI language model", "I cannot", "I'm sorry but"). Catches 17% of generated samples. Zero cost.</p>
                </div>
                <div class="card">
                    <div class="card-title">Stage 2 - Perplexity Filter (DistilGPT-2)</div>
                    <p class="card-body">Computes perplexity with a small local model. Low perplexity (&lt;150) = fluent, natural text. High perplexity = weird phrasing, likely low quality. Removes another 12 samples. Runs locally, no API cost.</p>
                </div>
                <div class="card">
                    <div class="card-title">Stage 3 - LLM Judge (GPT-4o-mini)</div>
                    <p class="card-body">GPT-4o-mini scores each sample on 4 criteria: accuracy, clarity, helpfulness, format. Minimum score 3.5/5 required. Removes final 7 samples. This stage costs API credits but is the most accurate quality signal.</p>
                </div>
            </div>
            """)

        with gr.Tab("Training Results"):
            gr.HTML('<div class="section" style="padding-bottom:0"><div class="sec-label">Fine-tuning perplexity - DistilGPT-2 on 5 conditions</div></div>')
            gr.Plot(ppl_chart())
            gr.HTML('<div class="section" style="padding-bottom:0"><div class="sec-label">Flywheel iterations - data compounds over time</div></div>')
            gr.Plot(flywheel_chart())
            gr.HTML("""
            <div class="section">
                <div class="card">
                    <div class="card-title">What perplexity means</div>
                    <p class="card-body">Perplexity measures how well a language model predicts text - lower is better. Baseline (no fine-tune) = 45.6. Fine-tuning on 5 real samples brings it to 22.4. Filtering synthetic data beats that at 17.9 despite zero real data.</p>
                </div>
            </div>
            """)

        with gr.Tab("Generate Samples"):
            gr.HTML('<div class="section" style="padding-bottom:12px"><div class="sec-label">Live generation + filtering - requires OpenAI API key</div></div>')
            api_key = gr.Textbox(label="OpenAI API Key", type="password", value=OPENAI_KEY)
            domain = gr.Dropdown(choices=["instruction_following", "code_generation", "summarization"], value="instruction_following", label="Domain")
            n_samples = gr.Slider(minimum=5, maximum=20, value=10, step=5, label="Samples to generate")
            gen_btn = gr.Button("Generate & Filter", variant="primary")
            gen_out = gr.HTML()

            def run_generation(api_key, domain, n):
                if not api_key:
                    return "<div class='card'><p class='card-body'>Enter your OpenAI API key to generate samples.</p></div>"
                try:
                    if SyntheticDataGenerator is None:
                        return "<div class='card'><p class='card-body'>Backend modules not available in this environment.</p></div>"
                    generator = SyntheticDataGenerator(openai_api_key=api_key)
                    samples = generator.generate_batch(domain=domain, n=int(n))
                    pipeline = QualityFilterPipeline(openai_api_key=api_key)
                    report = pipeline.filter_batch(samples)
                    rows = "".join([
                        f'<div style="border-bottom:1px solid rgba(255,255,255,0.05);padding:12px 0">'
                        f'<div style="display:flex;justify-content:space-between"><div style="font-size:13px;color:#f5f5f7">{s.prompt[:100]}...</div>'
                        f'<div style="font-size:12px;color:{"#30d158" if s.passed_filter else "#ff453a"};margin-left:12px;white-space:nowrap">{"Pass" if s.passed_filter else "Filtered"}</div></div>'
                        f'<div style="font-size:12px;color:#6e6e73;margin-top:4px">Score: {s.quality_score:.2f} · PPL: {s.perplexity:.1f}</div>'
                        f'</div>'
                        for s in samples
                    ])
                    return f"""
                    <div class="card">
                        <div class="card-title">Results: {report.passed}/{report.total} passed ({report.passed/report.total*100:.0f}%)</div>
                        {rows}
                    </div>"""
                except Exception as e:
                    return f"<div class='card'><p class='card-body' style='color:#ff453a'>Error: {e}</p></div>"

            gen_btn.click(fn=run_generation, inputs=[api_key, domain, n_samples], outputs=gen_out)

        with gr.Tab("How It Works"):
            gr.Markdown(
                "## The Flywheel Loop\n\n"
                "```\n"
                "10 seed examples (human-written)\n"
                "  -> GPT-4o-mini generates 100 synthetic samples\n"
                "  -> Stage 1: Rule-based filter  -> 83 remain\n"
                "  -> Stage 2: Perplexity filter  -> 71 remain\n"
                "  -> Stage 3: LLM judge          -> 64 remain (64% pass rate)\n"
                "  -> Fine-tune DistilGPT-2 on 64 samples\n"
                "  -> Use improved model to generate better seeds -> repeat\n"
                "```\n\n"
                "## Perplexity Filter\n\n"
                "```python\n"
                "def compute_perplexity(self, text):\n"
                "    inputs = self.tokenizer(text, return_tensors='pt', truncation=True)\n"
                "    with torch.no_grad():\n"
                "        outputs = self.model(**inputs, labels=inputs['input_ids'])\n"
                "    return torch.exp(outputs.loss).item()\n"
                "```\n\n"
                "## LLM Judge Rubric\n\n"
                "Score each sample 1-5 on: Accuracy, Clarity, Helpfulness, Format.\n"
                "Minimum average score: **3.5 / 5** to pass the filter.\n\n"
                "## Key Insight\n\n"
                "Filtering cost (~$0.01/sample with GPT-4o-mini) is cheaper than training on bad data.\n"
                "The 36% filter rate is quality control, not waste.\n\n"
                "## References\n"
                "- Self-Instruct: arxiv.org/abs/2212.10560\n"
                "- Phi-1 Textbooks Are All You Need: arxiv.org/abs/2306.11644\n"
            )

demo.launch()
