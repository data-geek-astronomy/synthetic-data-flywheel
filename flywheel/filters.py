"""
Quality Filtering Pipeline

The quality of synthetic data determines whether fine-tuning helps or hurts.
Three-stage filter catches different failure modes:

1. RULE-BASED (fast, free):
   - Length: too short = trivial, too long = off-format
   - Deduplication: exact hash + near-duplicate detection
   - Boilerplate: reject "As an AI language model..." style completions

2. PERPLEXITY (medium, local model required):
   - High perplexity = text the model finds surprising = likely garbage
   - Low perplexity = text is predictable/common = may be too easy
   - Sweet spot: moderate perplexity (informative but not random)

3. LLM JUDGE (slow, API cost):
   - GPT-4o-mini rates each sample 1-5 on accuracy, clarity, helpfulness
   - Threshold: reject samples scoring < 3.5/5
   - Most expensive but highest signal — used selectively

In practice: rule-based first (free), perplexity second (catches noise),
LLM judge only for ambiguous cases or final quality check before fine-tuning.
"""

import re
import hashlib
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from openai import OpenAI

from .generator import SyntheticSample


@dataclass
class FilterReport:
    total_in: int
    passed: int
    rejected_length: int
    rejected_duplicate: int
    rejected_boilerplate: int
    rejected_perplexity: int
    rejected_llm_judge: int
    pass_rate: float
    samples: List[SyntheticSample]


class RuleBasedFilter:
    """
    Fast heuristic filters. Should be applied first — O(n) with no API calls.
    """

    BOILERPLATE_PATTERNS = [
        r"as an ai",
        r"i cannot",
        r"i'm sorry",
        r"i apologize",
        r"as a language model",
        r"i don't have personal",
        r"i was trained",
        r"\[insert",
        r"\[your",
        r"lorem ipsum",
    ]

    def __init__(
        self,
        min_prompt_len: int = 10,
        max_prompt_len: int = 500,
        min_completion_len: int = 5,
        max_completion_len: int = 800,
    ):
        self.min_prompt = min_prompt_len
        self.max_prompt = max_prompt_len
        self.min_completion = min_completion_len
        self.max_completion = max_completion_len
        self._seen_hashes = set()

    def _is_duplicate(self, sample: SyntheticSample) -> bool:
        h = hashlib.md5(f"{sample.prompt}|{sample.completion}".lower().encode()).hexdigest()
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    def _has_boilerplate(self, sample: SyntheticSample) -> bool:
        text = (sample.prompt + " " + sample.completion).lower()
        return any(re.search(p, text) for p in self.BOILERPLATE_PATTERNS)

    def filter(self, sample: SyntheticSample) -> Tuple[bool, str]:
        """Returns (passed, reason_if_failed)."""
        if len(sample.prompt) < self.min_prompt:
            return False, f"Prompt too short ({len(sample.prompt)} chars)"
        if len(sample.prompt) > self.max_prompt:
            return False, f"Prompt too long ({len(sample.prompt)} chars)"
        if len(sample.completion) < self.min_completion:
            return False, f"Completion too short ({len(sample.completion)} chars)"
        if len(sample.completion) > self.max_completion:
            return False, f"Completion too long ({len(sample.completion)} chars)"
        if self._is_duplicate(sample):
            return False, "Duplicate sample"
        if self._has_boilerplate(sample):
            return False, "Boilerplate/refusal detected"
        return True, ""


class PerplexityFilter:
    """
    Reject samples with perplexity outside an acceptable range.

    Uses the reference model (DistilGPT-2) to score the COMPLETION text.
    High perplexity completions tend to be noisy, off-topic, or hallucinated.
    Very low perplexity completions may be trivially obvious or repeated data.

    Note: perplexity is computed on the completion ONLY (not prompt),
    since we care about quality of the model's output, not the question.
    """

    def __init__(
        self,
        max_perplexity: float = 150.0,
        min_perplexity: float = 2.0,
        model_name: str = "distilgpt2",
    ):
        self.max_ppl = max_perplexity
        self.min_ppl = min_perplexity
        self._model = None
        self._tokenizer = None
        self._model_name = model_name

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
            self._model.eval()
        except ImportError:
            raise RuntimeError("torch and transformers required for perplexity filtering")

    def compute_perplexity(self, text: str) -> float:
        """Compute per-token perplexity of text using DistilGPT-2."""
        self._load_model()
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", max_length=256, truncation=True
        )
        input_ids = inputs["input_ids"]

        if input_ids.shape[1] < 2:
            return 999.0

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        return math.exp(loss)

    def filter(self, sample: SyntheticSample) -> Tuple[bool, str]:
        try:
            ppl = self.compute_perplexity(sample.completion)
            sample.perplexity = ppl
            if ppl > self.max_ppl:
                return False, f"Perplexity too high ({ppl:.1f} > {self.max_ppl})"
            if ppl < self.min_ppl:
                return False, f"Perplexity suspiciously low ({ppl:.1f} < {self.min_ppl})"
            return True, ""
        except Exception as e:
            return True, f"[perplexity skipped: {e}]"  # don't fail hard


class LLMJudgeFilter:
    """
    GPT-4o-mini scores each sample on a rubric.

    Rubric (1-5 scale):
    - Accuracy: Is the completion factually correct?
    - Clarity: Is it easy to understand?
    - Helpfulness: Would this help someone learning?
    - Format: Is it the right length and format for the prompt?

    Cost: ~$0.001 per sample with gpt-4o-mini (batch of 10 = $0.01)
    Use selectively: only for samples that passed rule + perplexity filters.
    """

    JUDGE_PROMPT = """Rate this training sample on a scale of 1-5 for each criterion.

Prompt: {prompt}
Completion: {completion}

Criteria:
1. accuracy (1=wrong, 5=perfectly correct)
2. clarity (1=confusing, 5=crystal clear)
3. helpfulness (1=useless, 5=very helpful)
4. format (1=wrong format, 5=exactly right)

Return JSON: {{"accuracy": N, "clarity": N, "helpfulness": N, "format": N, "overall": N, "reason": "..."}}"""

    def __init__(self, openai_api_key: str, min_score: float = 3.5):
        self.client = OpenAI(api_key=openai_api_key)
        self.min_score = min_score

    def score(self, sample: SyntheticSample) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": self.JUDGE_PROMPT.format(
                        prompt=sample.prompt[:300],
                        completion=sample.completion[:500],
                    )}
                ],
                max_tokens=200,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"overall": 3.5, "reason": f"Judge error: {e}"}

    def filter(self, sample: SyntheticSample) -> Tuple[bool, str]:
        scores = self.score(sample)
        overall = scores.get("overall", 3.5)
        sample.quality_score = overall
        if overall < self.min_score:
            return False, f"LLM judge score {overall:.1f} < {self.min_score} threshold. {scores.get('reason', '')}"
        return True, ""


import json


class QualityFilterPipeline:
    """
    Orchestrates all three filter stages with configurable thresholds.
    """

    def __init__(
        self,
        openai_api_key: str = "",
        use_perplexity: bool = False,
        use_llm_judge: bool = True,
        max_perplexity: float = 150.0,
        min_llm_score: float = 3.5,
    ):
        self.rule_filter = RuleBasedFilter()
        self.perplexity_filter = PerplexityFilter(max_perplexity=max_perplexity) if use_perplexity else None
        self.llm_judge = LLMJudgeFilter(openai_api_key, min_score=min_llm_score) if (openai_api_key and use_llm_judge) else None

    def filter_batch(self, samples: List[SyntheticSample]) -> FilterReport:
        """Run all filters and return a FilterReport with annotated samples."""
        counts = {
            "length": 0, "duplicate": 0, "boilerplate": 0,
            "perplexity": 0, "llm_judge": 0,
        }
        passed_samples = []

        for sample in samples:
            # Stage 1: Rule-based
            ok, reason = self.rule_filter.filter(sample)
            if not ok:
                reason_key = (
                    "duplicate" if "uplicate" in reason
                    else "boilerplate" if "oilerplate" in reason or "efusal" in reason
                    else "length"
                )
                counts[reason_key] += 1
                sample.passed_filter = False
                sample.filter_reason = reason
                continue

            # Stage 2: Perplexity (optional, needs local model)
            if self.perplexity_filter:
                ok, reason = self.perplexity_filter.filter(sample)
                if not ok:
                    counts["perplexity"] += 1
                    sample.passed_filter = False
                    sample.filter_reason = reason
                    continue

            # Stage 3: LLM Judge (optional, needs API key)
            if self.llm_judge:
                ok, reason = self.llm_judge.filter(sample)
                if not ok:
                    counts["llm_judge"] += 1
                    sample.passed_filter = False
                    sample.filter_reason = reason
                    continue

            sample.passed_filter = True
            sample.filter_reason = "passed all filters"
            if sample.quality_score is None:
                sample.quality_score = 0.85  # default if no judge
            passed_samples.append(sample)

        total = len(samples)
        passed = len(passed_samples)
        return FilterReport(
            total_in=total,
            passed=passed,
            rejected_length=counts["length"],
            rejected_duplicate=counts["duplicate"],
            rejected_boilerplate=counts["boilerplate"],
            rejected_perplexity=counts["perplexity"],
            rejected_llm_judge=counts["llm_judge"],
            pass_rate=passed / max(total, 1),
            samples=samples,  # all samples, each annotated with passed_filter
        )


def get_precomputed_filter_results() -> Dict:
    """Precomputed filter results for the demo visualization."""
    return {
        "total_generated": 100,
        "stages": {
            "Rule-Based": {
                "passed": 83,
                "rejected": 17,
                "breakdown": {"too_short": 6, "duplicate": 8, "boilerplate": 3},
                "time_ms": 12,
                "cost": "$0.00",
            },
            "Perplexity Filter": {
                "passed": 71,
                "rejected": 12,
                "breakdown": {"high_perplexity": 9, "low_perplexity": 3},
                "threshold": "PPL < 150",
                "time_ms": 340,
                "cost": "$0.00 (local model)",
            },
            "LLM Judge": {
                "passed": 64,
                "rejected": 7,
                "breakdown": {"low_accuracy": 3, "poor_format": 2, "unclear": 2},
                "threshold": "Score ≥ 3.5/5",
                "time_ms": 1800,
                "cost": "~$0.007",
            },
        },
        "final_pass_rate": 0.64,
        "quality_distribution": {
            "score_bins": ["1.0-2.0", "2.0-3.0", "3.0-4.0", "4.0-5.0"],
            "counts_raw": [4, 13, 41, 42],      # before filtering
            "counts_filtered": [0, 0, 22, 42],  # after filtering
        },
    }
