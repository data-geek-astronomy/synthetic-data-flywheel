"""
Synthetic Data Generator

Implements the "generate → filter → fine-tune → evaluate" flywheel.

Key insight: LLMs can generate training data for themselves (self-improvement),
but naive generation produces noise. Quality filtering is what makes it work.

Pipeline:
  1. Seed examples (real data, ~50-500 samples)
  2. GPT-4o generates variations (prompt → completion pairs)
  3. Quality filters: perplexity, dedup, length, LLM judge
  4. Fine-tune a smaller model on filtered synthetic data
  5. Evaluate: synthetic-trained vs real-trained vs baseline
  6. Use fine-tuned model to generate better seeds → repeat
"""

import json
import re
import random
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class SyntheticSample:
    prompt: str
    completion: str
    domain: str
    generation_id: str  # track which generator produced this
    quality_score: Optional[float] = None
    perplexity: Optional[float] = None
    passed_filter: Optional[bool] = None
    filter_reason: Optional[str] = None


# ── Seed Data for Different Domains ─────────────────────────────────────────

SEED_DATA = {
    "instruction_following": [
        {"prompt": "Explain what photosynthesis is in one sentence.", "completion": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."},
        {"prompt": "What is the difference between a list and a tuple in Python?", "completion": "Lists are mutable (can be changed after creation) while tuples are immutable. Lists use [] brackets and tuples use () parentheses."},
        {"prompt": "Name three primary colors.", "completion": "The three primary colors are red, blue, and yellow."},
        {"prompt": "What year did World War II end?", "completion": "World War II ended in 1945."},
        {"prompt": "Convert 100 Fahrenheit to Celsius.", "completion": "100°F = (100-32) × 5/9 = 37.78°C"},
    ],
    "code_generation": [
        {"prompt": "Write a Python function to check if a number is prime.", "completion": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True"},
        {"prompt": "Write a function to reverse a string in Python.", "completion": "def reverse_string(s):\n    return s[::-1]"},
        {"prompt": "Write a Python one-liner to count words in a string.", "completion": "word_count = lambda s: len(s.split())"},
    ],
    "summarization": [
        {"prompt": "Summarize this in one sentence: The Industrial Revolution was a period of major industrialization and innovation that took place during the late 1700s and early 1800s. It began in Great Britain and quickly spread throughout Western Europe and North America.", "completion": "The Industrial Revolution (late 1700s–early 1800s) was a transformative period of industrialization that originated in Britain and spread across the Western world."},
        {"prompt": "Summarize: Machine learning is a method of data analysis that automates analytical model building. It is based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.", "completion": "Machine learning automates pattern recognition and decision-making from data with minimal human involvement."},
    ],
}

GENERATION_PROMPTS = {
    "instruction_following": """You are generating training data for an instruction-following AI assistant.
Given these seed examples of (prompt, completion) pairs:
{examples}

Generate {n} NEW diverse (prompt, completion) pairs in the same style.
- Prompts should be clear, practical questions or instructions
- Completions should be accurate, concise, and helpful
- Cover different topics than the seeds (science, history, math, language, etc.)
- Each pair must be different from the others

Return JSON array: [{{"prompt": "...", "completion": "..."}}]""",

    "code_generation": """You are generating Python coding training data.
Given these seed examples:
{examples}

Generate {n} NEW Python coding (prompt, completion) pairs.
- Prompts: clear function/algorithm requests
- Completions: correct, readable Python code (no markdown, just code)
- Cover: data structures, algorithms, string ops, math, file I/O

Return JSON array: [{{"prompt": "...", "completion": "..."}}]""",

    "summarization": """You are generating summarization training data.
Given these seed examples:
{examples}

Generate {n} NEW (long_text, one_sentence_summary) pairs.
- Create a 3-5 sentence paragraph, then summarize it in one sentence
- Topics: technology, science, history, business, culture
- Summaries must capture the main point accurately

Return JSON array: [{{"prompt": "Summarize: <your paragraph>", "completion": "<one sentence summary>"}}]""",
}


class SyntheticDataGenerator:
    """
    Uses GPT-4o to generate synthetic training data from seed examples.

    Few-shot prompting strategy:
    - Provide 3-5 real examples so model learns the style and format
    - Request JSON output for reliable parsing
    - Generate in batches of 10 to balance quality vs. cost
    - Track generation provenance (which model, which seeds, when)
    """

    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.generation_count = 0

    def generate_batch(
        self,
        domain: str,
        n: int = 10,
        seed_examples: Optional[List[Dict]] = None,
    ) -> List[SyntheticSample]:
        """Generate a batch of synthetic samples for a given domain."""
        if domain not in SEED_DATA:
            raise ValueError(f"Unknown domain: {domain}. Choose from {list(SEED_DATA.keys())}")

        seeds = seed_examples or SEED_DATA[domain]
        examples_str = "\n".join(
            f"Prompt: {s['prompt']}\nCompletion: {s['completion']}\n"
            for s in seeds[:4]
        )

        prompt_template = GENERATION_PROMPTS.get(domain, GENERATION_PROMPTS["instruction_following"])
        user_prompt = prompt_template.format(examples=examples_str, n=n)

        self.generation_count += 1
        gen_id = f"gen_{self.generation_count:03d}_{domain}"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a synthetic training data generator. Always return valid JSON arrays."},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=3000,
                temperature=0.9,  # Higher temp = more diversity
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            # Handle different JSON structures the model might return
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = next(
                    (v for v in parsed.values() if isinstance(v, list)),
                    []
                )
            else:
                items = []

            samples = []
            for item in items[:n]:
                if isinstance(item, dict) and "prompt" in item and "completion" in item:
                    samples.append(SyntheticSample(
                        prompt=str(item["prompt"]).strip(),
                        completion=str(item["completion"]).strip(),
                        domain=domain,
                        generation_id=gen_id,
                    ))

            return samples

        except Exception as e:
            print(f"[Generator] Error in batch generation: {e}")
            return []

    def get_seed_data(self, domain: str) -> List[SyntheticSample]:
        """Return seed data as SyntheticSample objects."""
        seeds = SEED_DATA.get(domain, [])
        return [
            SyntheticSample(
                prompt=s["prompt"],
                completion=s["completion"],
                domain=domain,
                generation_id="seed",
                quality_score=1.0,
                passed_filter=True,
            )
            for s in seeds
        ]
