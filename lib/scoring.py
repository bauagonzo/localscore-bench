"""Score calculation matching localscore's algorithm."""

from typing import List
from dataclasses import dataclass

from .benchmark import TestResult


@dataclass
class ScoreSummary:
    """Summary of benchmark results with calculated score."""
    avg_prompt_tps: float
    avg_gen_tps: float
    avg_ttft_ms: float
    performance_score: float


def calculate_score(results: List[TestResult]) -> ScoreSummary:
    """
    Calculate the localscore performance score.

    The score is a geometric mean of:
    - Average prompt tokens per second
    - Average generation tokens per second
    - 1000 / average time to first token (ms)

    Multiplied by 10 for readability.

    Formula: score = 10 * (avg_prompt_tps * avg_gen_tps * (1000 / avg_ttft_ms))^(1/3)
    """
    if not results:
        return ScoreSummary(0.0, 0.0, 0.0, 0.0)

    # Calculate averages
    avg_prompt_tps = sum(r.prompt_tps for r in results) / len(results)
    avg_gen_tps = sum(r.gen_tps for r in results) / len(results)
    avg_ttft_ms = sum(r.ttft_ms for r in results) / len(results)

    # Calculate geometric mean score
    if avg_ttft_ms > 0:
        ttft_factor = 1000 / avg_ttft_ms
    else:
        ttft_factor = 0

    if avg_prompt_tps > 0 and avg_gen_tps > 0 and ttft_factor > 0:
        score = 10 * (avg_prompt_tps * avg_gen_tps * ttft_factor) ** (1/3)
    else:
        score = 0

    return ScoreSummary(
        avg_prompt_tps=avg_prompt_tps,
        avg_gen_tps=avg_gen_tps,
        avg_ttft_ms=avg_ttft_ms,
        performance_score=score,
    )


def score_interpretation(score: float) -> str:
    """Return a human-readable interpretation of the score."""
    if score >= 1000:
        return "Excellent"
    elif score >= 500:
        return "Very Good"
    elif score >= 250:
        return "Good"
    elif score >= 100:
        return "Acceptable"
    else:
        return "Poor"
