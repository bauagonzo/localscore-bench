"""Test configurations matching localscore's baseline tests."""

from dataclasses import dataclass
from typing import List

@dataclass
class TestConfig:
    """A single test configuration."""
    n_prompt: int
    n_gen: int
    description: str

    @property
    def name(self) -> str:
        """Generate test name like 'pp1024+tg16'."""
        return f"pp{self.n_prompt}+tg{self.n_gen}"

    @property
    def ratio(self) -> str:
        """Get prompt:gen ratio as string."""
        if self.n_gen == 0:
            return "inf:1"
        r = self.n_prompt / self.n_gen
        if r >= 1:
            return f"{r:.0f}:1"
        return f"1:{1/r:.0f}"


# Baseline test configurations matching localscore
BASELINE_TESTS: List[TestConfig] = [
    TestConfig(1024, 16, "Title generation"),
    TestConfig(4096, 256, "Content summarization"),
    TestConfig(2048, 256, "Code review/fix"),
    TestConfig(2048, 768, "Standard code chat"),
    TestConfig(1024, 1024, "Code back-and-forth"),
    TestConfig(1280, 3072, "Reasoning over code"),
    TestConfig(384, 1152, "Code gen with interaction"),
    TestConfig(64, 1024, "Code gen/ideation"),
    TestConfig(16, 1536, "QA, Storytelling, Reasoning"),
]

# API endpoint for result submission
LOCALSCORE_API_URL = "https://www.localscore.ai/api/results"
LOCALSCORE_BASE_URL = "https://www.localscore.ai"
