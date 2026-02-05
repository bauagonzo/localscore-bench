"""Display and progress output matching localscore's format."""

import sys
from typing import List, Optional

from .benchmark import TestResult
from .scoring import ScoreSummary, score_interpretation


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    MAGENTA = "\033[1;35m"
    GREEN = "\033[32m"
    GREEN_BOLD = "\033[1;32m"
    GREEN_ITALIC = "\033[3;32m"
    CYAN = "\033[36m"
    CYAN_BOLD = "\033[1;36m"
    CYAN_ITALIC = "\033[3;36m"
    YELLOW = "\033[33m"
    YELLOW_BOLD = "\033[1;33m"
    YELLOW_ITALIC = "\033[3;33m"


# ASCII art digits for large score display
ASCII_DIGITS = {
    '0': [
        " ██████╗ ",
        "██╔═████╗",
        "██║██╔██║",
        "████╔╝██║",
        "╚██████╔╝",
        " ╚═════╝ ",
    ],
    '1': [
        " ██╗",
        "███║",
        "╚██║",
        " ██║",
        " ██║",
        " ╚═╝",
    ],
    '2': [
        "██████╗ ",
        "╚════██╗",
        " █████╔╝",
        "██╔═══╝ ",
        "███████╗",
        "╚══════╝",
    ],
    '3': [
        "██████╗ ",
        "╚════██╗",
        " █████╔╝",
        " ╚═══██╗",
        "██████╔╝",
        "╚═════╝ ",
    ],
    '4': [
        "██╗  ██╗",
        "██║  ██║",
        "███████║",
        "╚════██║",
        "     ██║",
        "     ╚═╝",
    ],
    '5': [
        "███████╗",
        "██╔════╝",
        "███████╗",
        "╚════██║",
        "███████║",
        "╚══════╝",
    ],
    '6': [
        " ██████╗ ",
        "██╔════╝ ",
        "███████╗ ",
        "██╔═══██╗",
        "╚██████╔╝",
        " ╚═════╝ ",
    ],
    '7': [
        "███████╗",
        "╚════██║",
        "    ██╔╝",
        "   ██╔╝ ",
        "   ██║  ",
        "   ╚═╝  ",
    ],
    '8': [
        " █████╗ ",
        "██╔══██╗",
        "╚█████╔╝",
        "██╔══██╗",
        "╚█████╔╝",
        " ╚════╝ ",
    ],
    '9': [
        " █████╗ ",
        "██╔══██╗",
        "╚██████║",
        " ╚═══██║",
        " █████╔╝",
        " ╚════╝ ",
    ],
}


def supports_color() -> bool:
    """Check if the terminal supports color."""
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def color(code: str) -> str:
    """Return color code if terminal supports it, empty string otherwise."""
    if supports_color():
        return code
    return ""


def print_large_number(num: int) -> None:
    """Print a number in large ASCII art."""
    digits = str(num)
    if not supports_color():
        print(f"LocalScore: {num}")
        return

    # Get the ASCII art for each digit
    lines = [[] for _ in range(6)]
    for d in digits:
        if d in ASCII_DIGITS:
            for i, line in enumerate(ASCII_DIGITS[d]):
                lines[i].append(line)

    # Print each line
    for line_parts in lines:
        print("".join(line_parts))


def print_header(gpu_info: str, model_info: str, total_width: int = 80) -> None:
    """Print the table header."""
    border = "+" + "-" * (total_width - 2) + "+"

    # Center the info strings
    gpu_centered = gpu_info.center(total_width - 2)
    model_centered = model_info.center(total_width - 2)

    print(border)
    print(f"|{gpu_centered}|")
    print(f"|{model_centered}|")
    print(border)

    # Column headers
    headers = ["test", "run", "avg time", "tokens", "pp t/s", "tg t/s", "ttft"]
    widths = [13, 6, 12, 16, 10, 10, 10]

    header_line = "|"
    separator_line = "|"
    for h, w in zip(headers, widths):
        header_line += f" {h:>{w}} |"
        separator_line += " " + "-" * w + " |"

    print(header_line)
    print(separator_line)


def print_test_row(result: TestResult, run: int, total_runs: int) -> None:
    """Print a single test result row."""
    test_name = result.config.name

    # Format time
    avg_time_ms = (result.pp_avg_ns + result.tg_avg_ns) / 1e6
    if avg_time_ms < 1000:
        time_str = f"{avg_time_ms:.2f} ms"
    else:
        time_str = f"{avg_time_ms / 1000:.2f} s"

    # Format tokens
    total_tokens = result.config.n_prompt + result.config.n_gen
    tokens_str = f"{total_tokens} / {total_tokens}"

    # Format TTFT
    if result.ttft_ms < 1000:
        ttft_str = f"{result.ttft_ms:.2f} ms"
    else:
        ttft_str = f"{result.ttft_ms / 1000:.2f} s"

    print(f"| {test_name:>13} | {run:>2}/{total_runs:<2} | {time_str:>12} | "
          f"{tokens_str:>16} | {result.prompt_tps:>10.2f} | {result.gen_tps:>10.2f} | "
          f"{ttft_str:>10} |")


def print_footer(total_width: int = 80) -> None:
    """Print the table footer."""
    border = "+" + "-" * (total_width - 2) + "+"
    print(border)


def print_progress(test_name: str, current: int, total: int) -> None:
    """Print progress indicator."""
    print(f"\rRunning test {current}/{total}: {test_name}...", end="", flush=True)


def clear_progress() -> None:
    """Clear the progress line."""
    print("\r" + " " * 60 + "\r", end="", flush=True)


def print_results_summary(summary: ScoreSummary, plaintext: bool = False) -> None:
    """Print the final results summary with score."""
    print()

    if plaintext or not supports_color():
        print(f"LocalScore: {int(summary.performance_score)}")
    else:
        print(f"{color(Colors.MAGENTA)}")
        print_large_number(int(summary.performance_score))
        print(f"{color(Colors.RESET)}")

    interpretation = score_interpretation(summary.performance_score)
    print()

    # Token Generation
    print(f"{color(Colors.GREEN)}Token Generation: \t "
          f"{color(Colors.GREEN_BOLD)}{summary.avg_gen_tps:.2f}"
          f"{color(Colors.RESET)} {color(Colors.GREEN_ITALIC)}tok/s{color(Colors.RESET)}")

    # Prompt Processing
    print(f"{color(Colors.CYAN)}Prompt Processing: \t "
          f"{color(Colors.CYAN_BOLD)}{summary.avg_prompt_tps:.2f}"
          f"{color(Colors.RESET)} {color(Colors.CYAN_ITALIC)}tok/s{color(Colors.RESET)}")

    # Time to First Token
    print(f"{color(Colors.YELLOW)}Time to First Token:\t "
          f"{color(Colors.YELLOW_BOLD)}{summary.avg_ttft_ms:.2f}"
          f"{color(Colors.RESET)} {color(Colors.YELLOW_ITALIC)}ms{color(Colors.RESET)}")

    print()


def print_test_results_table(results: List[TestResult], gpu_info: str, model_info: str) -> None:
    """Print all test results in a table format."""
    print_header(gpu_info, model_info)

    for i, result in enumerate(results, 1):
        print_test_row(result, i, len(results))

    print_footer()
