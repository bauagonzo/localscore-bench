#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
localscore-bench: A reimplementation of localscore using llama-bench.

This tool wraps llama-bench to provide the same benchmarking experience
as the original localscore tool, including score calculation and optional
result submission to localscore.ai.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from lib.config import BASELINE_TESTS, TestConfig
from lib.benchmark import LlamaBench, TestResult
from lib.scoring import calculate_score, ScoreSummary
from lib.display import (
    print_test_results_table,
    print_results_summary,
    print_progress,
    clear_progress,
)
from lib.submit import submit_results, get_user_confirmation


def find_llama_bench() -> Optional[Path]:
    """Try to find llama-bench in common locations."""
    script_dir = Path(__file__).parent.resolve()

    # Check relative to script
    candidates = [
        script_dir / "../llama-b7935/llama-bench",
        script_dir / "llama-bench",
        Path("./llama-bench"),
        Path("./llama-b7935/llama-bench"),
    ]

    # Check PATH
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for d in path_dirs:
        candidates.append(Path(d) / "llama-bench")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark LLM inference performance using llama-bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m model.gguf                    Run benchmark with default settings
  %(prog)s -m model.gguf -y                 Submit results without asking
  %(prog)s -m model.gguf -c                 Run on CPU only
  %(prog)s -m model.gguf --reps 4           Run each test 4 times
  %(prog)s -m model.gguf --llama-bench /path/to/llama-bench
        """,
    )

    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the GGUF model file",
    )

    parser.add_argument(
        "--llama-bench",
        dest="llama_bench_path",
        help="Path to llama-bench binary (auto-detected if not specified)",
    )

    parser.add_argument(
        "-c", "--cpu",
        action="store_true",
        help="Disable GPU acceleration (CPU only)",
    )

    parser.add_argument(
        "-g", "--gpu",
        choices=["auto", "amd", "apple", "nvidia", "disabled"],
        default="auto",
        help="GPU backend to use (default: auto)",
    )

    parser.add_argument(
        "-i", "--gpu-index",
        type=int,
        default=0,
        help="Select GPU by index (default: 0)",
    )

    parser.add_argument(
        "-t", "--threads",
        type=int,
        help="Number of CPU threads to use",
    )

    parser.add_argument(
        "-o", "--output",
        choices=["console", "json", "csv"],
        default="console",
        help="Output format (default: console)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--plaintext",
        action="store_true",
        help="Use plaintext output (no ASCII art)",
    )

    parser.add_argument(
        "-y", "--send-results",
        action="store_true",
        help="Submit results to localscore.ai without asking",
    )

    parser.add_argument(
        "-n", "--no-send-results",
        action="store_true",
        help="Do not submit results to localscore.ai",
    )

    parser.add_argument(
        "-e", "--extended",
        action="store_true",
        help="Run 4 repetitions (shortcut for --reps=4)",
    )

    parser.add_argument(
        "--long",
        action="store_true",
        help="Run 16 repetitions (shortcut for --reps=16)",
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Number of repetitions for each test (default: 1)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 3 representative tests instead of all 9",
    )

    parser.add_argument(
        "--save-csv",
        dest="save_csv",
        metavar="FILE",
        help="Save results to CSV file",
    )

    parser.add_argument(
        "--save-json",
        dest="save_json",
        metavar="FILE",
        help="Save results to JSON file",
    )

    parser.add_argument(
        "--save-console",
        dest="save_console",
        metavar="FILE",
        help="Save console output to text file",
    )

    return parser.parse_args()


def get_quick_tests() -> List[TestConfig]:
    """Return a subset of tests for quick benchmarking."""
    # Select representative tests: small, medium, large
    return [
        BASELINE_TESTS[0],  # pp1024+tg16 (prompt-heavy)
        BASELINE_TESTS[4],  # pp1024+tg1024 (balanced)
        BASELINE_TESTS[8],  # pp16+tg1536 (gen-heavy)
    ]


def run_benchmarks(
    bench: LlamaBench,
    tests: List[TestConfig],
    repetitions: int,
    verbose: bool = False,
) -> List[TestResult]:
    """Run all benchmark tests."""
    results = []

    for i, test_config in enumerate(tests, 1):
        print_progress(test_config.name, i, len(tests))

        if verbose:
            print()  # Newline for verbose output

        try:
            result = bench.run_test(test_config, repetitions)
            results.append(result)
            clear_progress()

            if verbose:
                print(f"  {test_config.name}: pp={result.prompt_tps:.2f} t/s, "
                      f"tg={result.gen_tps:.2f} t/s, ttft={result.ttft_ms:.2f} ms")

        except Exception as e:
            clear_progress()
            print(f"Error running {test_config.name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    return results


def output_json(results: List[TestResult], summary: ScoreSummary, build_info=None, file=None) -> str:
    """Output results in JSON format. Returns the JSON string."""
    import json
    from lib.submit import build_payload

    payload = build_payload(results, summary, build_info=build_info)
    json_str = json.dumps(payload, indent=2)

    if file:
        print(json_str, file=file)
    else:
        print(json_str)

    return json_str


def output_csv(results: List[TestResult], summary: ScoreSummary, file=None) -> str:
    """Output results in CSV format. Returns the CSV string."""
    import io
    output = io.StringIO()

    # Header
    output.write("test,n_prompt,n_gen,prompt_tps,gen_tps,ttft_ms\n")

    # Data rows
    for r in results:
        output.write(f"{r.config.name},{r.config.n_prompt},{r.config.n_gen},"
                     f"{r.prompt_tps:.2f},{r.gen_tps:.2f},{r.ttft_ms:.2f}\n")

    # Summary
    output.write("\n")
    output.write(f"# Summary\n")
    output.write(f"# avg_prompt_tps,{summary.avg_prompt_tps:.2f}\n")
    output.write(f"# avg_gen_tps,{summary.avg_gen_tps:.2f}\n")
    output.write(f"# avg_ttft_ms,{summary.avg_ttft_ms:.2f}\n")
    output.write(f"# performance_score,{summary.performance_score:.2f}\n")

    csv_str = output.getvalue()

    if file:
        print(csv_str, file=file, end='')
    else:
        print(csv_str, end='')

    return csv_str


def output_console_to_string(results: List[TestResult], summary: ScoreSummary,
                              gpu_info: str, model_info: str, plaintext: bool = False) -> str:
    """Capture console output to a string."""
    import io
    import sys

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        print()
        print_test_results_table(results, gpu_info, model_info)
        print_results_summary(summary, plaintext=True)  # Always plaintext for file
    finally:
        sys.stdout = old_stdout

    return buffer.getvalue()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine repetitions
    repetitions = args.reps
    if args.extended:
        repetitions = 4
    if args.long:
        repetitions = 16

    # Find llama-bench
    if args.llama_bench_path:
        llama_bench_path = Path(args.llama_bench_path)
    else:
        llama_bench_path = find_llama_bench()
        if not llama_bench_path:
            print("Error: Could not find llama-bench. Please specify with --llama-bench",
                  file=sys.stderr)
            return 1

    # Verify model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        return 1

    # Determine GPU settings
    n_gpu_layers = 99
    device = None

    if args.cpu or args.gpu == "disabled":
        n_gpu_layers = 0
    elif args.gpu_index is not None and args.gpu_index >= 0:
        # Use specific GPU by index (e.g., Vulkan1 for index 1)
        device = f"Vulkan{args.gpu_index}"
    elif args.gpu != "auto":
        # Map GPU type to device string if needed
        device = args.gpu

    # Select tests
    tests = get_quick_tests() if args.quick else BASELINE_TESTS

    # Create benchmark instance
    try:
        bench = LlamaBench(
            llama_bench_path=str(llama_bench_path),
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            device=device,
            gpu_index=args.gpu_index,
            threads=args.threads,
            verbose=args.verbose,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Print startup info
    if args.output == "console":
        print(f"Using llama-bench: {llama_bench_path}")
        print(f"Model: {model_path}")
        print(f"Running {len(tests)} tests with {repetitions} repetition(s) each")
        print()

    # Get build info first (we'll need it for submission)
    build_info = None
    try:
        build_info = bench.get_build_info()
    except Exception:
        pass  # Non-fatal, we can submit without it

    # Run benchmarks
    results = run_benchmarks(bench, tests, repetitions, args.verbose)

    if not results:
        print("Error: No benchmark results", file=sys.stderr)
        return 1

    # Calculate score
    summary = calculate_score(results)

    # Get display info
    if results:
        # Select correct GPU from comma-separated list based on gpu_index
        gpu_info_full = results[0].gpu_info or "CPU"
        gpu_parts = gpu_info_full.split(",")
        gpu_index = results[0].gpu_index
        if gpu_index < len(gpu_parts):
            gpu_info = gpu_parts[gpu_index].strip()
        else:
            gpu_info = gpu_parts[0].strip()
        model_info = results[0].model_type or model_path.name

        # Truncate long strings
        if len(gpu_info) > 60:
            gpu_info = gpu_info[:57] + "..."
        if len(model_info) > 60:
            model_info = model_info[:57] + "..."

    # Output results
    if args.output == "json":
        output_json(results, summary, build_info)
    elif args.output == "csv":
        output_csv(results, summary)
    else:
        # Console output
        print()
        print_test_results_table(results, gpu_info, model_info)
        print_results_summary(summary, args.plaintext)

    # Save to files if requested
    if args.save_csv:
        try:
            with open(args.save_csv, 'w') as f:
                output_csv(results, summary, file=f)
            print(f"CSV saved to: {args.save_csv}")
        except IOError as e:
            print(f"Error saving CSV: {e}", file=sys.stderr)

    if args.save_json:
        try:
            with open(args.save_json, 'w') as f:
                output_json(results, summary, build_info, file=f)
            print(f"JSON saved to: {args.save_json}")
        except IOError as e:
            print(f"Error saving JSON: {e}", file=sys.stderr)

    if args.save_console:
        try:
            console_output = output_console_to_string(results, summary, gpu_info, model_info, args.plaintext)
            with open(args.save_console, 'w') as f:
                f.write(console_output)
            print(f"Console output saved to: {args.save_console}")
        except IOError as e:
            print(f"Error saving console output: {e}", file=sys.stderr)

    # Handle result submission
    if not args.no_send_results:
        should_submit = False

        if args.send_results:
            should_submit = True
        elif args.output == "console":
            should_submit = get_user_confirmation()

        if should_submit:
            print("Submitting results...")
            success, result = submit_results(results, summary, build_info=build_info, verbose=args.verbose)

            if success:
                print(f"Result Link: {result}")
            else:
                print(f"Failed to submit results: {result}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
