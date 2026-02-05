"""Result submission to localscore.ai."""

import json
import platform
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple

from .config import LOCALSCORE_API_URL, LOCALSCORE_BASE_URL
from .benchmark import TestResult
from .scoring import ScoreSummary


def get_system_info() -> Dict[str, Any]:
    """Gather system information."""
    uname = platform.uname()

    # Try to get RAM info
    ram_gb = 0.0
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    # Convert from KB to GB
                    ram_kb = int(line.split()[1])
                    ram_gb = ram_kb / (1024 * 1024)
                    break
    except (FileNotFoundError, PermissionError):
        pass

    return {
        "cpu_name": uname.processor or uname.machine,
        "cpu_arch": uname.machine,
        "ram_gb": round(ram_gb, 2),
        "kernel_type": uname.system,
        "kernel_release": uname.release,
        "version": uname.version,
    }


def build_payload(
    results: List[TestResult],
    summary: ScoreSummary,
    system_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON payload for submission."""

    if not results:
        raise ValueError("No results to submit")

    first_result = results[0]

    # Runtime info
    runtime_info = {
        "name": "localscore-bench",
        "version": "1.0.0",
        "commit": "llama-bench-wrapper",
    }

    # System info
    if system_info is None:
        system_info = get_system_info()

    # Update with info from benchmark if available
    if first_result.cpu_info:
        system_info["cpu_name"] = first_result.cpu_info

    # Accelerator info
    gpu_parts = first_result.gpu_info.split(",") if first_result.gpu_info else ["CPU"]
    gpu_name = gpu_parts[0].strip() if gpu_parts else "Unknown"

    # Try to determine manufacturer
    manufacturer = "Unknown"
    if "NVIDIA" in gpu_name.upper():
        manufacturer = "NVIDIA"
    elif "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper():
        manufacturer = "AMD"
    elif "INTEL" in gpu_name.upper():
        manufacturer = "Intel"
    elif "APPLE" in gpu_name.upper():
        manufacturer = "Apple"

    accelerator_info = {
        "name": gpu_name,
        "manufacturer": manufacturer,
        "memory_gb": 0.0,  # Not available from llama-bench
        "type": "GPU" if first_result.gpu_info else "CPU",
    }

    # Build results array
    results_array = []
    for r in results:
        results_array.append({
            "n_prompt": r.config.n_prompt,
            "n_gen": r.config.n_gen,
            "prompt_tps": r.prompt_tps,
            "gen_tps": r.gen_tps,
            "ttft_ms": r.ttft_ms,
            "test_name": r.config.name,
        })

    # Results summary
    results_summary = {
        "avg_prompt_tps": summary.avg_prompt_tps,
        "avg_gen_tps": summary.avg_gen_tps,
        "avg_ttft_ms": summary.avg_ttft_ms,
        "performance_score": summary.performance_score,
    }

    return {
        "runtime_info": runtime_info,
        "system_info": system_info,
        "accelerator_info": accelerator_info,
        "results": results_array,
        "results_summary": results_summary,
    }


def submit_results(
    results: List[TestResult],
    summary: ScoreSummary,
    max_retries: int = 3,
    verbose: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Submit benchmark results to localscore.ai.

    Returns:
        Tuple of (success, result_url or error_message)
    """
    payload = build_payload(results, summary)
    payload_json = json.dumps(payload)

    if verbose:
        print(f"Submitting payload: {payload_json}")

    for attempt in range(max_retries):
        if attempt > 0:
            wait_time = 2 ** attempt
            print(f"Retry attempt {attempt + 1} of {max_retries} after {wait_time} seconds...")
            time.sleep(wait_time)

        try:
            req = urllib.request.Request(
                LOCALSCORE_API_URL,
                data=payload_json.encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'localscore-bench/1.0',
                },
                method='POST',
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    response_data = json.loads(response.read().decode('utf-8'))
                    if 'id' in response_data:
                        result_url = f"{LOCALSCORE_BASE_URL}/result/{response_data['id']}"
                        return True, result_url
                    return False, "Response missing 'id' field"
                else:
                    error_msg = f"HTTP {response.status}"
                    if attempt == max_retries - 1:
                        return False, error_msg

        except urllib.error.HTTPError as e:
            error_msg = f"HTTP Error {e.code}: {e.reason}"
            if attempt == max_retries - 1:
                return False, error_msg

        except urllib.error.URLError as e:
            error_msg = f"URL Error: {e.reason}"
            if attempt == max_retries - 1:
                return False, error_msg

        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {e}"
            if attempt == max_retries - 1:
                return False, error_msg

        except Exception as e:
            error_msg = f"Error: {e}"
            if attempt == max_retries - 1:
                return False, error_msg

    return False, f"Failed after {max_retries} attempts"


def get_user_confirmation() -> bool:
    """Ask user for confirmation before submitting results."""
    try:
        response = input("\nDo you want to submit your results to https://localscore.ai? "
                        "The results will be public (y/n): ")
        return response.lower() in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        return False
