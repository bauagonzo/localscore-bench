"""Result submission to localscore.ai."""

import json
import platform
import time
from typing import List, Dict, Any, Optional, Tuple

# Try to use requests library if available (better TLS handling)
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False

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
    build_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON payload for submission matching original localscore format."""

    if not results:
        raise ValueError("No results to submit")

    first_result = results[0]

    # Runtime info - use "llamafile" for compatibility with server
    runtime_info = {
        "name": "llamafile",
        "version": "1.0.0-llama-bench",
        "commit": build_info.get("build_commit", "unknown") if build_info else "unknown",
    }

    # System info
    if system_info is None:
        system_info = get_system_info()

    # Update with info from benchmark if available
    if first_result.cpu_info:
        system_info["cpu_name"] = first_result.cpu_info

    # Accelerator info - use correct GPU based on gpu_index
    gpu_parts = first_result.gpu_info.split(",") if first_result.gpu_info else ["CPU"]
    gpu_index = getattr(first_result, 'gpu_index', 0)
    if gpu_index < len(gpu_parts):
        gpu_name = gpu_parts[gpu_index].strip()
    else:
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

    # Build results array with all fields expected by the server
    results_array = []
    for r in results:
        # Get build info
        b_commit = build_info.get("build_commit", "") if build_info else ""
        b_number = build_info.get("build_number", 0) if build_info else 0

        # Calculate timing values
        avg_time_ms = (r.pp_avg_ns + r.tg_avg_ns) / 1e6

        # Combine samples from pp and tg tests
        combined_samples = r.pp_samples_ns + r.tg_samples_ns

        results_array.append({
            # Build info
            "build_commit": b_commit,
            "build_number": b_number,
            # Model info
            "model_name": r.model_type.split()[0] if r.model_type else "",
            "model_quant_str": r.model_type.split()[-1] if r.model_type else "",
            "model_params_str": "",
            "model_filename": r.model_filename,
            "model_type": r.model_type,
            "model_size": r.model_size,
            "model_n_params": r.model_n_params,
            # Test config
            "n_prompt": r.config.n_prompt,
            "n_gen": r.config.n_gen,
            "test_time": "",
            # Timing
            "avg_time_ms": avg_time_ms,
            "stddev_time_ms": 0,
            # Performance metrics
            "prompt_tps": r.prompt_tps,
            "prompt_tps_watt": 0.0,
            "prompt_tps_stddev": 0.0,
            "gen_tps": r.gen_tps,
            "gen_tps_watt": 0.0,
            "gen_tps_stddev": 0.0,
            # Other
            "name": r.config.name,
            "power_watts": 0.0,
            "ttft_ms": r.ttft_ms,
            "main_gpu": 0,
            "samples_ns": combined_samples,
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


def _submit_with_requests(payload_json: str, verbose: bool) -> Tuple[bool, str]:
    """Submit using requests library (matching official Python implementation)."""
    # Use json= parameter like official implementation, with minimal headers
    payload = json.loads(payload_json)

    try:
        response = requests.post(
            LOCALSCORE_API_URL,
            json=payload,  # Use json= like official implementation
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if verbose:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")

        if response.status_code == 200:
            data = response.json()
            if 'id' in data:
                return True, f"{LOCALSCORE_BASE_URL}/result/{data['id']}"
            return False, "Response missing 'id' field"
        else:
            return False, f"HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return False, f"Error: {e}"


def _submit_with_urllib(payload_json: str, verbose: bool) -> Tuple[bool, str]:
    """Submit using urllib (fallback)."""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'User-Agent': 'hurl/1.o (https://github.com/jart/cosmopolitan)',
    }

    try:
        req = urllib.request.Request(
            LOCALSCORE_API_URL,
            data=payload_json.encode('utf-8'),
            headers=headers,
            method='POST',
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if 'id' in data:
                    return True, f"{LOCALSCORE_BASE_URL}/result/{data['id']}"
                return False, "Response missing 'id' field"
            return False, f"HTTP {response.status}"

    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode('utf-8')
            return False, f"HTTP Error {e.code}: {e.reason} - {error_body}"
        except:
            return False, f"HTTP Error {e.code}: {e.reason}"

    except urllib.error.URLError as e:
        return False, f"URL Error: {e.reason}"

    except Exception as e:
        return False, f"Error: {e}"


def submit_results(
    results: List[TestResult],
    summary: ScoreSummary,
    build_info: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    verbose: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Submit benchmark results to localscore.ai.

    Returns:
        Tuple of (success, result_url or error_message)
    """
    payload = build_payload(results, summary, build_info=build_info)
    payload_json = json.dumps(payload)

    if verbose:
        print(f"Submitting payload: {payload_json}")

    # Choose submission method
    submit_func = _submit_with_requests if HAS_REQUESTS else _submit_with_urllib

    for attempt in range(max_retries):
        if attempt > 0:
            wait_time = 2 ** attempt
            print(f"Retry attempt {attempt + 1} of {max_retries} after {wait_time} seconds...")
            time.sleep(wait_time)

        success, result = submit_func(payload_json, verbose)

        if success:
            return True, result

        if verbose:
            print(f"Server response: {result}")

        if attempt == max_retries - 1:
            return False, result

    return False, f"Failed after {max_retries} attempts"


def get_user_confirmation() -> bool:
    """Ask user for confirmation before submitting results."""
    try:
        response = input("\nDo you want to submit your results to https://localscore.ai? "
                        "The results will be public (y/n): ")
        return response.lower() in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        return False
