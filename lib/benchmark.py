"""Wrapper for llama-bench binary."""

import json
import subprocess
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

from .config import TestConfig


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    n_prompt: int
    n_gen: int
    avg_ns: int
    avg_ts: float
    stddev_ns: int = 0
    stddev_ts: float = 0.0
    samples_ns: List[int] = field(default_factory=list)
    samples_ts: List[float] = field(default_factory=list)

    # System info (populated from first result)
    cpu_info: str = ""
    gpu_info: str = ""
    backends: str = ""
    model_type: str = ""
    model_filename: str = ""
    model_size: int = 0
    model_n_params: int = 0
    build_commit: str = ""
    build_number: int = 0


@dataclass
class TestResult:
    """Combined result for a localscore-style test (pp + tg)."""
    config: TestConfig
    prompt_tps: float
    gen_tps: float
    ttft_ms: float
    pp_avg_ns: int
    tg_avg_ns: int
    repetitions: int

    # System info
    cpu_info: str = ""
    gpu_info: str = ""
    gpu_index: int = 0
    model_type: str = ""
    model_filename: str = ""
    model_size: int = 0
    model_n_params: int = 0

    # Sample data
    pp_samples_ns: List[int] = field(default_factory=list)
    tg_samples_ns: List[int] = field(default_factory=list)


class LlamaBench:
    """Wrapper for the llama-bench binary."""

    def __init__(self, llama_bench_path: str, model_path: str,
                 n_gpu_layers: int = 99, device: Optional[str] = None,
                 gpu_index: int = 0, threads: Optional[int] = None,
                 verbose: bool = False):
        self.llama_bench_path = Path(llama_bench_path).resolve()
        self.model_path = Path(model_path).resolve()
        self.n_gpu_layers = n_gpu_layers
        self.device = device
        self.gpu_index = gpu_index
        self.threads = threads
        self.verbose = verbose

        if not self.llama_bench_path.exists():
            raise FileNotFoundError(f"llama-bench not found: {self.llama_bench_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    def _build_command(self, n_prompt: int, n_gen: int, repetitions: int) -> List[str]:
        """Build the llama-bench command."""
        cmd = [
            str(self.llama_bench_path),
            "-m", str(self.model_path),
            "-p", str(n_prompt),
            "-n", str(n_gen),
            "-r", str(repetitions),
            "-o", "json",
            "-ngl", str(self.n_gpu_layers),
        ]

        if self.device:
            cmd.extend(["-dev", self.device])

        if self.threads:
            cmd.extend(["-t", str(self.threads)])

        return cmd

    def _parse_json_output(self, output: str) -> List[BenchmarkResult]:
        """Parse JSON output from llama-bench."""
        # Find the JSON array in output (skip any log messages)
        json_start = output.find('[')
        if json_start == -1:
            raise ValueError("No JSON output found from llama-bench")

        json_str = output[json_start:]
        data = json.loads(json_str)

        results = []
        for item in data:
            result = BenchmarkResult(
                n_prompt=item.get("n_prompt", 0),
                n_gen=item.get("n_gen", 0),
                avg_ns=item.get("avg_ns", 0),
                avg_ts=item.get("avg_ts", 0.0),
                stddev_ns=item.get("stddev_ns", 0),
                stddev_ts=item.get("stddev_ts", 0.0),
                samples_ns=item.get("samples_ns", []),
                samples_ts=item.get("samples_ts", []),
                cpu_info=item.get("cpu_info", ""),
                gpu_info=item.get("gpu_info", ""),
                backends=item.get("backends", ""),
                model_type=item.get("model_type", ""),
                model_filename=item.get("model_filename", ""),
                model_size=item.get("model_size", 0),
                model_n_params=item.get("model_n_params", 0),
                build_commit=item.get("build_commit", ""),
                build_number=item.get("build_number", 0),
            )
            results.append(result)

        return results

    def run_single(self, n_prompt: int, n_gen: int, repetitions: int = 1) -> BenchmarkResult:
        """Run a single benchmark test."""
        cmd = self._build_command(n_prompt, n_gen, repetitions)

        if self.verbose:
            print(f"  Running: {' '.join(cmd)}")

        # Set library path to find shared libraries
        env = os.environ.copy()
        lib_dir = str(self.llama_bench_path.parent)
        if sys.platform == "win32":
            env["PATH"] = lib_dir + ";" + env.get("PATH", "")
        elif sys.platform == "darwin":
            env["DYLD_LIBRARY_PATH"] = lib_dir + ":" + env.get("DYLD_LIBRARY_PATH", "")
        else:
            env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=lib_dir,
        )

        if result.returncode != 0:
            raise RuntimeError(f"llama-bench failed: {result.stderr}")

        # Combine stdout and stderr for parsing (logs go to stderr)
        output = result.stdout
        results = self._parse_json_output(output)

        if not results:
            raise RuntimeError("No results from llama-bench")

        # Return the result matching our request
        for r in results:
            if r.n_prompt == n_prompt and r.n_gen == n_gen:
                return r

        # Fallback to first result
        return results[0]

    def run_test(self, config: TestConfig, repetitions: int = 1) -> TestResult:
        """Run a complete localscore-style test (separate pp and tg runs)."""
        # Run prompt processing test
        pp_result = self.run_single(config.n_prompt, 0, repetitions)

        # Run token generation test
        tg_result = self.run_single(0, config.n_gen, repetitions)

        # Calculate metrics
        prompt_tps = pp_result.avg_ts
        gen_tps = tg_result.avg_ts

        # TTFT = prompt processing time + time for first generated token
        pp_time_ms = pp_result.avg_ns / 1e6
        tg_time_per_token_ms = (tg_result.avg_ns / config.n_gen) / 1e6
        ttft_ms = pp_time_ms + tg_time_per_token_ms

        return TestResult(
            config=config,
            prompt_tps=prompt_tps,
            gen_tps=gen_tps,
            ttft_ms=ttft_ms,
            pp_avg_ns=pp_result.avg_ns,
            tg_avg_ns=tg_result.avg_ns,
            repetitions=repetitions,
            cpu_info=pp_result.cpu_info,
            gpu_info=pp_result.gpu_info,
            gpu_index=self.gpu_index,
            model_type=pp_result.model_type,
            model_filename=pp_result.model_filename,
            model_size=pp_result.model_size,
            model_n_params=pp_result.model_n_params,
            pp_samples_ns=pp_result.samples_ns,
            tg_samples_ns=tg_result.samples_ns,
        )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system info by running a minimal benchmark."""
        result = self.run_single(16, 0, 1)
        return {
            "cpu_info": result.cpu_info,
            "gpu_info": result.gpu_info,
            "backends": result.backends,
            "model_type": result.model_type,
            "model_filename": result.model_filename,
            "model_size": result.model_size,
            "model_n_params": result.model_n_params,
            "build_commit": result.build_commit,
            "build_number": result.build_number,
        }

    def get_build_info(self) -> Dict[str, Any]:
        """Get build info by running a minimal benchmark."""
        result = self.run_single(16, 0, 1)
        return {
            "build_commit": result.build_commit,
            "build_number": result.build_number,
        }
