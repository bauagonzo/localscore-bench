"""GPU monitoring via nvidia-smi during benchmark runs."""

import csv
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GpuMonitor:
    """Runs nvidia-smi in the background, logging GPU stats to CSV.

    Usage::

        mon = GpuMonitor(output_path="gpu_log.csv", interval_ms=100)
        mon.start()
        # ... run workload ...
        mon.stop()
    """

    output_path: str
    interval_ms: int = 100          # sampling interval (default 100 ms)
    gpu_index: int = 0              # which GPU to monitor
    _proc: Optional[subprocess.Popen] = None

    # nvidia-smi query fields
    QUERY_FIELDS = (
        "timestamp,"
        "gpu_bus_id,"
        "utilization.gpu,"        # GPU core utilization %
        "utilization.memory,"     # Memory controller utilization %
        "memory.used,"            # MiB
        "memory.total,"           # MiB
        "temperature.gpu,"        # °C
        "power.draw,"             # W
        "clocks.current.sm,"      # MHz
        "clocks.current.memory,"  # MHz
        "pcie.link.gen.current,"  # PCIe gen
        "pcie.link.width.current" # PCIe width
    )

    def start(self) -> None:
        """Start nvidia-smi background logging."""
        if self._proc is not None:
            return  # already running

        interval_s = self.interval_ms / 1000.0
        # nvidia-smi uses ms for -l but also supports fractional seconds via lms
        cmd = [
            "nvidia-smi",
            f"--query-gpu={self.QUERY_FIELDS}",
            "--format=csv",
            f"--loop-ms={self.interval_ms}",
            f"-i={self.gpu_index}",
        ]

        outfile = open(self.output_path, "w")
        self._proc = subprocess.Popen(
            cmd,
            stdout=outfile,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,  # own process group for clean kill
        )
        self._outfile = outfile

    def stop(self) -> Optional[str]:
        """Stop nvidia-smi logging.  Returns the output file path."""
        if self._proc is None:
            return None

        # Send SIGTERM to the process group
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

        self._proc.wait(timeout=5)
        self._outfile.close()
        self._proc = None

        return self.output_path

    def add_marker(self, label: str) -> None:
        """Write a comment marker into the CSV (for correlating with test phases).

        This appends directly to the file so it interleaves with nvidia-smi output.
        """
        try:
            with open(self.output_path, "a") as f:
                f.write(f"# MARKER: {label} @ {time.strftime('%Y/%m/%d %H:%M:%S.%f' if hasattr(time, 'strftime') else '%Y/%m/%d %H:%M:%S')}\n")
        except Exception:
            pass  # best effort
