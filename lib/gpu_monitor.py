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
    gpu_index: int = 0              # nvidia-smi GPU index (0 = first NVIDIA GPU)
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

        # Markers go to a sidecar file so they don't conflict with nvidia-smi stdout
        self._marker_path = self.output_path + ".markers"
        self._marker_file = open(self._marker_path, "w")
        self._marker_file.write("# Benchmark phase markers\n")
        self._marker_file.flush()

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

        # Close marker file
        if hasattr(self, "_marker_file") and self._marker_file:
            self._marker_file.close()

        # Merge markers into the main CSV as comments
        if hasattr(self, "_marker_path") and Path(self._marker_path).exists():
            markers = Path(self._marker_path).read_text()
            with open(self.output_path, "a") as f:
                f.write("\n")
                f.write(markers)
            Path(self._marker_path).unlink()

        return self.output_path

    def add_marker(self, label: str) -> None:
        """Write a marker with timestamp to the sidecar marker file."""
        try:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y/%m/%d %H:%M:%S.%f")
            if hasattr(self, "_marker_file") and self._marker_file:
                self._marker_file.write(f"# MARKER: {label} @ {ts}\n")
                self._marker_file.flush()
        except Exception:
            pass  # best effort
