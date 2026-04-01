"""Shared runtime and peak-RSS tracking utilities."""

from __future__ import annotations

import resource
import time


def _normalize_ru_maxrss(raw_value: int) -> int:
    """Convert ``ru_maxrss`` to bytes across platforms."""
    if raw_value <= 0:
        return 0
    # macOS reports bytes; Linux typically reports KiB.
    if raw_value < 1024 * 1024:
        return int(raw_value * 1024)
    return int(raw_value)


def peak_rss_bytes() -> int:
    """Return current process peak RSS in bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return _normalize_ru_maxrss(int(usage.ru_maxrss))


class PerformanceTracker:
    """Track wall-clock runtime and peak process RSS for one run."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._peak_rss_bytes = peak_rss_bytes()

    def sample(self) -> None:
        """Refresh the tracked peak RSS."""
        self._peak_rss_bytes = max(self._peak_rss_bytes, peak_rss_bytes())

    def metrics(self) -> dict[str, float | int]:
        """Return runtime and peak RSS metrics for the tracked run."""
        self.sample()
        return {
            "runtime_sec": float(time.perf_counter() - self._t0),
            "peak_rss_bytes": int(self._peak_rss_bytes),
        }
