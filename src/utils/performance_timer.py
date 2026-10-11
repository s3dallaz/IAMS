"""
Lightweight performance measurement utilities for Real-Time Attendance optimization.

Provides minimal-overhead timing instrumentation to identify actual bottlenecks.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    recent_samples: deque = field(default_factory=lambda: deque(maxlen=10))

    def add_sample(self, duration_ms: float) -> None:
        """Add a timing sample."""
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        self.recent_samples.append(duration_ms)

    @property
    def avg_ms(self) -> float:
        """Average duration in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def recent_avg_ms(self) -> float:
        """Average of recent samples."""
        return sum(self.recent_samples) / len(self.recent_samples) if self.recent_samples else 0.0


class PerformanceTimer:
    """
    Thread-safe performance measurement collector.

    Usage:
        timer = PerformanceTimer()

        # Manual timing
        with timer.time_context('operation_name'):
            do_work()

        # Or decorator style
        @timer.time_function('function_name')
        def my_function():
            pass

        # Get aggregated results
        stats = timer.get_stats()
        timer.log_summary()
    """

    def __init__(self):
        self._stats: Dict[str, TimingStats] = defaultdict(TimingStats)
        self._lock = threading.Lock()
        self._enabled = True

    def enable(self) -> None:
        """Enable timing collection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable timing collection."""
        self._enabled = False

    def time_context(self, operation_name: str):
        """Context manager for timing operations."""
        return _TimingContext(self, operation_name)

    def time_function(self, operation_name: str):
        """Decorator for timing functions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self._enabled:
                    return func(*args, **kwargs)

                with self.time_context(operation_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def record_time(self, operation_name: str, duration_ms: float) -> None:
        """Record a timing measurement."""
        if not self._enabled:
            return

        with self._lock:
            self._stats[operation_name].add_sample(duration_ms)

    def get_stats(self) -> Dict[str, TimingStats]:
        """Get current timing statistics."""
        with self._lock:
            return dict(self._stats)

    def reset(self) -> None:
        """Clear all timing statistics."""
        with self._lock:
            self._stats.clear()

    def log_summary(self, min_count: int = 5) -> None:
        """Log performance summary for operations with sufficient samples."""
        stats = self.get_stats()
        if not stats:
            return

        logger.info("=== PERFORMANCE SUMMARY ===")
        for name, timing in sorted(stats.items()):
            if timing.count >= min_count:
                logger.info(
                    f"{name}: count={timing.count}, "
                    f"avg={timing.avg_ms:.2f}ms, "
                    f"recent_avg={timing.recent_avg_ms:.2f}ms, "
                    f"min={timing.min_ms:.2f}ms, "
                    f"max={timing.max_ms:.2f}ms"
                )
        logger.info("=== END PERFORMANCE SUMMARY ===")

    def get_fps_estimate(self, operation_name: str) -> float:
        """Get estimated FPS based on recent average timing."""
        stats = self.get_stats().get(operation_name)
        if not stats or not stats.recent_samples:
            return 0.0

        avg_ms = stats.recent_avg_ms
        if avg_ms <= 0:
            return 0.0

        return 1000.0 / avg_ms


class _TimingContext:
    """Context manager for timing operations."""

    def __init__(self, timer: PerformanceTimer, operation_name: str):
        self.timer = timer
        self.operation_name = operation_name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        self.timer.record_time(self.operation_name, duration_ms)


# Global timer instance for convenient use
_global_timer = PerformanceTimer()


def get_timer() -> PerformanceTimer:
    """Get the global performance timer instance."""
    return _global_timer


def time_operation(operation_name: str):
    """Context manager shortcut using global timer."""
    return _global_timer.time_context(operation_name)


def log_performance_summary():
    """Log performance summary using global timer."""
    _global_timer.log_summary()


def reset_performance_stats():
    """Reset global timer statistics."""
    _global_timer.reset()