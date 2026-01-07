#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Timing instrumentation utilities for profiling control loops and identifying bottlenecks.

Usage:
    from lerobot.utils.timing_utils import LoopTimer

    timer = LoopTimer(target_fps=30)

    while running:
        timer.start_loop()

        with timer.measure("robot_read"):
            obs = robot.get_observation()

        with timer.measure("teleop_read"):
            action = teleop.get_action()

        with timer.measure("robot_write"):
            robot.send_action(action)

        with timer.measure("dataset_write"):
            dataset.add_frame(frame)

        timer.end_loop()
        stats = timer.get_stats()

        # stats contains timing breakdowns and jitter analysis
"""

import logging
import statistics
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for a single timing measurement."""
    name: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    last_ms: float = 0.0
    samples: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def std_ms(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(self.samples)

    @property
    def p95_ms(self) -> float:
        if len(self.samples) < 5:
            return self.max_ms
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[idx]

    def record(self, duration_ms: float) -> None:
        self.count += 1
        self.total_ms += duration_ms
        self.last_ms = duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        self.samples.append(duration_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "last_ms": round(self.last_ms, 2),
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.min_ms != float("inf") else 0.0,
            "max_ms": round(self.max_ms, 2),
            "std_ms": round(self.std_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
        }


class LoopTimer:
    """
    Timing instrumentation for control loops.

    Tracks individual operation timings and overall loop performance,
    providing insights into jitter and bottlenecks.
    """

    def __init__(self, target_fps: int = 30, history_size: int = 100):
        self.target_fps = target_fps
        self.target_loop_ms = 1000.0 / target_fps
        self.history_size = history_size

        # Per-operation timing
        self.timings: dict[str, TimingStats] = {}

        # Loop-level timing
        self.loop_stats = TimingStats(name="loop_total")
        self.loop_start_time: float | None = None
        self._current_measure_start: float | None = None

        # Jitter tracking
        self.loop_intervals: deque = deque(maxlen=history_size)
        self.last_loop_end: float | None = None

        # Deadline tracking
        self.deadline_misses = 0
        self.total_loops = 0

        # Enabled flag for zero-overhead when disabled
        self.enabled = True

    def start_loop(self) -> None:
        """Mark the start of a control loop iteration."""
        if not self.enabled:
            return

        now = time.perf_counter()

        # Track interval between loops (for jitter analysis)
        if self.last_loop_end is not None:
            interval_ms = (now - self.last_loop_end) * 1000
            self.loop_intervals.append(interval_ms)

        self.loop_start_time = now

    def end_loop(self) -> None:
        """Mark the end of a control loop iteration."""
        if not self.enabled or self.loop_start_time is None:
            return

        now = time.perf_counter()
        loop_duration_ms = (now - self.loop_start_time) * 1000

        self.loop_stats.record(loop_duration_ms)
        self.total_loops += 1

        if loop_duration_ms > self.target_loop_ms:
            self.deadline_misses += 1

        self.last_loop_end = now
        self.loop_start_time = None

    @contextmanager
    def measure(self, name: str):
        """Context manager to measure the duration of an operation."""
        if not self.enabled:
            yield
            return

        if name not in self.timings:
            self.timings[name] = TimingStats(name=name)

        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.timings[name].record(duration_ms)

    def record(self, name: str, duration_ms: float) -> None:
        """Manually record a timing measurement."""
        if not self.enabled:
            return

        if name not in self.timings:
            self.timings[name] = TimingStats(name=name)

        self.timings[name].record(duration_ms)

    def get_stats(self) -> dict[str, Any]:
        """
        Get comprehensive timing statistics.

        Returns a dictionary with:
        - target_fps: Target frames per second
        - actual_fps: Achieved FPS based on loop timing
        - loop: Loop timing statistics
        - operations: Per-operation timing breakdowns
        - jitter: Timing variability metrics
        - deadline: Deadline miss statistics
        """
        if not self.enabled:
            return {"enabled": False}

        # Calculate actual FPS
        if self.loop_stats.avg_ms > 0:
            actual_fps = 1000.0 / self.loop_stats.avg_ms
        else:
            actual_fps = 0.0

        # Calculate jitter (variation in loop intervals)
        if len(self.loop_intervals) >= 2:
            jitter_std_ms = statistics.stdev(self.loop_intervals)
            jitter_max_ms = max(self.loop_intervals) - min(self.loop_intervals)
        else:
            jitter_std_ms = 0.0
            jitter_max_ms = 0.0

        # Calculate time budget breakdown
        operations = {}
        total_op_time = 0.0
        for name, stats in self.timings.items():
            operations[name] = stats.to_dict()
            total_op_time += stats.avg_ms

        # Overhead is loop time minus measured operations
        overhead_ms = max(0, self.loop_stats.avg_ms - total_op_time)

        return {
            "enabled": True,
            "target_fps": self.target_fps,
            "target_loop_ms": round(self.target_loop_ms, 2),
            "actual_fps": round(actual_fps, 1),
            "loop": self.loop_stats.to_dict(),
            "operations": operations,
            "overhead_ms": round(overhead_ms, 2),
            "jitter": {
                "std_ms": round(jitter_std_ms, 2),
                "max_deviation_ms": round(jitter_max_ms, 2),
            },
            "deadline": {
                "total_loops": self.total_loops,
                "misses": self.deadline_misses,
                "miss_rate": round(self.deadline_misses / max(1, self.total_loops) * 100, 1),
            },
        }

    def get_compact_stats(self) -> dict[str, Any]:
        """Get a compact version of stats for real-time display."""
        if not self.enabled:
            return {"enabled": False}

        actual_fps = 1000.0 / self.loop_stats.avg_ms if self.loop_stats.avg_ms > 0 else 0.0

        # Get the slowest operations
        sorted_ops = sorted(
            self.timings.items(),
            key=lambda x: x[1].avg_ms,
            reverse=True
        )

        slowest = [
            {"name": name, "avg_ms": round(stats.avg_ms, 1), "last_ms": round(stats.last_ms, 1)}
            for name, stats in sorted_ops[:5]
        ]

        return {
            "fps": round(actual_fps, 1),
            "loop_ms": round(self.loop_stats.last_ms, 1),
            "target_ms": round(self.target_loop_ms, 1),
            "deadline_miss_rate": round(self.deadline_misses / max(1, self.total_loops) * 100, 1),
            "slowest_operations": slowest,
        }

    def log_summary(self) -> None:
        """Log a summary of timing statistics."""
        stats = self.get_stats()
        if not stats.get("enabled"):
            return

        logger.info(f"=== Loop Timing Summary ===")
        logger.info(f"Target: {stats['target_fps']} FPS ({stats['target_loop_ms']:.1f}ms)")
        logger.info(f"Actual: {stats['actual_fps']:.1f} FPS ({stats['loop']['avg_ms']:.1f}ms avg)")
        logger.info(f"Deadline misses: {stats['deadline']['misses']}/{stats['deadline']['total_loops']} ({stats['deadline']['miss_rate']:.1f}%)")
        logger.info(f"Jitter: {stats['jitter']['std_ms']:.1f}ms std, {stats['jitter']['max_deviation_ms']:.1f}ms max deviation")

        logger.info("Operation breakdown:")
        for name, op_stats in sorted(stats["operations"].items(), key=lambda x: -x[1]["avg_ms"]):
            pct = (op_stats["avg_ms"] / stats["target_loop_ms"]) * 100
            logger.info(f"  {name}: {op_stats['avg_ms']:.1f}ms avg ({pct:.0f}% of budget)")

        logger.info(f"Overhead: {stats['overhead_ms']:.1f}ms")

    def reset(self) -> None:
        """Reset all timing statistics."""
        self.timings.clear()
        self.loop_stats = TimingStats(name="loop_total")
        self.loop_intervals.clear()
        self.last_loop_end = None
        self.deadline_misses = 0
        self.total_loops = 0


# Global timer instance for easy access
_global_timer: LoopTimer | None = None


def get_global_timer() -> LoopTimer:
    """Get or create the global loop timer."""
    global _global_timer
    if _global_timer is None:
        _global_timer = LoopTimer()
    return _global_timer


def set_global_timer(timer: LoopTimer) -> None:
    """Set the global loop timer."""
    global _global_timer
    _global_timer = timer
