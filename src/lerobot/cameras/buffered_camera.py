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
Buffered camera wrapper for non-blocking frame access.

This wrapper maintains a frame buffer that is continuously updated by a background
thread, allowing the main control loop to always get the latest frame without blocking.

Key features:
- Non-blocking frame access (always returns immediately)
- Frame dropping when behind (keeps only latest frames)
- Timestamp tracking for frame freshness
- Statistics for monitoring camera performance
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.cameras.camera import Camera

logger = logging.getLogger(__name__)


@dataclass
class FrameStats:
    """Statistics for camera frame capture."""
    frames_captured: int = 0
    frames_dropped: int = 0
    last_capture_ms: float = 0.0
    avg_capture_ms: float = 0.0
    capture_times: deque = field(default_factory=lambda: deque(maxlen=30))

    def record_capture(self, duration_ms: float) -> None:
        self.frames_captured += 1
        self.last_capture_ms = duration_ms
        self.capture_times.append(duration_ms)
        if self.capture_times:
            self.avg_capture_ms = sum(self.capture_times) / len(self.capture_times)

    def record_drop(self) -> None:
        self.frames_dropped += 1

    @property
    def drop_rate(self) -> float:
        total = self.frames_captured + self.frames_dropped
        return self.frames_dropped / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_captured": self.frames_captured,
            "frames_dropped": self.frames_dropped,
            "drop_rate": round(self.drop_rate * 100, 1),
            "last_capture_ms": round(self.last_capture_ms, 1),
            "avg_capture_ms": round(self.avg_capture_ms, 1),
        }


@dataclass
class BufferedFrame:
    """A frame with metadata."""
    data: np.ndarray
    timestamp: float  # When the frame was captured
    frame_index: int  # Sequential frame counter


class BufferedCamera:
    """
    A wrapper around Camera that provides non-blocking frame access.

    Instead of blocking on camera reads, this wrapper maintains a buffer of the
    latest frame that is continuously updated by a background thread. The main
    thread can always get the latest frame without waiting.

    Usage:
        camera = OpenCVCamera(config)
        buffered = BufferedCamera(camera, buffer_size=2)
        buffered.start()

        # In control loop - never blocks!
        frame, timestamp = buffered.get_latest_frame()

        buffered.stop()
    """

    def __init__(
        self,
        camera: Camera,
        buffer_size: int = 2,
        drop_old_frames: bool = True,
    ):
        """
        Initialize the buffered camera.

        Args:
            camera: The underlying camera instance
            buffer_size: Number of frames to buffer (default 2 for double buffering)
            drop_old_frames: If True, drop old frames when buffer is full (default True)
        """
        self.camera = camera
        self.buffer_size = buffer_size
        self.drop_old_frames = drop_old_frames

        # Frame buffer
        self._buffer: deque[BufferedFrame] = deque(maxlen=buffer_size)
        self._latest_frame: BufferedFrame | None = None
        self._frame_lock = threading.Lock()
        self._frame_index = 0

        # Background thread
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

        # Statistics
        self.stats = FrameStats()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the background capture thread."""
        if self._running:
            return

        if not self.camera.is_connected:
            self.camera.connect()

        self._stop_event.clear()
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"BufferedCamera_{id(self)}",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info(f"BufferedCamera started for {self.camera}")

    def stop(self) -> None:
        """Stop the background capture thread."""
        if not self._running:
            return

        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self._running = False
        logger.info(f"BufferedCamera stopped for {self.camera}")

    def _capture_loop(self) -> None:
        """Background thread that continuously captures frames."""
        while not self._stop_event.is_set():
            try:
                start = time.perf_counter()

                # Read frame from camera
                frame_data = self.camera.read()

                duration_ms = (time.perf_counter() - start) * 1000
                self.stats.record_capture(duration_ms)

                # Create buffered frame
                frame = BufferedFrame(
                    data=frame_data,
                    timestamp=time.time(),
                    frame_index=self._frame_index,
                )
                self._frame_index += 1

                # Update buffer
                with self._frame_lock:
                    if self.drop_old_frames and len(self._buffer) >= self.buffer_size:
                        self.stats.record_drop()
                    self._buffer.append(frame)
                    self._latest_frame = frame

            except Exception as e:
                if not self._stop_event.is_set():
                    logger.warning(f"Error capturing frame: {e}")
                    time.sleep(0.01)  # Brief pause on error

    def get_latest_frame(self) -> tuple[np.ndarray | None, float]:
        """
        Get the latest available frame without blocking.

        Returns:
            Tuple of (frame_data, timestamp) or (None, 0.0) if no frame available
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.data.copy(), self._latest_frame.timestamp

    def get_latest_frame_if_new(self, last_index: int) -> tuple[np.ndarray | None, float, int]:
        """
        Get the latest frame only if it's newer than the given index.

        Args:
            last_index: The frame index of the last frame processed

        Returns:
            Tuple of (frame_data, timestamp, frame_index) or (None, 0.0, last_index) if no new frame
        """
        with self._frame_lock:
            if self._latest_frame is None or self._latest_frame.frame_index <= last_index:
                return None, 0.0, last_index
            return (
                self._latest_frame.data.copy(),
                self._latest_frame.timestamp,
                self._latest_frame.frame_index,
            )

    def get_frame_age_ms(self) -> float:
        """Get the age of the latest frame in milliseconds."""
        with self._frame_lock:
            if self._latest_frame is None:
                return float("inf")
            return (time.time() - self._latest_frame.timestamp) * 1000

    def get_stats(self) -> dict[str, Any]:
        """Get capture statistics."""
        stats = self.stats.to_dict()
        stats["frame_age_ms"] = round(self.get_frame_age_ms(), 1)
        stats["buffer_size"] = len(self._buffer)
        stats["is_running"] = self._running
        return stats


class BufferedCameraManager:
    """
    Manages multiple buffered cameras for a robot.

    Provides a convenient way to wrap all robot cameras with buffering
    and access them through a unified interface.
    """

    def __init__(self, cameras: dict[str, Camera], buffer_size: int = 2):
        """
        Initialize the manager with a dict of cameras.

        Args:
            cameras: Dictionary mapping camera names to Camera instances
            buffer_size: Buffer size for each camera
        """
        self.cameras = cameras
        self.buffered_cameras: dict[str, BufferedCamera] = {}

        for name, camera in cameras.items():
            self.buffered_cameras[name] = BufferedCamera(camera, buffer_size=buffer_size)

    def start_all(self) -> None:
        """Start all buffered cameras."""
        for buffered in self.buffered_cameras.values():
            buffered.start()

    def stop_all(self) -> None:
        """Stop all buffered cameras."""
        for buffered in self.buffered_cameras.values():
            buffered.stop()

    def get_all_frames(self) -> dict[str, np.ndarray | None]:
        """
        Get the latest frame from each camera.

        Returns:
            Dictionary mapping camera names to frame data (or None if no frame)
        """
        frames = {}
        for name, buffered in self.buffered_cameras.items():
            frame, _ = buffered.get_latest_frame()
            frames[name] = frame
        return frames

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all cameras."""
        return {name: buffered.get_stats() for name, buffered in self.buffered_cameras.items()}

    def __getitem__(self, name: str) -> BufferedCamera:
        return self.buffered_cameras[name]

    def __contains__(self, name: str) -> bool:
        return name in self.buffered_cameras

    def items(self):
        return self.buffered_cameras.items()

    def values(self):
        return self.buffered_cameras.values()

    def keys(self):
        return self.buffered_cameras.keys()
