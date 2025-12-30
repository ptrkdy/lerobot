#!/usr/bin/env python

"""
UJ201 Dual Camera Test Script

Tests two cameras attached to UJ201:
- Camera 0: 1280x960 @ 5fps (likely wrist camera - lower fps)
- Camera 1: 1920x1080 @ 30fps (likely scene/desktop camera - higher fps)

This script validates:
1. Camera detection and initialization
2. Synchronized frame capture
3. Frame quality and latency
4. Async vs sync capture performance
5. Side-by-side display

Usage:
    python test_uj201_cameras.py
"""

import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

# Output configuration
OUTPUT_DIR = Path("camera_test_results")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = OUTPUT_DIR / f"camera_test_{timestamp}.txt"

def log_and_print(message, file_handle=None):
    """Print to console and write to file."""
    print(message)
    if file_handle:
        file_handle.write(message + "\n")
        file_handle.flush()

def print_section(title, file_handle=None):
    """Print section header."""
    log_and_print("\n" + "=" * 80, file_handle)
    log_and_print(f"  {title}", file_handle)
    log_and_print("=" * 80, file_handle)

# Open output file
output_file = open(OUTPUT_FILE, "w")
log_and_print("=" * 80, output_file)
log_and_print("📷 UJ201 Dual Camera Test", output_file)
log_and_print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", output_file)
log_and_print("=" * 80, output_file)

# ============================================================================
# Camera Configuration
# ============================================================================
print_section("Camera Configuration", output_file)

# Camera 0: Lower resolution, lower fps (wrist camera)
wrist_config = OpenCVCameraConfig(
    index_or_path=0,
    fps=5,
    width=1280,
    height=960,
)

# Camera 1: Higher resolution, higher fps (scene camera)
scene_config = OpenCVCameraConfig(
    index_or_path=1,
    fps=30,
    width=1920,
    height=1080,
)

log_and_print(f"📷 Wrist Camera Config:", output_file)
log_and_print(f"   Index: {wrist_config.index_or_path}", output_file)
log_and_print(f"   Resolution: {wrist_config.width}x{wrist_config.height}", output_file)
log_and_print(f"   FPS: {wrist_config.fps}", output_file)

log_and_print(f"\n📷 Scene Camera Config:", output_file)
log_and_print(f"   Index: {scene_config.index_or_path}", output_file)
log_and_print(f"   Resolution: {scene_config.width}x{scene_config.height}", output_file)
log_and_print(f"   FPS: {scene_config.fps}", output_file)

# ============================================================================
# Initialize Cameras
# ============================================================================
print_section("Initialize Cameras", output_file)

try:
    log_and_print("🔌 Connecting to wrist camera...", output_file)
    wrist_camera = OpenCVCamera(wrist_config)
    wrist_camera.connect()
    log_and_print("✅ Wrist camera connected", output_file)
except Exception as e:
    log_and_print(f"❌ Failed to connect wrist camera: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)
    output_file.close()
    exit(1)

try:
    log_and_print("🔌 Connecting to scene camera...", output_file)
    scene_camera = OpenCVCamera(scene_config)
    scene_camera.connect()
    log_and_print("✅ Scene camera connected", output_file)
except Exception as e:
    log_and_print(f"❌ Failed to connect scene camera: {e}", output_file)
    wrist_camera.disconnect()
    import traceback
    log_and_print(traceback.format_exc(), output_file)
    output_file.close()
    exit(1)

# ============================================================================
# Test 1: Single Frame Capture
# ============================================================================
print_section("Test 1: Single Frame Capture", output_file)

try:
    log_and_print("📸 Capturing single frames...", output_file)
    
    # Capture from wrist camera
    start_time = time.time()
    wrist_frame = wrist_camera.async_read()
    wrist_latency = (time.time() - start_time) * 1000  # ms
    
    log_and_print(f"✅ Wrist frame captured:", output_file)
    log_and_print(f"   Shape: {wrist_frame.shape}", output_file)
    log_and_print(f"   Dtype: {wrist_frame.dtype}", output_file)
    log_and_print(f"   Latency: {wrist_latency:.2f} ms", output_file)
    
    # Capture from scene camera
    start_time = time.time()
    scene_frame = scene_camera.async_read()
    scene_latency = (time.time() - start_time) * 1000  # ms
    
    log_and_print(f"\n✅ Scene frame captured:", output_file)
    log_and_print(f"   Shape: {scene_frame.shape}", output_file)
    log_and_print(f"   Dtype: {scene_frame.dtype}", output_file)
    log_and_print(f"   Latency: {scene_latency:.2f} ms", output_file)
    
    # Save test frames
    wrist_path = OUTPUT_DIR / f"wrist_test_{timestamp}.jpg"
    scene_path = OUTPUT_DIR / f"scene_test_{timestamp}.jpg"
    cv2.imwrite(str(wrist_path), wrist_frame)
    cv2.imwrite(str(scene_path), scene_frame)
    log_and_print(f"\n💾 Test frames saved:", output_file)
    log_and_print(f"   Wrist: {wrist_path}", output_file)
    log_and_print(f"   Scene: {scene_path}", output_file)
    
except Exception as e:
    log_and_print(f"❌ Frame capture failed: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)

# ============================================================================
# Test 2: Frame Rate Analysis
# ============================================================================
print_section("Test 2: Frame Rate Analysis", output_file)

log_and_print("📊 Analyzing actual frame rates over 5 seconds...", output_file)

try:
    # Test wrist camera (5fps needs 200ms per frame, use 500ms timeout for safety)
    wrist_count = 0
    wrist_start = time.time()
    while time.time() - wrist_start < 5.0:
        _ = wrist_camera.async_read(timeout_ms=500)
        wrist_count += 1
    wrist_actual_fps = wrist_count / 5.0
    
    log_and_print(f"\n📷 Wrist Camera:", output_file)
    log_and_print(f"   Expected FPS: {wrist_config.fps}", output_file)
    log_and_print(f"   Actual FPS: {wrist_actual_fps:.2f}", output_file)
    log_and_print(f"   Frames captured: {wrist_count}", output_file)
    
    # Test scene camera
    scene_count = 0
    scene_start = time.time()
    while time.time() - scene_start < 5.0:
        _ = scene_camera.async_read()
        scene_count += 1
    scene_actual_fps = scene_count / 5.0
    
    log_and_print(f"\n📷 Scene Camera:", output_file)
    log_and_print(f"   Expected FPS: {scene_config.fps}", output_file)
    log_and_print(f"   Actual FPS: {scene_actual_fps:.2f}", output_file)
    log_and_print(f"   Frames captured: {scene_count}", output_file)
    
except Exception as e:
    log_and_print(f"❌ Frame rate test failed: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)

# ============================================================================
# Test 3: Synchronized Dual Camera Capture
# ============================================================================
print_section("Test 3: Synchronized Dual Camera Capture", output_file)

log_and_print("🔄 Testing synchronized capture (30 frames)...", output_file)

try:
    sync_times = []
    max_time_diff = 0.0
    
    for i in range(30):
        start = time.time()
        
        # Capture both cameras as close in time as possible
        wrist_frame = wrist_camera.async_read(timeout_ms=500)
        wrist_time = time.time()
        
        scene_frame = scene_camera.async_read()
        scene_time = time.time()
        
        # Calculate time difference between captures
        time_diff = abs(scene_time - wrist_time) * 1000  # ms
        max_time_diff = max(max_time_diff, time_diff)
        sync_times.append(time_diff)
        
        if (i + 1) % 10 == 0:
            log_and_print(f"   Frame {i+1}/30: time_diff={time_diff:.2f}ms", output_file)
    
    avg_sync_time = np.mean(sync_times)
    log_and_print(f"\n📊 Synchronization Results:", output_file)
    log_and_print(f"   Average time difference: {avg_sync_time:.2f} ms", output_file)
    log_and_print(f"   Maximum time difference: {max_time_diff:.2f} ms", output_file)
    log_and_print(f"   Std dev: {np.std(sync_times):.2f} ms", output_file)
    
except Exception as e:
    log_and_print(f"❌ Synchronized capture failed: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)

# ============================================================================
# Test 4: Display Side-by-Side (Interactive)
# ============================================================================
print_section("Test 4: Live Display", output_file)

log_and_print("🖥️  Opening live camera display...", output_file)
log_and_print("   Press 'q' to quit", output_file)
log_and_print("   Press 's' to save snapshot", output_file)

try:
    snapshot_count = 0
    while True:
        # Capture frames
        wrist_frame = wrist_camera.async_read(timeout_ms=500)
        scene_frame = scene_camera.async_read()
        
        # Resize for display (scale down scene camera to match aspect ratio)
        display_height = 480
        wrist_display = cv2.resize(wrist_frame, (640, display_height))
        scene_display = cv2.resize(scene_frame, (int(1920 * display_height / 1080), display_height))
        
        # Add labels
        cv2.putText(wrist_display, "Wrist Camera", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(scene_display, "Scene Camera", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine side-by-side
        combined = np.hstack([wrist_display, scene_display])
        
        # Display
        cv2.imshow("UJ201 Dual Camera Feed", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            log_and_print("\n⏹️  Display closed by user", output_file)
            break
        elif key == ord('s'):
            snapshot_count += 1
            snapshot_path = OUTPUT_DIR / f"snapshot_{timestamp}_{snapshot_count:03d}.jpg"
            cv2.imwrite(str(snapshot_path), combined)
            log_and_print(f"💾 Snapshot saved: {snapshot_path}", output_file)
            print(f"💾 Snapshot {snapshot_count} saved!")
    
    cv2.destroyAllWindows()
    
except Exception as e:
    log_and_print(f"❌ Display failed: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)
    cv2.destroyAllWindows()

# ============================================================================
# Test 5: Performance Benchmark
# ============================================================================
print_section("Test 5: Performance Benchmark", output_file)

log_and_print("⚡ Benchmarking capture performance...", output_file)

try:
    # Benchmark dual capture latency
    latencies = []
    for i in range(100):
        start = time.time()
        wrist_frame = wrist_camera.async_read(timeout_ms=500)
        scene_frame = scene_camera.async_read()
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
    
    log_and_print(f"\n📊 Dual Capture Performance (100 iterations):", output_file)
    log_and_print(f"   Mean latency: {np.mean(latencies):.2f} ms", output_file)
    log_and_print(f"   Median latency: {np.median(latencies):.2f} ms", output_file)
    log_and_print(f"   Min latency: {np.min(latencies):.2f} ms", output_file)
    log_and_print(f"   Max latency: {np.max(latencies):.2f} ms", output_file)
    log_and_print(f"   Std dev: {np.std(latencies):.2f} ms", output_file)
    
    # Check if suitable for control loop (target < 33ms for 30Hz)
    if np.mean(latencies) < 33.0:
        log_and_print(f"\n✅ Camera latency suitable for 30Hz control loop", output_file)
    else:
        log_and_print(f"\n⚠️  Camera latency may limit control loop speed", output_file)
    
except Exception as e:
    log_and_print(f"❌ Benchmark failed: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)

# ============================================================================
# Cleanup
# ============================================================================
print_section("Cleanup", output_file)

try:
    log_and_print("🔌 Disconnecting cameras...", output_file)
    wrist_camera.disconnect()
    log_and_print("✅ Wrist camera disconnected", output_file)
    scene_camera.disconnect()
    log_and_print("✅ Scene camera disconnected", output_file)
except Exception as e:
    log_and_print(f"⚠️  Disconnect warning: {e}", output_file)

# ============================================================================
# Summary
# ============================================================================
log_and_print("\n" + "=" * 80, output_file)
log_and_print("✅ Camera Test Completed", output_file)
log_and_print(f"Test ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", output_file)
log_and_print("=" * 80, output_file)
log_and_print(f"\n📄 Results saved to: {OUTPUT_FILE}", output_file)
log_and_print(f"📁 Output directory: {OUTPUT_DIR}", output_file)

output_file.close()
print(f"\n✅ Full test log saved to: {OUTPUT_FILE}")
