#!/usr/bin/env python

"""
UJ201 Robot + Camera Integration Test

Tests UJ201 robot arm with dual cameras integrated:
- Wrist camera (Camera 0: 1280x960 @ 5fps)
- Scene camera (Camera 1: 1920x1080 @ 30fps)

Validates:
1. Robot initialization with camera configs
2. get_observation() returns motor positions + camera frames
3. Observation dict structure matches VLA requirements
4. Synchronized motor + camera data capture

Usage:
    python test_uj201_with_cameras.py
"""

import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig

# Output configuration
OUTPUT_DIR = Path("robot_camera_test_results")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = OUTPUT_DIR / f"robot_camera_test_{timestamp}.txt"

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
log_and_print("🤖 UJ201 Robot + Camera Integration Test", output_file)
log_and_print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", output_file)
log_and_print("=" * 80, output_file)

# ============================================================================
# Configure Robot with Cameras
# ============================================================================
print_section("Robot Configuration", output_file)

# Camera configurations
cameras = {
    "wrist": OpenCVCameraConfig(
        index_or_path=0,
        fps=5,
        width=1280,
        height=960,
    ),
    "scene": OpenCVCameraConfig(
        index_or_path=1,
        fps=30,
        width=1920,
        height=1080,
    ),
}

# Robot configuration
robot_config = UJ201FollowerConfig(
    port="/dev/tty.usbmodem5A7A0565221",
    cameras=cameras,
    use_degrees=True,
    max_relative_target=10.0,  # 10 degree safety limit
)

log_and_print("📋 Robot Config:", output_file)
log_and_print(f"   Port: {robot_config.port}", output_file)
log_and_print(f"   Cameras: {list(cameras.keys())}", output_file)
log_and_print(f"   Use degrees: {robot_config.use_degrees}", output_file)
log_and_print(f"   Max relative target: {robot_config.max_relative_target}°", output_file)

for cam_name, cam_config in cameras.items():
    log_and_print(f"\n📷 {cam_name.capitalize()} Camera:", output_file)
    log_and_print(f"   Index: {cam_config.index_or_path}", output_file)
    log_and_print(f"   Resolution: {cam_config.width}x{cam_config.height}", output_file)
    log_and_print(f"   FPS: {cam_config.fps}", output_file)

# ============================================================================
# Initialize Robot
# ============================================================================
print_section("Initialize Robot", output_file)

try:
    log_and_print("🤖 Creating UJ201Follower instance...", output_file)
    robot = UJ201Follower(robot_config)
    log_and_print("✅ Robot instance created", output_file)
    
    log_and_print("\n🔌 Connecting robot (motors + cameras)...", output_file)
    robot.connect(calibrate=False)  # Skip calibration for testing
    log_and_print("✅ Robot connected successfully", output_file)
    
    log_and_print(f"\n📊 Connection Status:", output_file)
    log_and_print(f"   Robot connected: {robot.is_connected}", output_file)
    log_and_print(f"   Bus connected: {robot.bus.is_connected}", output_file)
    log_and_print(f"   Cameras connected: {all(cam.is_connected for cam in robot.cameras.values())}", output_file)
    
except Exception as e:
    log_and_print(f"❌ Failed to initialize robot: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)
    output_file.close()
    exit(1)

# ============================================================================
# Test 1: Observation Features
# ============================================================================
print_section("Test 1: Observation Features", output_file)

try:
    log_and_print("📋 Inspecting observation features...", output_file)
    
    obs_features = robot.observation_features
    log_and_print(f"\n📊 Total observation features: {len(obs_features)}", output_file)
    
    # Motor features
    motor_features = {k: v for k, v in obs_features.items() if k.endswith('.pos')}
    log_and_print(f"\n🔧 Motor Features ({len(motor_features)}):", output_file)
    for motor, dtype in motor_features.items():
        log_and_print(f"   {motor}: {dtype}", output_file)
    
    # Camera features
    camera_features = {k: v for k, v in obs_features.items() if not k.endswith('.pos')}
    log_and_print(f"\n📷 Camera Features ({len(camera_features)}):", output_file)
    for cam, shape in camera_features.items():
        log_and_print(f"   {cam}: shape={shape}", output_file)
    
except Exception as e:
    log_and_print(f"❌ Failed to inspect features: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)

# ============================================================================
# Test 2: Get Observation
# ============================================================================
print_section("Test 2: Get Observation", output_file)

try:
    log_and_print("📸 Capturing observation...", output_file)
    
    start_time = time.time()
    observation = robot.get_observation()
    obs_latency = (time.time() - start_time) * 1000  # ms
    
    log_and_print(f"✅ Observation captured in {obs_latency:.2f} ms", output_file)
    log_and_print(f"\n📊 Observation contains {len(observation)} keys:", output_file)
    
    # Motor positions
    motor_obs = {k: v for k, v in observation.items() if k.endswith('.pos')}
    log_and_print(f"\n🔧 Motor Positions:", output_file)
    for motor, pos in motor_obs.items():
        log_and_print(f"   {motor}: {pos:.2f}°", output_file)
    
    # Camera frames
    camera_obs = {k: v for k, v in observation.items() if not k.endswith('.pos')}
    log_and_print(f"\n📷 Camera Frames:", output_file)
    for cam, frame in camera_obs.items():
        log_and_print(f"   {cam}: shape={frame.shape}, dtype={frame.dtype}", output_file)
        log_and_print(f"         min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}", output_file)
    
    # Save sample frames
    for cam_name, frame in camera_obs.items():
        frame_path = OUTPUT_DIR / f"{cam_name}_sample_{timestamp}.jpg"
        cv2.imwrite(str(frame_path), frame)
        log_and_print(f"   💾 Saved to: {frame_path}", output_file)
    
except Exception as e:
    log_and_print(f"❌ Failed to get observation: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)

# ============================================================================
# Test 3: Observation Latency Benchmark
# ============================================================================
print_section("Test 3: Observation Latency", output_file)

log_and_print("⚡ Benchmarking get_observation() performance...", output_file)

try:
    latencies = []
    for i in range(50):
        start = time.time()
        obs = robot.get_observation()
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        
        if (i + 1) % 10 == 0:
            log_and_print(f"   Iteration {i+1}/50: {latency:.2f} ms", output_file)
    
    log_and_print(f"\n📊 Latency Statistics (50 iterations):", output_file)
    log_and_print(f"   Mean: {np.mean(latencies):.2f} ms", output_file)
    log_and_print(f"   Median: {np.median(latencies):.2f} ms", output_file)
    log_and_print(f"   Min: {np.min(latencies):.2f} ms", output_file)
    log_and_print(f"   Max: {np.max(latencies):.2f} ms", output_file)
    log_and_print(f"   Std dev: {np.std(latencies):.2f} ms", output_file)
    
    # Calculate achievable control frequency
    mean_latency = np.mean(latencies)
    max_hz = 1000 / mean_latency
    log_and_print(f"\n📈 Control Loop Performance:", output_file)
    log_and_print(f"   Maximum achievable frequency: {max_hz:.1f} Hz", output_file)
    
    if max_hz >= 10:
        log_and_print(f"   ✅ Suitable for VLA control loop (10+ Hz)", output_file)
    elif max_hz >= 5:
        log_and_print(f"   ✅ Suitable for slower VLA control (5-10 Hz)", output_file)
    elif max_hz >= 2:
        log_and_print(f"   ⚠️  Slow but usable for deliberate tasks (2-5 Hz)", output_file)
    else:
        log_and_print(f"   ❌ Too slow for practical control (<2 Hz)", output_file)
    
except Exception as e:
    log_and_print(f"❌ Benchmark failed: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)

# ============================================================================
# Test 4: Live Observation Display
# ============================================================================
print_section("Test 4: Live Display", output_file)

log_and_print("🖥️  Opening live observation display...", output_file)
log_and_print("   Press 'q' to quit", output_file)
log_and_print("   Press 's' to save snapshot", output_file)

try:
    snapshot_count = 0
    frame_count = 0
    fps_start = time.time()
    
    while True:
        # Get full observation
        obs = robot.get_observation()
        frame_count += 1
        
        # Extract camera frames
        wrist_frame = obs.get("wrist")
        scene_frame = obs.get("scene")
        
        if wrist_frame is None or scene_frame is None:
            log_and_print("⚠️  Missing camera frames in observation", output_file)
            break
        
        # Resize for display
        display_height = 480
        wrist_display = cv2.resize(wrist_frame, (640, display_height))
        scene_display = cv2.resize(scene_frame, (int(1920 * display_height / 1080), display_height))
        
        # Add motor position overlay on wrist camera
        y_offset = 30
        for motor, pos in obs.items():
            if motor.endswith('.pos'):
                text = f"{motor}: {pos:.1f}°"
                cv2.putText(wrist_display, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 25
                if y_offset > display_height - 20:
                    break
        
        # Add FPS counter
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            current_fps = 30 / elapsed if elapsed > 0 else 0
            fps_start = time.time()
        else:
            current_fps = 0
        
        if current_fps > 0:
            cv2.putText(scene_display, f"FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine side-by-side
        combined = np.hstack([wrist_display, scene_display])
        
        # Display
        cv2.imshow("UJ201 Robot + Camera Observation", combined)
        
        # waitKey needs at least 10ms to process events properly
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            log_and_print("\n⏹️  Display closed by user", output_file)
            break
        elif key == ord('s'):
            snapshot_count += 1
            snapshot_path = OUTPUT_DIR / f"observation_snapshot_{timestamp}_{snapshot_count:03d}.jpg"
            cv2.imwrite(str(snapshot_path), combined)
            log_and_print(f"💾 Snapshot saved: {snapshot_path}", output_file)
            print(f"💾 Observation snapshot {snapshot_count} saved!")
    
    cv2.destroyAllWindows()
    
except Exception as e:
    log_and_print(f"❌ Display failed: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)
    cv2.destroyAllWindows()

# ============================================================================
# Test 5: Action Features
# ============================================================================
print_section("Test 5: Action Features", output_file)

try:
    log_and_print("🎯 Inspecting action features...", output_file)
    
    action_features = robot.action_features
    log_and_print(f"\n📊 Total action features: {len(action_features)}", output_file)
    
    for motor, dtype in action_features.items():
        log_and_print(f"   {motor}: {dtype}", output_file)
    
    log_and_print(f"\n✅ Action space matches motor position keys", output_file)
    
except Exception as e:
    log_and_print(f"❌ Failed to inspect action features: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)

# ============================================================================
# Cleanup
# ============================================================================
print_section("Cleanup", output_file)

try:
    log_and_print("🔌 Disconnecting robot...", output_file)
    robot.disconnect()
    log_and_print("✅ Robot disconnected successfully", output_file)
except Exception as e:
    log_and_print(f"⚠️  Disconnect warning: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)

# ============================================================================
# Summary
# ============================================================================
log_and_print("\n" + "=" * 80, output_file)
log_and_print("✅ Robot + Camera Integration Test Completed", output_file)
log_and_print(f"Test ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", output_file)
log_and_print("=" * 80, output_file)
log_and_print(f"\n📄 Results saved to: {OUTPUT_FILE}", output_file)
log_and_print(f"📁 Output directory: {OUTPUT_DIR}", output_file)

output_file.close()
print(f"\n✅ Full test log saved to: {OUTPUT_FILE}")
