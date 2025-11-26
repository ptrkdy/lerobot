#!/usr/bin/env python

"""
UJ201 + SmolVLA Inference Test Script

Tests vision-language-action (VLA) inference loop with UJ201 robot:
- Dual camera input (wrist + scene)
- SmolVLA pretrained model
- Full preprocessing/postprocessing pipeline
- Action execution on robot

This validates the complete VLA control loop:
  get_observation → build_inference_frame → preprocess → 
  select_action → postprocess → send_action

Usage:
    python test_uj201_vla_inference.py

Requirements:
    - UJ201 robot connected
    - Both cameras (wrist + scene) connected
    - SmolVLA model downloaded (lerobot/smolvla_base)
    - PyTorch with MPS/CUDA support (recommended)
"""

import time
import torch
from datetime import datetime
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig

# ============================================================================
# Configuration
# ============================================================================

# Device configuration (MPS for Mac M1/M2, CUDA for NVIDIA, CPU fallback)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU (slower inference)")

# Model configuration
MODEL_ID = "lerobot/smolvla_base"  # 450M parameter SmolVLA model

# Robot configuration
ROBOT_PORT = "/dev/tty.usbmodem5A7A0565221"
ROBOT_ID = "None"  # Set to your robot ID if you have calibration files

# Camera configuration
# NOTE: SmolVLA resizes images to 512x512 automatically with padding
# Camera keys MUST match those used during training
CAMERAS = {
    "wrist": OpenCVCameraConfig(
        name="wrist",
        camera_index=0,
        width=1280,
        height=960,
        fps=5,
    ),
    "scene": OpenCVCameraConfig(
        name="scene", 
        camera_index=1,
        width=1920,
        height=1080,
        fps=30,
    ),
}

# Task specification (natural language instruction)
TASK = "pick up the object and place it in the box"  # Modify for your task
ROBOT_TYPE = "uj201_follower"  # For multi-embodiment datasets

# Test parameters
MAX_EPISODES = 3
MAX_STEPS_PER_EPISODE = 50  # SmolVLA outputs 50-step action chunks by default

# Logging
OUTPUT_DIR = Path("vla_inference_results")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = OUTPUT_DIR / f"vla_inference_{timestamp}.txt"

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

# ============================================================================
# Main Script
# ============================================================================

output_file = open(OUTPUT_FILE, "w")
log_and_print("=" * 80, output_file)
log_and_print("🤖 UJ201 + SmolVLA Inference Test", output_file)
log_and_print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", output_file)
log_and_print("=" * 80, output_file)

# ============================================================================
# Load SmolVLA Model
# ============================================================================
print_section("Loading SmolVLA Model", output_file)

try:
    log_and_print(f"📦 Loading model: {MODEL_ID}", output_file)
    log_and_print(f"   Device: {device}", output_file)
    
    start_time = time.time()
    model = SmolVLAPolicy.from_pretrained(MODEL_ID)
    model = model.to(device)
    load_time = time.time() - start_time
    
    log_and_print(f"✅ Model loaded in {load_time:.2f}s", output_file)
    log_and_print(f"\n📊 Model Config:", output_file)
    log_and_print(f"   Chunk size: {model.config.chunk_size}", output_file)
    log_and_print(f"   N observation steps: {model.config.n_obs_steps}", output_file)
    log_and_print(f"   Max state dim: {model.config.max_state_dim}", output_file)
    
except Exception as e:
    log_and_print(f"❌ Failed to load model: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)
    output_file.close()
    exit(1)

# ============================================================================
# Create Preprocessor and Postprocessor
# ============================================================================
print_section("Creating Processors", output_file)

try:
    log_and_print("🔧 Building preprocessor and postprocessor...", output_file)
    
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        MODEL_ID,
        # Override device for preprocessor (important for MPS)
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    
    log_and_print("✅ Processors created", output_file)
    log_and_print(f"   Preprocessor: {type(preprocess).__name__}", output_file)
    log_and_print(f"   Postprocessor: {type(postprocess).__name__}", output_file)
    
except Exception as e:
    log_and_print(f"❌ Failed to create processors: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)
    output_file.close()
    exit(1)

# ============================================================================
# Initialize Robot
# ============================================================================
print_section("Initialize Robot", output_file)

robot_config = UJ201FollowerConfig(
    port=ROBOT_PORT,
    cameras=CAMERAS,
    use_degrees=True,
    max_relative_target=10.0,  # Safety limit: 10 degrees per step
)

try:
    log_and_print("🤖 Creating UJ201Follower...", output_file)
    robot = UJ201Follower(robot_config)
    
    log_and_print("🔌 Connecting robot (motors + cameras)...", output_file)
    robot.connect(calibrate=False)
    
    log_and_print("✅ Robot connected", output_file)
    log_and_print(f"   Motors connected: {robot.bus.is_connected}", output_file)
    log_and_print(f"   Cameras connected: {all(cam.is_connected for cam in robot.cameras.values())}", output_file)
    
except Exception as e:
    log_and_print(f"❌ Failed to initialize robot: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)
    output_file.close()
    exit(1)

# ============================================================================
# Prepare Dataset Features
# ============================================================================
print_section("Dataset Features", output_file)

try:
    log_and_print("📋 Building dataset feature mappings...", output_file)
    
    # Convert robot features to dataset format
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    
    log_and_print(f"✅ Dataset features created:", output_file)
    log_and_print(f"   Action features: {len(action_features)}", output_file)
    log_and_print(f"   Observation features: {len(obs_features)}", output_file)
    log_and_print(f"   Total: {len(dataset_features)}", output_file)
    
    # Log feature details
    log_and_print(f"\n📊 Action Features:", output_file)
    for key, value in action_features.items():
        log_and_print(f"   {key}: {value}", output_file)
    
    log_and_print(f"\n📊 Observation Features:", output_file)
    for key, value in list(obs_features.items())[:5]:  # Show first 5
        log_and_print(f"   {key}: {value}", output_file)
    if len(obs_features) > 5:
        log_and_print(f"   ... and {len(obs_features) - 5} more", output_file)
    
except Exception as e:
    log_and_print(f"❌ Failed to create dataset features: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)
    robot.disconnect()
    output_file.close()
    exit(1)

# ============================================================================
# VLA Inference Loop
# ============================================================================
print_section("VLA Inference Loop", output_file)

log_and_print(f"🎯 Task: \"{TASK}\"", output_file)
log_and_print(f"🤖 Robot Type: {ROBOT_TYPE}", output_file)
log_and_print(f"📊 Episodes: {MAX_EPISODES}", output_file)
log_and_print(f"📊 Steps per episode: {MAX_STEPS_PER_EPISODE}", output_file)

try:
    total_steps = 0
    inference_times = []
    action_times = []
    
    for episode in range(MAX_EPISODES):
        log_and_print(f"\n{'='*60}", output_file)
        log_and_print(f"Episode {episode + 1}/{MAX_EPISODES}", output_file)
        log_and_print(f"{'='*60}", output_file)
        
        episode_start = time.time()
        
        for step in range(MAX_STEPS_PER_EPISODE):
            step_start = time.time()
            
            # 1. Get observation from robot
            obs = robot.get_observation()
            
            # 2. Build inference frame
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=dataset_features,
                device=device,
                task=TASK,
                robot_type=ROBOT_TYPE,
            )
            
            # 3. Preprocess
            obs_processed = preprocess(obs_frame)
            
            # 4. Model inference
            inference_start = time.time()
            with torch.no_grad():
                action = model.select_action(obs_processed)
            inference_time = (time.time() - inference_start) * 1000
            inference_times.append(inference_time)
            
            # 5. Postprocess
            action = postprocess(action)
            
            # 6. Convert to robot action format
            robot_action = make_robot_action(action, dataset_features)
            
            # 7. Send action to robot
            action_start = time.time()
            robot.send_action(robot_action)
            action_time = (time.time() - action_start) * 1000
            action_times.append(action_time)
            
            step_time = (time.time() - step_start) * 1000
            total_steps += 1
            
            # Log progress every 10 steps
            if (step + 1) % 10 == 0:
                log_and_print(
                    f"   Step {step+1:3d}/{MAX_STEPS_PER_EPISODE}: "
                    f"inference={inference_time:.1f}ms, "
                    f"action={action_time:.1f}ms, "
                    f"total={step_time:.1f}ms",
                    output_file
                )
        
        episode_time = time.time() - episode_start
        log_and_print(f"\n✅ Episode {episode + 1} completed in {episode_time:.2f}s", output_file)
        
        if episode < MAX_EPISODES - 1:
            log_and_print("\n🔄 Starting new episode...", output_file)
    
    # ========================================================================
    # Performance Summary
    # ========================================================================
    print_section("Performance Summary", output_file)
    
    import numpy as np
    
    log_and_print(f"📊 Total Steps: {total_steps}", output_file)
    log_and_print(f"\n⚡ Inference Performance:", output_file)
    log_and_print(f"   Mean: {np.mean(inference_times):.2f} ms", output_file)
    log_and_print(f"   Median: {np.median(inference_times):.2f} ms", output_file)
    log_and_print(f"   Min: {np.min(inference_times):.2f} ms", output_file)
    log_and_print(f"   Max: {np.max(inference_times):.2f} ms", output_file)
    
    log_and_print(f"\n🔧 Action Send Performance:", output_file)
    log_and_print(f"   Mean: {np.mean(action_times):.2f} ms", output_file)
    log_and_print(f"   Median: {np.median(action_times):.2f} ms", output_file)
    
    total_cycle_time = np.mean(inference_times) + np.mean(action_times)
    achievable_hz = 1000 / total_cycle_time
    
    log_and_print(f"\n📈 Control Loop:", output_file)
    log_and_print(f"   Average cycle time: {total_cycle_time:.2f} ms", output_file)
    log_and_print(f"   Achievable frequency: {achievable_hz:.1f} Hz", output_file)
    
    if achievable_hz >= 10:
        log_and_print(f"   ✅ Suitable for real-time VLA control", output_file)
    else:
        log_and_print(f"   ⚠️  May be too slow for smooth control", output_file)
    
except Exception as e:
    log_and_print(f"\n❌ Inference loop failed: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)

# ============================================================================
# Cleanup
# ============================================================================
print_section("Cleanup", output_file)

try:
    log_and_print("🔌 Disconnecting robot...", output_file)
    robot.disconnect()
    log_and_print("✅ Robot disconnected", output_file)
except Exception as e:
    log_and_print(f"⚠️  Disconnect warning: {e}", output_file)

# ============================================================================
# Summary
# ============================================================================
log_and_print("\n" + "=" * 80, output_file)
log_and_print("✅ VLA Inference Test Completed", output_file)
log_and_print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", output_file)
log_and_print("=" * 80, output_file)
log_and_print(f"\n📄 Results saved to: {OUTPUT_FILE}", output_file)

output_file.close()
print(f"\n✅ Full test log saved to: {OUTPUT_FILE}")
