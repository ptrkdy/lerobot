#!/usr/bin/env python

"""
UJ201 Data Collection Script for SmolVLA Fine-tuning

Records demonstration episodes with:
- Dual camera observations (camera1, camera2)
- 9-motor joint positions
- Leader-follower teleoperation
- Automatic episode saving

The collected data can be used to fine-tune SmolVLA on UJ201-specific tasks.

Usage:
    python collect_uj201_data.py --output-dir data/uj201_demos --num-episodes 50

Requirements:
    - UJ201 Leader arm for teleoperation
    - UJ201 Follower arm
    - Both cameras connected
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import cv2

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig

# Default configuration
DEFAULT_OUTPUT_DIR = Path("data/uj201_demos")
DEFAULT_NUM_EPISODES = 50
DEFAULT_EPISODE_LENGTH = 30  # seconds
DEFAULT_FPS = 5  # Match wrist camera FPS for consistent timing
LEADER_PORT = "/dev/tty.usbmodem58760431631"  # Update with your leader port
FOLLOWER_PORT = "/dev/tty.usbmodem5A7A0565221"

def print_header(title):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def setup_robot(port, robot_id, cameras):
    """Initialize robot with cameras."""
    config = UJ201FollowerConfig(
        port=port,
        cameras=cameras,
        use_degrees=True,
        max_relative_target=None,  # No safety limit during teleoperation
    )
    
    robot = UJ201Follower(config)
    robot.connect(calibrate=False)
    return robot

def save_episode(episode_data, output_dir, episode_num, task_description):
    """Save episode data to disk."""
    episode_dir = output_dir / f"episode_{episode_num:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "episode_num": episode_num,
        "task": task_description,
        "num_frames": len(episode_data["observations"]),
        "fps": DEFAULT_FPS,
        "timestamp": datetime.now().isoformat(),
        "robot": "uj201_follower",
        "cameras": ["camera1", "camera2"],
        "motors": [
            "shoulder", "gearbox", "universal_joint",
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper"
        ],
    }
    
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save observations and actions as numpy arrays
    np.savez_compressed(
        episode_dir / "data.npz",
        observations=np.array(episode_data["observations"]),
        actions=np.array(episode_data["actions"]),
        motor_names=episode_data["motor_names"],
    )
    
    # Save camera frames as video files
    save_video(episode_data["camera1_frames"], episode_dir / "camera1.mp4", DEFAULT_FPS)
    save_video(episode_data["camera2_frames"], episode_dir / "camera2.mp4", DEFAULT_FPS)
    
    print(f"✅ Episode {episode_num} saved to {episode_dir}")
    print(f"   Frames: {len(episode_data['observations'])}")
    print(f"   Duration: {len(episode_data['observations']) / DEFAULT_FPS:.1f}s")

def save_video(frames, output_path, fps):
    """Save frames as video file."""
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()

def collect_episode(follower, leader, episode_num, task_description, episode_length):
    """Collect a single demonstration episode."""
    print_header(f"Episode {episode_num}: {task_description}")
    
    input("Position the robot at the starting state and press ENTER to begin recording...")
    
    episode_data = {
        "observations": [],
        "actions": [],
        "camera1_frames": [],
        "camera2_frames": [],
        "motor_names": [],
    }
    
    print(f"🔴 Recording for {episode_length} seconds...")
    print("   Perform the demonstration now!")
    
    start_time = time.time()
    frame_count = 0
    expected_frame_time = 1.0 / DEFAULT_FPS
    
    try:
        while time.time() - start_time < episode_length:
            frame_start = time.time()
            
            # Get leader position (action)
            leader_obs = leader.get_observation()
            action = {
                key: value for key, value in leader_obs.items()
                if key.endswith(".pos")
            }
            
            # Get follower observation (state + cameras)
            follower_obs = follower.get_observation()
            
            # Extract motor positions
            motor_obs = {
                key: value for key, value in follower_obs.items()
                if key.endswith(".pos")
            }
            
            # Store data
            episode_data["observations"].append(list(motor_obs.values()))
            episode_data["actions"].append(list(action.values()))
            episode_data["camera1_frames"].append(follower_obs["camera1"])
            episode_data["camera2_frames"].append(follower_obs["camera2"])
            
            # Store motor names (once)
            if not episode_data["motor_names"]:
                episode_data["motor_names"] = list(motor_obs.keys())
            
            # Send leader position to follower
            follower.send_action(action)
            
            frame_count += 1
            
            # Maintain consistent FPS
            elapsed = time.time() - frame_start
            if elapsed < expected_frame_time:
                time.sleep(expected_frame_time - elapsed)
            
            # Progress indicator
            if frame_count % 10 == 0:
                print(f"   Frame {frame_count} ({time.time() - start_time:.1f}s)")
    
    except KeyboardInterrupt:
        print("\n⏹️  Recording stopped by user")
    
    actual_duration = time.time() - start_time
    actual_fps = frame_count / actual_duration
    
    print(f"\n✅ Recording complete")
    print(f"   Frames captured: {frame_count}")
    print(f"   Actual duration: {actual_duration:.1f}s")
    print(f"   Actual FPS: {actual_fps:.1f}")
    
    return episode_data

def main():
    parser = argparse.ArgumentParser(description="Collect UJ201 demonstration data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                       help="Directory to save collected data")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES,
                       help="Number of episodes to collect")
    parser.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_LENGTH,
                       help="Length of each episode in seconds")
    parser.add_argument("--task", type=str, default="pick up the red cube",
                       help="Task description for this data collection session")
    parser.add_argument("--leader-port", type=str, default=LEADER_PORT,
                       help="Serial port for leader arm")
    parser.add_argument("--follower-port", type=str, default=FOLLOWER_PORT,
                       help="Serial port for follower arm")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🎥 UJ201 Data Collection for SmolVLA Fine-tuning")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Number of episodes: {args.num_episodes}")
    print(f"   Episode length: {args.episode_length}s")
    print(f"   Recording FPS: {DEFAULT_FPS}")
    print(f"   Task: {args.task}")
    print(f"   Leader port: {args.leader_port}")
    print(f"   Follower port: {args.follower_port}")
    
    # Camera configuration (matching SmolVLA expectations)
    cameras = {
        "camera1": OpenCVCameraConfig(
            index_or_path=0,
            width=1280,
            height=960,
            fps=5,
        ),
        "camera2": OpenCVCameraConfig(
            index_or_path=1,
            width=1920,
            height=1080,
            fps=30,
        ),
    }
    
    # Initialize robots
    print_header("Initializing Robots")
    
    try:
        print("🔌 Connecting to follower arm...")
        follower = setup_robot(args.follower_port, "follower", cameras)
        print("✅ Follower connected")
        
        print("\n🔌 Connecting to leader arm...")
        leader = setup_robot(args.leader_port, "leader", {})  # Leader doesn't need cameras
        print("✅ Leader connected")
        
    except Exception as e:
        print(f"\n❌ Failed to initialize robots: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Collect episodes
    print_header("Data Collection")
    print(f"\n⚠️  IMPORTANT:")
    print(f"   - Move the LEADER arm to control the FOLLOWER")
    print(f"   - Follower will mirror leader movements")
    print(f"   - Press Ctrl+C during recording to stop early")
    print(f"   - You can skip episodes by pressing Ctrl+C at the start prompt")
    
    collected_episodes = 0
    
    try:
        for episode_num in range(1, args.num_episodes + 1):
            try:
                episode_data = collect_episode(
                    follower, leader, episode_num, args.task, args.episode_length
                )
                
                save_episode(episode_data, args.output_dir, episode_num, args.task)
                collected_episodes += 1
                
                print(f"\n✅ Progress: {collected_episodes}/{args.num_episodes} episodes collected")
                
                if episode_num < args.num_episodes:
                    response = input("\nContinue to next episode? (yes/no): ")
                    if response.lower() != 'yes':
                        print("⏹️  Data collection stopped by user")
                        break
            
            except KeyboardInterrupt:
                print("\n⏭️  Skipping episode")
                continue
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Data collection interrupted")
    
    finally:
        # Cleanup
        print_header("Cleanup")
        try:
            follower.disconnect()
            leader.disconnect()
            print("✅ Robots disconnected")
        except:
            pass
    
    # Summary
    print_header("Collection Summary")
    print(f"✅ Data collection complete!")
    print(f"   Episodes collected: {collected_episodes}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Task: {args.task}")
    print(f"\n📊 Next steps:")
    print(f"   1. Review collected data: ls {args.output_dir}")
    print(f"   2. Convert to LeRobot dataset format")
    print(f"   3. Upload to HuggingFace: huggingface-cli upload")
    print(f"   4. Fine-tune SmolVLA: lerobot-train --policy.path=lerobot/smolvla_base")

if __name__ == "__main__":
    main()
