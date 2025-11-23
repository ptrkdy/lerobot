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
Example script demonstrating UJ201 follower robot usage for recording demonstrations.

This script shows how to:
1. Configure the UJ201 follower robot and leader teleoperator
2. Set up cameras and dataset recording
3. Record demonstration episodes
4. Handle calibration and safety

Usage:
    python examples/uj201_follower/record.py
"""

import time

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.uj201_leader import UJ201Leader, UJ201LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# Configuration
NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
HF_REPO_ID = "your_username/uj201_dataset"  # Replace with your HF username
TASK_DESCRIPTION = "UJ201 demonstration task"

# Hardware configuration
robot_config = UJ201FollowerConfig(
    port="/dev/ttyACM0",  # Update to your follower port
    id="my_uj201_follower",
    cameras={
        "front": OpenCVCameraConfig(
            index_or_path=0,
            width=640,
            height=480,
            fps=30
        )
    }
)

teleop_config = UJ201LeaderConfig(
    port="/dev/ttyACM1",  # Update to your leader port
    id="my_uj201_leader"
)

# Initialize robot and teleoperator
robot = UJ201Follower(robot_config)
leader_arm = UJ201Leader(teleop_config)

# TODO: Update this example to use pipelines
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, ACTION)
obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_REPO_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Connect the robot and teleoperator
robot.connect()
leader_arm.connect()

# Initialize the keyboard listener and rerun visualization
listener, events = init_keyboard_listener()
init_rerun(session_name="uj201_record")

if not robot.is_connected or not leader_arm.is_connected:
    raise ValueError("Robot or teleop is not connected!")

print("Starting UJ201 record loop...")
print(f"Robot: {robot}")
print(f"Teleoperator: {leader_arm}")
print(f"Features: {len(dataset_features)} total")
print(f"Motors: {len(robot.action_features)} joints")

recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {recorded_episodes}")

    # Main record loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        teleop=[leader_arm],
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            dataset=dataset,
            teleop=[leader_arm],
            control_time_s=RESET_TIME_SEC,
            single_task="Reset environment to starting position",
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording this episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    recorded_episodes += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
leader_arm.disconnect()
dataset.push_to_hub()

print(f"Successfully recorded {recorded_episodes} episodes!")
print(f"Dataset saved to: {HF_REPO_ID}")
print("You can now train a policy using:")
print(f"lerobot-train --dataset.repo_id={HF_REPO_ID} --policy.type=act")