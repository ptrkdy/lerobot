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
Example script demonstrating UJ201 follower robot policy evaluation.

This script shows how to:
1. Load a trained policy for the UJ201 robot
2. Run inference and record evaluation episodes
3. Visualize the robot's autonomous behavior

Usage:
    python examples/uj201_follower/evaluate.py
"""

import time

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# Configuration
NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 60
HF_MODEL_ID = "your_username/uj201_policy"  # Replace with your trained policy
HF_DATASET_ID = "your_username/eval_uj201"  # Replace with your eval dataset repo
TASK_DESCRIPTION = "UJ201 policy evaluation"

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

# Initialize robot
robot = UJ201Follower(robot_config)

# Load the trained policy
policy = PreTrainedPolicy.from_pretrained(HF_MODEL_ID)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, ACTION)
obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
dataset_features = {**action_features, **obs_features}

# Create evaluation dataset
dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Build Policy Processors
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=HF_MODEL_ID,
    dataset_stats=dataset.meta.stats,
    # The inference device is automatically set to match the detected hardware
    preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
)

# Connect the robot
robot.connect()

# TODO: Update this example to use pipelines
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# Initialize the keyboard listener and rerun visualization
listener, events = init_keyboard_listener()
init_rerun(session_name="uj201_evaluate")

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

print("Starting UJ201 evaluation loop...")
print(f"Robot: {robot}")
print(f"Policy: {HF_MODEL_ID}")
print(f"Motors: {len(robot.action_features)} joints")

recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Running inference, recording eval episode {recorded_episodes} of {NUM_EPISODES}")

    # Main evaluation loop with policy inference
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
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
        time.sleep(2)  # Brief pause for reset

    if events["rerecord_episode"]:
        log_say("Re-recording this episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    recorded_episodes += 1

# Clean up
log_say("Stop evaluation")
robot.disconnect()
dataset.push_to_hub()

print(f"Successfully recorded {recorded_episodes} evaluation episodes!")
print(f"Evaluation dataset saved to: {HF_DATASET_ID}")
print("Review the results in rerun or the dataset viewer.")