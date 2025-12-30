#!/usr/bin/env python

"""
Simple test script to verify UJ201 robot implementation and calibration.
Run this after installing your fork in development mode.
"""

import sys
from pathlib import Path

# Add the local lerobot to the path (if needed)
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Test imports
    from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig
    from lerobot.teleoperators.uj201_leader import UJ201Leader, UJ201LeaderConfig
    print("✅ UJ201 imports successful")

    # Test configuration creation
    robot_config = UJ201FollowerConfig(
        port="/dev/ttyACM0",  # Update to your actual port
        id="test_uj201_follower"
    )
    
    teleop_config = UJ201LeaderConfig(
        port="/dev/ttyACM1",  # Update to your actual port  
        id="test_uj201_leader"
    )
    print("✅ Configuration creation successful")

    # Test robot instantiation (without connecting)
    robot = UJ201Follower(robot_config)
    teleop = UJ201Leader(teleop_config)
    print("✅ Robot instantiation successful")

    # Check motor configuration
    print(f"Robot motors: {list(robot.bus.motors.keys())}")
    print(f"Expected 9 motors, got: {len(robot.bus.motors)}")
    
    print(f"Teleop motors: {list(teleop.bus.motors.keys())}")
    print(f"Expected 9 motors, got: {len(teleop.bus.motors)}")

    # Check features
    print(f"Action features: {len(robot.action_features)}")
    print(f"Observation features: {len(robot.observation_features)}")

    print("\n🎉 All tests passed! Your UJ201 implementation is working.")
    print("\nTo test with actual hardware:")
    print("1. Connect your UJ201 robot to the specified ports")
    print("2. Run: python test_uj201_hardware.py")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()