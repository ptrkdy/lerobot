#!/usr/bin/env python

"""
Debug script to test UJ201 device connections individually.
Run this to identify which device is causing the hang.
"""

import sys
import traceback

def test_robot_connection():
    print("Testing UJ201 Follower connection...")
    try:
        from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig
        
        config = UJ201FollowerConfig(
            port="/dev/tty.usbmodem5A7A0565221",
            id="debug_follower"
        )
        
        print(f"✅ Config created: {config}")
        robot = UJ201Follower(config)
        print(f"✅ Robot instantiated: {robot}")
        
        print("🔌 Attempting to connect...")
        robot.connect(calibrate=False)  # Skip calibration for debugging
        print("✅ Robot connected successfully!")
        
        print("📊 Testing observation read...")
        obs = robot.get_observation()
        print(f"✅ Observation keys: {list(obs.keys())}")
        
        robot.disconnect()
        print("✅ Robot disconnected successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Robot test failed: {e}")
        traceback.print_exc()
        return False

def test_teleop_connection():
    print("\nTesting UJ201 Leader connection...")
    try:
        from lerobot.teleoperators.uj201_leader import UJ201Leader, UJ201LeaderConfig
        
        config = UJ201LeaderConfig(
            port="/dev/tty.usbmodem5A680092181",
            id="debug_leader"
        )
        
        print(f"✅ Config created: {config}")
        teleop = UJ201Leader(config)
        print(f"✅ Teleoperator instantiated: {teleop}")
        
        print("🔌 Attempting to connect...")
        teleop.connect(calibrate=False)  # Skip calibration for debugging
        print("✅ Teleoperator connected successfully!")
        
        print("📊 Testing action read...")
        action = teleop.get_action()
        print(f"✅ Action keys: {list(action.keys())}")
        
        teleop.disconnect()
        print("✅ Teleoperator disconnected successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Teleoperator test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 UJ201 Connection Debug Test")
    print("=" * 50)
    
    robot_ok = test_robot_connection()
    teleop_ok = test_teleop_connection()
    
    print("\n" + "=" * 50)
    print("🎯 Summary:")
    print(f"Robot (Follower): {'✅ PASS' if robot_ok else '❌ FAIL'}")
    print(f"Teleoperator (Leader): {'✅ PASS' if teleop_ok else '❌ FAIL'}")
    
    if robot_ok and teleop_ok:
        print("\n🎉 Both devices working! The issue might be in the teleoperate script.")
    else:
        print("\n🔧 Fix the failing device(s) before running teleoperation.")