#!/usr/bin/env python

"""
Debug UJ201 teleoperate hanging issue.
This script tests the full connection process step by step.
"""

import sys
import traceback
import time

def test_minimal_connection():
    print("🧪 Testing minimal connection (no calibration)...")
    try:
        from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig
        from lerobot.teleoperators.uj201_leader import UJ201Leader, UJ201LeaderConfig
        
        # Create configs
        robot_config = UJ201FollowerConfig(
            port="/dev/tty.usbmodem5A7A0565221",
            id="debug_follower_min"
        )
        
        teleop_config = UJ201LeaderConfig(
            port="/dev/tty.usbmodem5A680092181",
            id="debug_leader_min"
        )
        
        print("✅ Configs created")
        
        # Create instances
        robot = UJ201Follower(robot_config)
        teleop = UJ201Leader(teleop_config)
        
        print("✅ Instances created")
        
        # Test connections without calibration
        print("🔌 Connecting robot (no calibration)...")
        robot.connect(calibrate=False)
        print("✅ Robot connected!")
        
        print("🔌 Connecting teleoperator (no calibration)...")
        teleop.connect(calibrate=False)
        print("✅ Teleoperator connected!")
        
        # Test if basic methods work
        print("📊 Testing basic functionality...")
        
        print("  Getting robot observation...")
        obs = robot.get_observation()
        print(f"  ✅ Robot observation keys: {list(obs.keys())}")
        
        print("  Getting teleop action...")
        action = teleop.get_action()
        print(f"  ✅ Teleop action keys: {list(action.keys())}")
        
        print("  Sending action to robot...")
        robot.send_action(action)
        print("  ✅ Action sent successfully!")
        
        # Cleanup
        teleop.disconnect()
        robot.disconnect()
        print("✅ Disconnected successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        traceback.print_exc()
        try:
            if 'teleop' in locals():
                teleop.disconnect()
            if 'robot' in locals():
                robot.disconnect()
        except:
            pass
        return False

def test_teleoperate_loop():
    print("\n🔄 Testing actual teleoperation loop...")
    try:
        from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig
        from lerobot.teleoperators.uj201_leader import UJ201Leader, UJ201LeaderConfig
        
        # Create configs
        robot_config = UJ201FollowerConfig(
            port="/dev/tty.usbmodem5A7A0565221",
            id="debug_follower_loop"
        )
        
        teleop_config = UJ201LeaderConfig(
            port="/dev/tty.usbmodem5A680092181",
            id="debug_leader_loop"
        )
        
        # Create instances
        robot = UJ201Follower(robot_config)
        teleop = UJ201Leader(teleop_config)
        
        # Connect (without calibration to avoid hang)
        robot.connect(calibrate=False)
        teleop.connect(calibrate=False)
        
        print("✅ Both devices connected, starting teleoperation loop...")
        
        # Run a short teleoperation loop
        for i in range(10):
            print(f"  Loop iteration {i+1}/10")
            
            # Get robot observation
            obs = robot.get_observation()
            
            # Get teleop action
            action = teleop.get_action()
            
            # Send action to robot
            robot.send_action(action)
            
            # Small delay
            time.sleep(0.1)
            
            if i == 4:
                print("  ✅ Halfway through loop - all good!")
        
        print("✅ Teleoperation loop completed successfully!")
        
        # Cleanup
        teleop.disconnect()
        robot.disconnect()
        print("✅ Cleanup completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Teleoperation loop failed: {e}")
        traceback.print_exc()
        try:
            if 'teleop' in locals():
                teleop.disconnect()
            if 'robot' in locals():
                robot.disconnect()
        except:
            pass
        return False

def test_with_processors():
    print("\n🔧 Testing with processors (like real teleoperate)...")
    try:
        from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig
        from lerobot.teleoperators.uj201_leader import UJ201Leader, UJ201LeaderConfig
        from lerobot.processor import make_default_processors
        
        # Create configs
        robot_config = UJ201FollowerConfig(
            port="/dev/tty.usbmodem5A7A0565221",
            id="debug_follower_proc"
        )
        
        teleop_config = UJ201LeaderConfig(
            port="/dev/tty.usbmodem5A680092181",
            id="debug_leader_proc"
        )
        
        # Create instances
        robot = UJ201Follower(robot_config)
        teleop = UJ201Leader(teleop_config)
        
        print("✅ Creating processors...")
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
        
        # Connect
        robot.connect(calibrate=False)
        teleop.connect(calibrate=False)
        
        print("✅ Testing with processors...")
        
        # Test processor pipeline (like teleoperate script)
        for i in range(5):
            print(f"  Processor loop {i+1}/5")
            
            # Get robot observation
            obs = robot.get_observation()
            
            # Get teleop action
            raw_action = teleop.get_action()
            
            # Process teleop action through pipeline
            teleop_action = teleop_action_processor((raw_action, obs))
            
            # Process action for robot through pipeline
            robot_action_to_send = robot_action_processor((teleop_action, obs))
            
            # Send processed action to robot
            robot.send_action(robot_action_to_send)
            
            time.sleep(0.1)
        
        print("✅ Processor pipeline test completed!")
        
        # Cleanup
        teleop.disconnect()
        robot.disconnect()
        
        return True
        
    except Exception as e:
        print(f"❌ Processor test failed: {e}")
        traceback.print_exc()
        try:
            if 'teleop' in locals():
                teleop.disconnect()
            if 'robot' in locals():
                robot.disconnect()
        except:
            pass
        return False

if __name__ == "__main__":
    print("🔧 UJ201 Teleoperation Debug Test")
    print("=" * 60)
    
    # Test 1: Basic connection
    print("Step 1: Testing basic connection without calibration")
    basic_ok = test_minimal_connection()
    
    if not basic_ok:
        print("\n❌ Basic connection failed - check hardware!")
        sys.exit(1)
    
    # Test 2: Teleoperation loop
    print("\nStep 2: Testing teleoperation loop")
    loop_ok = test_teleoperate_loop()
    
    if not loop_ok:
        print("\n❌ Teleoperation loop failed!")
        sys.exit(1)
    
    # Test 3: With processors (like real teleoperate script)
    print("\nStep 3: Testing with processors")
    proc_ok = test_with_processors()
    
    print("\n" + "=" * 60)
    print("🎯 Debug Summary:")
    print(f"Basic connection: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"Teleoperation loop: {'✅ PASS' if loop_ok else '❌ FAIL'}")
    print(f"With processors: {'✅ PASS' if proc_ok else '❌ FAIL'}")
    
    if basic_ok and loop_ok and proc_ok:
        print("\n🎉 All debug tests passed!")
        print("💡 The issue might be:")
        print("  1. Calibration process hanging (try with calibrate=False)")
        print("  2. Camera initialization (try without cameras)")
        print("  3. Display/rerun initialization (try with display_data=false)")
    else:
        print(f"\n🔧 Debug identified the issue at step {['basic', 'loop', 'processor'][([basic_ok, loop_ok, proc_ok].index(False))]}")