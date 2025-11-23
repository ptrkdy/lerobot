#!/usr/bin/env python

"""
Debug script for UJ201 wrist_roll servo issues.

This script provides comprehensive diagnostics for the wrist_roll joint:
- Motor connectivity and health checks
- Calibration verification
- Manual position testing
- Leader-follower synchronization test
- Communication diagnostics

Usage:
    python debug_wrist_roll.py
"""

import time
import json
from pathlib import Path
from lerobot.robots.uj201_follower import UJ201Follower, UJ201FollowerConfig
from lerobot.teleoperators.uj201_leader import UJ201Leader, UJ201LeaderConfig
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

print("=" * 80)
print("🔧 UJ201 Wrist Roll Joint Diagnostic Tool")
print("=" * 80)

# Configuration
FOLLOWER_PORT = "/dev/tty.usbmodem5A7A0565221"
LEADER_PORT = "/dev/tty.usbmodem5A680092181"
ROBOT_ID = "None"  # This matches your calibration file name
WRIST_ROLL_MOTOR_ID = 8  # Motor ID for wrist_roll

def load_calibration(device_type, robot_id):
    """Load calibration file for a device."""
    calib_dir = Path.home() / ".cache/huggingface/lerobot/calibration" / device_type
    calib_file = calib_dir / f"{robot_id}.json"
    
    if calib_file.exists():
        with open(calib_file) as f:
            return json.load(f)
    else:
        print(f"⚠️  No calibration file found at {calib_file}")
        return None

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

# ============================================================================
# TEST 1: Check Calibration Files
# ============================================================================
print_section("TEST 1: Checking Calibration Files")

follower_calib = load_calibration("robots/uj201_follower", ROBOT_ID)
leader_calib = load_calibration("teleoperators/uj201_leader", ROBOT_ID)

if follower_calib:
    print(f"✅ Follower calibration loaded")
    if "wrist_roll" in follower_calib.get("homing_offset", {}):
        print(f"   Wrist Roll Homing Offset: {follower_calib['homing_offset']['wrist_roll']}")
    else:
        print(f"   ⚠️  No wrist_roll homing offset found in calibration")
else:
    print(f"❌ Follower calibration not found")

if leader_calib:
    print(f"✅ Leader calibration loaded")
    if "wrist_roll" in leader_calib.get("homing_offset", {}):
        print(f"   Wrist Roll Homing Offset: {leader_calib['homing_offset']['wrist_roll']}")
    else:
        print(f"   ⚠️  No wrist_roll homing offset found in calibration")
else:
    print(f"❌ Leader calibration not found")

# ============================================================================
# TEST 2: Direct Motor Bus Communication
# ============================================================================
print_section("TEST 2: Testing Direct Motor Communication")

# Create minimal motor configuration for wrist_roll only (used by both tests)
norm_mode_body = MotorNormMode.RANGE_M100_100
wrist_roll_config = {
    "wrist_roll": Motor(WRIST_ROLL_MOTOR_ID, "sts3215", norm_mode_body)
}

print(f"\n🔍 Testing Follower Motor ID {WRIST_ROLL_MOTOR_ID}...")
try:
    follower_bus = FeetechMotorsBus(port=FOLLOWER_PORT, motors=wrist_roll_config)
    follower_bus.connect()
    print(f"✅ Follower bus connected")
    
    # Test ping
    print(f"   Testing ping to motor {WRIST_ROLL_MOTOR_ID}...", end=" ")
    ping_result = follower_bus.ping(WRIST_ROLL_MOTOR_ID)
    if ping_result:
        print("✅ PING successful")
    else:
        print("❌ PING failed")
    
    # Test read position
    try:
        print(f"   Reading current position...", end=" ")
        position = follower_bus.read("Present_Position", WRIST_ROLL_MOTOR_ID)
        print(f"✅ Position: {position}")
    except Exception as e:
        print(f"❌ Failed to read position: {e}")
    
    # Test read current
    try:
        print(f"   Reading current...", end=" ")
        current = follower_bus.read("Present_Current", WRIST_ROLL_MOTOR_ID)
        print(f"✅ Current: {current}")
    except Exception as e:
        print(f"❌ Failed to read current: {e}")
    
    # Test read temperature
    try:
        print(f"   Reading temperature...", end=" ")
        temp = follower_bus.read("Present_Temperature", WRIST_ROLL_MOTOR_ID)
        print(f"✅ Temperature: {temp}°C")
    except Exception as e:
        print(f"❌ Failed to read temperature: {e}")
    
    # Test read voltage
    try:
        print(f"   Reading voltage...", end=" ")
        voltage = follower_bus.read("Present_Voltage", WRIST_ROLL_MOTOR_ID)
        print(f"✅ Voltage: {voltage}V")
    except Exception as e:
        print(f"❌ Failed to read voltage: {e}")
    
    # Check torque status
    try:
        print(f"   Checking torque enable...", end=" ")
        torque = follower_bus.read("Torque_Enable", WRIST_ROLL_MOTOR_ID)
        print(f"✅ Torque Enable: {torque}")
    except Exception as e:
        print(f"❌ Failed to read torque status: {e}")
    
    follower_bus.disconnect()
    print(f"✅ Follower bus disconnected")
    
except Exception as e:
    print(f"❌ Follower motor test failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🔍 Testing Leader Motor ID {WRIST_ROLL_MOTOR_ID}...")
try:
    leader_bus = FeetechMotorsBus(port=LEADER_PORT, motors=wrist_roll_config)
    leader_bus.connect()
    print(f"✅ Leader bus connected")
    
    # Test ping
    print(f"   Testing ping to motor {WRIST_ROLL_MOTOR_ID}...", end=" ")
    ping_result = leader_bus.ping(WRIST_ROLL_MOTOR_ID)
    if ping_result:
        print("✅ PING successful")
    else:
        print("❌ PING failed")
    
    # Test read position
    try:
        print(f"   Reading current position...", end=" ")
        position = leader_bus.read("Present_Position", WRIST_ROLL_MOTOR_ID)
        print(f"✅ Position: {position}")
    except Exception as e:
        print(f"❌ Failed to read position: {e}")
    
    leader_bus.disconnect()
    print(f"✅ Leader bus disconnected")
    
except Exception as e:
    print(f"❌ Leader motor test failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 3: Robot-Level Communication
# ============================================================================
print_section("TEST 3: Testing Robot-Level Communication")

print("\n🤖 Testing Follower Robot...")
try:
    follower_config = UJ201FollowerConfig(
        port=FOLLOWER_PORT,
        id=ROBOT_ID
    )
    follower = UJ201Follower(follower_config)
    follower.connect(calibrate=False)
    print("✅ Follower robot connected")
    
    # Get observation
    print("   Reading observation...", end=" ")
    obs = follower.get_observation()
    print(f"✅ Observation keys: {list(obs.keys())}")
    
    if "wrist_roll" in obs:
        print(f"   Wrist Roll Position: {obs['wrist_roll']}")
    else:
        print(f"   ⚠️  'wrist_roll' not in observation")
    
    follower.disconnect()
    print("✅ Follower robot disconnected")
    
except Exception as e:
    print(f"❌ Follower robot test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎮 Testing Leader Teleoperator...")
try:
    leader_config = UJ201LeaderConfig(
        port=LEADER_PORT,
        id=ROBOT_ID
    )
    leader = UJ201Leader(leader_config)
    leader.connect(calibrate=False)
    print("✅ Leader teleoperator connected")
    
    # Get action
    print("   Reading action...", end=" ")
    action = leader.get_action()
    print(f"✅ Action keys: {list(action.keys())}")
    
    if "wrist_roll" in action:
        print(f"   Wrist Roll Action: {action['wrist_roll']}")
    else:
        print(f"   ⚠️  'wrist_roll' not in action")
    
    leader.disconnect()
    print("✅ Leader teleoperator disconnected")
    
except Exception as e:
    print(f"❌ Leader teleoperator test failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 4: Manual Position Test
# ============================================================================
print_section("TEST 4: Manual Position Test")

print("\n⚠️  This test will attempt to move the wrist_roll motor.")
response = input("Do you want to proceed? (yes/no): ")

if response.lower() == 'yes':
    print("\n🔧 Testing manual position control...")
    try:
        follower_config = UJ201FollowerConfig(
            port=FOLLOWER_PORT,
            id=ROBOT_ID
        )
        follower = UJ201Follower(follower_config)
        follower.connect(calibrate=False)
        
        # Get current position
        current_obs = follower.get_observation()
        current_pos = current_obs.get("wrist_roll", 0.0)
        print(f"   Current wrist_roll position: {current_pos}")
        
        # Test small movements
        test_positions = [current_pos + 0.1, current_pos - 0.1, current_pos]
        
        for i, target_pos in enumerate(test_positions, 1):
            print(f"\n   Test {i}/3: Moving to {target_pos:.3f}...")
            
            # Create action with only wrist_roll
            action = {key: current_obs[key] for key in current_obs.keys()}
            action["wrist_roll"] = target_pos
            
            # Send command
            follower.send_action(action)
            time.sleep(0.5)
            
            # Read back position
            new_obs = follower.get_observation()
            actual_pos = new_obs.get("wrist_roll", 0.0)
            print(f"   Actual position: {actual_pos:.3f}")
            
            error = abs(actual_pos - target_pos)
            if error < 0.05:
                print(f"   ✅ Position reached (error: {error:.3f})")
            else:
                print(f"   ⚠️  Large error: {error:.3f}")
        
        follower.disconnect()
        print("\n✅ Manual position test completed")
        
    except Exception as e:
        print(f"❌ Manual position test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   ⏭️  Skipped manual position test")

# ============================================================================
# TEST 5: Leader-Follower Synchronization
# ============================================================================
print_section("TEST 5: Leader-Follower Synchronization Test")

print("\n⚠️  This test will read from leader and send to follower.")
response = input("Do you want to proceed? (yes/no): ")

if response.lower() == 'yes':
    print("\n🔄 Testing leader-follower sync for wrist_roll...")
    try:
        # Connect both
        follower_config = UJ201FollowerConfig(port=FOLLOWER_PORT, id=ROBOT_ID)
        leader_config = UJ201LeaderConfig(port=LEADER_PORT, id=ROBOT_ID)
        
        follower = UJ201Follower(follower_config)
        leader = UJ201Leader(leader_config)
        
        follower.connect(calibrate=False)
        leader.connect(calibrate=False)
        
        print("✅ Both devices connected")
        print("\n📊 Monitoring wrist_roll synchronization (5 seconds)...")
        print("   Move the leader wrist_roll and observe follower response\n")
        
        start_time = time.time()
        while time.time() - start_time < 5.0:
            # Get leader action
            leader_action = leader.get_action()
            leader_wrist = leader_action.get("wrist_roll", 0.0)
            
            # Send to follower
            follower.send_action(leader_action)
            
            # Read follower position
            follower_obs = follower.get_observation()
            follower_wrist = follower_obs.get("wrist_roll", 0.0)
            
            # Calculate error
            error = abs(follower_wrist - leader_wrist)
            
            # Print status
            print(f"   Leader: {leader_wrist:6.3f} | Follower: {follower_wrist:6.3f} | Error: {error:6.3f}", end="\r")
            
            time.sleep(0.02)  # 50 Hz
        
        print("\n")
        follower.disconnect()
        leader.disconnect()
        print("✅ Synchronization test completed")
        
    except Exception as e:
        print(f"❌ Synchronization test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   ⏭️  Skipped synchronization test")

# ============================================================================
# Summary
# ============================================================================
print_section("🎯 Diagnostic Summary")

print("""
Review the test results above to diagnose wrist_roll issues:

1. Calibration Files: Ensure homing offsets are present
2. Motor Communication: Verify ping and reads are successful
3. Robot Communication: Check observation and action include wrist_roll
4. Manual Position: Confirm motor responds to commands
5. Synchronization: Validate leader-follower tracking

Common Issues:
- Motor not responding: Check wiring and motor ID
- Large position errors: Recalibrate the motor
- Missing in observation: Check robot configuration
- Sync issues: Verify leader and follower calibration match
""")

print("=" * 80)
print("✅ Diagnostic script completed")
print("=" * 80)
