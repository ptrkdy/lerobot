#!/usr/bin/env python

"""
Basic Feetech motor communication test.
Tests direct serial communication with motors using Feetech protocol.
"""

import time
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

print("=" * 80)
print("🔌 Basic Feetech Motor Communication Test")
print("=" * 80)

FOLLOWER_PORT = "/dev/tty.usbmodem5A7A0565221"

# Test with minimal motor configuration
print(f"\n1️⃣  Testing with minimal motor config...")

# Create dummy calibration for all motors (keyed by motor NAME, not ID)
from lerobot.motors.motors_bus import MotorCalibration

motor_names_list = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"]

dummy_calibration = {
    name: MotorCalibration(
        id=i,
        drive_mode=0,
        homing_offset=0,
        range_min=0,
        range_max=4095
    ) for i, name in enumerate(motor_names_list, 1)
}

# Try different motor configurations to find what's actually connected
test_configs = [
    # Config 1: Standard UJ201 motors (IDs 1-9) with calibration
    {
        "m1": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "m2": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "m3": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "m4": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "m5": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "m6": Motor(6, "sts3215", MotorNormMode.DEGREES),
        "m7": Motor(7, "sts3215", MotorNormMode.DEGREES),
        "m8": Motor(8, "sts3215", MotorNormMode.DEGREES),
        "m9": Motor(9, "sts3215", MotorNormMode.DEGREES),
    }
]

for config_num, motors_config in enumerate(test_configs, 1):
    print(f"\n🔍 Trying configuration {config_num}...")
    try:
        bus = FeetechMotorsBus(port=FOLLOWER_PORT, motors=motors_config, calibration=dummy_calibration)
        print(f"   Creating bus... ✅")
        
        bus.connect()
        print(f"   Connected to port ✅")
    
        print(f"\n   Bus properties:")
        print(f"   - Port: {bus.port}")
        print(f"   - Connected: {bus.is_connected}")
        
        # Try to scan for motors using their configured names
        print(f"\n2️⃣  Reading motors using configured names...")
        found = []
        
        motor_names = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"]
        
        for motor_name in motor_names:
            motor_id = int(motor_name[1])  # Extract ID from name
            try:
                result = bus.read("Present_Position", motor_name)
                print(f"   ✅ {motor_name} (ID {motor_id}) found! Position: {result}°")
                found.append(motor_name)
            except KeyError as e:
                print(f"   ⚠️  {motor_name}: KeyError - {e}")
            except Exception as e:
                print(f"   ❌ {motor_name}: {type(e).__name__} - {e}")
        
        print(f"\n📊 Found {len(found)} motors: {found}")
        
        bus.disconnect()
        print(f"\n✅ Test complete")
        break  # Success, exit loop
        
    except Exception as e:
        print(f"   ❌ Configuration {config_num} failed: {e}")
        continue  # Try next configuration

print("=" * 80)
