#!/usr/bin/env python

"""
Comprehensive motor scan to find which motors actually respond.
This will ping all possible motor IDs to find what's connected.
"""

import time
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

FOLLOWER_PORT = "/dev/tty.usbmodem5A7A0565221"

print("=" * 80)
print("🔍 Scanning for Active Motors on Follower Port")
print("=" * 80)

# Create a minimal bus with one motor just to connect
dummy_motor = {"dummy": Motor(1, "sts3215", MotorNormMode.DEGREES)}

try:
    print(f"\n🔌 Connecting to {FOLLOWER_PORT}...")
    bus = FeetechMotorsBus(port=FOLLOWER_PORT, motors=dummy_motor)
    bus.connect()
    print("✅ Bus connected successfully\n")
    
    print("🔍 Scanning motor IDs 1-12...")
    print(f"{'ID':<5} {'Ping':<8} {'Position':<12} {'Voltage':<10} {'Status':<10}")
    print("-" * 60)
    
    found_motors = []
    
    for motor_id in range(1, 13):
        try:
            # Try to ping
            ping_success = bus.ping(motor_id)
            
            if ping_success:
                # Try to read position
                try:
                    position = bus.read("Present_Position", motor_id, num_retry=1)
                    position_str = f"{position}"
                except Exception as e:
                    position_str = f"Error"
                
                # Try to read voltage
                try:
                    voltage = bus.read("Present_Voltage", motor_id, num_retry=1)
                    voltage_str = f"{voltage}V"
                except Exception as e:
                    voltage_str = "Error"
                
                status = "✅ FOUND"
                found_motors.append(motor_id)
                print(f"{motor_id:<5} {'✅':<8} {position_str:<12} {voltage_str:<10} {status:<10}")
            else:
                print(f"{motor_id:<5} {'❌':<8} {'-':<12} {'-':<10} {'No response':<10}")
                
        except Exception as e:
            print(f"{motor_id:<5} {'❌':<8} {'-':<12} {'-':<10} {f'Error: {e}':<10}")
    
    print("\n" + "=" * 80)
    print(f"📊 Summary: Found {len(found_motors)} motor(s): {found_motors}")
    print("=" * 80)
    
    # If motors found, try to read detailed info
    if found_motors:
        print("\n🔍 Detailed Information for Found Motors:\n")
        
        for motor_id in found_motors:
            print(f"Motor ID {motor_id}:")
            
            # Try reading various registers
            registers = [
                ("Torque_Enable", "Torque"),
                ("Present_Position", "Position"),
                ("Goal_Position", "Goal"),
                ("Present_Current", "Current"),
                ("Present_Voltage", "Voltage"),
                ("Present_Temperature", "Temp"),
                ("Moving", "Moving"),
            ]
            
            for reg_name, display_name in registers:
                try:
                    value = bus.read(reg_name, motor_id, num_retry=1)
                    print(f"   {display_name}: {value}")
                except Exception as e:
                    print(f"   {display_name}: ❌ {e}")
            print()
    
    bus.disconnect()
    print("✅ Scan complete")
    
except Exception as e:
    print(f"❌ Scan failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
