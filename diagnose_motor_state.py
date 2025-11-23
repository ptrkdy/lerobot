#!/usr/bin/env python

"""
Diagnostic script to check motor configuration and state.
Checks torque enable, operating mode, and other critical parameters.
"""

import time
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

FOLLOWER_PORT = "/dev/tty.usbmodem5A7A0565221"

print("=" * 80)
print("🔍 Motor State Diagnostic Tool")
print("=" * 80)

# Motors to check
motors_to_check = {
    "shoulder": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "gearbox": Motor(2, "sts3215", MotorNormMode.DEGREES),
}

try:
    print(f"\n🔌 Connecting to {FOLLOWER_PORT}...")
    bus = FeetechMotorsBus(port=FOLLOWER_PORT, motors=motors_to_check)
    bus.connect()
    print("✅ Connected successfully\n")
    
    for motor_name, motor_obj in motors_to_check.items():
        motor_id = motor_obj.id
        print("=" * 80)
        print(f"📍 Motor: {motor_name} (ID: {motor_id})")
        print("=" * 80)
        
        # Check torque enable
        try:
            torque = bus.read("Torque_Enable", motor_id)
            print(f"   Torque Enable: {torque} {'✅ ENABLED' if torque else '❌ DISABLED'}")
        except Exception as e:
            print(f"   Torque Enable: ❌ Error reading - {e}")
        
        # Check operating mode
        try:
            mode = bus.read("Operating_Mode", motor_id)
            mode_str = {0: "Current", 1: "Velocity", 3: "Position", 4: "Extended Position", 
                       5: "Current-based Position", 16: "PWM"}.get(mode, f"Unknown ({mode})")
            print(f"   Operating Mode: {mode_str}")
        except Exception as e:
            print(f"   Operating Mode: Error reading - {e}")
        
        # Check current position
        try:
            position = bus.read("Present_Position", motor_id)
            print(f"   Present Position: {position}")
        except Exception as e:
            print(f"   Present Position: Error reading - {e}")
        
        # Check goal position
        try:
            goal = bus.read("Goal_Position", motor_id)
            print(f"   Goal Position: {goal}")
        except Exception as e:
            print(f"   Goal Position: Error reading - {e}")
        
        # Check if moving
        try:
            moving = bus.read("Moving", motor_id)
            print(f"   Moving: {moving} {'🔄 YES' if moving else '⏸️  NO'}")
        except Exception as e:
            print(f"   Moving: Error reading - {e}")
        
        # Check current
        try:
            current = bus.read("Present_Current", motor_id)
            print(f"   Present Current: {current} mA")
        except Exception as e:
            print(f"   Present Current: Error reading - {e}")
        
        # Check voltage
        try:
            voltage = bus.read("Present_Voltage", motor_id)
            print(f"   Present Voltage: {voltage} V")
        except Exception as e:
            print(f"   Present Voltage: Error reading - {e}")
        
        # Check temperature
        try:
            temp = bus.read("Present_Temperature", motor_id)
            print(f"   Temperature: {temp}°C")
        except Exception as e:
            print(f"   Temperature: Error reading - {e}")
        
        # Check hardware error status
        try:
            hw_error = bus.read("Hardware_Error_Status", motor_id)
            print(f"   Hardware Error: {hw_error} {'❌ ERROR' if hw_error else '✅ OK'}")
        except Exception as e:
            print(f"   Hardware Error: Error reading - {e}")
        
        print()
    
    # Try to enable torque if disabled
    print("=" * 80)
    print("🔧 Attempting to enable torque on both motors...")
    print("=" * 80)
    
    for motor_name, motor_obj in motors_to_check.items():
        try:
            motor_id = motor_obj.id
            current_torque = bus.read("Torque_Enable", motor_id)
            
            if not current_torque:
                print(f"\n   {motor_name}: Torque currently disabled, enabling...")
                bus.write("Torque_Enable", motor_id, 1)
                time.sleep(0.1)
                new_torque = bus.read("Torque_Enable", motor_id)
                if new_torque:
                    print(f"   {motor_name}: ✅ Torque enabled successfully")
                else:
                    print(f"   {motor_name}: ❌ Failed to enable torque")
            else:
                print(f"   {motor_name}: Torque already enabled")
        except Exception as e:
            print(f"   {motor_name}: ❌ Error enabling torque - {e}")
    
    bus.disconnect()
    print("\n✅ Diagnostic complete")
    
except Exception as e:
    print(f"❌ Diagnostic failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
