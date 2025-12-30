#!/usr/bin/env python

"""
Scan for available motor IDs on UJ201 hardware.
This helps identify which motors are actually connected.
"""

import sys
import traceback
from lerobot.motors.feetech import FeetechMotorsBus

def scan_motors_on_port(port):
    """Scan for motors on a specific port."""
    print(f"\n🔍 Scanning motors on port: {port}")
    
    try:
        # Create a minimal bus without predefined motors
        bus = FeetechMotorsBus(port=port, motors={})
        print("  Attempting to connect to bus...")
        bus.connect()
        
        print("✅ Port connected successfully")
        
        # First, try to ping with broadcast
        print("  Testing broadcast communication...")
        try:
            # Try a simple ping to see if any motor responds
            result = bus.ping(254)  # 254 is often broadcast ID
            print(f"  Broadcast ping result: {result}")
        except Exception as e:
            print(f"  Broadcast ping failed: {e}")
        
        # Scan for motor IDs 0-254 (full range)
        found_motors = []
        print("  Scanning motor IDs (this may take a moment)...")
        
        # Test key ID ranges
        test_ranges = [
            range(1, 10),    # Expected UJ201 range
            range(10, 20),   # Extended range
            range(20, 30),   # Higher range
            range(100, 110), # Much higher range
            [0, 254]         # Special IDs
        ]
        
        for id_range in test_ranges:
            for motor_id in id_range:
                try:
                    print(f"    Checking ID {motor_id}...", end="", flush=True)
                    
                    # Try ping first (less intrusive)
                    ping_result = bus.ping(motor_id)
                    if ping_result:
                        print(f" 🏓 PING ✅")
                        
                        # If ping works, try to read model number
                        try:
                            model = bus.read("Model_Number", motor_id, num_retry=1)
                            print(f"      Model: {model}")
                            found_motors.append(motor_id)
                        except Exception as e:
                            print(f"      Model read failed: {e}")
                            # Still count as found if ping worked
                            found_motors.append(motor_id)
                    else:
                        print(f" ❌")
                        
                except Exception as e:
                    print(f" ❌ ({str(e)[:30]})")
                    continue
        
        # Try alternative communication test
        if not found_motors:
            print("\n  🔧 No motors found with standard method, trying alternative approach...")
            try:
                # Try to read from the bus at hardware level
                bus_status = bus.is_connected
                print(f"  Bus connection status: {bus_status}")
                
                # Check if we can do any low-level communication
                for test_id in [1, 2, 254]:
                    try:
                        print(f"    Testing raw communication with ID {test_id}...")
                        # Try a very basic read
                        result = bus._read_bytes(test_id, 0, 1, num_retry=1)
                        print(f"    Raw read from ID {test_id}: {result}")
                        if result is not None:
                            found_motors.append(test_id)
                    except Exception as e:
                        print(f"    Raw read from ID {test_id} failed: {e}")
                        
            except Exception as e:
                print(f"  Alternative approach failed: {e}")
        
        bus.disconnect()
        
        print(f"\n📋 Summary for {port}:")
        print(f"Found {len(found_motors)} motors: {found_motors}")
        
        return found_motors
        
    except Exception as e:
        print(f"❌ Failed to scan {port}: {e}")
        traceback.print_exc()
        return []

def test_basic_connectivity():
    """Test if the ports are accessible at all."""
    print("🔌 Testing basic port connectivity...")
    
    follower_port = "/dev/tty.usbmodem5A7A0565221"
    leader_port = "/dev/tty.usbmodem5A680092181"
    
    import os
    
    # Check if ports exist
    for port_name, port in [("Follower", follower_port), ("Leader", leader_port)]:
        print(f"  {port_name} port ({port}):", end=" ")
        
        if os.path.exists(port):
            print("✅ Exists")
            
            # Check if readable
            try:
                with open(port, 'rb') as f:
                    print(f"    ✅ Accessible")
            except PermissionError:
                print(f"    ⚠️  Permission denied - try: sudo chmod 666 {port}")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        else:
            print("❌ Does not exist")
            
            # Suggest available ports
            print("    Available tty ports:")
            try:
                tty_ports = [p for p in os.listdir('/dev') if p.startswith('tty.usbmodem')]
                for p in tty_ports[:5]:  # Show first 5
                    print(f"      /dev/{p}")
            except:
                print("      Could not list ports")

def scan_both_ports():
    """Scan both UJ201 ports for motors."""
    print("🔧 UJ201 Motor ID Scanner")
    print("=" * 50)
    
    # First test basic connectivity
    test_basic_connectivity()
    
    follower_port = "/dev/tty.usbmodem5A7A0565221"
    leader_port = "/dev/tty.usbmodem5A680092181"
    
    follower_motors = scan_motors_on_port(follower_port)
    leader_motors = scan_motors_on_port(leader_port)
    
    print("\n" + "=" * 50)
    print("🎯 Final Summary:")
    print(f"Follower ({follower_port}): {follower_motors}")
    print(f"Leader ({leader_port}): {leader_motors}")
    
    # Expected motor IDs for UJ201
    expected_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    print(f"\nExpected IDs: {expected_ids}")
    
    follower_missing = [id for id in expected_ids if id not in follower_motors]
    leader_missing = [id for id in expected_ids if id not in leader_motors]
    
    if follower_missing:
        print(f"⚠️  Follower missing IDs: {follower_missing}")
    else:
        print("✅ Follower has all expected IDs")
        
    if leader_missing:
        print(f"⚠️  Leader missing IDs: {leader_missing}")
    else:
        print("✅ Leader has all expected IDs")
    
    return follower_motors, leader_motors

def suggest_motor_mapping(follower_motors, leader_motors):
    """Suggest motor mapping based on found motors."""
    print(f"\n💡 Motor Mapping Suggestions:")
    print("=" * 30)
    
    motor_names = ["shoulder", "gearbox", "universal_joint", "shoulder_pan", 
                   "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    if len(follower_motors) >= 6:  # At least SO101 motors
        print("🤖 For Follower:")
        for i, motor_id in enumerate(follower_motors[:9]):  # Take first 9 found
            motor_name = motor_names[i] if i < len(motor_names) else f"motor_{i+1}"
            print(f"  ID {motor_id}: {motor_name}")
    
    if len(leader_motors) >= 6:  # At least SO101 motors  
        print("\n🎮 For Leader:")
        for i, motor_id in enumerate(leader_motors[:9]):  # Take first 9 found
            motor_name = motor_names[i] if i < len(motor_names) else f"motor_{i+1}"
            print(f"  ID {motor_id}: {motor_name}")

if __name__ == "__main__":
    try:
        follower_motors, leader_motors = scan_both_ports()
        suggest_motor_mapping(follower_motors, leader_motors)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Scan interrupted by user")
    except Exception as e:
        print(f"\n❌ Scan failed: {e}")
        traceback.print_exc()