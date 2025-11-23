#!/usr/bin/env python

"""
Test script for UJ201 gearbox and shoulder joint rotation.

UPDATED Motor ID Mapping (daisy chain physical IDs vs joint indices):
- Shoulder joint (config joint 7) = Physical Motor ID 3
- Gearbox joint (config joint 8) = Physical Motor ID 2

IMPORTANT: Gearbox and shoulder are mechanically coupled!
- Gearbox moves 2x shoulder in opposite direction
- They must move in coordination to avoid mechanical stress

Usage:
    python test_gearbox_shoulder.py
"""

import time
import numpy as np
from datetime import datetime
from pathlib import Path
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration

# Output file configuration
OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = OUTPUT_DIR / f"gearbox_shoulder_test_{timestamp}.txt"

def log_and_print(message, file_handle=None):
    """Print to console and write to file."""
    print(message)
    if file_handle:
        file_handle.write(message + "\n")
        file_handle.flush()

# Open output file
output_file = open(OUTPUT_FILE, "w")
log_and_print("=" * 80, output_file)
log_and_print("🔄 UJ201 Gearbox & Shoulder Joint Rotation Test", output_file)
log_and_print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", output_file)
log_and_print("=" * 80, output_file)

# Configuration
FOLLOWER_PORT = "/dev/tty.usbmodem5A7A0565221"

# Mechanical coupling constants
GEAR_RATIO = 2.0
OPPOSITE_DIRECTION = True

# Test parameters
ROTATION_RANGE = 30.0  # Total range in degrees (increased for even larger step size)
NUM_STEPS = 10         # Number of test increments
STEP_DURATION = 1.0
NEGATIVE_DIRECTION = True  # Move in negative direction to avoid frame obstruction

def calculate_coupled_position(shoulder_pos, gearbox_initial, shoulder_initial):
    """Calculate gearbox position from shoulder movement (2x opposite)."""
    shoulder_delta = shoulder_pos - shoulder_initial
    gearbox_delta = GEAR_RATIO * shoulder_delta  # Same direction as shoulder
    return gearbox_initial + gearbox_delta

def calculate_coupled_shoulder(gearbox_pos, shoulder_initial, gearbox_initial):
    """Calculate shoulder position from gearbox movement."""
    gearbox_delta = gearbox_pos - gearbox_initial
    shoulder_delta = gearbox_delta / GEAR_RATIO  # Same direction, not opposite
    return shoulder_initial + shoulder_delta

def print_section(title):
    """Print section header."""
    log_and_print("\n" + "=" * 80, output_file)
    log_and_print(f"  {title}", output_file)
    log_and_print("=" * 80, output_file)

# ============================================================================
# Connect to Motors
# ============================================================================
print_section("Connecting to Motors")

norm_mode = MotorNormMode.DEGREES
motors_config = {
    "shoulder": Motor(3, "sts3215", norm_mode),  # Physical ID 3 = Joint 7 (shoulder)
    "gearbox": Motor(2, "sts3215", norm_mode),   # Physical ID 2 = Joint 8 (gearbox)
}

# Create calibration for physical motors 2 and 3
calibration = {
    "shoulder": MotorCalibration(id=3, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
    "gearbox": MotorCalibration(id=2, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
}

try:
    log_and_print(f"🔌 Connecting to {FOLLOWER_PORT}...", output_file)
    bus = FeetechMotorsBus(port=FOLLOWER_PORT, motors=motors_config, calibration=calibration)
    bus.connect()
    log_and_print(f"✅ Connected successfully!", output_file)
except Exception as e:
    log_and_print(f"❌ Failed to connect: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)
    output_file.close()
    exit(1)

# ============================================================================
# Read Initial Positions
# ============================================================================
print_section("Reading Initial Positions")

try:
    shoulder_initial = bus.read("Present_Position", "shoulder")
    gearbox_initial = bus.read("Present_Position", "gearbox")
    
    log_and_print(f"📊 Initial Positions:", output_file)
    log_and_print(f"   Shoulder (physical ID 3): {shoulder_initial:.2f}°", output_file)
    log_and_print(f"   Gearbox (physical ID 2):  {gearbox_initial:.2f}°", output_file)
except Exception as e:
    log_and_print(f"❌ Failed to read initial positions: {e}", output_file)
    bus.disconnect()
    output_file.close()
    exit(1)

# ============================================================================
# Test Selection Menu
# ============================================================================
print_section("Test Selection")

log_and_print("\nAvailable tests:", output_file)
log_and_print("  1. Gearbox-driven test (gearbox moves, shoulder follows coupling)", output_file)
log_and_print("  2. Shoulder-driven test (shoulder moves, gearbox follows coupling)", output_file)
log_and_print("  3. Simultaneous coordinated test (both move together)", output_file)
log_and_print("  4. Run all tests", output_file)

test_choice = input("\nSelect test to run (1/2/3/4): ")
log_and_print(f"User selected: {test_choice}", output_file)

run_gearbox = test_choice in ['1', '4']
run_shoulder = test_choice in ['2', '4']
run_simultaneous = test_choice in ['3', '4']

# ============================================================================
# Test 1: Gearbox Rotation (Physical Motor ID 2)
# ============================================================================
if run_gearbox:
    print_section("Test 1: Gearbox-Driven Joint Rotation (Physical ID 2)")

    log_and_print(f"\n⚠️  IMPORTANT: Motors are mechanically coupled!", output_file)
    log_and_print(f"   - Gearbox (phys ID 2) moves 2x shoulder (phys ID 3) in same direction", output_file)
    log_and_print(f"\n   This will rotate gearbox -{ROTATION_RANGE:.1f}° (from initial to lower position)", output_file)
    log_and_print(f"   Shoulder will move -{ROTATION_RANGE/2:.1f}° (coupled, same direction)", output_file)
    log_and_print(f"   Using {NUM_STEPS} test increments in each direction", output_file)
    
    if test_choice != '4':
        response = input("\nProceed with gearbox test? (yes/no): ")
        log_and_print(f"User response: {response}", output_file)
    else:
        response = 'yes'
        log_and_print(f"Auto-proceeding (running all tests)", output_file)

    if response.lower() == 'yes':
        log_and_print(f"\n🔄 Rotating gearbox (negative direction, moving down from initial)...", output_file)
    
    try:
        # Move in negative direction (e.g., 36.53° -> 26.53°)
        gearbox_positions = np.concatenate([
            np.linspace(gearbox_initial, gearbox_initial - ROTATION_RANGE, NUM_STEPS),  # Move negative
            np.linspace(gearbox_initial - ROTATION_RANGE, gearbox_initial, NUM_STEPS)   # Return to initial
        ])
        
        log_and_print(f"\n{'Step':<6} {'GB Target°':<12} {'GB Actual°':<12} {'SH Target°':<12} {'SH Actual°':<12} {'Status':<6}", output_file)
        log_and_print("-" * 70, output_file)
        
        for i, gearbox_target in enumerate(gearbox_positions, 1):
            shoulder_target = calculate_coupled_shoulder(gearbox_target, shoulder_initial, gearbox_initial)
            
            bus.write("Goal_Position", "gearbox", gearbox_target)
            bus.write("Goal_Position", "shoulder", shoulder_target)
            time.sleep(STEP_DURATION)
            
            gb_actual = bus.read("Present_Position", "gearbox")
            sh_actual = bus.read("Present_Position", "shoulder")
            
            gb_error = abs(gb_actual - gearbox_target)
            sh_error = abs(sh_actual - shoulder_target)
            status = "✅" if (gb_error < 2.0 and sh_error < 2.0) else "⚠️"
            
            log_and_print(f"{i:<6} {gearbox_target:<12.2f} {gb_actual:<12.2f} {shoulder_target:<12.2f} {sh_actual:<12.2f} {status:<6}", output_file)
        
        log_and_print(f"\n✅ Gearbox test completed", output_file)
    except Exception as e:
        log_and_print(f"❌ Gearbox test failed: {e}", output_file)
        import traceback
        log_and_print(traceback.format_exc(), output_file)
    else:
        log_and_print("⏭️  Skipped gearbox test", output_file)

    # Return to initial
    log_and_print(f"\n🏠 Returning to initial...", output_file)
    try:
        bus.write("Goal_Position", "gearbox", gearbox_initial)
        bus.write("Goal_Position", "shoulder", shoulder_initial)
        time.sleep(1.0)
        log_and_print(f"✅ Returned", output_file)
    except Exception as e:
        log_and_print(f"⚠️  Failed: {e}", output_file)

# ============================================================================
# Test 2: Shoulder Rotation (Physical Motor ID 3)
# ============================================================================
if run_shoulder:
    print_section("Test 2: Shoulder-Driven Joint Rotation (Physical ID 3)")

    log_and_print(f"\n⚠️  This will rotate shoulder -{ROTATION_RANGE:.1f}° (from initial to lower position)", output_file)
    log_and_print(f"   Gearbox will move -{ROTATION_RANGE*2:.1f}° (coupled, same direction)", output_file)
    log_and_print(f"   Using {NUM_STEPS} test increments in each direction", output_file)
    
    if test_choice != '4':
        response = input("\nProceed with shoulder test? (yes/no): ")
        log_and_print(f"User response: {response}", output_file)
    else:
        response = 'yes'
        log_and_print(f"Auto-proceeding (running all tests)", output_file)

    if response.lower() == 'yes':
        log_and_print(f"\n🔄 Rotating shoulder (negative direction, moving down from initial)...", output_file)
    
    try:
        # Move in negative direction (e.g., -34.68° -> -44.68°)
        shoulder_positions = np.concatenate([
            np.linspace(shoulder_initial, shoulder_initial - ROTATION_RANGE, NUM_STEPS),  # Move negative
            np.linspace(shoulder_initial - ROTATION_RANGE, shoulder_initial, NUM_STEPS)   # Return to initial
        ])
        
        log_and_print(f"\n{'Step':<6} {'SH Target°':<12} {'SH Actual°':<12} {'GB Target°':<12} {'GB Actual°':<12} {'Status':<6}", output_file)
        log_and_print("-" * 70, output_file)
        
        for i, shoulder_target in enumerate(shoulder_positions, 1):
            gearbox_target = calculate_coupled_position(shoulder_target, gearbox_initial, shoulder_initial)
            
            bus.write("Goal_Position", "shoulder", shoulder_target)
            bus.write("Goal_Position", "gearbox", gearbox_target)
            time.sleep(STEP_DURATION)
            
            sh_actual = bus.read("Present_Position", "shoulder")
            gb_actual = bus.read("Present_Position", "gearbox")
            
            sh_error = abs(sh_actual - shoulder_target)
            gb_error = abs(gb_actual - gearbox_target)
            status = "✅" if (sh_error < 2.0 and gb_error < 2.0) else "⚠️"
            
            log_and_print(f"{i:<6} {shoulder_target:<12.2f} {sh_actual:<12.2f} {gearbox_target:<12.2f} {gb_actual:<12.2f} {status:<6}", output_file)
        
        log_and_print(f"\n✅ Shoulder test completed", output_file)
    except Exception as e:
        log_and_print(f"❌ Shoulder test failed: {e}", output_file)
        import traceback
        log_and_print(traceback.format_exc(), output_file)
    else:
        log_and_print("⏭️  Skipped shoulder test", output_file)

    # Return to initial
    log_and_print(f"\n🏠 Returning to initial...", output_file)
    try:
        bus.write("Goal_Position", "gearbox", gearbox_initial)
        bus.write("Goal_Position", "shoulder", shoulder_initial)
        time.sleep(1.0)
        log_and_print(f"✅ Returned", output_file)
    except Exception as e:
        log_and_print(f"⚠️  Failed: {e}", output_file)

# ============================================================================
# Test 3: Simultaneous Coordinated Movement
# ============================================================================
if run_simultaneous:
    print_section("Test 3: Simultaneous Coordinated Movement")

    log_and_print(f"\n⚠️  Both motors move together in coordinated pattern", output_file)
    log_and_print(f"   - Shoulder moves +{ROTATION_RANGE:.1f}° (positive direction)", output_file)
    log_and_print(f"   - Gearbox moves -{ROTATION_RANGE*2:.1f}° (negative direction, 2x shoulder)", output_file)
    log_and_print(f"   - Tests coordinated movement with opposite directions at 2:1 ratio", output_file)
    log_and_print(f"   Using {NUM_STEPS} test increments in each direction", output_file)
    
    if test_choice != '4':
        response = input("\nProceed with simultaneous test? (yes/no): ")
        log_and_print(f"User response: {response}", output_file)
    else:
        response = 'yes'
        log_and_print(f"Auto-proceeding (running all tests)", output_file)

    if response.lower() == 'yes':
        log_and_print(f"\n🔄 Moving both motors simultaneously...", output_file)
        
        try:
            # Motors move in opposite directions at 2:1 ratio (gearbox moves 2x shoulder)
            shoulder_positions = np.concatenate([
                np.linspace(shoulder_initial, shoulder_initial + ROTATION_RANGE, NUM_STEPS),  # Shoulder positive
                np.linspace(shoulder_initial + ROTATION_RANGE, shoulder_initial, NUM_STEPS)   # Return to initial
            ])
            
            gearbox_positions = np.concatenate([
                np.linspace(gearbox_initial, gearbox_initial - ROTATION_RANGE*2, NUM_STEPS),  # Gearbox negative (2x shoulder)
                np.linspace(gearbox_initial - ROTATION_RANGE*2, gearbox_initial, NUM_STEPS)   # Return to initial
            ])
            
            log_and_print(f"\n{'Step':<6} {'SH Target°':<12} {'SH Actual°':<12} {'GB Target°':<12} {'GB Actual°':<12} {'Δ Ratio':<10} {'Status':<6}", output_file)
            log_and_print("-" * 80, output_file)
            
            for i, (shoulder_target, gearbox_target) in enumerate(zip(shoulder_positions, gearbox_positions), 1):
                bus.write("Goal_Position", "shoulder", shoulder_target)
                bus.write("Goal_Position", "gearbox", gearbox_target)
                time.sleep(STEP_DURATION)
                
                sh_actual = bus.read("Present_Position", "shoulder")
                gb_actual = bus.read("Present_Position", "gearbox")
                
                sh_error = abs(sh_actual - shoulder_target)
                gb_error = abs(gb_actual - gearbox_target)
                
                # Check movement ratio for opposite directions (should be negative ratio)
                sh_delta = sh_actual - shoulder_initial
                gb_delta = gb_actual - gearbox_initial
                ratio = (gb_delta / sh_delta) if abs(sh_delta) > 0.1 else 0.0  # Expect negative for opposite
                
                status = "✅" if (sh_error < 2.0 and gb_error < 2.0) else "⚠️"
                
                log_and_print(f"{i:<6} {shoulder_target:<12.2f} {sh_actual:<12.2f} {gearbox_target:<12.2f} {gb_actual:<12.2f} {ratio:<10.2f} {status:<6}", output_file)
            
            log_and_print(f"\n✅ Simultaneous test completed", output_file)
        except Exception as e:
            log_and_print(f"❌ Simultaneous test failed: {e}", output_file)
            import traceback
            log_and_print(traceback.format_exc(), output_file)
    else:
        log_and_print("⏭️  Skipped simultaneous test", output_file)

# ============================================================================
# Cleanup
# ============================================================================
print_section("Cleanup")

try:
    log_and_print("🏠 Returning to initial positions...", output_file)
    bus.write("Goal_Position", "shoulder", shoulder_initial)
    bus.write("Goal_Position", "gearbox", gearbox_initial)
    time.sleep(1.0)
    
    final_shoulder = bus.read("Present_Position", "shoulder")
    final_gearbox = bus.read("Present_Position", "gearbox")
    
    log_and_print(f"\n📊 Final Positions:", output_file)
    log_and_print(f"   Shoulder: {final_shoulder:.2f}° (initial: {shoulder_initial:.2f}°)", output_file)
    log_and_print(f"   Gearbox:  {final_gearbox:.2f}° (initial: {gearbox_initial:.2f}°)", output_file)
    
    bus.disconnect()
    log_and_print(f"\n✅ Disconnected successfully", output_file)
except Exception as e:
    log_and_print(f"⚠️  Cleanup failed: {e}", output_file)
    import traceback
    log_and_print(traceback.format_exc(), output_file)

log_and_print("\n" + "=" * 80, output_file)
log_and_print("✅ Test completed", output_file)
log_and_print(f"Test ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", output_file)
log_and_print("=" * 80, output_file)
log_and_print(f"\n📄 Results saved to: {OUTPUT_FILE}", output_file)

output_file.close()
print(f"\n✅ Full test log saved to: {OUTPUT_FILE}")
