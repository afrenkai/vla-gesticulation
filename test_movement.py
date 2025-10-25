#!/usr/bin/env python3
"""
Simple test script for LeRobot device movement (left and right)
Tests basic motor control on LeKiwi robot
"""

import time
import numpy as np
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot


def test_left_right_movement(robot, duration=2.0, amplitude=0.5):
    """
    Test simple left-right movement on the robot
    
    Args:
        robot: LeRobot robot instance
        duration: Duration for each movement in seconds
        amplitude: Movement amplitude (0.0 to 1.0)
    """
    print("Starting left-right movement test...")
    print(f"Duration per movement: {duration}s, Amplitude: {amplitude}")
    
    # Get current position
    initial_state = robot.get_state()
    print(f"Initial state: {initial_state}")
    
    # Define rest/center position (all zeros)
    rest_position = np.zeros(robot.num_motors)
    
    # Define left position (negative shoulder pan)
    left_position = rest_position.copy()
    left_position[0] = -amplitude  # First motor: shoulder pan left
    
    # Define right position (positive shoulder pan)
    right_position = rest_position.copy()
    right_position[0] = amplitude  # First motor: shoulder pan right
    
    try:
        # Move to center
        print("\n1. Moving to center position...")
        robot.send_action(rest_position)
        time.sleep(duration)
        
        # Move left
        print("2. Moving LEFT...")
        robot.send_action(left_position)
        time.sleep(duration)
        
        # Move to center
        print("3. Moving to center...")
        robot.send_action(rest_position)
        time.sleep(duration)
        
        # Move right
        print("4. Moving RIGHT...")
        robot.send_action(right_position)
        time.sleep(duration)
        
        # Return to center
        print("5. Returning to center...")
        robot.send_action(rest_position)
        time.sleep(duration)
        
        # Repeat the sequence 3 times
        print("\n--- Repeating sequence 3 times ---")
        for i in range(3):
            print(f"\nCycle {i+1}/3:")
            
            # Left
            print("  -> LEFT")
            robot.send_action(left_position)
            time.sleep(duration)
            
            # Right
            print("  -> RIGHT")
            robot.send_action(right_position)
            time.sleep(duration)
        
        # Final return to center
        print("\nReturning to rest position...")
        robot.send_action(rest_position)
        time.sleep(1.0)
        
        print("\n✓ Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user!")
        print("Returning to rest position...")
        robot.send_action(rest_position)
        time.sleep(1.0)
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        print("Attempting to return to rest position...")
        try:
            robot.send_action(rest_position)
            time.sleep(1.0)
        except:
            print("Could not return to rest position")


def test_smooth_sweep(robot, duration=5.0, amplitude=0.8, steps=20):
    """
    Test smooth sweeping motion from left to right
    
    Args:
        robot: LeRobot robot instance
        duration: Total duration for sweep in seconds
        amplitude: Movement amplitude (0.0 to 1.0)
        steps: Number of steps in the sweep
    """
    print("\nStarting smooth sweep test...")
    print(f"Duration: {duration}s, Amplitude: {amplitude}, Steps: {steps}")
    
    rest_position = np.zeros(robot.num_motors)
    
    try:
        # Sweep from left to right
        print("Sweeping from LEFT to RIGHT...")
        for i in range(steps):
            # Calculate position along sweep (-amplitude to +amplitude)
            progress = i / (steps - 1)  # 0.0 to 1.0
            position_value = -amplitude + (2 * amplitude * progress)
            
            position = rest_position.copy()
            position[0] = position_value
            
            robot.send_action(position)
            time.sleep(duration / steps)
        
        # Sweep back from right to left
        print("Sweeping from RIGHT to LEFT...")
        for i in range(steps):
            progress = i / (steps - 1)
            position_value = amplitude - (2 * amplitude * progress)
            
            position = rest_position.copy()
            position[0] = position_value
            
            robot.send_action(position)
            time.sleep(duration / steps)
        
        # Return to center
        print("Returning to center...")
        robot.send_action(rest_position)
        time.sleep(1.0)
        
        print("✓ Sweep test completed!")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted!")
        robot.send_action(rest_position)
    except Exception as e:
        print(f"✗ Error: {e}")
        robot.send_action(rest_position)


def main():
    """Main test function"""
    print("=" * 60)
    print("LeRobot Device Movement Test - Left/Right")
    print("=" * 60)
    
    # Robot configuration - adjust these for your setup
    ROBOT_PORT = "/dev/ttyACM0"  # Change this to your robot's port
    
    print(f"\nConnecting to robot on port: {ROBOT_PORT}")
    
    from lerobot.common.robot_devices.robots.configs import LeKiwiRobotConfig
    config = LeKiwiRobotConfig(port=ROBOT_PORT)
    robot = make_robot(config)
        
        # # Option 2: Direct instantiation (replace with your robot class)
        # # For this example, we'll show the structure
        # print("Note: Please configure the robot connection based on your hardware")
        # print("Uncomment the appropriate robot initialization code")
        
        # # Placeholder - replace with actual robot initialization
        # print("\nTo use this script:")
        # print("1. Import your robot class (e.g., LeKiwiRobot)")
        # print("2. Initialize with proper config")
        # print("3. Call test_left_right_movement(robot)")
        # print("\nExample:")
        # print("  from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot")
        # print("  robot = ManipulatorRobot(port=ROBOT_PORT)")
        # print("  robot.connect()")
        # print("  test_left_right_movement(robot, duration=2.0, amplitude=0.5)")
        # print("  test_smooth_sweep(robot, duration=5.0, amplitude=0.8)")
        # print("  robot.disconnect()")
        
        # # Uncomment below when you have robot configured:
        # """
    robot.connect()
        
    test_left_right_movement(robot, duration=2.0, amplitude=0.5)
        
    print("\n" + "=" * 60)
        
        # Test 2: Smooth sweeping motion
    test_smooth_sweep(robot, duration=5.0, amplitude=0.8, steps=20)
        
        # Cleanup
    robot.disconnect()
    print("\nRobot disconnected. Test complete!")

if __name__ == "__main__":
    main()
