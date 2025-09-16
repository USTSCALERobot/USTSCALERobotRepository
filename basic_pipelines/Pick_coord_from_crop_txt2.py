import kinematics as kin
import numpy as np
import phx
import time
import math
import re

# Turn on Phoenix system and initialize resting position
phx.turn_on()
phx.rest_position()

def transform_coordinates(x1, y1):
    """Transform coordinates from System 1 (0-1 scale) to System 2 (10-30 in X, -10 to 10 in Y)."""
    x2 = 20 * x1 + 10
    y2 = 20 * y1 - 10
    return x2, y2


def read_coordinates_from_file(filename):
    # Open the file and read its contents
    with open(filename, 'r') as file:
        content = file.read()

    # Use regular expression to find the coordinates in the format (x, y)
    match_coordinates = re.search(r"Chip Middle Point: \((\d+\.\d+), (\d+\.\d+)\)", content)
    if match_coordinates:
        chip_x = float(match_coordinates.group(1))
        chip_y = float(match_coordinates.group(2))
    else:
        raise ValueError("Chip Middle Point coordinates not found in the file.")
    
    # Use regular expression to find the angle of error
    match_angle = re.search(r"Angle of error: (\d+\.\d+)°", content)
    if match_angle:
        additional_angle = float(match_angle.group(1))
    else:
        raise ValueError("Angle of error not found in the file.")

    return chip_x, chip_y, additional_angle

 
    
# Calculations (Angle & Motor Steps Functions)
def calculate_angle(x, y):
    """Calculate the angle based on X and Y, adjusting the gripper direction according to the dominant vector."""
    if abs(y) > abs(x):  # If Y is greater in magnitude, use Y as the dominant direction
        angle_rad = math.atan2(x, y)
        angle_rad -= (math.pi)/4  # This gives the angle from the positive Y-axis
        angle_deg = math.degrees(angle_rad) + 45  # Convert the angle from radians to degrees
        
        if angle_deg < 0:
            angle_deg += 360  # Normalize to the range [0, 360]
    else:
        angle_rad = math.atan2(y, x)
        angle_rad -= math.pi  # Adjust to align with the negative X-axis (westward direction)
        angle_deg = math.degrees(angle_rad)
        
        if angle_deg < 0:
            angle_deg += 360  # Normalize to the range [0, 360]

    return angle_deg

def angle_to_motor_steps(angle_deg):
    """Convert the angle (in degrees) to the stepper motor position (0-512 range)."""
    motor_position = (angle_deg / 180) * 512
    return round(motor_position)

def adjust_gripper_angle(current_angle, additional_angle):
    """Adjust the gripper's angle by a given additional angle."""
    adjusted_angle = current_angle + additional_angle
    print(f"Gripper adjusted by {additional_angle} degrees. New angle: {adjusted_angle:.2f}")
    return adjusted_angle

def go_to_pos(pickup_pos, theta0_4):
    try:
        joint_angles = kin.ik3(pickup_pos)  # Attempt to get joint angles for the position
        theta4 = kin.calculate_theta_4(joint_angles, theta0_4)
        phx.set_wrist(theta4)
        phx.set_wse(joint_angles)
        
        phx.wait_for_completion()
    except ValueError as e:
        # This will catch invalid kinematics solutions (e.g., position out of reach)
        print(f"Error: Unable to reach position {pickup_pos}.")
        print(f"Details: {e}")
        return False  # Indicate failure to reach the position
    return True  # Successful move

def move_to_position_with_z_adjustment(pickup_pos, theta0_4, z_adjustment=15):
    """Move to a position, first adjusting Z by a given amount."""
    intermediate_pos = pickup_pos.copy()
    intermediate_pos[2] += z_adjustment  # Adjust Z by the given amount
    print(f"Moving to intermediate position (Z first): {intermediate_pos}")
    go_to_pos(intermediate_pos, theta0_4)
    print(f"Moving to final position: {pickup_pos}")
    go_to_pos(pickup_pos, theta0_4)

def set_gripper(position):
    """Set the gripper position (0-512)."""
    phx.set_gripper(position)

def pick_up(x, y, additional_angle=0):
    """Move to the pick up location, adjust Z, close the gripper, and move up after pick-up."""
    # Define the initial pick-up position (Z = 20)
    pickup_pos = [x, y, 20]  
    # The initial position for theta_4 (the wrist angle)
    theta0_4 = -95  # theta4 = -95 for pick up

    print(f"Picking up from position: {pickup_pos}, with theta4: {theta0_4}")
    
    # Calculate the gripper angle based on X, Y
    angle = calculate_angle(x, y)
    print(f"The gripper is adjusted to angle: {angle:.2f} degrees")
    
    # Adjust gripper angle with the additional user-defined rotation
    adjusted_angle = adjust_gripper_angle(angle, additional_angle)
    
    # Convert the adjusted angle to motor steps and set gripper
    gripper_position = angle_to_motor_steps(adjusted_angle)
    set_gripper(gripper_position)

    # Move to (X, Y, 25) first (with theta_4 angle set)
    print(f"Moving to the position (X, Y, 25) with theta_4 set.")
    intermediate_pos = [x, y, 25]  # Set Z to 25
    go_to_pos(intermediate_pos, theta0_4)  # Move to (X, Y, 25)

    # Now move down to (X, Y, 20) to pick up the object
    print(f"Moving down to pick up position (X, Y, 20).")
    go_to_pos(pickup_pos, theta0_4)  # Move down to (X, Y, 20)

    # Close the gripper after reaching the position
    phx.close_gripper2()
    print("Gripper closed at the pick up location.")
    time.sleep(1.5)  # Add 1.5 second delay after pick up
    
    # After pick-up, move the arm up along the Z-axis only (e.g., move to (X, Y, 25))
    print(f"Moving up to clear the area: (X, Y, 25).")
    intermediate_pos[2] = 25  # Set Z back to 25
    go_to_pos(intermediate_pos, theta0_4)  # Move up to (X, Y, 25)


def drop_off(x, y):
    """Move to the drop off location, adjust Z, open the gripper, and go to a fixed position before going to rest."""
    drop_off_pos = [x, y, 15]  # Z = 25 for drop off (initial position)
    theta0_4 = -90  # theta4 = -90 for drop off

    print(f"Dropping off at position: {drop_off_pos}, with theta4: {theta0_4}")

    # Calculate the gripper angle based on X, Y
    angle = calculate_angle(x, y)
    print(f"The gripper is adjusted to angle: {angle:.2f} degrees")

    # Convert the angle to motor steps and set gripper
    gripper_position = angle_to_motor_steps(angle)
    set_gripper(gripper_position)

    # Move to the drop off position with Z adjustment (Z first, then X/Y)
    print(f"Moving to drop-off position with Z adjustment first.")
    move_to_position_with_z_adjustment(drop_off_pos, 0)
    
    # Open the gripper after reaching the position
    phx.open_gripper2()
    print("Gripper opened at the drop off location.")
    time.sleep(2.5)  # Add 2.5 second delay after drop off

    # **Always move to fixed position (0, 10, 25)**
    print("Moving to fixed position (0, 10, 25)...")
    fixed_position = [10, 0, 25]
    go_to_pos(fixed_position, 90)

    # After reaching (0, 10, 25), move to rest position
    print("Moving directly to rest position...")
    phx.rest_position()  # Go straight to rest position


def main():
    while True:
        filename = "/home/scalepi/Desktop/savephototest/latest_detection.txt"
        try:
            chip_middle_x, chip_middle_y, text_file_angle = read_coordinates_from_file(filename)
        except ValueError as e:
            print(e)
            return
        transformed_x, transformed_y = transform_coordinates(chip_middle_x, chip_middle_y)
        print(f"Original coordinates in System 1: ({chip_middle_x}, {chip_middle_y})")
        print(f"Transformed coordinates in System 2: ({transformed_x}, {transformed_y})")
        print(f"Angle from text file: {text_file_angle} degrees")
        pick_up(transformed_x, transformed_y, text_file_angle)
        drop_off(0, -15)
        print("Returning to resting position after all drop-offs...")
        phx.rest_position()
        continue_input = input("\nDo you want to repeat the process? (y/n): ").strip().lower()
        if continue_input != 'y':
            print("Ending process.")
            break 
        time.sleep(0.1)
if __name__ == "__main__":
    main()

