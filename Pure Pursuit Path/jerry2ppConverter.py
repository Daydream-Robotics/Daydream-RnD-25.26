import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
args = parser.parse_args()

# print()

def convert_jerryio_to_robot(file_path):
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    with open(file_path, 'r') as f:
        file_content = f.read()

    lines = file_content.strip().split('\n')
    raw_points = []
    initial_heading = 0.0
    
    for line in lines:
        if line and (line[0].isdigit() or line[0] == '-'):
            if ',' in line:
                parts = line.split(',')
                if not raw_points and len(parts) >= 4:
                    initial_heading = float(parts[3])
                raw_points.append((float(parts[0]), float(parts[1])))
    
    if not raw_points:
        return "Error: No path points found in file."

    # Establish the Origin (Robot's 0,0 is the first point)
    origin_x, origin_y = raw_points[0]

    # Calculate heading vectors based on VEX GPS (0=North/+Y, 90=East/+X)
    theta = math.radians(initial_heading)
    fwd_x = math.sin(theta)
    fwd_y = math.cos(theta)
    right_x = math.cos(theta)
    right_y = -math.sin(theta)

    cpp_output = "std::vector<Position> path = {\n"
    formatted_points = []

    for px, py in raw_points:
        # Calculate displacement from start in centimeters
        dx_cm = px - origin_x
        dy_cm = py - origin_y
        
        # Convert centimeters to inches
        dx_in = dx_cm / 2.54
        dy_in = dy_cm / 2.54
        
        # Transform to Robot's local frame (pos x is forward, pos y is right)
        robot_x = dx_in * fwd_x + dy_in * fwd_y
        robot_y = dx_in * right_x + dy_in * right_y
        
        formatted_points.append(f"    {{{robot_x:.2f}, {robot_y:.2f}}}")

    cpp_output += ",\n".join(formatted_points)
    cpp_output += "\n};"
    
    return cpp_output

result = convert_jerryio_to_robot(args.file if args.file else 'path.jerryio.txt')
print(result)