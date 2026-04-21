import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--file", default='path.jerryio.txt', type=str, help="Use custom filename")
args = parser.parse_args()

def convert_jerryio_to_robot(file_path):

    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    with open(file_path, 'r') as f:
        file_content = f.read()

    lines = file_content.strip().split('\n')
    
    path_names = []
    all_paths = []
    current_raw_points = []
    current_heading = 0.0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # checks if the line is the start of a new path
        if "#PATH-POINTS-START" in line:
            path_name = line.removeprefix("#PATH-POINTS-START").strip().upper().replace(' ', '_')
            if path_name in path_names:
                return f"Error: Duplicate path name '{path_name}' found."
            path_names.append(path_name)

            # if there are saved points from a previous path, save to all_paths
            if current_raw_points:
                all_paths.append((current_raw_points, current_heading))
                current_raw_points = []
                current_heading = 0.0

        # checks if the line contains a coordinate number (@BranStile the dashed numbers are negative numbers)
        elif line[0].isdigit() or line[0] == '-':
            if ',' in line:
                parts = line.split(',')
                if not current_raw_points and len(parts) >= 4:
                    current_heading = float(parts[3])
                current_raw_points.append((float(parts[0]), float(parts[1])))
                
    # save final path
    if current_raw_points:
        all_paths.append((current_raw_points, current_heading))
        
    # check for no paths
    if not all_paths:
        return "Error: No path points found in file."

    enum_members = ",\n    ".join(path_names)
    cpp_output = f"#pragma once\n\
#include \"odometry.hpp\"\n\
#include \"arclengthSplining.hpp\"\n\
#include <vector>\n\
\n\
#ifndef PATHS_HPP\n\
#define PATHS_HPP 3.14159265358979323846\n\
\n\
enum class PathName : uint32_t {{\n\
    {enum_members}\n\
}};\n\
\n\
inline std::vector<std::vector<Position>> raw_paths = {{\n"

    # Set robots orign and intial heading
    if all_paths[0]:
        raw_points, initial_heading = all_paths[0]
        
        origin_x, origin_y = raw_points[0]

        # get heading vectors based off initial heading
        theta = math.radians(initial_heading)
        fwd_x = math.sin(theta)
        fwd_y = math.cos(theta)
        right_x = math.cos(theta)
        right_y = -math.sin(theta)

    
    # parse each path and save to cpp file
    for path_index, (raw_points, initial_heading) in enumerate(all_paths):
        del raw_points[-1]

        cpp_output += f"    {{\n"
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
            
            formatted_points.append(f"        {{{robot_x:.2f}, {robot_y:.2f}}}")

        cpp_output += ",\n".join(formatted_points)
        cpp_output += "\n    }"
        if path_index < len(all_paths) - 1:
            cpp_output += ","
        cpp_output += "\n"

    cpp_output += "};\n\nstd::vector<ALS_Path> buildAllPaths(double sampleSpacing = 0.25);\n\n#endif"
    return cpp_output.strip()

result = convert_jerryio_to_robot(args.file if args.file else 'path.jerryio.txt')
with open("paths.hpp", 'w') as pf:
    pf.write(result)
