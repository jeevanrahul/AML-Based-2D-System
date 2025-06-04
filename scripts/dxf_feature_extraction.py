import ezdxf
import math
import numpy as np

filename = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\raw\rectangle.dxf"
doc = ezdxf.readfile(filename)
msp = doc.modelspace()

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

def perimeter(points):
    n = len(points)
    peri = 0
    for i in range(n):
        peri += distance(points[i], points[(i + 1) % n])
    return peri

def aspect_ratio(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if height == 0:
        return 0
    
    return width / height

def vertex_angles(points):
    angles=[]
    n = len(points)
    for i in range(n):
        p0 = np.array(points[i - 1])
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % n])

        v1 = p0 - p1
        v2 = p2 - p1

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)
        angles.append(np.degrees(angle))
    return angles

def is_convex(points):
    n = len(points)
    if n < 4:
        return True
    signs = []

    for i in range(n):
        dx1 = points[(i + 1)%n][0] - points[i][0]
        dy1 = points[(i+1)%n][1] - points[i][1]
        dx2 = points[(i+2)%n][0] - points[(i+1)%n][0]
        dy2 = points[(i+2)%n][1] - points[(i+1)%n][1]
        cross = dx1*dy2 - dy1*dx2
        signs.append(cross > 0)
    return all(signs) or not any(signs)

def compactness(area, perimeter):
    if perimeter == 0:
        return 0
    return (4 * math.pi * area) / (perimeter**2)

def bounding_box_area(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

def centroid(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs)/len(xs), sum(ys)/len(ys))

def count_acute_obtuse_angles(angles):
    acute = sum(1 for a in angles if a < 90)
    obtuse = sum(1 for a in angles if a > 90)
    return acute, obtuse

def slope(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0:
        return float('inf')
    return dy / dx

def orientation_angle(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle % 360

polylines_data = []
circles_data = []
lines_data = []
arcs_data = []

for entity in msp:
    if entity.dxftype() == 'LWPOLYLINE':
        points = [(point[0], point[1]) for point in entity.get_points()]
        edges = [distance(points[i], points[i+1]) for i in range(len(points) - 1)]
        if points[0] != points[-1]:
            edges.append(distance(points[-1], points[0]))
        
        area_val = polygon_area(points)
        peri_val = perimeter(points)
        angles = vertex_angles(points)

        poly_data = {
            "points": [(float(x), float(y)) for x, y in points],
            "num_points": int(len(points)),
            "edge_lengths": [float(distance(points[i], points[(i + 1) % len(points)])) for i in range(len(points))],
            "area": float(polygon_area(points)),
            "perimeter": float(perimeter(points)),
            "aspect_ratio": float(aspect_ratio(points)),
            "vertex_angles": [float(a) for a in vertex_angles(points)],
            "is_convex": bool(is_convex(points)),
            "compactness": float(compactness(polygon_area(points), perimeter(points))),
            "bounding_box_area": float(bounding_box_area(points)),
            "centroid_x": float(centroid(points)[0]),
            "centroid_y": float(centroid(points)[1]),
            "acute_angle_count": int(count_acute_obtuse_angles(vertex_angles(points))[0]),
            "obtuse_angle_count": int(count_acute_obtuse_angles(vertex_angles(points))[1]),
}

        polylines_data.append(poly_data)

    elif entity.dxftype() == 'CIRCLE':
        center = (entity.dxf.center[0], entity.dxf.center[1])
        radius = entity.dxf.radius
        circle_data = {
            'center': center,
            'radius': radius,
            'diameter': 2 * radius,
            'circumference': 2 * math.pi * radius,
            'area': math.pi * radius**2
        }
        circles_data.append(circle_data)

    elif entity.dxftype() == 'LINE':
        start = (entity.dxf.start[0], entity.dxf.start[1])
        end = (entity.dxf.end[0], entity.dxf.end[1])
        length_val = distance(start, end)
        line_data = {
            'start': start,
            'end': end,
            'length': length_val,
            'slope': slope(start, end),
            'orientation_angle': orientation_angle(start, end)
        }
        lines_data.append(line_data)

    elif entity.dxftype() == 'ARC':
        center = (entity.dxf.center[0], entity.dxf.center[1])
        radius = entity.dxf.radius
        start_angle = entity.dxf.start_angle
        end_angle = entity.dxf.end_angle
        angle_rad = math.radians(end_angle - start_angle)
        arc_length = abs(radius * angle_rad)
        arc_data = {
            'center': center,
            'radius': radius,
            'start_angle': start_angle,
            'end_angle': end_angle,
            'arc_length': arc_length
        }
        arcs_data.append(arc_data)


print(f"Polylines found: {len(polylines_data)}")
print(f"Circles found: {len(circles_data)}")
print(f"Lines found: {len(lines_data)}")
print(f"Arcs found: {len(arcs_data)}")

if polylines_data:
    example = polylines_data[0]
    print("\nExample Polyline Features:")
    for k, v in example.items():
        print(f"{k}: {v}")

if lines_data:
    example_line = lines_data[0]
    print("\nExample Line Features:")
    for k, v in example_line.items():
        print(f"{k}: {v}")

import pandas as pd
import os

save_folder = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\processed"

all_shapes_data = polylines_data + circles_data + lines_data + arcs_data

if not all_shapes_data:
    print("No shape features found to save.")
else:
    save_input = input("Do you want to save the extracted features to a CSV file? (yes/no): ").strip().lower()
    if save_input == "yes":
        os.makedirs(save_folder, exist_ok=True)

        filename = input("Enter the filename (without extension): ").strip()
        if not filename:
            print("Invalid filename. Features not saved.")
        else:
            file_path = os.path.join(save_folder, filename + ".csv")

            df = pd.DataFrame(all_shapes_data)
            df.to_csv(file_path, index=False)
            print(f"Features saved successfully at: {file_path}")
    else:
        print("Features not saved.")
