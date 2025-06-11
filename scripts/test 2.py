import ezdxf
import math
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

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
    angles = []
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
        dx1 = points[(i + 1) % n][0] - points[i][0]
        dy1 = points[(i + 1) % n][1] - points[i][1]
        dx2 = points[(i + 2) % n][0] - points[(i + 1) % n][0]
        dy2 = points[(i + 2) % n][1] - points[(i + 1) % n][1]
        cross = dx1 * dy2 - dy1 * dx2
        signs.append(cross > 0)
    return all(signs) or not any(signs)

def compactness(area, perimeter):
    if perimeter == 0:
        return 0
    return (4 * math.pi * area) / (perimeter ** 2)

def bounding_box_area(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

def centroid(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

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

def process_dxf_file(filepath, save_features_folder, save_norm_folder):
    print(f"\nProcessing: {filepath}")
    try:
        doc = ezdxf.readfile(filepath)
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return

    msp = doc.modelspace()

    polylines_data = []
    circles_data = []
    lines_data = []
    arcs_data = []

    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE':
            points = [(point[0], point[1]) for point in entity.get_points()]
            area_val = polygon_area(points)
            peri_val = perimeter(points)
            angles = vertex_angles(points)

            poly_data = {
                "points": [(float(x), float(y)) for x, y in points],
                "num_points": int(len(points)),
                "edge_lengths": [float(distance(points[i], points[(i + 1) % len(points)])) for i in range(len(points))],
                "area": float(area_val),
                "perimeter": float(peri_val),
                "aspect_ratio": float(aspect_ratio(points)),
                "vertex_angles": [float(a) for a in angles],
                "is_convex": bool(is_convex(points)),
                "compactness": float(compactness(area_val, peri_val)),
                "bounding_box_area": float(bounding_box_area(points)),
                "centroid_x": float(centroid(points)[0]),
                "centroid_y": float(centroid(points)[1]),
                "acute_angle_count": int(count_acute_obtuse_angles(angles)[0]),
                "obtuse_angle_count": int(count_acute_obtuse_angles(angles)[1]),
            }
            polylines_data.append(poly_data)

        elif entity.dxftype() == 'CIRCLE':
            center = (entity.dxf.center[0], entity.dxf.center[1])
            radius = entity.dxf.radius
            circle_data = {
                'center_x': float(center[0]),
                'center_y': float(center[1]),
                'radius': float(radius),
                'diameter': 2 * radius,
                'circumference': 2 * math.pi * radius,
                'area': math.pi * radius ** 2
            }
            circles_data.append(circle_data)

        elif entity.dxftype() == 'LINE':
            start = (entity.dxf.start[0], entity.dxf.start[1])
            end = (entity.dxf.end[0], entity.dxf.end[1])
            length_val = distance(start, end)
            line_data = {
                'start_x': float(start[0]),
                'start_y': float(start[1]),
                'end_x': float(end[0]),
                'end_y': float(end[1]),
                'length': float(length_val),
                'slope': float(slope(start, end)),
                'orientation_angle': float(orientation_angle(start, end))
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
                'center_x': float(center[0]),
                'center_y': float(center[1]),
                'radius': float(radius),
                'start_angle': float(start_angle),
                'end_angle': float(end_angle),
                'arc_length': float(arc_length)
            }
            arcs_data.append(arc_data)

    all_shapes_data = polylines_data + circles_data + lines_data + arcs_data

    if not all_shapes_data:
        print(f"No shapes found in {filepath}. Skipping save.")
        return

    # Base filename without extension
    base_name = os.path.splitext(os.path.basename(filepath))[0]

    # Save extracted features CSV
    extracted_file_path = os.path.join(save_features_folder, f"features_{base_name}.csv")
    df = pd.DataFrame(all_shapes_data)
    df.to_csv(extracted_file_path, index=False)
    print(f"Extracted features saved at: {extracted_file_path}")

    # Normalize numeric columns and save
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        print(f"No numeric features to normalize in {filepath}.")
        return

    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(numeric_df)
    normalized_df = pd.DataFrame(normalized_array, columns=numeric_df.columns)

    normalized_file_path = os.path.join(save_norm_folder, f"normalized_{base_name}.csv")
    normalized_df.to_csv(normalized_file_path, index=False)
    print(f"Normalized features saved at: {normalized_file_path}")


def process_folder(folder_path):
    save_features_folder = os.path.join(r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\processed", "processed_features")
    save_norm_folder = os.path.join(r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\processed", "processed_normalized")

    os.makedirs(save_features_folder, exist_ok=True)
    os.makedirs(save_norm_folder, exist_ok=True)

    files = os.listdir(folder_path)
    dxf_files = [f for f in files if f.lower().endswith('.dxf')]

    if not dxf_files:
        print("No DXF files found in the folder.")
        return

    for dxf_file in dxf_files:
        full_path = os.path.join(folder_path, dxf_file)
        process_dxf_file(full_path, save_features_folder, save_norm_folder)

if __name__ == "__main__":
    folder_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\raw"
    if not os.path.isdir(folder_path):
        print("Invalid folder path! Please enter a valid directory.")
    else:
        process_folder(folder_path)