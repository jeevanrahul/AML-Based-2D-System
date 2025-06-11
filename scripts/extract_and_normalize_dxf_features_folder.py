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
    return sum(distance(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))

def aspect_ratio(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return 0 if height == 0 else width / height

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
        angles.append(np.degrees(np.arccos(cos_angle)))
    return angles

def is_convex(points):
    if len(points) < 4:
        return True
    signs = []
    for i in range(len(points)):
        dx1 = points[(i + 1) % len(points)][0] - points[i][0]
        dy1 = points[(i + 1) % len(points)][1] - points[i][1]
        dx2 = points[(i + 2) % len(points)][0] - points[(i + 1) % len(points)][0]
        dy2 = points[(i + 2) % len(points)][1] - points[(i + 1) % len(points)][1]
        cross = dx1 * dy2 - dy1 * dx2
        signs.append(cross > 0)
    return all(signs) or not any(signs)

def compactness(area, perimeter):
    return 0 if perimeter == 0 else (4 * math.pi * area) / (perimeter ** 2)

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
    return float('inf') if dx == 0 else dy / dx

def orientation_angle(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return math.degrees(math.atan2(dy, dx)) % 360

# Global storage
extracted_dfs = []
file_names = []

def process_dxf_file(filepath, save_features_folder):
    print(f"\nProcessing: {filepath}")
    try:
        doc = ezdxf.readfile(filepath)
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return None

    msp = doc.modelspace()

    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE':
            points = [(point[0], point[1]) for point in entity.get_points()]
            area_val = polygon_area(points)
            peri_val = perimeter(points)
            angles = vertex_angles(points)

            data = {
                "num_points": len(points),
                "area": area_val,
                "perimeter": peri_val,
                "aspect_ratio": aspect_ratio(points),
                "is_convex": int(is_convex(points)),
                "compactness": compactness(area_val, peri_val),
                "bounding_box_area": bounding_box_area(points),
                "centroid_x": centroid(points)[0],
                "centroid_y": centroid(points)[1],
                "acute_angle_count": count_acute_obtuse_angles(angles)[0],
                "obtuse_angle_count": count_acute_obtuse_angles(angles)[1],
            }
            df = pd.DataFrame([data])
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            feature_path = os.path.join(save_features_folder, f"features_{base_name}.csv")
            df.to_csv(feature_path, index=False)
            print(f"Extracted features saved at: {feature_path}")
            return df, base_name

    print(f"No valid shapes found in {filepath}.")
    return None

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

    global extracted_dfs, file_names
    extracted_dfs = []
    file_names = []

    for dxf_file in dxf_files:
        full_path = os.path.join(folder_path, dxf_file)
        result = process_dxf_file(full_path, save_features_folder)
        if result:
            df, base_name = result
            extracted_dfs.append(df)
            file_names.append(base_name)

    if not extracted_dfs:
        print("No features extracted from any files.")
        return

    # Combine all data
    combined_df = pd.concat(extracted_dfs, ignore_index=True)
    numeric_df = combined_df.select_dtypes(include=[np.number]).copy()

    # Remove constant columns
    numeric_df = numeric_df.loc[:, (numeric_df.max() != numeric_df.min())]
    if numeric_df.empty:
        print("All numeric features are constant. Skipping normalization.")
        return

    # Normalize
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(numeric_df)
    normalized_df = pd.DataFrame(normalized_array, columns=numeric_df.columns)

    # Save normalized rows into their respective files
    for i, base_name in enumerate(file_names):
        out_df = normalized_df.iloc[[i]].reset_index(drop=True)
        out_path = os.path.join(save_norm_folder, f"normalized_{base_name}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Normalized features saved at: {out_path}")

if __name__ == "__main__":
    folder_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\raw"
    if not os.path.isdir(folder_path):
        print("Invalid folder path! Please enter a valid directory.")
    else:
        process_folder(folder_path)
