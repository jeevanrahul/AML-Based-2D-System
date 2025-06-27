import ezdxf
import math
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# ---------- Geometry Utilities ----------
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def polygon_area(points):
    return abs(sum(p[0]*points[(i+1)%len(points)][1] - points[(i+1)%len(points)][0]*p[1]
                   for i, p in enumerate(points))) / 2

def perimeter(points):
    return sum(distance(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))

def aspect_ratio(points):
    xs, ys = zip(*points)
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return width / height if height else 0

def vertex_angles(points):
    angles = []
    n = len(points)
    for i in range(n):
        p0, p1, p2 = np.array(points[i-1]), np.array(points[i]), np.array(points[(i+1)%n])
        v1, v2 = p0 - p1, p2 - p1
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_angle = np.clip(np.dot(v1, v2) / denom if denom else 1.0, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_angle)))
    return angles

def is_convex(points):
    signs = []
    for i in range(len(points)):
        dx1, dy1 = points[(i+1)%len(points)][0] - points[i][0], points[(i+1)%len(points)][1] - points[i][1]
        dx2, dy2 = points[(i+2)%len(points)][0] - points[(i+1)%len(points)][0], points[(i+2)%len(points)][1] - points[(i+1)%len(points)][1]
        cross = dx1 * dy2 - dy1 * dx2
        signs.append(cross > 0)
    return all(signs) or not any(signs)

def compactness(area, peri):
    return (4 * math.pi * area) / (peri ** 2) if peri else 0

def bounding_box_area(points):
    xs, ys = zip(*points)
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

def centroid(points):
    xs, ys = zip(*points)
    return sum(xs) / len(xs), sum(ys) / len(ys)

def count_acute_obtuse_angles(angles):
    acute = sum(1 for a in angles if a < 90)
    obtuse = sum(1 for a in angles if a > 90)
    return acute, obtuse

# ---------- Feature Extraction Core ----------
class ShapeFeatureExtractor:
    def __init__(self, save_features_folder):
        self.save_features_folder = save_features_folder

    def extract_from_entity(self, entity):
        points = self._get_entity_points(entity)
        if not points:
            return None

        area = polygon_area(points)
        peri = perimeter(points)
        asp = aspect_ratio(points)
        angles = vertex_angles(points)
        acute, obtuse = count_acute_obtuse_angles(angles)
        cx, cy = centroid(points)

        return {
            "num_points": len(points),
            "area": area,
            "perimeter": peri,
            "aspect_ratio": asp,
            "is_convex": int(is_convex(points)),
            "compactness": compactness(area, peri),
            "bounding_box_area": bounding_box_area(points),
            "centroid_x": cx,
            "centroid_y": cy,
            "acute_angle_count": acute,
            "obtuse_angle_count": obtuse,
        }

    def _get_entity_points(self, entity):
        try:
            if entity.dxftype() == "LWPOLYLINE":
                raw = [(p[0], p[1]) for p in entity.get_points()]
                return raw[:-1] if len(raw) > 1 and raw[0] == raw[-1] else raw

            elif entity.dxftype() == "CIRCLE":
                center, r = entity.dxf.center, entity.dxf.radius
                return [(center[0] + r * math.cos(a), center[1] + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 36)]

            elif entity.dxftype() == "ELLIPSE":
                center, major_axis, ratio = entity.dxf.center, entity.dxf.major_axis, entity.dxf.ratio
                start, end = entity.dxf.start_param, entity.dxf.end_param
                angle = math.atan2(major_axis[1], major_axis[0])
                major_len = math.hypot(major_axis[0], major_axis[1])
                return [
                    (
                        major_len * math.cos(t) * math.cos(angle) - major_len * ratio * math.sin(t) * math.sin(angle) + center[0],
                        major_len * math.cos(t) * math.sin(angle) + major_len * ratio * math.sin(t) * math.cos(angle) + center[1]
                    )
                    for t in np.linspace(start, end, 36)
                ]
        except Exception as e:
            print(f"⚠️ Error extracting entity points: {e}")
        return []

# ---------- Main Workflow ----------
def process_dxf_file(filepath, save_features_folder):
    print(f"\n📂 Processing: {filepath}")
    try:
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
    except Exception as e:
        print(f"❌ Failed to read {filepath}: {e}")
        return None

    extractor = ShapeFeatureExtractor(save_features_folder)

    for entity in msp:
        data = extractor.extract_from_entity(entity)
        if data:
            df = pd.DataFrame([data])
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            out_path = os.path.join(save_features_folder, f"features_{base_name}.csv")
            df.to_csv(out_path, index=False)
            print(f"✅ Extracted features saved at: {out_path}")
            return df, base_name

    print("⚠️ No valid shape found in file.")
    return None

def process_folder(folder_path):
    save_features_folder = os.path.join("data", "processed", "processed_features")
    save_norm_folder = os.path.join("data", "processed", "processed_normalized")
    os.makedirs(save_features_folder, exist_ok=True)
    os.makedirs(save_norm_folder, exist_ok=True)

    dxf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".dxf")]
    if not dxf_files:
        print("❌ No DXF files found.")
        return

    dfs, names = [], []
    for file in dxf_files:
        result = process_dxf_file(os.path.join(folder_path, file), save_features_folder)
        if result:
            df, name = result
            dfs.append(df)
            names.append(name)

    if not dfs:
        print("❌ No features extracted from any file.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    numeric_df = combined.select_dtypes(include=[np.number])
    numeric_df = numeric_df.loc[:, (numeric_df.max() != numeric_df.min())]

    if numeric_df.empty:
        print("⚠️ All numeric features are constant. Skipping normalization.")
        return

    scaler = MinMaxScaler()
    norm = scaler.fit_transform(numeric_df)
    norm_df = pd.DataFrame(norm, columns=numeric_df.columns)

    for i, name in enumerate(names):
        out = norm_df.iloc[[i]].reset_index(drop=True)
        out_path = os.path.join(save_norm_folder, f"normalized_{name}.csv")
        out.to_csv(out_path, index=False)
        print(f"✅ Normalized features saved at: {out_path}")

# ---------- Entry ----------
if __name__ == "__main__":
    folder_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\raw\shapes_dataset"
    if not os.path.isdir(folder_path):
        print("❌ Invalid folder path.")
    else:
        process_folder(folder_path)
