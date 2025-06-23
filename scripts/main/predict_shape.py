import ezdxf
import math
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# --- Feature Extraction Utilities ---
def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2.0

def perimeter(points):
    return sum(distance(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))

def aspect_ratio(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return width / height if height != 0 else 0

def bounding_box_area(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

def centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return (sum(x_coords) / len(points), sum(y_coords) / len(points))

def compactness(area, peri):
    return 4 * math.pi * area / (peri ** 2) if peri != 0 else 0

def vertex_angles(points):
    def angle(a, b, c):
        ab = np.array([b[0] - a[0], b[1] - a[1]])
        cb = np.array([b[0] - c[0], b[1] - c[1]])
        cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        return np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * 180 / np.pi

    angles = []
    for i in range(len(points)):
        angles.append(angle(points[i - 1], points[i], points[(i + 1) % len(points)]))
    return angles

def count_acute_obtuse_angles(angles):
    acute = sum(1 for a in angles if a < 90)
    obtuse = sum(1 for a in angles if a > 90)
    return acute, obtuse

# --- Main Prediction Function ---
def extract_features_from_dxf(filepath):
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()

    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE':
            points = [(point[0], point[1]) for point in entity.get_points()]
            area_val = polygon_area(points)
            peri_val = perimeter(points)
            angles = vertex_angles(points)

            features = {
                "num_points": len(points),
                "area": area_val,
                "perimeter": peri_val,
                "aspect_ratio": aspect_ratio(points),
                "compactness": compactness(area_val, peri_val),
                "bounding_box_area": bounding_box_area(points),
                "centroid_x": centroid(points)[0],
                "centroid_y": centroid(points)[1],
                "acute_angle_count": count_acute_obtuse_angles(angles)[0],
                "obtuse_angle_count": count_acute_obtuse_angles(angles)[1],
            }
            return pd.DataFrame([features])
    return None

# --- Run Prediction ---
def predict_shape(dxf_file_path):
    print("\n🔍 Extracting features from:", dxf_file_path)
    
    features_df = extract_features_from_dxf(dxf_file_path)
    if features_df is None:
        print("❌ No LWPOLYLINE entity found.")
        return

    # Load trained model and scaler
    model = joblib.load(r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\models\random_forest_model.pkl")
    scaler = joblib.load(r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\models\minmax_scaler.pkl")
    encoder = joblib.load(r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\models\label_encoder.pkl")

    # Apply same normalization
    features_scaled = scaler.transform(features_df)

    # Predict
    prediction = model.predict(features_scaled)
    predicted_label = encoder.inverse_transform(prediction)
    print(f"✅ Predicted shape type: {predicted_label[0]}")

# --- Example usage ---
if __name__ == "__main__":
    # Change path below to any test .dxf file
    test_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\raw\shapes_dataset\triangle_30.dxf"
    predict_shape(test_path)
