import ezdxf
import joblib
import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- File Paths ---
INPUT_DXF = r"data/raw/shapes_inside_box/sample_shapes_in_box.dxf"
OUTPUT_DXF = r"data/output/arranged_output.dxf"
MODEL_PATH = r"models/random_forest_model.pkl"
ENCODER_PATH = r"models/label_encoder.pkl"
SCALER_PATH = r"models/minmax_scaler.pkl"
FEATURE_COLUMNS_PATH = r"models/feature_columns.txt"

SPACING = 4  # mm

# --- Feature Extraction Utilities ---
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def polygon_area(points):
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + [points[0]])))

def perimeter(points):
    return sum(distance(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))

def aspect_ratio(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return width / height if height != 0 else 0

def bounding_box(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def get_circle_features(entity):
    center = entity.dxf.center
    radius = entity.dxf.radius
    points = [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 36)]
    return points

def vertex_angles(points):
    angles = []
    n = len(points)
    for i in range(n):
        p0 = np.array(points[i - 1])
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % n])
        v1 = p0 - p1
        v2 = p2 - p1
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_angle = np.dot(v1, v2) / denom if denom != 0 else 1.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_angle)))
    return angles

def count_acute_obtuse_angles(angles):
    acute = sum(1 for a in angles if a < 90)
    obtuse = sum(1 for a in angles if a > 90)
    return acute, obtuse

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

def centroid(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def extract_features(entity):
    if entity.dxftype() == 'CIRCLE':
        points = get_circle_features(entity)
    elif entity.dxftype() == 'LWPOLYLINE':
        points = [(p[0], p[1]) for p in entity.get_points()]
    else:
        return None, None

    area = polygon_area(points)
    peri = perimeter(points)
    asp = aspect_ratio(points)
    box_area = (max(p[0] for p in points) - min(p[0] for p in points)) * (max(p[1] for p in points) - min(p[1] for p in points))
    angles = vertex_angles(points)
    acute, obtuse = count_acute_obtuse_angles(angles)
    cx, cy = centroid(points)

    return {
        "num_points": len(points),
        "area": area,
        "perimeter": peri,
        "aspect_ratio": asp,
        "bounding_box_area": box_area,
        "centroid_x": cx,
        "centroid_y": cy,
        "is_convex": int(is_convex(points)),
        "compactness": compactness(area, peri),
        "acute_angle_count": acute,
        "obtuse_angle_count": obtuse,
    }, points

# --- Main Script ---
print("\U0001F504 Loading model and scaler...")
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# Load expected feature columns
with open(FEATURE_COLUMNS_PATH, "r") as f:
    expected_cols = [line.strip() for line in f]

print("\U0001F4C2 Reading DXF:", INPUT_DXF)
doc = ezdxf.readfile(INPUT_DXF)
msp = doc.modelspace()

shapes = []
boundary = None

# --- Step 1: Detect boundary and extract shapes ---
for e in msp:
    if e.dxftype() == "LWPOLYLINE" and e.closed:
        pts = [(p[0], p[1]) for p in e.get_points()]
        width = max(p[0] for p in pts) - min(p[0] for p in pts)
        height = max(p[1] for p in pts) - min(p[1] for p in pts)
        if not boundary or width * height > boundary["width"] * boundary["height"]:
            boundary = {"entity": e, "x": min(p[0] for p in pts), "y": min(p[1] for p in pts), "width": width, "height": height}
        else:
            features, points = extract_features(e)
            if features:
                shapes.append({"entity": e, "features": features, "points": points})
    elif e.dxftype() == "CIRCLE":
        features, points = extract_features(e)
        if features:
            shapes.append({"entity": e, "features": features, "points": points})

if not boundary:
    raise ValueError("\u274C No rectangle boundary found!")

# --- Step 2: Predict shape types ---
df = pd.DataFrame([s["features"] for s in shapes])

# Reorder columns to match training
for col in expected_cols:
    if col not in df.columns:
        df[col] = 0

df = df[expected_cols]
df_scaled = scaler.transform(df)
preds = model.predict(df_scaled)
labels = encoder.inverse_transform(preds)

for shape, label in zip(shapes, labels):
    shape["label"] = label
    print(f"\U0001F9E0 Predicted shape: {label}")

# --- Step 3: Clear everything except boundary
for e in list(msp):
    if e != boundary["entity"]:
        msp.delete_entity(e)

# --- Step 4: Arrange shapes inside the box
x_cursor = boundary["x"] + SPACING
y_cursor = boundary["y"] + SPACING
row_max_height = 0

for shape in shapes:
    min_x, min_y, max_x, max_y = bounding_box(shape["points"])
    w = max_x - min_x
    h = max_y - min_y

    if x_cursor + w + SPACING > boundary["x"] + boundary["width"]:
        x_cursor = boundary["x"] + SPACING
        y_cursor += row_max_height + SPACING
        row_max_height = 0

    if y_cursor + h + SPACING > boundary["y"] + boundary["height"]:
        print("\u26A0\uFE0F Not enough vertical space. Skipping shape.")
        continue

    dx = x_cursor - min_x
    dy = y_cursor - min_y

    if shape["label"] == "circle":
        r = w / 2
        msp.add_circle(center=(x_cursor + r, y_cursor + r), radius=r)
    else:
        new_pts = [(p[0] + dx, p[1] + dy) for p in shape["points"]]
        msp.add_lwpolyline(new_pts, close=True)

    x_cursor += w + SPACING
    row_max_height = max(row_max_height, h)

# --- Step 5: Save output ---
doc.saveas(OUTPUT_DXF)
print("\u2705 Done! Saved arranged file to:", OUTPUT_DXF)
