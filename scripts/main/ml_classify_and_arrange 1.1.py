import ezdxf
import joblib
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# === Config Paths ===
INPUT_DXF = r"data/raw/shapes_inside_box/sample.dxf"
OUTPUT_DXF = r"data/output/sample_output.dxf"
MODEL_PATH = r"models/random_forest_model.pkl"
ENCODER_PATH = r"models/label_encoder.pkl"
SCALER_PATH = r"models/minmax_scaler.pkl"
FEATURE_COLUMNS_PATH = r"models/feature_columns.txt"
SPACING = 4  # mm

# === Geometry Utilities ===
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def polygon_area(points):
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + [points[0]])))

def perimeter(points):
    return sum(distance(points[i], points[(i + 1) % len(points)]) for i in range(len(points)))

def aspect_ratio(points):
    xs, ys = zip(*points)
    return (max(xs) - min(xs)) / (max(ys) - min(ys)) if (max(ys) - min(ys)) != 0 else 0

def bounding_box(points):
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)

def get_circle_points(entity):
    center = entity.dxf.center
    radius = entity.dxf.radius
    return [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 36)]

def get_polyline_points(entity):
    return [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices()]

def vertex_angles(points):
    angles = []
    n = len(points)
    for i in range(n):
        p0, p1, p2 = np.array(points[i - 1]), np.array(points[i]), np.array(points[(i + 1) % n])
        v1, v2 = p0 - p1, p2 - p1
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_angle = np.clip(np.dot(v1, v2) / denom if denom != 0 else 1.0, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_angle)))
    return angles

def count_acute_obtuse_angles(angles):
    return sum(a < 90 for a in angles), sum(a > 90 for a in angles)

def is_convex(points):
    if len(points) < 4:
        return True
    signs = []
    for i in range(len(points)):
        dx1 = points[(i + 1) % len(points)][0] - points[i][0]
        dy1 = points[(i + 1) % len(points)][1] - points[i][1]
        dx2 = points[(i + 2) % len(points)][0] - points[(i + 1) % len(points)][0]
        dy2 = points[(i + 2) % len(points)][1] - points[(i + 1) % len(points)][1]
        signs.append((dx1 * dy2 - dy1 * dx2) > 0)
    return all(signs) or not any(signs)

def compactness(area, peri):
    return (4 * math.pi * area) / (peri ** 2) if peri != 0 else 0

def centroid(points):
    xs, ys = zip(*points)
    return sum(xs) / len(xs), sum(ys) / len(ys)

def extract_features(entity):
    if entity.dxftype() == 'CIRCLE':
        points = get_circle_points(entity)
    elif entity.dxftype() == 'LWPOLYLINE':
        points = [(p[0], p[1]) for p in entity.get_points()]
    elif entity.dxftype() == 'POLYLINE':
        points = get_polyline_points(entity)
    else:
        return None, None

    if len(points) < 3:
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

def apply_rule_based_label(features, predicted):
    if features["num_points"] == 4:
        if 0.9 < features["aspect_ratio"] < 1.1:
            return "square"
        return "rectangle"
    if features["num_points"] >= 30 and features["compactness"] > 0.9:
        return "circle"
    return predicted

# === Load Model and Scaler ===
print("🔄 Loading model and scaler...")
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATURE_COLUMNS_PATH) as f:
    expected_cols = [line.strip() for line in f]

# === Parse DXF ===
print("📂 Reading DXF:", INPUT_DXF)
doc = ezdxf.readfile(INPUT_DXF)
msp = doc.modelspace()

shapes = []
boundary = None
for e in msp:
    if e.dxftype() in ["LWPOLYLINE", "POLYLINE"] and e.closed:
        pts = [(p[0], p[1]) for p in e.get_points()] if e.dxftype() == "LWPOLYLINE" else get_polyline_points(e)
        width, height = max(p[0] for p in pts) - min(p[0] for p in pts), max(p[1] for p in pts) - min(p[1] for p in pts)
        if not boundary or width * height > boundary["width"] * boundary["height"]:
            boundary = {"entity": e, "x": min(p[0] for p in pts), "y": min(p[1] for p in pts), "width": width, "height": height}
        else:
            f, pts = extract_features(e)
            if f: shapes.append({"entity": e, "features": f, "points": pts})
    elif e.dxftype() == "CIRCLE":
        f, pts = extract_features(e)
        if f: shapes.append({"entity": e, "features": f, "points": pts})

if not boundary:
    raise ValueError("❌ No rectangle boundary found!")

# === Predict and Correct Labels ===
df = pd.DataFrame([s["features"] for s in shapes])
df = df.reindex(columns=expected_cols, fill_value=0)
df_scaled = scaler.transform(df)
preds = model.predict(df_scaled)
labels = encoder.inverse_transform(preds)
final_labels = []
for s, pred in zip(shapes, labels):
    corrected = apply_rule_based_label(s["features"], pred)
    s["label"] = corrected
    final_labels.append(corrected)
    print(f"🧠 Predicted: {pred} → Final: {corrected}")

df["predicted_label"] = labels
df["final_label"] = final_labels
df.to_csv("debug_features.csv", index=False)

# === Clear Existing Shapes ===
for e in list(msp):
    if e != boundary["entity"]:
        msp.delete_entity(e)

# === Arrange Inside Rectangle ===
x_cursor, y_cursor = boundary["x"] + SPACING, boundary["y"] + SPACING
row_max_height = 0
for shape in shapes:
    min_x, min_y, max_x, max_y = bounding_box(shape["points"])
    w, h = max_x - min_x, max_y - min_y
    if x_cursor + w + SPACING > boundary["x"] + boundary["width"]:
        x_cursor = boundary["x"] + SPACING
        y_cursor += row_max_height + SPACING
        row_max_height = 0
    if y_cursor + h + SPACING > boundary["y"] + boundary["height"]:
        print(f"⚠️ Skipping shape ({shape['label']}) due to space constraints.")
        continue
    dx, dy = x_cursor - min_x, y_cursor - min_y
    new_pts = [(p[0] + dx, p[1] + dy) for p in shape["points"]]
    msp.add_lwpolyline(new_pts, close=True)
    x_cursor += w + SPACING
    row_max_height = max(row_max_height, h)

# === Save Output ===
doc.saveas(OUTPUT_DXF)
print("✅ Done! Saved arranged file to:", OUTPUT_DXF)
