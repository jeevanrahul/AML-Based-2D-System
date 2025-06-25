import ezdxf
import math
import numpy as np
import pandas as pd

# Change this to your test input
dxf_path = r"data\raw\shapes_inside_box\own.dxf"

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

def bounding_box_area(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

def get_circle_points(center, radius):
    return [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 36)]

# Load DXF and extract features
doc = ezdxf.readfile(dxf_path)
msp = doc.modelspace()

features_list = []
for entity in msp:
    if entity.dxftype() == 'CIRCLE':
        center = entity.dxf.center
        radius = entity.dxf.radius
        points = get_circle_points(center, radius)
    elif entity.dxftype() == 'LWPOLYLINE' and entity.closed:
        points = [(p[0], p[1]) for p in entity.get_points()]
    else:
        continue

    area = polygon_area(points)
    peri = perimeter(points)
    angles = vertex_angles(points)
    acute, obtuse = count_acute_obtuse_angles(angles)

    features = {
        "num_points": len(points),
        "area": area,
        "perimeter": peri,
        "aspect_ratio": aspect_ratio(points),
        "compactness": compactness(area, peri),
        "bounding_box_area": bounding_box_area(points),
        "centroid_x": centroid(points)[0],
        "centroid_y": centroid(points)[1],
        "acute_angle_count": acute,
        "obtuse_angle_count": obtuse
    }
    features_list.append(features)

df = pd.DataFrame(features_list)
print(df)
