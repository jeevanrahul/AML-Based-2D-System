import ezdxf
import math
import numpy as np

filename = ""
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

        angle = np.arcoss(cos_angle)
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