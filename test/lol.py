import ezdxf
import math

# Create new DXF
doc = ezdxf.new(setup=True)
msp = doc.modelspace()
doc.units = ezdxf.units.MM

# Draw bounding box (200mm x 150mm)
box_x, box_y = 10, 10
box_width, box_height = 200, 150
msp.add_lwpolyline([
    (box_x, box_y),
    (box_x + box_width, box_y),
    (box_x + box_width, box_y + box_height),
    (box_x, box_y + box_height),
    (box_x, box_y)
], dxfattribs={"closed": True})

# Function to create regular polygon points
def regular_polygon(cx, cy, sides, radius):
    return [
        (cx + radius * math.cos(2 * math.pi * i / sides),
         cy + radius * math.sin(2 * math.pi * i / sides))
        for i in range(sides)
    ]

# Add shapes in a grid
for i in range(2):  # 2 rows
    for j in range(3):  # 3 columns
        ox = 20 + j * 60
        oy = 20 + i * 60

        # 1. Circle
        msp.add_circle(center=(ox, oy), radius=5)

        # 2. Rectangle
        msp.add_lwpolyline([
            (ox + 12, oy),
            (ox + 32, oy),
            (ox + 32, oy + 10),
            (ox + 12, oy + 10),
            (ox + 12, oy)
        ], dxfattribs={"closed": True})

        # 3. Triangle
        msp.add_lwpolyline([
            (ox + 40, oy),
            (ox + 50, oy),
            (ox + 45, oy + 10),
            (ox + 40, oy)
        ], dxfattribs={"closed": True})

        # 4. Square
        msp.add_lwpolyline([
            (ox + 55, oy),
            (ox + 65, oy),
            (ox + 65, oy + 10),
            (ox + 55, oy + 10),
            (ox + 55, oy)
        ], dxfattribs={"closed": True})

        # 5. Pentagon
        pent = regular_polygon(ox + 75, oy + 5, sides=5, radius=5)
        pent.append(pent[0])
        msp.add_lwpolyline(pent, dxfattribs={"closed": True})

        # 6. Star (10-point)
        star_points = []
        cx, cy = ox + 90, oy + 5
        for k in range(10):
            angle = math.radians(k * 36)
            r = 5 if k % 2 == 0 else 2.5
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            star_points.append((x, y))
        star_points.append(star_points[0])
        msp.add_lwpolyline(star_points, dxfattribs={"closed": True})

# Save it
doc.saveas("sample_shapes_in_box.dxf")
print("✅ DXF file with all shapes created: sample_shapes_in_box.dxf")
