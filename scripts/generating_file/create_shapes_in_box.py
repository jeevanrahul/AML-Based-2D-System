import ezdxf
import math

# Create new DXF document
doc = ezdxf.new(setup=True)
msp = doc.modelspace()

# Units in mm
doc.units = ezdxf.units.MM

# Define bounding rectangle (200mm x 150mm)
rect_x, rect_y = 10, 10
rect_w, rect_h = 200, 150
msp.add_lwpolyline([
    (rect_x, rect_y),
    (rect_x + rect_w, rect_y),
    (rect_x + rect_w, rect_y + rect_h),
    (rect_x, rect_y + rect_h),
    (rect_x, rect_y)
], dxfattribs={"closed": True})

# Add unarranged shapes (scattered positions)

# Circle
msp.add_circle(center=(40, 40), radius=10)

# Square (30x30)
msp.add_lwpolyline([
    (100, 30),
    (130, 30),
    (130, 60),
    (100, 60),
    (100, 30)
], dxfattribs={"closed": True})

# Rectangle (40x20)
msp.add_lwpolyline([
    (60, 90),
    (100, 90),
    (100, 110),
    (60, 110),
    (60, 90)
], dxfattribs={"closed": True})

# Triangle (equilateral)
msp.add_lwpolyline([
    (140, 100),
    (155, 125.98),
    (170, 100),
    (140, 100)
], dxfattribs={"closed": True})

# Pentagon
angle = 2 * math.pi / 5
pentagon_center = (70, 50)
pentagon_radius = 15
pentagon_points = [
    (pentagon_center[0] + pentagon_radius * math.cos(i * angle),
     pentagon_center[1] + pentagon_radius * math.sin(i * angle))
    for i in range(5)
]
pentagon_points.append(pentagon_points[0])
msp.add_lwpolyline(pentagon_points, dxfattribs={"closed": True})

# Save the DXF file
doc.saveas(r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\raw\shapes_inside_box\sample_shapes_in_box.dxf")
print("✅ DXF with unarranged shapes saved.")
