import ezdxf
import math

FILENAME = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\sample_shapes_in_box.dxf"
OUTPUT = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\processed\test output\shapes_inside_rectangle_arranged.dxf"
DEFAULT_SPACING = 4

def detect_shape_type(polyline):
    points = list(polyline.get_points())
    count = len(points) - 1
    if count == 3:
        return "triangle"
    elif count == 4:
        return "rectangle"
    elif count == 5:
        return "pentagon"
    elif count == 10:
        return "star"
    else:
        return "polygon"

# Load DXF
doc = ezdxf.readfile(FILENAME)
msp = doc.modelspace()

boundary = None
shapes = []

# STEP 1: DETECT SHAPES
for entity in msp:
    if entity.dxftype() == "LWPOLYLINE" and entity.closed:
        pts = list(entity.get_points())
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        shape_type = detect_shape_type(entity)
        if shape_type == "rectangle" and (not boundary or w * h > boundary["width"] * boundary["height"]):
            boundary = {
                "entity": entity,
                "x": min(xs),
                "y": min(ys),
                "width": w,
                "height": h
            }
        else:
            shapes.append({
                "type": shape_type,
                "width": w,
                "height": h
            })
    elif entity.dxftype() == "CIRCLE":
        center = entity.dxf.center
        r = entity.dxf.radius
        shapes.append({
            "type": "circle",
            "radius": r,
            "width": r * 2,
            "height": r * 2
        })

# STEP 2: DELETE OLD SHAPES (except boundary)
for entity in list(msp):
    if entity != boundary["entity"]:
        msp.delete_entity(entity)

# STEP 3: ARRANGE SHAPES
current_x = boundary["x"] + DEFAULT_SPACING
current_y = boundary["y"] + DEFAULT_SPACING
row_max_height = 0
last_shape_type = None

for shape in shapes:
    w, h = shape["width"], shape["height"]

    if current_x + w + DEFAULT_SPACING > boundary["x"] + boundary["width"]:
        current_x = boundary["x"] + DEFAULT_SPACING
        current_y += row_max_height + DEFAULT_SPACING
        row_max_height = 0

    if current_y + h + DEFAULT_SPACING > boundary["y"] + boundary["height"]:
        print("⚠️ Skipped shape — not enough vertical space")
        continue

    if shape["type"] == "circle":
        center = (current_x + shape["radius"], current_y + shape["radius"])
        msp.add_circle(center=center, radius=shape["radius"])
    else:
        msp.add_lwpolyline([
            (current_x, current_y),
            (current_x + w, current_y),
            (current_x + w, current_y + h),
            (current_x, current_y + h),
            (current_x, current_y)
        ], dxfattribs={"closed": True})

    current_x += w + DEFAULT_SPACING
    row_max_height = max(row_max_height, h)
    last_shape_type = shape["type"]

# SAVE FILE
doc.saveas(OUTPUT)
print(f"✅ Successfully arranged and saved to: {OUTPUT}")
