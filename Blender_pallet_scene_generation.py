import os
import bpy
import json
import random
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# =========================================================
# SETTINGS
# =========================================================
base_dir = "/Users/j.engels/Desktop/My_masters/code_yolov5/yolov5/dataset/var1"


#image_dir = os.path.join(base_dir, "images", "train")
#label_dir = os.path.join(base_dir, "labels", "train")
#meta_dir = os.path.join(base_dir, "metadata")
#pc_dir = os.path.join(base_dir, "pointcloud", "train")
#seg_dir = os.path.join(base_dir, "masks", "train")


#image_dir = os.path.join(base_dir, "images", "val")
#label_dir = os.path.join(base_dir, "labels", "val")
#meta_dir = os.path.join(base_dir, "metadata", "val")
#pc_dir = os.path.join(base_dir, "pointcloud", "val")
#seg_dir = os.path.join(base_dir, "masks", "val")

image_dir = os.path.join(base_dir, "images", "test")
label_dir = os.path.join(base_dir, "labels", "test")
meta_dir = os.path.join(base_dir, "metadata", "test")
pc_dir = os.path.join(base_dir, "pointcloud", "test")
seg_dir = os.path.join(base_dir, "masks", "test")
fused_pc_dir = os.path.join(base_dir, "pointcloud_fused", "test")
os.makedirs(fused_pc_dir, exist_ok=True)

os.makedirs(pc_dir, exist_ok=True)
os.makedirs(seg_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)

NUM_SCENES = 100
START_INDEX = 0

# =========================================================
# GLOBAL PARAMETERS
# =========================================================
PALLET_L = 1.2
PALLET_W = 0.8
PALLET_H = 0.144

MIN_PARENT_COVERAGE = 0.85
HEIGHT_TOLERANCE = 0.035

GRID_COLS = 5
GRID_ROWS = 3

BOX_GAP = 0.002
MIN_SUPPORT_RATIO = 0.85

margin_x = 0.010
margin_y = 0.010

usable_x_left = -PALLET_L / 2 + margin_x
usable_x_right = PALLET_L / 2 - margin_x
usable_y_bottom = -PALLET_W / 2 + margin_y
usable_y_top = PALLET_W / 2 - margin_y

usable_w = usable_x_right - usable_x_left
usable_d = usable_y_top - usable_y_bottom

CELL_W = usable_w / GRID_COLS
CELL_D = usable_d / GRID_ROWS

# =========================================================
# HELPERS
# =========================================================
def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block)


def make_material(name, color, roughness=0.9):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat

def make_mask_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = 1.0

    output = nodes.new(type="ShaderNodeOutputMaterial")

    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    return mat


def add_box_object(name, location, dimensions, material=None):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2)
    if material is not None:
        obj.data.materials.append(material)
    return obj


def setup_scene(image_path):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 95
    scene.render.filepath = image_path
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100

    scene.eevee.use_shadows = True
    scene.eevee.taa_render_samples = 32
    
    scene.view_layers["ViewLayer"].use_pass_object_index = True

    world = scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.96, 0.96, 0.96, 1.0)
    bg.inputs[1].default_value = 0.9

    return scene

def setup_segmentation_output(scene, output_path):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new(type='CompositorNodeRLayers')

    id_mask = tree.nodes.new(type='CompositorNodeIDMask')
    id_mask.index = 1  # будем менять динамически

    composite = tree.nodes.new(type='CompositorNodeComposite')
    file_output = tree.nodes.new(type='CompositorNodeOutputFile')

    file_output.base_path = ""
    file_output.file_slots[0].path = output_path
    file_output.format.file_format = 'PNG'

    tree.links.new(render_layers.outputs['IndexOB'], id_mask.inputs['ID value'])
    tree.links.new(id_mask.outputs['Alpha'], file_output.inputs[0])
    
def id_to_color(idx):
    rnd = random.Random(idx)
    return (
        rnd.uniform(0.25, 1.0),
        rnd.uniform(0.25, 1.0),
        rnd.uniform(0.25, 1.0)
    )
    
def save_instance_mask(scene, camera, box_objects, filepath):
    original_filepath = scene.render.filepath
    original_format = scene.render.image_settings.file_format

    original_materials = {}
    original_world_color = None

    # black world background
    world = scene.world
    if world and world.use_nodes:
        bg = world.node_tree.nodes["Background"]
        original_world_color = bg.inputs[0].default_value[:]
        bg.inputs[0].default_value = (0, 0, 0, 1)

    # hide non-box objects
    hidden_objects = []
    for obj in bpy.context.scene.objects:
        if not obj.name.startswith("Box_") and obj.type != "CAMERA":
            hidden_objects.append(obj)
            obj.hide_render = True

    # give each box unique grayscale material
    for obj, meta in box_objects:
        original_materials[obj.name] = list(obj.data.materials)

        obj.data.materials.clear()

        color = id_to_color(meta["id"])
        mat = make_mask_material(f"MaskMat_{meta['id']}", color)
        obj.data.materials.append(mat)

    scene.camera = camera
    scene.render.filepath = filepath
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    bpy.ops.render.render(write_still=True)

    # restore
    scene.render.filepath = original_filepath
    scene.render.image_settings.file_format = original_format

    for obj, meta in box_objects:
        obj.data.materials.clear()
        for mat in original_materials[obj.name]:
            obj.data.materials.append(mat)

    for obj in hidden_objects:
        obj.hide_render = False

    if world and world.use_nodes and original_world_color is not None:
        bg = world.node_tree.nodes["Background"]
        bg.inputs[0].default_value = original_world_color
    

def create_materials():
    floor_mat = make_material("FloorMat", (0.90, 0.90, 0.90), 0.95)
    wood_mat = make_material("WoodMat", (0.58, 0.40, 0.22), 0.95)

    box_palette = [
        (0.82, 0.68, 0.44),
        (0.76, 0.60, 0.38),
        (0.72, 0.56, 0.34),
        (0.85, 0.72, 0.50),
        (0.68, 0.53, 0.32),
        (0.79, 0.64, 0.41),
    ]

    box_materials = []
    for i, color in enumerate(box_palette):
        box_materials.append(make_material(f"BoxMatPalette_{i}", color, 0.92))

    return floor_mat, wood_mat, box_materials


def create_floor(floor_mat):
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.data.materials.append(floor_mat)


def create_pallet(wood_mat):
    top_board_h = 0.022
    top_board_z = PALLET_H - top_board_h / 2
    top_board_width = 0.12
    top_board_positions_y = [-0.32, -0.16, 0.0, 0.16, 0.32]

    for i, y in enumerate(top_board_positions_y):
        add_box_object(
            name=f"PalletTopBoard_{i}",
            location=(0, y, top_board_z),
            dimensions=(1.2, top_board_width, top_board_h),
            material=wood_mat
        )

    bottom_board_h = 0.022
    bottom_board_z = bottom_board_h / 2
    bottom_board_width = 0.12
    bottom_board_positions_x = [-0.42, 0.0, 0.42]

    for i, x in enumerate(bottom_board_positions_x):
        add_box_object(
            name=f"PalletBottomBoard_{i}",
            location=(x, 0, bottom_board_z),
            dimensions=(bottom_board_width, 0.8, bottom_board_h),
            material=wood_mat
        )

    block_h = PALLET_H - top_board_h - bottom_board_h
    block_z = bottom_board_h + block_h / 2
    block_positions = [
        (-0.42, -0.28), (0.0, -0.28), (0.42, -0.28),
        (-0.42,  0.00), (0.0,  0.00), (0.42,  0.00),
        (-0.42,  0.28), (0.0,  0.28), (0.42,  0.28),
    ]

    for i, (x, y) in enumerate(block_positions):
        add_box_object(
            name=f"PalletBlock_{i}",
            location=(x, y, block_z),
            dimensions=(0.10, 0.10, block_h),
            material=wood_mat
        )


def create_camera_at_view(scene, view_name):
    camera_configs = {
    "left": {
        "location": (-1.8, 1.7, 2.0),
        "rotation": (1.05, 0.0, -2.40),
        "lens": 45
    },
    "center": {
        "location": (0.0, 0.0, 2.7),
        "rotation": (0.0, 0.0, 0.0),
        "lens": 45
    },
    "right": {
        "location": (1.8, -1.7, 2.0),
        "rotation": (1.05, 0.0, 0.80),
        "lens": 45
    }
}

    cfg = camera_configs[view_name]

    bpy.ops.object.camera_add(
        location=cfg["location"],
        rotation=cfg["rotation"]
    )

    camera = bpy.context.active_object
    camera.name = f"Camera_{view_name}"
    camera.data.lens = cfg["lens"]
    scene.camera = camera

    return camera


def create_lights():
    bpy.ops.object.light_add(type='AREA', location=(2.2, -2.0, 3.0))
    light1 = bpy.context.active_object
    light1.data.energy = 1450
    light1.data.shape = 'RECTANGLE'
    light1.data.size = 3.0
    light1.data.size_y = 3.0

    bpy.ops.object.light_add(type='AREA', location=(-2.0, 1.6, 2.5))
    light2 = bpy.context.active_object
    light2.data.energy = 850
    light2.data.shape = 'RECTANGLE'
    light2.data.size = 2.5
    light2.data.size_y = 2.5


def get_yolo_bbox(obj, cam, scene):
    mat = obj.matrix_world
    coords = [mat @ Vector(corner) for corner in obj.bound_box]
    coords_2d = [world_to_camera_view(scene, cam, coord) for coord in coords]

    xs = [co.x for co in coords_2d]
    ys = [co.y for co in coords_2d]
    zs = [co.z for co in coords_2d]

    if all(z < 0 for z in zs):
        return None

    min_x = max(min(xs), 0.0)
    max_x = min(max(xs), 1.0)
    min_y = max(min(ys), 0.0)
    max_y = min(max(ys), 1.0)

    if max_x <= min_x or max_y <= min_y:
        return None

    x_center = (min_x + max_x) / 2
    y_center = 1 - ((min_y + max_y) / 2)
    width = max_x - min_x
    height = max_y - min_y

    return x_center, y_center, width, height


def project_world_point_to_image(world_point, cam, scene):
    co_2d = world_to_camera_view(scene, cam, Vector(world_point))
    if co_2d.z < 0:
        return None

    render_scale = scene.render.resolution_percentage / 100.0
    width = int(scene.render.resolution_x * render_scale)
    height = int(scene.render.resolution_y * render_scale)

    x_norm = co_2d.x
    y_norm = 1.0 - co_2d.y
    u_px = int(round(x_norm * width))
    v_px = int(round(y_norm * height))

    return {
        "norm": [round(x_norm, 6), round(y_norm, 6)],
        "px": [u_px, v_px]
    }


def cell_bounds(col, row):
    x0 = usable_x_left + col * CELL_W
    x1 = x0 + CELL_W
    y0 = usable_y_bottom + row * CELL_D
    y1 = y0 + CELL_D
    return x0, x1, y0, y1


def tile_fits(used, col, row, tw, th):
    if col + tw > GRID_COLS or row + th > GRID_ROWS:
        return False

    for rr in range(row, row + th):
        for cc in range(col, col + tw):
            if used[rr][cc]:
                return False

    return True


def mark_tile(used, col, row, tw, th):
    for rr in range(row, row + th):
        for cc in range(col, col + tw):
            used[rr][cc] = True


def get_cells_for_tile(col, row, tw, th):
    return [(cc, rr) for rr in range(row, row + th) for cc in range(col, col + tw)]


def tile_size_class(tw, th):
    area = tw * th
    if area == 1:
        return "small"
    elif area == 2:
        return "medium"
    else:
        return "large"


def choose_height(size_class, layer_idx):
    base_by_layer = {
        1: 0.24,
        2: 0.20,
        3: 0.16
    }

    base = base_by_layer.get(layer_idx, 0.18)

    if size_class == "small":
        return random.uniform(base * 0.75, base * 1.05)
    elif size_class == "medium":
        return random.uniform(base * 0.90, base * 1.15)
    else:
        return random.uniform(base * 1.00, base * 1.30)


def get_preferred_tiles_for_layer(layer_idx):
    r = random.random()

    if layer_idx == 1:
        if r < 0.40:
            return [(3, 2), (2, 2), (3, 1), (2, 1), (1, 2), (1, 1)]
        elif r < 0.80:
            return [(2, 2), (2, 1), (1, 2), (3, 1), (1, 1)]
        else:
            return [(2, 1), (1, 2), (1, 1)]

    elif layer_idx == 2:
        if r < 0.55:
            return [(2, 1), (1, 2), (2, 2), (1, 1)]
        elif r < 0.85:
            return [(1, 2), (2, 1), (3, 1), (1, 1)]
        else:
            return [(2, 2), (3, 1), (1, 3), (1, 1)]

    else:
        if r < 0.65:
            return [(1, 1), (1, 2), (2, 1)]
        elif r < 0.90:
            return [(2, 1), (1, 2), (1, 1), (2, 2)]
        else:
            return [(3, 1), (1, 3), (2, 2), (1, 1)]


def tile_world_box_var(col, row, tw, th, size_class, gap=BOX_GAP):
    x0, _, y0, _ = cell_bounds(col, row)
    _, x1, _, y1 = cell_bounds(col + tw - 1, row + th - 1)

    full_w = (x1 - x0) - gap
    full_d = (y1 - y0) - gap

    if size_class == "small":
        shrink_w = random.choice([0.04, 0.06, 0.08, 0.10])
        shrink_d = random.choice([0.04, 0.06, 0.08, 0.10])

    elif size_class == "medium":
        shrink_w = random.choice([0.02, 0.04, 0.06, 0.08])
        shrink_d = random.choice([0.02, 0.04, 0.06, 0.08])

    else:
        shrink_w = random.choice([0.00, 0.02, 0.04, 0.06])
        shrink_d = random.choice([0.00, 0.02, 0.04, 0.06])

    w = max(0.08, full_w * (1 - shrink_w))
    d = max(0.08, full_d * (1 - shrink_d))

    free_x = full_w - w
    free_y = full_d - d

    offset_x = random.uniform(-free_x / 2, free_x / 2)
    offset_y = random.uniform(-free_y / 2, free_y / 2)

    base_cx = (x0 + x1) / 2
    base_cy = (y0 + y1) / 2

    x = base_cx + offset_x
    y = base_cy + offset_y

    return x, y, w, d


def generate_layer_pattern(layer_idx):
    used = [[False for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    tiles = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if used[row][col]:
                continue

            preferred = get_preferred_tiles_for_layer(layer_idx)

            chosen = None
            for tw, th in preferred:
                if tile_fits(used, col, row, tw, th):
                    chosen = (tw, th)
                    break

            if chosen is None:
                chosen = (1, 1)

            tw, th = chosen
            mark_tile(used, col, row, tw, th)
            tiles.append((col, row, tw, th))

    return tiles


def rect_intersection_area(a, b):
    x_overlap = max(0.0, min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"]))
    y_overlap = max(0.0, min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"]))
    return x_overlap * y_overlap


def build_box_rect_from_center_dims(x, y, w, d):
    return {
        "x_min": x - w / 2,
        "x_max": x + w / 2,
        "y_min": y - d / 2,
        "y_max": y + d / 2
    }


def get_support_info(x, y, w, d, below_layer):
    top_rect = build_box_rect_from_center_dims(x, y, w, d)
    top_area = w * d

    overlaps = []

    for below in below_layer:
        bx, by, _ = below["center"]
        bw, bd, _ = below["dimensions"]
        below_top_z = below["top_surface_center"][2]

        rect_below = build_box_rect_from_center_dims(bx, by, bw, bd)
        area = rect_intersection_area(top_rect, rect_below)

        if area > 0:
            overlaps.append({
                "id": below["id"],
                "area": area,
                "top_z": below_top_z
            })

    if not overlaps:
        return [], 0.0, None

    max_top_z = max(item["top_z"] for item in overlaps)

    usable_supports = [
        item for item in overlaps
        if max_top_z - item["top_z"] <= HEIGHT_TOLERANCE
    ]

    support_area = sum(item["area"] for item in usable_supports)
    support_ratio = support_area / top_area

    if support_ratio < MIN_PARENT_COVERAGE:
        return [], support_ratio, None

    support_ids = [item["id"] for item in usable_supports]

    # the upper box rests on the highest contacted surface
    return support_ids, support_ratio, max_top_z


def create_box(center, dims, layer_idx, camera, scene, box_materials,
               box_data, box_objects, box_id,
               support_box_ids=None, size_class=None, footprint_cells=None, support_ratio=None):

    x, y, z = center
    w, d, h = dims

    bpy.ops.mesh.primitive_cube_add(location=(x, y, z))
    obj = bpy.context.active_object
    obj.name = f"Box_{box_id}"
    obj.scale = (w / 2, d / 2, h / 2)
    obj.rotation_euler[2] = 0.0
    obj.data.materials.append(random.choice(box_materials))
    obj.pass_index = box_id

    top_surface_center = [x, y, z + h / 2]
    grasp_point = top_surface_center.copy()
    grasp_2d = project_world_point_to_image(grasp_point, camera, scene) if camera is not None else None

    meta = {
        "id": box_id,
        "name": obj.name,
        "layer": layer_idx,
        "size_class": size_class,
        "center": [round(x, 4), round(y, 4), round(z, 4)],
        "dimensions": [round(w, 4), round(d, 4), round(h, 4)],
        "rotation_z_deg": 0.0,
        "support_box_ids": support_box_ids if support_box_ids else [],
        "support_ratio": round(support_ratio, 4) if support_ratio is not None else None,
        "footprint_cells": footprint_cells if footprint_cells else [],
        "top_surface_center": [round(v, 4) for v in top_surface_center],
        "top_surface_size": [round(w, 4), round(d, 4)],
        "grasp_point": [round(v, 4) for v in grasp_point],
        "grasp_point_2d_norm": grasp_2d["norm"] if grasp_2d else None,
        "grasp_point_2d_px": grasp_2d["px"] if grasp_2d else None
    }

    box_data.append(meta)
    box_objects.append((obj, meta))

    return meta


def build_layer(layer_idx, below_layer, camera, scene, box_materials, box_data, box_objects, box_id_start):
    layer = []
    tiles = generate_layer_pattern(layer_idx)
    box_id = box_id_start

    for col, row, tw, th in tiles:
        size_class = tile_size_class(tw, th)
        cells = get_cells_for_tile(col, row, tw, th)

        if below_layer is None:
            x, y, w, d = tile_world_box_var(col, row, tw, th, size_class)
            h = choose_height(size_class, layer_idx)
            z = PALLET_H + h / 2

            meta = create_box(
                center=(x, y, z),
                dims=(w, d, h),
                layer_idx=layer_idx,
                camera=camera,
                scene=scene,
                box_materials=box_materials,
                box_data=box_data,
                box_objects=box_objects,
                box_id=box_id,
                support_box_ids=[],
                size_class=size_class,
                footprint_cells=cells,
                support_ratio=1.0
            )

            layer.append(meta)
            box_id += 1

        else:
            placed = False
            best_candidate = None
            best_ratio = -1.0

            for _ in range(80):
                x, y, w, d = tile_world_box_var(col, row, tw, th, size_class)
                support_ids, support_ratio, support_top_z = get_support_info(x, y, w, d, below_layer)

                if support_ratio > best_ratio:
                    best_ratio = support_ratio
                    best_candidate = (x, y, w, d, support_ids, support_ratio, support_top_z)

                if support_ratio >= MIN_SUPPORT_RATIO and support_top_z is not None:
                    h = choose_height(size_class, layer_idx)
                    z = support_top_z + h / 2

                    meta = create_box(
                        center=(x, y, z),
                        dims=(w, d, h),
                        layer_idx=layer_idx,
                        camera=camera,
                        scene=scene,
                        box_materials=box_materials,
                        box_data=box_data,
                        box_objects=box_objects,
                        box_id=box_id,
                        support_box_ids=support_ids,
                        size_class=size_class,
                        footprint_cells=cells,
                        support_ratio=support_ratio
                    )

                    layer.append(meta)
                    box_id += 1
                    placed = True
                    break

            if not placed:
                continue
#                x, y, w, d, _, _, _ = best_candidate

                for shrink in [0.95, 0.90, 0.85, 0.80, 0.75]:
                    ws = w * shrink
                    ds = d * shrink
                    support_ids, support_ratio, support_top_z = get_support_info(x, y, ws, ds, below_layer)

                    if support_ratio >= MIN_SUPPORT_RATIO and support_top_z is not None:
                        h = choose_height(size_class, layer_idx)
                        z = support_top_z + h / 2

                        meta = create_box(
                            center=(x, y, z),
                            dims=(ws, ds, h),
                            layer_idx=layer_idx,
                            camera=camera,
                            scene=scene,
                            box_materials=box_materials,
                            box_data=box_data,
                            box_objects=box_objects,
                            box_id=box_id,
                            support_box_ids=support_ids,
                            size_class=size_class,
                            footprint_cells=cells,
                            support_ratio=support_ratio
                        )

                        layer.append(meta)
                        box_id += 1
                        break

    return layer, box_id

def build_supported_layer(previous_layer, layer_idx, camera, scene,
                          box_materials, box_data, box_objects, box_id_start):
    layer = []
    box_id = box_id_start
    tiles = generate_layer_pattern(layer_idx)

    for col, row, tw, th in tiles:
        size_class = tile_size_class(tw, th)
        cells = get_cells_for_tile(col, row, tw, th)

        placed = False

        for _ in range(150):
            x, y, w, d = tile_world_box_var(col, row, tw, th, size_class)

            support_ids, support_ratio, support_top_z = get_support_info(
                x, y, w, d, previous_layer
            )

            if support_top_z is None:
                continue

            if support_ratio >= MIN_PARENT_COVERAGE:
                h = choose_height(size_class, layer_idx)
                z = support_top_z + h / 2

                meta = create_box(
                    center=(x, y, z),
                    dims=(w, d, h),
                    layer_idx=layer_idx,
                    camera=camera,
                    scene=scene,
                    box_materials=box_materials,
                    box_data=box_data,
                    box_objects=box_objects,
                    box_id=box_id,
                    support_box_ids=support_ids,
                    size_class=size_class,
                    footprint_cells=cells,
                    support_ratio=support_ratio
                )

                meta["support_height_tolerance"] = HEIGHT_TOLERANCE
                layer.append(meta)
                box_id += 1
                placed = True
                break

        if not placed:
            continue

    return layer, box_id

  


def build_child_layer_on_previous(previous_layer, layer_idx, camera, scene, box_materials, box_data, box_objects, box_id_start):
    """
    Builds a full child layer by filling the top surface of every box in the previous layer.
    Each parent box receives several smaller boxes that cover at least 90% of its top surface.
    """

    new_layer = []
    box_id = box_id_start

    for parent in previous_layer:
        children, box_id, coverage_ratio = pack_children_on_parent(
            parent=parent,
            layer_idx=layer_idx,
            camera=camera,
            scene=scene,
            box_materials=box_materials,
            box_data=box_data,
            box_objects=box_objects,
            box_id_start=box_id
        )

        parent[f"covered_by_layer_{layer_idx}_ratio"] = round(coverage_ratio, 4)
        parent[f"covered_by_layer_{layer_idx}_box_ids"] = [c["id"] for c in children]

        new_layer.extend(children)

    return new_layer, box_id

def export_camera_point_cloud(scene, camera, filepath, view_id=0, max_distance=6.0, step=4, only_boxes=True):
    """
    Exports visible point cloud from the active camera view.
    Points are saved in world coordinates.

    Each point contains:
    x, y, z, object_id, view_id

    object_id:
        0 = non-box object or unknown
        N = Box_N
    """

    depsgraph = bpy.context.evaluated_depsgraph_get()

    render_scale = scene.render.resolution_percentage / 100.0
    width = int(scene.render.resolution_x * render_scale)
    height = int(scene.render.resolution_y * render_scale)

    points = []

    view_frame = camera.data.view_frame(scene=scene)
    frame = [camera.matrix_world @ corner for corner in view_frame]

    top_left = frame[0]
    bottom_left = frame[1]
    bottom_right = frame[2]
    top_right = frame[3]

    origin = camera.matrix_world.translation

    for v in range(0, height, step):
        for u in range(0, width, step):
            x_ndc = (u + 0.5) / width
            y_ndc = 1.0 - ((v + 0.5) / height)

            top_point = top_left.lerp(top_right, x_ndc)
            bottom_point = bottom_left.lerp(bottom_right, x_ndc)
            point_on_near = bottom_point.lerp(top_point, y_ndc)

            ray_dir = (point_on_near - origin).normalized()

            hit, location, normal, index, obj, matrix = scene.ray_cast(
                depsgraph, origin, ray_dir, distance=max_distance
            )

            if not hit:
                continue

            object_id = 0

            if obj is not None and obj.name.startswith("Box_"):
                try:
                    object_id = int(obj.name.replace("Box_", "").split(".")[0])
                except:
                    object_id = 0

            if only_boxes and object_id == 0:
                continue

            points.append((
                location.x,
                location.y,
                location.z,
                object_id,
                view_id
            ))

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property int object_id\n")
        f.write("property int view_id\n")
        f.write("end_header\n")

        for x, y, z, object_id, view_id in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {object_id} {view_id}\n")

    return {
        "path": filepath,
        "num_points": len(points),
        "step": step,
        "max_distance": max_distance,
        "points": points
    }
    
def save_fused_point_cloud(filepath, all_points):
    """
    Saves fused multi-view point cloud.
    Input points format:
    x, y, z, object_id, view_id
    """

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property int object_id\n")
        f.write("property int view_id\n")
        f.write("end_header\n")

        for x, y, z, object_id, view_id in all_points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {object_id} {view_id}\n")

    return {
        "path": filepath,
        "num_points": len(all_points)
    }


def generate_one_scene(scene_index):
    scene_id = f"scene_{scene_index:04d}"
    json_path = os.path.join(meta_dir, f"{scene_id}.json")

    random.seed(113 + scene_index)

    clean_scene()

    # temporary filepath, потом для каждого ракурса будет свой путь
    temp_image_path = os.path.join(image_dir, f"{scene_id}_temp.jpg")
    scene = setup_scene(temp_image_path)

    floor_mat, wood_mat, box_materials = create_materials()

    create_floor(floor_mat)
    create_pallet(wood_mat)
    create_lights()

    box_data = []
    box_objects = []
    box_id = 1

    # 1) создаём коробки один раз
    layer1, box_id = build_layer(
        1, None, None, scene, box_materials, box_data, box_objects, box_id
    )

    layer2, box_id = build_layer(
        2, layer1, None, scene, box_materials, box_data, box_objects, box_id
    )

    layer3, box_id = build_layer(
        3, layer2, None, scene, box_materials, box_data, box_objects, box_id
    )

    views_metadata = {}
    
    all_fused_points = []
    view_id_map = {
        "left": 1,
        "center": 2,
        "right": 3
    }

    # 2) одну и ту же сцену снимаем с трёх камер
    for view_name in ["left", "center", "right"]:
        camera = create_camera_at_view(scene, view_name)

        view_image_path = os.path.join(image_dir, f"{scene_id}_{view_name}.jpg")
        view_label_path = os.path.join(label_dir, f"{scene_id}_{view_name}.txt")
        view_pc_path = os.path.join(pc_dir, f"{scene_id}_{view_name}.ply")
        view_mask_prefix = os.path.join(seg_dir, f"{scene_id}_{view_name}")

        scene.render.filepath = view_image_path
        view_mask_path = os.path.join(seg_dir, f"{scene_id}_{view_name}.png")

        # 3) обновляем 2D grasp point для конкретного ракурса
        for obj, meta in box_objects:
            gp = meta["grasp_point"]
            grasp_2d = project_world_point_to_image(gp, camera, scene)
            meta[f"grasp_point_2d_{view_name}"] = grasp_2d

        # 4) рендерим картинку для этого ракурса
        bpy.ops.render.render(write_still=True)
        save_instance_mask(scene, camera, box_objects, view_mask_path)


        pc_info = export_camera_point_cloud(
            scene=scene,
            camera=camera,
            filepath=view_pc_path,
            view_id=view_id_map[view_name],
            max_distance=6.0,
            step=3,
            only_boxes=True
        )

        all_fused_points.extend(pc_info["points"])
        # 5) создаём YOLO labels для этого ракурса
        label_lines = []

        for obj, meta in box_objects:
            bbox = get_yolo_bbox(obj, camera, scene)
            if bbox is None:
                continue

            x_center, y_center, width, height = bbox

            label_lines.append(
                f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

            # сохраняем bbox отдельно для каждого view в JSON
            meta[f"yolo_bbox_{view_name}"] = {
                "class_id": 0,
                "x_center": round(x_center, 6),
                "y_center": round(y_center, 6),
                "width": round(width, 6),
                "height": round(height, 6)
            }

        with open(view_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        views_metadata[view_name] = {
            "image_path": view_image_path,
            "label_path": view_label_path,
            "mask_path": view_mask_path,
            "point_cloud_path": view_pc_path,
            "point_cloud_points": pc_info["num_points"],
            "point_cloud_step": pc_info["step"],
            "camera": {
                "location": [
                    round(camera.location.x, 4),
                    round(camera.location.y, 4),
                    round(camera.location.z, 4)
                ],
                "rotation_euler": [
                    round(camera.rotation_euler.x, 4),
                    round(camera.rotation_euler.y, 4),
                    round(camera.rotation_euler.z, 4)
                ],
                "lens": camera.data.lens,
                "sensor_width": camera.data.sensor_width,
                "sensor_height": camera.data.sensor_height,
                "sensor_fit": camera.data.sensor_fit,
                "shift_x": camera.data.shift_x,
                "shift_y": camera.data.shift_y
            }
        }
    fused_pc_path = os.path.join(fused_pc_dir, f"{scene_id}_fused.ply")

    fused_pc_info = save_fused_point_cloud(
        filepath=fused_pc_path,
        all_points=all_fused_points
    )

    # 6) один общий JSON для всей сцены
    metadata = {
        "scene_id": scene_id,
        "json_path": json_path,
        "views": views_metadata,
        "fused_point_cloud": {
            "path": fused_pc_path,
            "points": fused_pc_info["num_points"],
            "views_used": ["left", "center", "right"],
            "description": "Fused multi-view point cloud generated by combining visible points from left, center, and right camera views in world coordinates."
        },
        "constraints": {
            "min_parent_coverage": MIN_PARENT_COVERAGE,
            "height_tolerance": HEIGHT_TOLERANCE,
            "rule": "Each scene is captured from three fixed camera views: left, center, and right."
        },
        "pallet": {
            "dimensions": [PALLET_L, PALLET_W, PALLET_H],
            "top_surface_z": PALLET_H
        },
        "grid": {
            "cols": GRID_COLS,
            "rows": GRID_ROWS,
            "cell_w": round(CELL_W, 4),
            "cell_d": round(CELL_D, 4)
        },
        "box_counts": {
            "layer1": len(layer1),
            "layer2": len(layer2),
            "layer3": len(layer3),
            "total": len(box_data)
        },
        "boxes": box_data
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Done {scene_id}")
    print("Views saved:")
    print(" - left")
    print(" - center")
    print(" - right")
    print("JSON:", json_path)
    print("Layer1:", len(layer1))
    print("Layer2:", len(layer2))
    print("Layer3:", len(layer3))
    print("Total boxes:", len(box_data))


# =========================================================
# MAIN LOOP
# =========================================================
for i in range(START_INDEX, START_INDEX + NUM_SCENES):
    generate_one_scene(i)

print("All scenes generated.")
