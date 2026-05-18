import os
import json
import math
import numpy as np
import open3d as o3d
import time

# =========================
# CONFIG
# =========================

BASE_DIR = "/Users/j.engels/Desktop/My_masters/code_yolov5/yolov5/dataset/var1"

SPLIT = "test"

IMAGE_DIR = os.path.join(BASE_DIR, "images", SPLIT)
# POINTCLOUD_DIR = os.path.join(BASE_DIR, "pointcloud", SPLIT)
POINTCLOUD_DIR = os.path.join(BASE_DIR, "pointcloud_fused", SPLIT)
# METADATA_DIR = os.path.join(BASE_DIR, "metadata")
METADATA_DIR = os.path.join(BASE_DIR, "metadata", SPLIT)


PRED_LABEL_DIR = os.path.join(
    BASE_DIR,
    "runs/detect/predict_test_for_3d/labels"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "geometry_results_fused", SPLIT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

VOXEL_SIZE = 0.01


# =========================
# HELPERS
# =========================

def parse_scene_and_view(image_name):
    stem = os.path.splitext(image_name)[0]
    parts = stem.split("_")

    view_name = parts[-1]
    scene_id = "_".join(parts[:-1])

    if view_name not in ["left", "center", "right"]:
        raise ValueError(f"Unknown view name in file: {image_name}")

    return stem, scene_id, view_name


def read_yolo_predictions(label_path):
    predictions = []

    if not os.path.exists(label_path):
        return predictions

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 6:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            confidence = float(parts[5])

            x1 = (x_center - width / 2) * IMAGE_WIDTH
            y1 = (y_center - height / 2) * IMAGE_HEIGHT
            x2 = (x_center + width / 2) * IMAGE_WIDTH
            y2 = (y_center + height / 2) * IMAGE_HEIGHT

            predictions.append({
                "class_id": class_id,
                "confidence": confidence,
                "bbox_xyxy": [x1, y1, x2, y2]
            })

    return predictions


def euler_xyz_to_rotation_matrix(rx, ry, rz):
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def camera_intrinsics_from_blender(camera_meta, width_px, height_px):
    lens_mm = camera_meta["lens"]
    sensor_width_mm = camera_meta.get("sensor_width", 36.0)

    shift_x = camera_meta.get("shift_x", 0.0)
    shift_y = camera_meta.get("shift_y", 0.0)

    fx = lens_mm / sensor_width_mm * width_px
    fy = fx

    cx = width_px * (0.5 - shift_x)
    cy = height_px * (0.5 + shift_y)

    return fx, fy, cx, cy


def project_world_points_to_image(points_world, camera_meta):
    cam_loc = np.array(camera_meta["location"], dtype=np.float64)
    rx, ry, rz = camera_meta["rotation_euler"]

    R_world_cam = euler_xyz_to_rotation_matrix(rx, ry, rz)
    R_cam_world = R_world_cam.T

    fx, fy, cx, cy = camera_intrinsics_from_blender(
        camera_meta,
        IMAGE_WIDTH,
        IMAGE_HEIGHT
    )

    local = (points_world - cam_loc) @ R_cam_world.T

    X = local[:, 0]
    Y = -local[:, 1]
    Z = -local[:, 2]

    valid = Z > 0

    u = fx * X / (Z + 1e-9) + cx
    v = fy * Y / (Z + 1e-9) + cy

    valid &= (u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT)

    pixels = np.stack([u, v], axis=1)

    return pixels, valid


def yolo_norm_to_xyxy_pixels(yolo_bbox):
    xc = yolo_bbox["x_center"] * IMAGE_WIDTH
    yc = yolo_bbox["y_center"] * IMAGE_HEIGHT
    w = yolo_bbox["width"] * IMAGE_WIDTH
    h = yolo_bbox["height"] * IMAGE_HEIGHT

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    return [x1, y1, x2, y2]


def iou_2d(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    return inter / (area_a + area_b - inter + 1e-9)


def centroid_error(pred_center, gt_center):
    return float(np.linalg.norm(np.array(pred_center) - np.array(gt_center)))


def box3d_from_center_dims(center, dims):
    center = np.array(center, dtype=np.float64)
    dims = np.array(dims, dtype=np.float64)
    half = dims / 2.0
    return np.concatenate([center - half, center + half])


def iou_3d(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    z1 = max(box_a[2], box_b[2])

    x2 = min(box_a[3], box_b[3])
    y2 = min(box_a[4], box_b[4])
    z2 = min(box_a[5], box_b[5])

    inter = (
        max(0, x2 - x1) *
        max(0, y2 - y1) *
        max(0, z2 - z1)
    )

    vol_a = (
        max(0, box_a[3] - box_a[0]) *
        max(0, box_a[4] - box_a[1]) *
        max(0, box_a[5] - box_a[2])
    )

    vol_b = (
        max(0, box_b[3] - box_b[0]) *
        max(0, box_b[4] - box_b[1]) *
        max(0, box_b[5] - box_b[2])
    )

    return float(inter / (vol_a + vol_b - inter + 1e-9))


# =========================
# POINT CLOUD PROCESSING
# =========================

def crop_points_by_bbox(points, pixels, valid_mask, bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy

    mask = (
        valid_mask &
        (pixels[:, 0] >= x1) &
        (pixels[:, 0] <= x2) &
        (pixels[:, 1] >= y1) &
        (pixels[:, 1] <= y2)
    )

    return points[mask]


def refine_point_cloud(points):
    if len(points) < 20:
        return None, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    if len(pcd.points) < 20:
        return None, None

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    if len(pcd.points) < 20:
        return None, None

    print(f"[DBSCAN] points before clustering: {len(pcd.points)}", flush=True)

    labels = np.array(
        pcd.cluster_dbscan(
            eps=0.025,
            min_points=10,
            print_progress=False
        )
    )

    if labels.size > 0 and labels.max() >= 0:
        valid_labels = labels[labels >= 0]
        largest_label = np.bincount(valid_labels).argmax()
        cluster_indices = np.where(labels == largest_label)[0]
        pcd = pcd.select_by_index(cluster_indices)

    if len(pcd.points) < 20:
        return None, None

    aabb = pcd.get_axis_aligned_bounding_box()

    geometry = {
        "center": [round(float(v), 5) for v in aabb.get_center()],
        "dimensions": [round(float(v), 5) for v in aabb.get_extent()],
        "num_points": int(len(pcd.points))
    }

    return pcd, geometry


def match_prediction_to_gt(pred_bbox, gt_boxes, view_name):
    best_gt = None
    best_iou = 0.0

    bbox_key = f"yolo_bbox_{view_name}"

    for gt in gt_boxes:
        if bbox_key not in gt:
            continue

        gt_bbox_px = yolo_norm_to_xyxy_pixels(gt[bbox_key])
        score = iou_2d(pred_bbox, gt_bbox_px)

        if score > best_iou:
            best_iou = score
            best_gt = gt

    return best_gt, best_iou


# =========================
# MAIN PIPELINE
# =========================

def process_one_scene(image_name):
    print(f"\n[START] {image_name}", flush=True)

    stem, scene_id, view_name = parse_scene_and_view(image_name)
    print("[1] Parsed scene/view", flush=True)

    image_path = os.path.join(IMAGE_DIR, image_name)
    # pointcloud_path = os.path.join(POINTCLOUD_DIR, f"{stem}.ply")
    pointcloud_path = os.path.join(POINTCLOUD_DIR, f"{scene_id}_fused.ply")
    metadata_path = os.path.join(METADATA_DIR, f"{scene_id}.json")
    pred_label_path = os.path.join(PRED_LABEL_DIR, f"{stem}.txt")

    print("[2] Paths prepared", flush=True)

    if not os.path.exists(pointcloud_path):
        print(f"[SKIP] No point cloud: {pointcloud_path}", flush=True)
        return None

    if not os.path.exists(metadata_path):
        print(f"[SKIP] No metadata: {metadata_path}", flush=True)
        return None

    if not os.path.exists(pred_label_path):
        print(f"[SKIP] No YOLO prediction label: {pred_label_path}", flush=True)
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("[3] Metadata loaded", flush=True)

    if "views" not in metadata or view_name not in metadata["views"]:
        print(f"[SKIP] View metadata not found: {scene_id}, {view_name}", flush=True)
        return None

    camera_meta = metadata["views"][view_name]["camera"]
    gt_boxes = metadata["boxes"]

    print("[4] Loading YOLO predictions from txt...", flush=True)
    predictions = read_yolo_predictions(pred_label_path)
    print(f"[5] Predictions loaded: {len(predictions)}", flush=True)

    if len(predictions) == 0:
        print(f"[SKIP] No predictions in label file: {pred_label_path}", flush=True)
        return None

    print("[6] Loading point cloud...", flush=True)
    pcd_full = o3d.io.read_point_cloud(pointcloud_path)
    points = np.asarray(pcd_full.points)
    print(f"[7] Point cloud loaded. Points: {len(points)}", flush=True)

    if len(points) == 0:
        print(f"[SKIP] Empty point cloud: {pointcloud_path}", flush=True)
        return None

    print("[8] Projecting points to image...", flush=True)
    pixels, valid_mask = project_world_points_to_image(points, camera_meta)
    print("[9] Projection done", flush=True)

    scene_results = {
        "scene_id": scene_id,
        "view": view_name,
        "image_path": image_path,
        "fused_pointcloud_path": pointcloud_path,
        "prediction_label_path": pred_label_path,
        "num_predictions": int(len(predictions)),
        "objects": []
    }

    for idx, pred in enumerate(predictions):
        print(f"[10] Processing object {idx}", flush=True)

        pred_bbox = pred["bbox_xyxy"]
        conf = pred["confidence"]

        object_points = crop_points_by_bbox(points, pixels, valid_mask, pred_bbox)
        print(f"[11] Cropped object points: {len(object_points)}", flush=True)

        refined_pcd, pred_geom = refine_point_cloud(object_points)

        if pred_geom is None:
            print(f"[SKIP] Object {idx}: insufficient/refined points", flush=True)
            continue

        gt, gt_iou2d = match_prediction_to_gt(pred_bbox, gt_boxes, view_name)

        object_result = {
            "prediction_id": int(idx),
            "confidence": round(float(conf), 4),
            "bbox_xyxy": [round(float(v), 2) for v in pred_bbox],
            "predicted_3d": pred_geom,
            "matched_gt_id": None,
            "gt_2d_iou": round(float(gt_iou2d), 4),
            "centroid_error": None,
            "iou_3d": None
        }

        if gt is not None:
            object_result["matched_gt_id"] = gt["id"]

            gt_center = gt["center"]
            gt_dims = gt["dimensions"]

            object_result["centroid_error"] = round(
                centroid_error(pred_geom["center"], gt_center),
                5
            )

            pred_box3d = box3d_from_center_dims(
                pred_geom["center"],
                pred_geom["dimensions"]
            )

            gt_box3d = box3d_from_center_dims(
                gt_center,
                gt_dims
            )

            object_result["iou_3d"] = round(
                iou_3d(pred_box3d, gt_box3d),
                5
            )

        out_ply = os.path.join(
            OUTPUT_DIR,
            f"{stem}_object_{idx:03d}.ply"
        )

        o3d.io.write_point_cloud(out_ply, refined_pcd)

        scene_results["objects"].append(object_result)

    out_json = os.path.join(OUTPUT_DIR, f"{stem}_geometry.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scene_results, f, indent=4)

    print(f"[DONE] {stem}: {len(scene_results['objects'])} geometry objects saved", flush=True)

    return scene_results


def main():
    image_names = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    scene_id = "scene_0000"

    scene_images = [
        f"{scene_id}_left.jpg",
        f"{scene_id}_center.jpg",
        f"{scene_id}_right.jpg"
    ]

    print(f"Measuring runtime for pallet scene: {scene_id}", flush=True)

    start_time = time.perf_counter()

    for image_name in scene_images:
        process_one_scene(image_name)

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / len(scene_images)

    print("\n===== RUNTIME RESULT =====")
    print(f"Scene: {scene_id}")
    print(f"Views processed: {len(scene_images)}")
    print(f"Total time for one pallet / three views: {total_time:.2f} seconds")
    print(f"Average time per view: {avg_time:.2f} seconds")

if __name__ == "__main__":
    main()
# python run_geometry_from_yolo_predictions.py
# python check_geometry_result.py
# python check_geometry_all_results.py