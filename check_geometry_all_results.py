import os
import json
import pandas as pd

BASE_DIR = "/Users/j.engels/Desktop/My_masters/code_yolov5/yolov5/dataset/var1"
SPLIT = "test"

RESULTS_DIR = os.path.join(BASE_DIR, "geometry_results_fused", SPLIT)

rows = []

json_files = sorted([
    f for f in os.listdir(RESULTS_DIR)
    if f.endswith("_geometry.json")
])

print(f"Found geometry result files: {len(json_files)}")

for filename in json_files:
    path = os.path.join(RESULTS_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scene_id = data.get("scene_id")
    view = data.get("view")

    for obj in data.get("objects", []):
        pred_3d = obj.get("predicted_3d", {})

        rows.append({
            "file": filename,
            "scene_id": scene_id,
            "view": view,
            "prediction_id": obj.get("prediction_id"),
            "confidence": obj.get("confidence"),
            "matched_gt_id": obj.get("matched_gt_id"),
            "gt_2d_iou": obj.get("gt_2d_iou"),
            "centroid_error": obj.get("centroid_error"),
            "iou_3d": obj.get("iou_3d"),
            "num_points": pred_3d.get("num_points"),
            "center": pred_3d.get("center"),
            "dimensions": pred_3d.get("dimensions")
        })

df = pd.DataFrame(rows)

if df.empty:
    print("No object results found.")
    raise SystemExit

for col in ["confidence", "gt_2d_iou", "centroid_error", "iou_3d", "num_points"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# OVERALL SUMMARY
# =========================

summary_rows = []

summary_rows.append({
    "group": "overall",
    "num_result_files": len(json_files),
    "num_objects": len(df),
    "mean_confidence": df["confidence"].mean(),
    "mean_2d_iou": df["gt_2d_iou"].mean(),
    "median_2d_iou": df["gt_2d_iou"].median(),
    "mean_centroid_error": df["centroid_error"].mean(),
    "median_centroid_error": df["centroid_error"].median(),
    "mean_3d_iou": df["iou_3d"].mean(),
    "median_3d_iou": df["iou_3d"].median(),
    "mean_num_points": df["num_points"].mean(),
    "median_num_points": df["num_points"].median()
})

# =========================
# SUMMARY BY VIEW
# =========================

for view, group_df in df.groupby("view"):
    summary_rows.append({
        "group": f"view_{view}",
        "num_result_files": group_df["file"].nunique(),
        "num_objects": len(group_df),
        "mean_confidence": group_df["confidence"].mean(),
        "mean_2d_iou": group_df["gt_2d_iou"].mean(),
        "median_2d_iou": group_df["gt_2d_iou"].median(),
        "mean_centroid_error": group_df["centroid_error"].mean(),
        "median_centroid_error": group_df["centroid_error"].median(),
        "mean_3d_iou": group_df["iou_3d"].mean(),
        "median_3d_iou": group_df["iou_3d"].median(),
        "mean_num_points": group_df["num_points"].mean(),
        "median_num_points": group_df["num_points"].median()
    })

summary_df = pd.DataFrame(summary_rows)

# =========================
# PRINT RESULTS
# =========================

print("\n==============================")
print("SUMMARY METRICS")
print("==============================")
print(summary_df.round(5))

print("\n==============================")
print("OVERALL MAIN VALUES")
print("==============================")
overall = summary_df[summary_df["group"] == "overall"].iloc[0]

print("Total result files:", int(overall["num_result_files"]))
print("Total predicted objects:", int(overall["num_objects"]))
print("Mean confidence:", round(overall["mean_confidence"], 5))
print("Mean 2D IoU:", round(overall["mean_2d_iou"], 5))
print("Mean centroid error:", round(overall["mean_centroid_error"], 5))
print("Mean 3D IoU:", round(overall["mean_3d_iou"], 5))
print("Mean object points:", round(overall["mean_num_points"], 2))

# =========================
# SAVE CSV FILES
# =========================

details_csv = os.path.join(RESULTS_DIR, "aggregated_geometry_results.csv")
summary_csv = os.path.join(RESULTS_DIR, "summary_geometry_metrics.csv")

df.to_csv(details_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

print("\nDetailed object-level CSV saved to:")
print(details_csv)

print("\nSummary metrics CSV saved to:")
print(summary_csv)