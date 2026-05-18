# MSs_88578
Computer vision pipeline for synthetic pallet dataset generation, YOLOv10n box detection, 3D geometry reconstruction, and evaluation.

# YOLOv10n-Based 2D–3D Perception Pipeline for Palletized Box Scenes

## Overview

This repository contains a complete implementation of a synthetic 2D–3D perception pipeline for palletized logistics scenes. The project combines Blender-generated synthetic datasets, YOLOv10n object detection, and point-cloud-based 3D geometry estimation for irregular palletized box arrangements.

The implementation includes:

- synthetic dataset generation in Blender;
- RGB image rendering;
- YOLO-format annotation generation;
- instance segmentation mask generation;
- point cloud generation;
- YOLOv10n training and inference;
- 2D–3D fusion pipeline;
- point cloud filtering and clustering;
- 3D geometry estimation and evaluation.

---

# Repository Structure

```text
dataset/var1/

images/
├── train
├── val
└── test

labels/
├── train
├── val
└── test

masks/
├── train
├── val
└── test

pointcloud/
├── train
├── val
└── test

metadata/
runs/detect/
geometry_results/test/
```

Installation
1. Install Blender

Download Blender from:

https://www.blender.org/download/

Blender is used for procedural pallet-scene generation and dataset creation.
https://drive.google.com/file/d/134seawx58YHSI8hfFuEFLJO_6S9P7sTk/view?usp=sharing  

2. Create Python Environment

Create and activate a virtual environment.

Install required libraries:

ultralytics
open3d
numpy
pandas
opencv-python
matplotlib

Ultralytics repository:

https://github.com/ultralytics/ultralytics

Dataset Generation

Run:

Blender_pallet_scene_generation.py

The script generates:

RGB images;
YOLO bounding-box annotations;
segmentation masks;
point clouds;
JSON metadata.

Each scene is rendered from three camera views:

left
center
right

The generated dataset is automatically separated into:

train
validation
test
YOLOv10n Training

Run:

train_yolov10_var1.py

Training configuration:

Parameter	Value
Model	YOLOv10n
Epochs	50
Image size	640×640
Batch size	8
Confidence threshold	0.90

Training duration:

~11–11.5 hours

Final test results:

Metric	Result
Precision	0.9299
Recall	0.8872
mAP@50	0.9688
mAP@50–95	0.9176
YOLO Prediction

Run:

predict_yolov10_labels.py

The script generates YOLO prediction labels for the test dataset.

Predictions are saved into:

runs/detect/predict_test_for_3d/
3D Geometry Pipeline

Run:

run_geometry_from_yolo_predictions.py

The pipeline performs:

YOLO prediction loading;
point cloud loading;
3D-to-2D point projection;
object-level point extraction;
point cloud filtering;
DBSCAN clustering;
3D geometry estimation.

Results are saved into:

geometry_results/test/
Evaluation

Run:

check_geometry_all_results.py

The evaluation computes:

2D IoU;
3D IoU;
centroid error;
geometry statistics.
Runtime Performance
Measurement	Result
One pallet (3 views)	0.32 s
Average per view	0.11 s
Approximate FPS	9.1 FPS
Important Note

The current implementation does not use a dedicated 3D neural network.

YOLOv10n is used for 2D object detection, while point clouds provide geometric information for 3D estimation.

Current Limitation

The primary limitation of the current implementation is sensitivity to RGB–point-cloud alignment, which affects 3D geometry estimation quality.

Future Work

Potential future improvements include:

segmentation-mask-based point extraction;
YOLACT-style segmentation;
oriented 3D bounding boxes;
multi-view point cloud fusion;
grasp-point estimation;
sequential depalletization;
realistic warehouse simulation.
