from ultralytics import YOLO

BASE_DIR = "/Users/j.engels/Desktop/My_masters/code_yolov5/yolov5/dataset/var1"

MODEL_PATH = f"{BASE_DIR}/runs/detect/yolov10_var1_boxes/weights/best.pt"

model = YOLO(MODEL_PATH)

model.predict(
    source=f"{BASE_DIR}/images/test",
    imgsz=640,
    conf=0.90,
    iou=0.4,
    max_det=40,
    save=True,
    save_txt=True,
    save_conf=True,
    project=f"{BASE_DIR}/runs/detect",
    name="predict_test_for_3d",
    exist_ok=True,
    device="cpu"
)

print("YOLO predictions saved.")