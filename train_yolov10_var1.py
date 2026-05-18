from ultralytics import YOLO

print("START TRAINING...")

model = YOLO("yolov10n.pt")

print("MODEL LOADED")

model.train(
    data="/Users/j.engels/Desktop/My_masters/code_yolov5/yolov5/dataset/var1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",
    name="yolov10_var1_boxes"
)
print("TRAINING FINISHED")