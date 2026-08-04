from ultralytics import YOLO

# Load the pretrained YOLOv8 Nano model
model = YOLO("models/pre_trained/yolov8n.pt")

# Train the model on the license plate dataset
model.train(
    data="dataset/YOLO_dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=4,
    device="cpu",
    project="runs",
    name="license_plate_detector",
    patience=5,
    workers=0
)