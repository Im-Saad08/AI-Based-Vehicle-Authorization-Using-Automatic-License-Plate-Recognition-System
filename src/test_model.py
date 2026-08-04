from ultralytics import YOLO

# Load the trained model
model = YOLO("run_model/detect/runs/license_plate_detector/weights/best.pt")

# Run detection on one image
results = model.predict(
    source="dataset/YOLO_dataset/images/train/N204.jpeg",
    save=True,
    conf=0.5
)

print("Detection completed successfully!")