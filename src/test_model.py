from ultralytics import YOLO

# Load the trained model
model = YOLO("models/trained/rbflw_y8_best.pt")

# Run detection on one image
results = model.predict(
    source="img/output/test_plate_detector/frame_106_detected.jpg",
    save=True,
    conf=0.5
)

print("Detection completed successfully!")