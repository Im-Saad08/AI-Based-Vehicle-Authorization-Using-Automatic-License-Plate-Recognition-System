from ultralytics import YOLO
import cv2

MODEL_PATH = "models/trained/yolo11n_best.pt"
VIDEO_PATH = "img/input/video.mp4"

model = YOLO(MODEL_PATH)

video = cv2.VideoCapture(VIDEO_PATH)

frame_number = 0

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame_number += 1

    results = model.predict(
        source=frame,
        classes=[2, 3, 5, 7],
        conf=0.25,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        print(
            f"Frame {frame_number}: "
            f"{len(result.boxes)} vehicles detected"
        )

    if frame_number >= 30:
        break

video.release()

print("\nVehicle prediction test completed.")