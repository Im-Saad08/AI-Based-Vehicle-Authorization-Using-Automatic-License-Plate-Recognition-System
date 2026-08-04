from ultralytics import YOLO
import cv2
import os


# ----------------------------------------
# Model path
# ----------------------------------------
MODEL_PATH = "run_model/detect/runs/license_plate_detector/weights/best.pt"


# ----------------------------------------
# Output folders
# ----------------------------------------
DETECTION_FOLDER = "img/output/detections"
CROPPED_FOLDER = "img/output/cropped_plates"


# ----------------------------------------
# Load YOLO model once
# ----------------------------------------
model = YOLO(MODEL_PATH)


# ----------------------------------------
# Detect and crop all license plates
# ----------------------------------------
def detect_and_crop(image_path):

    # ----------------------------------------
    # Read original image
    # ----------------------------------------
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )

    # ----------------------------------------
    # Keep a clean copy of the original image
    # This image will be used for cropping
    # ----------------------------------------
    original_image = image.copy()

    # ----------------------------------------
    # Create output folders if needed
    # ----------------------------------------
    os.makedirs(
        DETECTION_FOLDER,
        exist_ok=True
    )

    os.makedirs(
        CROPPED_FOLDER,
        exist_ok=True
    )

    # ----------------------------------------
    # Extract image name
    #
    # Example:
    # img/input/car_3.jpg
    #        ↓
    # car_3
    # ----------------------------------------
    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # ----------------------------------------
    # Run YOLO detection
    # ----------------------------------------
    results = model.predict(
        source=image_path,
        conf=0.5,
        save=False
    )

    # ----------------------------------------
    # Store cropped plate information
    # ----------------------------------------
    cropped_plates = []

    # ----------------------------------------
    # Plate counter
    # ----------------------------------------
    plate_count = 1

    # ----------------------------------------
    # Process all detections
    # ----------------------------------------
    for result in results:

        for box in result.boxes:

            # ----------------------------------------
            # Get bounding box coordinates
            # ----------------------------------------
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # ----------------------------------------
            # Get YOLO detection confidence
            # ----------------------------------------
            detection_confidence = float(
                box.conf[0]
            )

            # ----------------------------------------
            # Crop plate from CLEAN original image
            # ----------------------------------------
            plate_crop = original_image[
                y1:y2,
                x1:x2
            ].copy()

            # ----------------------------------------
            # Create cropped plate filename
            #
            # Example:
            # car_3_plate_1.png
            # car_3_plate_2.png
            # ----------------------------------------
            crop_filename = (
                f"{image_name}_plate_{plate_count}.png"
            )

            crop_path = os.path.join(
                CROPPED_FOLDER,
                crop_filename
            )

            # ----------------------------------------
            # Save clean cropped plate
            # ----------------------------------------
            cv2.imwrite(
                crop_path,
                plate_crop
            )

            # ----------------------------------------
            # Draw bounding box ONLY on visualization image
            # ----------------------------------------
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ----------------------------------------
            # Store information for main.py
            # ----------------------------------------
            cropped_plates.append({

                "image_name":
                    crop_filename,

                "image":
                    plate_crop,

                "yolo_confidence":
                    detection_confidence
            })

            # ----------------------------------------
            # Move to next plate
            # ----------------------------------------
            plate_count += 1

    # ----------------------------------------
    # Save detected image
    # This image contains bounding boxes
    # ----------------------------------------
    detected_filename = (
        f"{image_name}_detected.png"
    )

    detected_path = os.path.join(
        DETECTION_FOLDER,
        detected_filename
    )

    cv2.imwrite(
        detected_path,
        image
    )

    # ----------------------------------------
    # Print detection summary
    # ----------------------------------------
    print(
        f"\nYOLO detected "
        f"{len(cropped_plates)} plate(s)."
    )

    print(
        f"Detection image saved at: "
        f"{detected_path}"
    )

    # ----------------------------------------
    # Return all detected plates
    # ----------------------------------------
    return cropped_plates