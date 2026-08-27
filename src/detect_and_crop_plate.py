from ultralytics import YOLO
import cv2
import os
import time


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "trained",
    "rbflw_y8_best.pt"
)


# ============================================================
# OUTPUT FOLDERS
# ============================================================

DETECTION_FOLDER = os.path.join(
    PROJECT_ROOT,
    "img",
    "output",
    "detections"
)

CROPPED_FOLDER = os.path.join(
    PROJECT_ROOT,
    "img",
    "output",
    "cropped_plates"
)

PREDICT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "img",
    "output",
    "predict"
)


# ============================================================
# LOAD LICENSE PLATE MODEL
# ============================================================

print(
    f"\nLoading license plate model:"
)

print(
    MODEL_PATH
)

model = YOLO(
    MODEL_PATH
)


# ============================================================
# DETECT AND CROP LICENSE PLATES
# ============================================================

def detect_and_crop(
    image_input,
    save_output=True,
    image_name=None
):

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    if isinstance(
        image_input,
        str
    ):

        image = cv2.imread(
            image_input
        )

        if image is None:

            raise FileNotFoundError(
                f"Unable to load image: "
                f"{image_input}"
            )

        if image_name is None:

            image_name = os.path.splitext(
                os.path.basename(
                    image_input
                )
            )[0]

    else:

        image = image_input.copy()

        if image_name is None:

            # Unique name for video/webcam frames
            image_name = (
                f"frame_"
                f"{int(time.time() * 1000)}"
            )


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    original_image = image.copy()


    # ========================================================
    # CREATE OUTPUT FOLDERS
    # ========================================================

    if save_output:

        os.makedirs(
            DETECTION_FOLDER,
            exist_ok=True
        )

        os.makedirs(
            CROPPED_FOLDER,
            exist_ok=True
        )

        os.makedirs(
            PREDICT_FOLDER,
            exist_ok=True
        )


    # ========================================================
    # LICENSE PLATE DETECTION
    # ========================================================

    results = model.predict(

        source=image,

        conf=0.5,

        save=False,

        verbose=False

    )


    # ========================================================
    # STORE DETECTED PLATES
    # ========================================================

    cropped_plates = []


    # ========================================================
    # PLATE COUNTER
    # ========================================================

    plate_count = 1


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    for result in results:

        if result.boxes is None:

            continue


        if len(result.boxes) == 0:

            continue


        for box in result.boxes:

            # =================================================
            # BOUNDING BOX
            # =================================================

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # =================================================
            # YOLO CONFIDENCE
            # =================================================

            detection_confidence = float(
                box.conf[0]
            )


            # =================================================
            # IMAGE DIMENSIONS
            # =================================================

            height, width = (
                original_image.shape[:2]
            )


            # =================================================
            # KEEP COORDINATES INSIDE IMAGE
            # =================================================

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                width,
                x2
            )

            y2 = min(
                height,
                y2
            )


            # =================================================
            # VALIDATE BOX & MINIMUM SIZE (DISTANCE FILTER)
            # =================================================

            if x2 <= x1 or y2 <= y1:

                continue

            box_width = x2 - x1
            box_height = y2 - y1

            MIN_PLATE_WIDTH = 50   # Minimum width in pixels for readable plate
            MIN_PLATE_HEIGHT = 15  # Minimum height in pixels for readable plate

            if box_width < MIN_PLATE_WIDTH or box_height < MIN_PLATE_HEIGHT:
                print(
                    f"Skipping plate crop: Box dimensions too small "
                    f"({box_width}x{box_height}px). Vehicle is too far."
                )
                continue


            # =================================================
            # CROP PLATE WITH MARGIN PADDING
            # =================================================
            # Use 40% vertical padding for all plates to ensure 2-line plates
            # have enough height for OCR split logic. YOLO detections tend to
            # be wider than the actual plate text area (aspect ratio ~0.48),
            # so we need more vertical padding to reach aspect ratio > 0.7.
            aspect_ratio = box_height / max(1, box_width)
            is_tall_plate = aspect_ratio > 0.55

            pad_w = int(box_width * 0.10)
            pad_h = int(box_height * 0.40)  # Use 40% for all plates

            crop_x1 = max(0, x1 - pad_w)
            crop_y1 = max(0, y1 - pad_h)
            crop_x2 = min(width, x2 + pad_w)
            crop_y2 = min(height, y2 + pad_h)

            plate_crop = original_image[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ].copy()


            if plate_crop.size == 0:

                continue


            # =================================================
            # UNIQUE PLATE FILENAME
            # =================================================

            crop_filename = (

                f"{image_name}_"

                f"plate_{plate_count}_"

                f"conf_{detection_confidence:.2f}"

                ".png"

            )


            # =================================================
            # SAVE CROPPED PLATE
            # =================================================

            if save_output:

                crop_path = os.path.join(

                    CROPPED_FOLDER,

                    crop_filename

                )


                cv2.imwrite(

                    crop_path,

                    plate_crop

                )


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            label = (

                f"Plate "
                f"{detection_confidence:.2f}"

            )


            cv2.rectangle(

                image,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2

            )


            cv2.putText(

                image,

                label,

                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 0),

                2

            )


            # =================================================
            # STORE PLATE INFORMATION
            # =================================================

            cropped_plates.append({

                "image_name":
                    crop_filename,

                "image":
                    plate_crop,

                "yolo_confidence":
                    detection_confidence,

                "bbox": [

                    x1,
                    y1,
                    x2,
                    y2

                ],

                "is_tall_plate":
                    is_tall_plate

            })


            plate_count += 1


    # ========================================================
    # SAVE DETECTION OUTPUT
    # ========================================================

    if save_output:

        # ====================================================
        # DETECTION IMAGE
        # ====================================================

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


        # ====================================================
        # YOLO-STYLE PREDICTION IMAGE
        # ====================================================

        predict_filename = (

            f"{image_name}_predict.png"

        )


        predict_path = os.path.join(

            PREDICT_FOLDER,

            predict_filename

        )


        cv2.imwrite(

            predict_path,

            image

        )


        # ====================================================
        # CONSOLE OUTPUT
        # ====================================================

        print(

            f"\nYOLO detected "
            f"{len(cropped_plates)} "
            f"plate(s)."

        )


        print(

            f"Detection image saved at: "
            f"{detected_path}"

        )


        print(

            f"Prediction image saved at: "
            f"{predict_path}"

        )


        if cropped_plates:

            print(
                "\nDetected plates:"
            )


            for plate in cropped_plates:

                print(

                    f"  "
                    f"{plate['image_name']} "
                    f"Conf="
                    f"{plate['yolo_confidence']:.2f}"

                )


    # ========================================================
    # RETURN PLATES
    # ========================================================

    return cropped_plates