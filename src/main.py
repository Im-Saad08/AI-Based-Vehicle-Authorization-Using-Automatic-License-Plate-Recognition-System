from detect_and_crop_plate import detect_and_crop
from recognize_plate import recognize_plate
from authorize_vehicle import authorize_vehicle
from enhance_plate import enhance_plate
from logger import log_entry

import os


# ----------------------------------------
# Input
# Supports:
# 1. Single image
# 2. Folder
# 3. List of selected images
# ----------------------------------------

INPUT_PATH = "img/input/Cars/DSC_1049.jpg"
#INPUT_PATH = "img/output/enhanced_plates/contrast_enhanced.png"

# Examples:
#
# INPUT_PATH = "img/input/car_3.jpg"
#
# INPUT_PATH = "img/input"
#
# INPUT_PATH = [
#     "img/input/car_1.jpg",
#     "img/input/car_3.jpg",
#     "img/input/car_5.jpg"
# ]


# ----------------------------------------
# Prepare image list
# ----------------------------------------

image_paths = []

# Case 1: Multiple selected images
if isinstance(INPUT_PATH, list):

    for image in INPUT_PATH:

        if os.path.isfile(image):

            image_paths.append(image)

        else:

            print(f"Image not found: {image}")


# Case 2: Single image
elif os.path.isfile(INPUT_PATH):

    image_paths = [INPUT_PATH]


# Case 3: Folder
elif os.path.isdir(INPUT_PATH):

    image_paths = [

        os.path.join(INPUT_PATH, file)

        for file in sorted(os.listdir(INPUT_PATH))

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]


# Invalid input
else:

    print("Invalid input path.")
    exit()


# ----------------------------------------
# Check if images exist
# ----------------------------------------

if not image_paths:

    print("No valid images found.")
    exit()


# ----------------------------------------
# Start system
# ----------------------------------------

print("\n" + "=" * 60)
print("VEHICLE AUTHORIZATION SYSTEM")
print("=" * 60)


# ----------------------------------------
# Process every image
# ----------------------------------------

for IMAGE_PATH in image_paths:

    print(
        f"\nProcessing image: {IMAGE_PATH}"
    )

    # ----------------------------------------
    # Step 1:
    # Detect and crop license plates
    # ----------------------------------------

    cropped_plates = detect_and_crop(
        IMAGE_PATH
    )

    # ----------------------------------------
    # Check if plates were detected
    # ----------------------------------------

    if len(cropped_plates) == 0:

        print(
            "\nNo license plates detected."
        )

        continue


    # ----------------------------------------
    # Process each detected plate
    # ----------------------------------------

    for plate in cropped_plates:

        image_name = plate[
            "image_name"
        ]

        yolo_confidence = plate[
            "yolo_confidence"
        ]

        plate_image = plate[
            "image"
        ]


        print("\n" + "-" * 60)

        print(
            f"Processing: {image_name}"
        )

        # ----------------------------------------
        # Step 2:
        # OCR
        # ----------------------------------------

        ocr_result = recognize_plate(
            plate_image
        )

        detected_plate = ocr_result[
            "plate_text"
        ]

        ocr_confidence = ocr_result[
            "confidence"
        ]