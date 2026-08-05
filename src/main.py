from detect_and_crop_plate import detect_and_crop
from recognize_plate import recognize_plate
from authorize_vehicle import authorize_vehicle
from logger import log_entry

import os


# ----------------------------------------
# Input
# Supports:
# 1. Single image
# 2. Folder
# 3. List of selected images
# ----------------------------------------

INPUT_PATH = "img/evaluation/authorized"

# Examples:
#
# INPUT_PATH = "img/input"
#
# INPUT_PATH = [
#     "img/input/car1.jpg",
#     "img/input/car2.jpg"
# ]


# ----------------------------------------
# Prepare image list
# ----------------------------------------

image_paths = []

if isinstance(INPUT_PATH, list):

    for image in INPUT_PATH:

        if os.path.isfile(image):

            image_paths.append(image)

        else:

            print(f"Image not found: {image}")

elif os.path.isfile(INPUT_PATH):

    image_paths = [INPUT_PATH]

elif os.path.isdir(INPUT_PATH):

    image_paths = [

        os.path.join(INPUT_PATH, file)

        for file in sorted(os.listdir(INPUT_PATH))

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )

    ]

else:

    print("Invalid input path.")
    exit()


# ----------------------------------------
# Check images
# ----------------------------------------

if not image_paths:

    print("No valid images found.")
    exit()


# ----------------------------------------
# Start
# ----------------------------------------

print("\n" + "=" * 60)
print("VEHICLE AUTHORIZATION SYSTEM")
print("=" * 60)


# ----------------------------------------
# Process images
# ----------------------------------------

for IMAGE_PATH in image_paths:

    print(f"\nProcessing image: {IMAGE_PATH}")

    cropped_plates = detect_and_crop(
        IMAGE_PATH
    )

    if len(cropped_plates) == 0:

        print("\nNo license plates detected.")
        continue

    for plate in cropped_plates:

        image_name = plate["image_name"]

        yolo_confidence = plate["yolo_confidence"]

        plate_image = plate["image"]

        print("\n" + "-" * 60)

        print(
            f"Processing: {image_name}"
        )

        # ----------------------------------------
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

        # ----------------------------------------
        # Authorization
        # ----------------------------------------

        authorization = authorize_vehicle(
            detected_plate
        )

        status = authorization[
            "status"
        ]

        owner = authorization[
            "owner"
        ]

        employee_id = authorization[
            "employee_id"
        ]

        department = authorization[
            "department"
        ]

        vehicle_type = authorization[
            "vehicle_type"
        ]

        registration_year = authorization[
            "registration_year"
        ]

        # ----------------------------------------
        # Display Result
        # ----------------------------------------

        print()

        print(
            f"Detected Plate      : {detected_plate}"
        )

        print(
            f"Registration Year   : {registration_year}"
        )

        print(
            f"OCR Confidence      : {ocr_confidence:.2f}"
        )

        print(
            f"YOLO Confidence     : {yolo_confidence:.2f}"
        )

        print()

        print(
            f"STATUS              : {status.upper()}"
        )

        print(
            f"Owner               : {owner}"
        )

        print(
            f"Employee ID         : {employee_id}"
        )

        print(
            f"Department          : {department}"
        )

        print(
            f"Vehicle Type        : {vehicle_type}"
        )

        # ----------------------------------------
        # Log Entry
        # ----------------------------------------

        log_entry(
            image_name=image_name,
            plate_number=detected_plate,
            registration_year=registration_year,
            confidence=ocr_confidence,
            status=status,
            owner=owner,
            employee_id=employee_id,
            department=department,
            vehicle_type=vehicle_type
        )


print("\n" + "=" * 60)
print("PROCESSING COMPLETED")
print("=" * 60)