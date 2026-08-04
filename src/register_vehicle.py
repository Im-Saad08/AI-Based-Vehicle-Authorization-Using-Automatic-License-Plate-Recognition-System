from ultralytics import YOLO
import easyocr
import pandas as pd
import cv2
import os

from normalize_plate import normalize_plate_text


# ==================================================
# FILE PATHS
# ==================================================

MODEL_PATH = (
    "run_model/detect/runs/"
    "license_plate_detector/weights/best.pt"
)

VEHICLES_FILE = "data/vehicles.csv"


# ==================================================
# INITIALIZE MODELS
# ==================================================

print("\nLoading YOLO model...")

yolo_model = YOLO(
    MODEL_PATH
)

print("Loading EasyOCR...")

ocr_reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("Models loaded successfully.")


# ==================================================
# ENTER EMPLOYEE / VEHICLE INFORMATION
# ==================================================

print("\n" + "=" * 50)
print("NEW VEHICLE REGISTRATION")
print("=" * 50)

employee_id = input(
    "Employee ID    : "
).strip()

name = input(
    "Name           : "
).strip()

department = input(
    "Department     : "
).strip()

vehicle_type = input(
    "Vehicle Type   : "
).strip()

image_path = input(
    "Vehicle Image Path: "
).strip()


# ==================================================
# CHECK IMAGE
# ==================================================

if not os.path.exists(image_path):

    print(
        "\nError: Image file does not exist."
    )

    exit()


image = cv2.imread(
    image_path
)

if image is None:

    print(
        "\nError: Unable to read the image."
    )

    exit()


# ==================================================
# RUN YOLO LICENSE PLATE DETECTION
# ==================================================

print(
    "\nDetecting license plate..."
)

results = yolo_model.predict(
    source=image_path,
    conf=0.5,
    save=False
)


# ==================================================
# COLLECT DETECTED PLATES
# ==================================================

detected_plates = []

for result in results:

    for box in result.boxes:

        # ------------------------------------------
        # Bounding box coordinates
        # ------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # ------------------------------------------
        # YOLO confidence
        # ------------------------------------------

        detection_confidence = float(
            box.conf[0]
        )

        # ------------------------------------------
        # Crop from original clean image
        # ------------------------------------------

        plate_crop = image[
            y1:y2,
            x1:x2
        ].copy()

        detected_plates.append(
            (
                plate_crop,
                detection_confidence
            )
        )


# ==================================================
# CHECK IF PLATE WAS DETECTED
# ==================================================

if len(detected_plates) == 0:

    print(
        "\nNo license plate detected."
    )

    print(
        "Please provide a clearer "
        "vehicle image."
    )

    exit()


# ==================================================
# SELECT HIGHEST-CONFIDENCE PLATE
# ==================================================

# Registration is intended for one vehicle
# at a time, so select the strongest detection.

detected_plates.sort(
    key=lambda x: x[1],
    reverse=True
)

plate_crop = detected_plates[0][0]

yolo_confidence = detected_plates[0][1]


print(
    "\nLicense plate detected."
)

print(
    f"YOLO Detection Confidence: "
    f"{yolo_confidence:.2f}"
)


if len(detected_plates) > 1:

    print(
        f"Warning: {len(detected_plates)} "
        "license plates detected."
    )

    print(
        "Using the highest-confidence "
        "plate for registration."
    )


# ==================================================
# RUN EASYOCR
# ==================================================

print(
    "\nReading license plate "
    "using EasyOCR..."
)

ocr_results = ocr_reader.readtext(
    plate_crop
)


# ==================================================
# CHECK OCR RESULT
# ==================================================

if len(ocr_results) == 0:

    print(
        "\nEasyOCR could not read "
        "any text."
    )

    detected_text = ""

    ocr_confidence = 0.0


else:

    detected_texts = []

    confidence_scores = []

    print(
        "\nOCR Results:"
    )

    for detection in ocr_results:

        bbox, text, confidence = detection

        detected_texts.append(
            text
        )

        confidence_scores.append(
            confidence
        )

        print(
            f"Text       : {text}"
        )

        print(
            f"Confidence : "
            f"{confidence:.2f}"
        )


    # ------------------------------------------
    # Combine raw OCR text
    # ------------------------------------------

    detected_text = " ".join(
        detected_texts
    )


    # ------------------------------------------
    # Minimum OCR confidence
    # ------------------------------------------

    ocr_confidence = min(
        confidence_scores
    )


# ==================================================
# NORMALIZE OCR RESULT
# ==================================================

normalized_result = normalize_plate_text(
    detected_text
)

# Canonical plate number
detected_plate = normalized_result[
    "plate_number"
]

# Registration year
detected_registration_year = (
    normalized_result[
        "registration_year"
    ]
)


# ==================================================
# SHOW DETECTED REGISTRATION INFORMATION
# ==================================================

print("\n" + "=" * 50)
print("DETECTED REGISTRATION INFORMATION")
print("=" * 50)

print(
    f"Raw OCR Text           : "
    f"{detected_text}"
)

print(
    f"Plate Number           : "
    f"{detected_plate}"
)

print(
    f"Registration Year      : "
    f"{detected_registration_year}"
)

print(
    f"OCR Confidence         : "
    f"{ocr_confidence:.2f}"
)

print(
    f"YOLO Confidence        : "
    f"{yolo_confidence:.2f}"
)


# ==================================================
# CONFIRM OR CORRECT DETAILS
# ==================================================

print(
    "\nAre the detected "
    "registration details correct?"
)

confirmation = input(
    "Enter Y to confirm or "
    "N to correct: "
).strip().upper()


if confirmation == "Y":

    final_plate = detected_plate

    final_registration_year = (
        detected_registration_year
    )


elif confirmation == "N":

    # ------------------------------------------
    # Manually correct plate number
    # ------------------------------------------

    corrected_plate = input(
        "\nEnter the correct "
        "license plate number: "
    ).strip()


    # ------------------------------------------
    # Normalize manually entered plate
    # ------------------------------------------

    corrected_result = (
        normalize_plate_text(
            corrected_plate
        )
    )

    final_plate = corrected_result[
        "plate_number"
    ]


    # ------------------------------------------
    # Ask for registration year
    # ------------------------------------------

    final_registration_year = input(
        "Enter the registration year "
        "(e.g., 08) or press Enter "
        "to leave blank: "
    ).strip()


else:

    print(
        "\nInvalid input."
    )

    print(
        "Registration cancelled."
    )

    exit()


# ==================================================
# CHECK FINAL PLATE
# ==================================================

if final_plate == "":

    print(
        "\nError: Plate number cannot "
        "be empty."
    )

    exit()


# ==================================================
# CHECK CSV EXISTS
# ==================================================

if not os.path.exists(
    VEHICLES_FILE
):

    print(
        "\nError: vehicles.csv "
        "not found."
    )

    exit()


# ==================================================
# READ VEHICLE DATABASE
# ==================================================

vehicles_data = pd.read_csv(
    VEHICLES_FILE
)


# ==================================================
# CHECK REQUIRED COLUMNS
# ==================================================

required_columns = [
    "Employee_ID",
    "Name",
    "Department",
    "Vehicle_Type",
    "Plate_Number",
    "Registration_Year",
    "Status"
]


missing_columns = [

    column

    for column in required_columns

    if column
    not in vehicles_data.columns
]


if missing_columns:

    print(
        "\nError: vehicles.csv is missing "
        "the following columns:"
    )

    for column in missing_columns:

        print(
            f"- {column}"
        )

    exit()


# ==================================================
# NORMALIZE EXISTING DATABASE PLATES
# ==================================================

vehicles_data[
    "Normalized_Plate_Number"
] = (

    vehicles_data[
        "Plate_Number"
    ]
    .astype(str)
    .apply(
        lambda x:
        normalize_plate_text(
            x
        )[
            "plate_number"
        ]
    )
)


# ==================================================
# CHECK DUPLICATE EMPLOYEE ID
# ==================================================

employee_exists = vehicles_data[

    vehicles_data[
        "Employee_ID"
    ]
    .astype(str)
    .str.strip()
    .str.upper()

    ==

    employee_id.upper()
]


if not employee_exists.empty:

    print(
        "\nRegistration failed:"
    )

    print(
        "This Employee ID is "
        "already registered."
    )

    exit()


# ==================================================
# CHECK DUPLICATE PLATE NUMBER
# ==================================================

plate_exists = vehicles_data[

    vehicles_data[
        "Normalized_Plate_Number"
    ]

    ==

    final_plate.upper()
]


if not plate_exists.empty:

    print(
        "\nRegistration failed:"
    )

    print(
        "This license plate is "
        "already registered."
    )

    exit()


# ==================================================
# CREATE NEW VEHICLE RECORD
# ==================================================

new_vehicle = pd.DataFrame([

    {

        "Employee_ID":
            employee_id,

        "Name":
            name,

        "Department":
            department,

        "Vehicle_Type":
            vehicle_type,

        "Plate_Number":
            final_plate,

        "Registration_Year":
            final_registration_year,

        "Status":
            "Authorized"

    }

])


# ==================================================
# ADD NEW VEHICLE TO DATABASE
# ==================================================

# Keep only the original CSV columns
# so the helper column used for
# duplicate checking is not saved.

new_columns = [
    "Employee_ID",
    "Name",
    "Department",
    "Vehicle_Type",
    "Plate_Number",
    "Registration_Year",
    "Status"
]


vehicles_data = vehicles_data[
    new_columns
]


# Append new record
vehicles_data = pd.concat(

    [
        vehicles_data,
        new_vehicle
    ],

    ignore_index=True
)


# Save updated database
vehicles_data.to_csv(

    VEHICLES_FILE,

    index=False
)


# ==================================================
# FINAL REGISTRATION MESSAGE
# ==================================================

print("\n" + "=" * 50)
print(
    "VEHICLE REGISTERED SUCCESSFULLY"
)
print("=" * 50)

print(
    f"Employee ID       : "
    f"{employee_id}"
)

print(
    f"Name              : "
    f"{name}"
)

print(
    f"Plate Number      : "
    f"{final_plate}"
)

print(
    f"Registration Year : "
    f"{final_registration_year}"
)

print(
    "Status            : Authorized"
)

print(
    f"\nUpdated database: "
    f"{VEHICLES_FILE}"
)