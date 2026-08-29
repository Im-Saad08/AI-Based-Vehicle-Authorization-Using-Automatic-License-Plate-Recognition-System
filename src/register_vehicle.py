import pandas as pd
import os

from detect_and_crop_plate import detect_and_crop
from recognize_plate import recognize_plate
from normalize_plate import normalize_plate_text


# ==================================================
# FILE PATHS
# ==================================================

VEHICLES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "vehicles.csv"
)


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


# ==================================================
# RUN SHARED DETECTION + CROP PIPELINE
# ==================================================

print(
    "\nDetecting license plate..."
)

# Use shared detect_and_crop (same as video/webcam modes)
# save_output=False to avoid saving temp files for registration
cropped_plates = detect_and_crop(
    image_input=image_path,
    save_output=False
)


# ==================================================
# CHECK IF PLATE WAS DETECTED
# ==================================================

if len(cropped_plates) == 0:

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

cropped_plates.sort(
    key=lambda x: x["yolo_confidence"],
    reverse=True
)

best_plate = cropped_plates[0]

plate_crop = best_plate["image"]
yolo_confidence = best_plate["yolo_confidence"]


print(
    "\nLicense plate detected."
)

print(
    f"YOLO Detection Confidence: "
    f"{yolo_confidence:.2f}"
)

if len(cropped_plates) > 1:

    print(
        f"Warning: {len(cropped_plates)} "
        "license plates detected."
    )

    print(
        "Using the highest-confidence "
        "plate for registration."
    )


# ==================================================
# RUN PADDLEOCR RECOGNITION-ONLY PIPELINE
# ==================================================

print(
    "\nReading license plate "
    "using PaddleOCR (recognition-only)..."
)

ocr_result = recognize_plate(plate_crop)

detected_text = ocr_result["plate_text"]
ocr_confidence = ocr_result["confidence"]

if not detected_text:
    print(
        "\nPaddleOCR could not read "
        "any text."
    )
    detected_text = ""
    ocr_confidence = 0.0


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

print("\n" + "=" * 50)
print("ADMIN CONFIRMATION")
print("=" * 50)

print(
    f"Detected Plate : {detected_plate}"
)

print(
    "\nPress ENTER to accept the detected plate."
)

print(
    "Or type the correct plate number."
)

corrected_plate = input(
    "\nPlate: "
).strip()


# ------------------------------------------
# Admin accepted OCR result
# ------------------------------------------

if corrected_plate == "":

    final_plate = detected_plate


# ------------------------------------------
# Admin corrected OCR result
# ------------------------------------------

else:

    corrected_result = normalize_plate_text(
        corrected_plate
    )

    final_plate = corrected_result[
        "plate_number"
    ]
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
    "Status            : Authorized"
)

print(
    f"\nUpdated database: "
    f"{VEHICLES_FILE}"
)