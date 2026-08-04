import pandas as pd

from normalize_plate import normalize_plate_text


# ----------------------------------------
# Authorized vehicle database
# ----------------------------------------
VEHICLES_FILE = "data/vehicles.csv"


# ----------------------------------------
# Load vehicle database once
# ----------------------------------------
vehicles_data = pd.read_csv(
    VEHICLES_FILE
)


# ----------------------------------------
# Create normalized plate column
# ----------------------------------------
vehicles_data["Normalized_Plate_Number"] = (
    vehicles_data["Plate_Number"]
    .astype(str)
    .apply(
        lambda x:
        normalize_plate_text(x)["plate_number"]
    )
)


# ----------------------------------------
# Authorize vehicle
# ----------------------------------------
def authorize_vehicle(
    plate_number
):

    # ----------------------------------------
    # Normalize OCR result
    # ----------------------------------------
    normalized_ocr = normalize_plate_text(
        plate_number
    )

    # Canonical plate number
    normalized_plate = normalized_ocr[
        "plate_number"
    ]

    # Registration year detected by OCR
    ocr_registration_year = normalized_ocr[
        "registration_year"
    ]

    # ----------------------------------------
    # Empty OCR result
    # ----------------------------------------
    if (
        normalized_plate == ""
        or normalized_plate.lower() == "nan"
    ):

        return {
            "status": "Unable to Read",
            "owner": "Unknown",
            "employee_id": "Unknown",
            "department": "Unknown",
            "vehicle_type": "Unknown",
            "registration_year": ocr_registration_year
        }

    # ----------------------------------------
    # Search for matching plate
    #
    # Match is based on:
    # 1. Normalized plate number
    # 2. Vehicle status = Authorized
    #
    # Registration year is NOT used
    # for authorization matching.
    # ----------------------------------------
    matched_vehicle = vehicles_data[
        (
            vehicles_data[
                "Normalized_Plate_Number"
            ]
            ==
            normalized_plate
        )
        &
        (
            vehicles_data["Status"]
            .astype(str)
            .str.strip()
            .str.upper()
            ==
            "AUTHORIZED"
        )
    ]

    # ----------------------------------------
    # Vehicle found and authorized
    # ----------------------------------------
    if not matched_vehicle.empty:

        vehicle = matched_vehicle.iloc[0]

        # Get stored registration year
        stored_registration_year = str(
            vehicle.get(
                "Registration_Year",
                ""
            )
        ).strip()

        # If stored year is empty,
        # use OCR-detected year for display
        if (
            stored_registration_year == ""
            or stored_registration_year.lower()
            == "nan"
        ):

            final_registration_year = (
                ocr_registration_year
            )

        else:

            final_registration_year = (
                stored_registration_year
            )

        return {
            "status": "Authorized",
            "owner": vehicle["Name"],
            "employee_id":
                vehicle["Employee_ID"],
            "department":
                vehicle["Department"],
            "vehicle_type":
                vehicle["Vehicle_Type"],
            "registration_year":
                final_registration_year
        }

    # ----------------------------------------
    # Vehicle not found
    # OR vehicle is not authorized
    # ----------------------------------------
    return {
        "status": "Unauthorized",
        "owner": "Unknown",
        "employee_id": "Unknown",
        "department": "Unknown",
        "vehicle_type": "Unknown",
        "registration_year":
            ocr_registration_year
    }