import pandas as pd
import os


# ----------------------------------------
# Authorized vehicle database
# ----------------------------------------

VEHICLES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "vehicles.csv"
)


# ----------------------------------------
# Load vehicle database once
# ----------------------------------------

vehicles_data = pd.read_csv(
    VEHICLES_FILE
)


# ----------------------------------------
# Create normalized plate column for database
# (database plates are raw formats like "MN 4524", "Ka-09 Ma 2662")
# ----------------------------------------

from normalize_plate import normalize_plate_text

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
    """
    plate_number is already normalized by recognize_plate (token-aware corrected, joined as ABC123).
    Do NOT re-apply normalize_plate_text to avoid corrupting already-corrected plates
    (e.g., "MLE4008" would become "M1E4008" if re-tokenized as single token).
    """
    # ----------------------------------------
    # Empty OCR result
    # ----------------------------------------

    if not plate_number or plate_number.lower() == "nan":

        return {

            "status": "Unable to Read",

            "normalized_plate": ""
        }

    # The plate_number from recognize_plate is already in normalized format (ABC123)
    # Use it directly for database lookup
    normalized_plate = plate_number

    # ----------------------------------------
    # Search authorized database
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
    # Vehicle Found
    # ----------------------------------------

    if not matched_vehicle.empty:

        return {

            "status": "Authorized",

            "normalized_plate": normalized_plate
        }

    # ----------------------------------------
    # Vehicle Not Found
    # ----------------------------------------

    return {

        "status": "Unauthorized",

        "normalized_plate": normalized_plate
    }