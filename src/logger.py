import csv
import os
from datetime import datetime


# ----------------------------------------
# Project base directory
# ----------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ----------------------------------------
# Entry log file
# Always points to:
# Vehicle Authorization System/data/entry_log.csv
# ----------------------------------------
LOG_FILE = os.path.join(
    BASE_DIR,
    "data",
    "entry_log.csv"
)


# ----------------------------------------
# Log one vehicle entry attempt
# Simplified: only log plate number (ABC123 format, no dash)
# ----------------------------------------
def log_entry(
    image_name,
    plate_number,
    confidence,
    status
):

    # ----------------------------------------
    # Get current date and time
    # ----------------------------------------
    current_time = datetime.now()

    date = current_time.strftime(
        "%Y-%m-%d"
    )

    time = current_time.strftime(
        "%H:%M:%S"
    )


    # ----------------------------------------
    # Make sure data folder exists
    # ----------------------------------------
    os.makedirs(
        os.path.dirname(LOG_FILE),
        exist_ok=True
    )


    # ----------------------------------------
    # Show exact file being updated
    # ----------------------------------------
    print(
        f"Logging to: {os.path.abspath(LOG_FILE)}"
    )


    # ----------------------------------------
    # Check whether log file already exists
    # ----------------------------------------
    file_exists = os.path.exists(
        LOG_FILE
    )


    # ----------------------------------------
    # Open CSV in append mode
    # ----------------------------------------
    with open(
        LOG_FILE,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )


        # ----------------------------------------
        # Create header if file does not exist
        # ----------------------------------------
        if not file_exists:

            writer.writerow([
                "Date",
                "Time",
                "Image",
                "Plate_Number",
                "Confidence",
                "Status"
            ])


        # ----------------------------------------
        # Add vehicle entry (only plate_number in ABC123 format)
        # ----------------------------------------
        writer.writerow([
            date,
            time,
            image_name,
            plate_number,
            f"{confidence*100:.1f}%",
            status
        ])


    # ----------------------------------------
    # Confirm successful logging
    # ----------------------------------------
    print(
        f"Entry logged successfully: "
        f"{plate_number}"
    )