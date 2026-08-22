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
# ----------------------------------------
def log_entry(
    image_name,
    plate_number,
    confidence,
    status,
    owner="Unknown",
    employee_id="Unknown",
    department="Unknown",
    vehicle_type="Unknown",
    registration_year=""
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
                "Registration_Year",
                "Confidence",
                "Status",
                "Owner",
                "Employee_ID",
                "Department",
                "Vehicle_Type"
            ])


        # ----------------------------------------
        # Add vehicle entry
        # ----------------------------------------
        writer.writerow([
            date,
            time,
            image_name,
            plate_number,
            registration_year,
            confidence,
            status,
            owner,
            employee_id,
            department,
            vehicle_type
        ])


    # ----------------------------------------
    # Confirm successful logging
    # ----------------------------------------
    print(
        f"Entry logged successfully: "
        f"{plate_number}"
    )