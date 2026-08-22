# ============================================================
# VEHICLE AUTHORIZATION SYSTEM
# MAIN PIPELINE
# ============================================================

import os
import sys
import cv2
import re

from ultralytics import YOLO


# ============================================================
# PROJECT ROOT
# ============================================================

# main.py is inside:
# Vehicle Authorization System-V2/src/main.py
#
# Therefore the project root is one directory above src.

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SRC_DIR
)

# Make project root the working directory.
# This keeps all relative paths consistent.
os.chdir(PROJECT_ROOT)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from detect_and_crop_plate import detect_and_crop
from recognize_plate import recognize_plate
from authorize_vehicle import authorize_vehicle
from logger import log_entry


# ============================================================
# INPUT MODE
# ============================================================

# Available:
#
# "image"
# "video"
# "webcam"
#
# Change ONLY this variable when changing input type.

INPUT_MODE = "image"


# ============================================================
# INPUT PATH
# ============================================================

# IMAGE:
# INPUT_PATH = "img/input/car.jpg"
#
# VIDEO:
# INPUT_PATH = "img/input/video.mp4"
#
# WEBCAM:
# INPUT_PATH = 0

INPUT_PATH = "img/input/Cars/car_1.jpg"


# ============================================================
# VEHICLE MODEL
# ============================================================

VEHICLE_MODEL_PATH = (
    "models/pre_trained/yolo11n.pt"
)


# ============================================================
# VERIFY IMPORTANT PATHS
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PATH CONFIGURATION"
)

print(
    "=" * 70
)

print(
    f"Project root : {PROJECT_ROOT}"
)

print(
    f"Input mode   : {INPUT_MODE}"
)

print(
    f"Input path   : {INPUT_PATH}"
)

print(
    f"Input exists : {os.path.exists(INPUT_PATH)}"
)

print(
    f"Vehicle model: {VEHICLE_MODEL_PATH}"
)

print(
    f"Model exists : {os.path.exists(VEHICLE_MODEL_PATH)}"
)

print(
    "=" * 70
)


# ============================================================
# CHECK INPUT
# ============================================================

if INPUT_MODE != "webcam":

    if not os.path.exists(INPUT_PATH):

        print(
            "\nERROR:"
        )

        print(
            f"Input does not exist:"
        )

        print(
            os.path.abspath(INPUT_PATH)
        )

        sys.exit(1)


# ============================================================
# LOAD VEHICLE MODEL
# ============================================================

print(
    "\nLoading vehicle detection model..."
)

vehicle_model = YOLO(
    VEHICLE_MODEL_PATH
)

print(
    "Vehicle model loaded successfully."
)


# ============================================================
# VEHICLE MODEL INFORMATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "VEHICLE MODEL INFORMATION"
)

print(
    "=" * 70
)

print(
    f"Vehicle model: {VEHICLE_MODEL_PATH}"
)

print(
    f"Vehicle classes: {vehicle_model.names}"
)

print(
    "=" * 70
)


# ============================================================
# VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES = [

    2,      # car
    3,      # motorcycle
    5,      # bus
    7       # truck

]


# ============================================================
# VIDEO SETTINGS
# ============================================================

# 1 = process every frame
# 2 = process every second frame
# 3 = process every third frame
#
# For testing OCR/tracking, keep this at 1.

FRAME_SKIP = 1


# ============================================================
# DETECTION SETTINGS
# ============================================================

VEHICLE_CONFIDENCE = 0.40


# ============================================================
# OCR CONSENSUS SETTINGS
# ============================================================

MIN_OCR_CONFIDENCE = 0.10

MIN_OCR_OBSERVATIONS = 3

MIN_PLATE_REPETITIONS = 2

MAX_OCR_HISTORY = 10


# ============================================================
# TRACKING / LOGGING
# ============================================================

logged_track_ids = set()


# ============================================================
# OCR HISTORY
# ============================================================

ocr_history = {}


# ============================================================
# STATISTICS
# ============================================================

tracked_vehicle_ids = set()

vehicle_detection_count = 0

vehicle_detection_with_id_count = 0

plate_detection_count = 0

ocr_success_count = 0

ocr_failure_count = 0

finalization_attempt_count = 0

authorized_count = 0

unauthorized_count = 0


# ============================================================
# NORMALIZE OCR TEXT
# ============================================================

def normalize_plate_text(text):

    if not text:

        return ""

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# CHECK OCR RESULT
# ============================================================

def is_useful_ocr(
    text,
    confidence
):

    if not text:

        return False

    normalized = normalize_plate_text(
        text
    )

    if not normalized:

        return False

    if len(normalized) < 3:

        return False

    if confidence < MIN_OCR_CONFIDENCE:

        return False

    return True


# ============================================================
# ADD OCR OBSERVATION
# ============================================================

def add_ocr_observation(
    track_id,
    text,
    confidence,
    score
):

    if track_id not in ocr_history:

        ocr_history[track_id] = []

    normalized_text = normalize_plate_text(
        text
    )

    if not is_useful_ocr(
        text,
        confidence
    ):

        return

    ocr_history[track_id].append({

        "text": normalized_text,

        "confidence": confidence,

        "score": score

    })

    if len(
        ocr_history[track_id]
    ) > MAX_OCR_HISTORY:

        ocr_history[track_id] = (
            ocr_history[track_id]
            [-MAX_OCR_HISTORY:]
        )


# ============================================================
# GET OCR CONSENSUS
# ============================================================

def get_consensus_plate(track_id):

    if track_id not in ocr_history:

        return None

    history = ocr_history[track_id]

    if len(history) < MIN_OCR_OBSERVATIONS:

        return None

    grouped = {}

    for observation in history:

        text = observation["text"]

        if text not in grouped:

            grouped[text] = []

        grouped[text].append(
            observation
        )

    candidates = []

    for text, observations in grouped.items():

        repetition_count = len(
            observations
        )

        average_confidence = sum(

            item["confidence"]

            for item in observations

        ) / repetition_count

        average_score = sum(

            item["score"]

            for item in observations

        ) / repetition_count

        consensus_score = (

            repetition_count * 50

            + average_confidence * 100

            + average_score * 0.25

        )

        candidates.append({

            "text": text,

            "repetitions":
                repetition_count,

            "average_confidence":
                average_confidence,

            "average_score":
                average_score,

            "consensus_score":
                consensus_score

        })

    candidates.sort(

        key=lambda item:
        item["consensus_score"],

        reverse=True

    )

    if not candidates:

        return None

    best = candidates[0]

    if (
        best["repetitions"]
        < MIN_PLATE_REPETITIONS
    ):

        return None

    if (
        best["average_confidence"]
        < MIN_OCR_CONFIDENCE
    ):

        return None

    return best


# ============================================================
# PRINT OCR HISTORY
# ============================================================

def print_ocr_history(track_id):

    if track_id not in ocr_history:

        print(
            f"Track ID {track_id}: "
            "NO USEFUL OCR OBSERVATIONS"
        )

        return

    history = ocr_history[track_id]

    print(
        f"\nTrack ID {track_id}"
    )

    print(
        "-" * 60
    )

    for index, item in enumerate(
        history,
        start=1
    ):

        print(

            f"{index:02d}. "

            f"{item['text']:15}"

            f"Conf={item['confidence']:.2f} "

            f"Score={item['score']:.2f}"

        )


# ============================================================
# PRINT ALL OCR HISTORY
# ============================================================

def print_all_ocr_history():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "OCR HISTORY BY TRACK ID"
    )

    print(
        "=" * 70
    )

    if not ocr_history:

        print(
            "No OCR history was collected."
        )

        print(
            "=" * 70
        )

        return

    for track_id in sorted(
        ocr_history.keys()
    ):

        print_ocr_history(
            track_id
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "END OCR HISTORY DIAGNOSTIC"
    )

    print(
        "=" * 70
    )


# ============================================================
# FINALIZE VEHICLE
# ============================================================

def finalize_vehicle(
    track_id,
    plate_yolo_confidence,
    image_name,
    plate_image_name
):

    global authorized_count
    global unauthorized_count
    global finalization_attempt_count

    finalization_attempt_count += 1

    consensus = get_consensus_plate(
        track_id
    )

    if consensus is None:

        return False

    detected_plate = consensus["text"]

    ocr_confidence = consensus[
        "average_confidence"
    ]

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL OCR CONSENSUS"
    )

    print(
        "=" * 60
    )

    print(
        f"Vehicle Track ID    : "
        f"{track_id}"
    )

    print(
        f"Final Plate         : "
        f"{detected_plate}"
    )

    print(
        f"Repetitions         : "
        f"{consensus['repetitions']}"
    )

    print(
        f"Average OCR Conf.   : "
        f"{ocr_confidence:.2f}"
    )

    print(
        f"Average OCR Score   : "
        f"{consensus['average_score']:.2f}"
    )

    print_ocr_history(
        track_id
    )

    # ========================================================
    # AUTHORIZATION
    # ========================================================

    authorization = authorize_vehicle(
        detected_plate
    )

    status = authorization["status"]

    owner = authorization["owner"]

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

    # ========================================================
    # STATISTICS
    # ========================================================

    if status.lower() == "authorized":

        authorized_count += 1

    else:

        unauthorized_count += 1

    # ========================================================
    # DISPLAY
    # ========================================================

    print()

    print(
        f"Detected Plate       : "
        f"{detected_plate}"
    )

    print(
        f"Registration Year    : "
        f"{registration_year}"
    )

    print(
        f"OCR Confidence       : "
        f"{ocr_confidence:.2f}"
    )

    print(
        f"Plate YOLO Confidence: "
        f"{plate_yolo_confidence:.2f}"
    )

    print()

    print(
        f"STATUS               : "
        f"{status.upper()}"
    )

    print(
        f"Owner                : "
        f"{owner}"
    )

    print(
        f"Employee ID          : "
        f"{employee_id}"
    )

    print(
        f"Department           : "
        f"{department}"
    )

    print(
        f"Vehicle Type         : "
        f"{vehicle_type}"
    )

    # ========================================================
    # LOG
    # ========================================================

    log_entry(

        image_name=plate_image_name,

        plate_number=detected_plate,

        registration_year=registration_year,

        confidence=ocr_confidence,

        status=status,

        owner=owner,

        employee_id=employee_id,

        department=department,

        vehicle_type=vehicle_type

    )

    print(
        "\nVehicle entry logged successfully."
    )

    return True


# ============================================================
# PROCESS PLATE
# ============================================================

def process_plate(
    plate,
    image_name,
    track_id=None
):

    global ocr_success_count
    global ocr_failure_count

    plate_image = plate["image"]

    plate_image_name = plate[
        "image_name"
    ]

    plate_yolo_confidence = plate[
        "yolo_confidence"
    ]

    print(
        "\n"
        + "-" * 60
    )

    print(
        f"Processing: {image_name}"
    )

    if track_id is not None:

        print(
            f"Vehicle Track ID    : "
            f"{track_id}"
        )

    # ========================================================
    # OCR
    # ========================================================

    ocr_result = recognize_plate(
        plate_image
    )

    detected_plate = ocr_result[
        "plate_text"
    ]

    ocr_confidence = ocr_result[
        "confidence"
    ]

    ocr_score = ocr_result.get(
        "score",
        0
    )

    # ========================================================
    # OCR FAILURE
    # ========================================================

    if not detected_plate:

        ocr_failure_count += 1

        print(
            "\nNo readable plate text detected."
        )

        if track_id is not None:

            print(
                "Vehicle will be checked again "
                "in a later frame."
            )

        return False

    # ========================================================
    # OCR SUCCESS
    # ========================================================

    ocr_success_count += 1

    # ========================================================
    # IMAGE MODE
    # ========================================================

    if track_id is None:

        authorization = authorize_vehicle(
            detected_plate
        )

        status = authorization["status"]

        owner = authorization["owner"]

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

        print()

        print(
            f"Detected Plate       : "
            f"{detected_plate}"
        )

        print(
            f"Registration Year    : "
            f"{registration_year}"
        )

        print(
            f"OCR Confidence       : "
            f"{ocr_confidence:.2f}"
        )

        print(
            f"Plate YOLO Confidence: "
            f"{plate_yolo_confidence:.2f}"
        )

        print()

        print(
            f"STATUS               : "
            f"{status.upper()}"
        )

        print(
            f"Owner                : "
            f"{owner}"
        )

        print(
            f"Employee ID          : "
            f"{employee_id}"
        )

        print(
            f"Department           : "
            f"{department}"
        )

        print(
            f"Vehicle Type         : "
            f"{vehicle_type}"
        )

        log_entry(

            image_name=plate_image_name,

            plate_number=detected_plate,

            registration_year=registration_year,

            confidence=ocr_confidence,

            status=status,

            owner=owner,

            employee_id=employee_id,

            department=department,

            vehicle_type=vehicle_type

        )

        return True

    # ========================================================
    # VIDEO / WEBCAM
    # ========================================================

    add_ocr_observation(

        track_id,

        detected_plate,

        ocr_confidence,

        ocr_score

    )

    print()

    print(
        f"OCR Observation     : "
        f"{detected_plate}"
    )

    print(
        f"OCR Confidence      : "
        f"{ocr_confidence:.2f}"
    )

    print(
        f"OCR Score           : "
        f"{ocr_score:.2f}"
    )

    consensus = get_consensus_plate(
        track_id
    )

    if consensus is not None:

        print()

        print(
            f"Current consensus   : "
            f"{consensus['text']}"
        )

        print(
            f"Repetitions         : "
            f"{consensus['repetitions']}"
        )

    else:

        print(
            "\nNot enough OCR evidence yet."
        )

    return False


# ============================================================
# PROCESS NORMAL IMAGE
# ============================================================

def process_image(
    image,
    image_name
):

    cropped_plates = detect_and_crop(

        image,

        save_output=True

    )

    if len(cropped_plates) == 0:

        print(
            "\nNo license plates detected."
        )

        return

    for plate in cropped_plates:

        process_plate(

            plate,

            image_name,

            track_id=None

        )


# ============================================================
# PROCESS TRACKING FRAME
# ============================================================

def process_tracking_frame(
    frame,
    frame_name
):

    global vehicle_detection_count
    global vehicle_detection_with_id_count
    global plate_detection_count

    # ========================================================
    # VEHICLE DETECTION + TRACKING
    # ========================================================

    results = vehicle_model.track(

        source=frame,

        persist=True,

        tracker="bytetrack.yaml",

        classes=VEHICLE_CLASSES,

        conf=VEHICLE_CONFIDENCE,

        verbose=False

    )

    # ========================================================
    # PROCESS RESULTS
    # ========================================================

    for result in results:

        if result.boxes is None:

            continue

        if len(result.boxes) == 0:

            continue

        boxes = result.boxes.xyxy

        vehicle_detection_count += len(
            boxes
        )

        # ====================================================
        # TRACK IDS
        # ====================================================

        if result.boxes.id is None:

            print(
                f"Vehicle detected in "
                f"{frame_name}, but no tracking ID."
            )

            continue

        track_ids = result.boxes.id

        vehicle_detection_with_id_count += len(
            track_ids
        )

        # ====================================================
        # EACH VEHICLE
        # ====================================================

        for box, track_id_tensor in zip(

            boxes,

            track_ids

        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )

            track_id = int(
                track_id_tensor
            )

            tracked_vehicle_ids.add(
                track_id
            )

            print(
                f"\nVehicle detected | "
                f"Frame: {frame_name} | "
                f"Track ID: {track_id}"
            )

            # =================================================
            # ALREADY LOGGED
            # =================================================

            if track_id in logged_track_ids:

                print(
                    f"Track ID {track_id} "
                    f"already finalized."
                )

                continue

            # =================================================
            # FRAME BOUNDARIES
            # =================================================

            height, width = frame.shape[:2]

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

            if x2 <= x1 or y2 <= y1:

                continue

            # =================================================
            # VEHICLE CROP
            # =================================================

            vehicle_crop = frame[
                y1:y2,
                x1:x2
            ].copy()

            if vehicle_crop.size == 0:

                continue

            # =================================================
            # LICENSE PLATE DETECTION
            # =================================================

            print(
                f"Running license plate detector "
                f"for Track ID {track_id}..."
            )

            cropped_plates = detect_and_crop(

                vehicle_crop,

                save_output=False

            )

            # =================================================
            # NO PLATE
            # =================================================

            if len(cropped_plates) == 0:

                print(
                    f"No license plate detected "
                    f"for Track ID {track_id}."
                )

                continue

            # =================================================
            # PLATE COUNT
            # =================================================

            plate_detection_count += len(
                cropped_plates
            )

            print(
                f"License plates detected: "
                f"{len(cropped_plates)}"
            )

            # =================================================
            # PROCESS PLATES
            # =================================================

            for plate in cropped_plates:

                process_plate(

                    plate,

                    frame_name,

                    track_id=track_id

                )

            # =================================================
            # TRY FINALIZATION
            # =================================================

            if track_id not in ocr_history:

                continue

            consensus = get_consensus_plate(
                track_id
            )

            if consensus is None:

                continue

            # =================================================
            # FINALIZE
            # =================================================

            last_plate = cropped_plates[-1]

            finalized = finalize_vehicle(

                track_id,

                plate_yolo_confidence=(
                    last_plate[
                        "yolo_confidence"
                    ]
                ),

                image_name=frame_name,

                plate_image_name=(
                    last_plate[
                        "image_name"
                    ]
                )

            )

            if finalized:

                logged_track_ids.add(
                    track_id
                )

                print()

                print(
                    f"Vehicle Track ID "
                    f"{track_id} completed."
                )


# ============================================================
# START SYSTEM
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "VEHICLE AUTHORIZATION SYSTEM"
)

print(
    "=" * 70
)


# ============================================================
# IMAGE MODE
# ============================================================

if INPUT_MODE == "image":

    image_paths = []

    if isinstance(
        INPUT_PATH,
        list
    ):

        for image in INPUT_PATH:

            if os.path.isfile(image):

                image_paths.append(image)

            else:

                print(
                    f"Image not found: {image}"
                )

    elif os.path.isfile(INPUT_PATH):

        image_paths = [
            INPUT_PATH
        ]

    elif os.path.isdir(INPUT_PATH):

        image_paths = [

            os.path.join(
                INPUT_PATH,
                file
            )

            for file in sorted(
                os.listdir(INPUT_PATH)
            )

            if file.lower().endswith(

                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )

            )

        ]

    else:

        print(
            "Invalid image input."
        )

        sys.exit(1)

    if not image_paths:

        print(
            "No valid images found."
        )

        sys.exit(1)

    for image_path in image_paths:

        print(
            f"\nProcessing image: "
            f"{image_path}"
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                f"Unable to load: "
                f"{image_path}"
            )

            continue

        image_name = os.path.basename(
            image_path
        )

        process_image(
            image,
            image_name
        )


# ============================================================
# VIDEO MODE
# ============================================================

elif INPUT_MODE == "video":

    video_path = os.path.abspath(
        INPUT_PATH
    )

    print(
        f"\nOpening video:"
    )

    print(
        video_path
    )

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():

        print(
            "\nERROR: Unable to open video."
        )

        print(
            f"Path attempted:"
        )

        print(
            video_path
        )

        sys.exit(1)

    print(
        "\nVIDEO OPENED: True"
    )

    print(
        "\nVehicle detection + tracking enabled."
    )

    print(
        "\nVehicle classes:"
    )

    print(
        "Car / Motorcycle / Bus / Truck"
    )

    print(
        f"\nProcessing every "
        f"{FRAME_SKIP}th frame."
    )

    frame_number = 0

    # ========================================================
    # READ VIDEO
    # ========================================================

    while True:

        ret, frame = video.read()

        if not ret:

            break

        frame_number += 1

        if (
            frame_number
            % FRAME_SKIP
            != 0
        ):

            continue

        print(
            f"\nProcessing video frame: "
            f"{frame_number}"
        )

        process_tracking_frame(

            frame,

            f"video_frame_{frame_number}"

        )

    # ========================================================
    # RELEASE
    # ========================================================

    video.release()

    # ========================================================
    # OCR HISTORY
    # ========================================================

    print_all_ocr_history()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "VIDEO TESTING REPORT"
    )

    print(
        "=" * 60
    )

    print(
        f"Total frames in video       : "
        f"{frame_number}"
    )

    print(
        f"Frames processed            : "
        f"{frame_number // FRAME_SKIP}"
    )

    print(
        f"Unique vehicles detected    : "
        f"{len(tracked_vehicle_ids)}"
    )

    print(
        f"Vehicle detections processed: "
        f"{vehicle_detection_count}"
    )

    print(
        f"Vehicle detections with ID  : "
        f"{vehicle_detection_with_id_count}"
    )

    print(
        f"License plates detected     : "
        f"{plate_detection_count}"
    )

    print(
        f"OCR successful              : "
        f"{ocr_success_count}"
    )

    print(
        f"OCR failed                  : "
        f"{ocr_failure_count}"
    )

    print(
        f"Finalization attempts       : "
        f"{finalization_attempt_count}"
    )

    print(
        f"Authorized vehicles         : "
        f"{authorized_count}"
    )

    print(
        f"Unauthorized vehicles       : "
        f"{unauthorized_count}"
    )

    print(
        f"Unique vehicles logged      : "
        f"{len(logged_track_ids)}"
    )

    print(
        "=" * 60
    )


# ============================================================
# WEBCAM MODE
# ============================================================

elif INPUT_MODE == "webcam":

    camera_source = INPUT_PATH

    # If INPUT_PATH is not an integer,
    # default to webcam 0.

    if not isinstance(
        camera_source,
        int
    ):

        camera_source = 0

    camera = cv2.VideoCapture(
        camera_source
    )

    if not camera.isOpened():

        print(
            "Unable to open webcam."
        )

        sys.exit(1)

    print(
        "\nVehicle detection + tracking enabled."
    )

    print(
        "\nWebcam processing started."
    )

    print(
        "\nVehicle classes:"
    )

    print(
        "Car / Motorcycle / Bus / Truck"
    )

    print(
        f"\nProcessing every "
        f"{FRAME_SKIP}th frame."
    )

    print(
        "\nPress CTRL+C in terminal "
        "to stop."
    )

    frame_number = 0

    while True:

        ret, frame = camera.read()

        if not ret:

            print(
                "Unable to read webcam frame."
            )

            break

        frame_number += 1

        if (
            frame_number
            % FRAME_SKIP
            != 0
        ):

            continue

        print(
            f"\nProcessing webcam frame: "
            f"{frame_number}"
        )

        process_tracking_frame(

            frame,

            f"webcam_frame_{frame_number}"

        )

    camera.release()

    print_all_ocr_history()

    print(
        f"\nTotal webcam frames processed: "
        f"{frame_number}"
    )

    print(
        f"Unique vehicles detected: "
        f"{len(tracked_vehicle_ids)}"
    )

    print(
        f"Unique vehicles logged: "
        f"{len(logged_track_ids)}"
    )


# ============================================================
# INVALID MODE
# ============================================================

else:

    print(
        "\nInvalid INPUT_MODE."
    )

    print(
        "Available modes:"
    )

    print(
        "image"
    )

    print(
        "video"
    )

    print(
        "webcam"
    )

    sys.exit(1)


# ============================================================
# END
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PROCESSING COMPLETED"
)

print(
    "=" * 70
)