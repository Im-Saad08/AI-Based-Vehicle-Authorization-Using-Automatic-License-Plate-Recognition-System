# ============================================================
# VEHICLE AUTHORIZATION SYSTEM
# MAIN PIPELINE
# ============================================================

import os
import sys
import cv2
import re
import time

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
from normalize_plate import format_plate_display


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

INPUT_MODE = "video"
#video processing got improved, from processing 10sec/321f in ~1hr to 2 minutes

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

INPUT_PATH = "img/input/video.mp4"


# ============================================================
# LICENSE PLATE MODEL
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "trained",
    "rbflw_y8_best.pt"
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
    f"Plate model  : {MODEL_PATH}"
)

print(
    f"Model exists : {os.path.exists(MODEL_PATH)}"
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
# LOAD LICENSE PLATE MODEL
# ============================================================

print(
    "\nLoading license plate detection model..."
)

plate_model = YOLO(
    MODEL_PATH
)

print(
    "License plate model loaded successfully."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "LICENSE PLATE MODEL INFORMATION"
)

print(
    "=" * 70
)

print(
    f"Plate model  : {MODEL_PATH}"
)

print(
    f"Model classes: {plate_model.names}"
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
# 3 = process every third frame (CPU efficiency: skip 2/3 frames)
#
# For testing OCR/tracking, keep this at 1.

FRAME_SKIP = 3


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


# ============================================================
# PER-TRACK OCR GATING (CPU Efficiency Rule 3)
# ============================================================

# Maximum OCR attempts per track ID before giving up
MAX_OCR_ATTEMPTS_PER_TRACK = 3

# Confidence threshold at which we consider OCR "good enough" and stop retrying
# Once a track reaches this confidence, we skip OCR for that track entirely
OCR_CONFIDENCE_THRESHOLD = 0.50

# High confidence accept: a single OCR read at or above this confidence
# finalizes the track immediately, without waiting for consensus.
HIGH_CONFIDENCE_ACCEPT = 0.85

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
# PER-TRACK OCR ATTEMPT TRACKING (CPU Efficiency Rule 3)
# ============================================================

# Track how many OCR attempts have been made per track ID
ocr_attempts_per_track = {}

# Track whether a track has reached the confidence threshold
track_reached_confidence_threshold = set()


# ============================================================
# STATISTICS
# ============================================================

tracked_vehicle_ids = set()

vehicle_detection_count = 0

vehicle_detection_with_id_count = 0

plate_detection_count = 0

ocr_success_count = 0

ocr_failure_count = 0

ocr_call_count = 0

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
    plate_image_name,
    override_plate=None,
    override_confidence=None
):

    global authorized_count
    global unauthorized_count
    global finalization_attempt_count

    finalization_attempt_count += 1

    # High-confidence bypass: use provided override values
    if override_plate is not None and override_confidence is not None:
        detected_plate = override_plate
        ocr_confidence = override_confidence
        consensus = None
    else:
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

    if consensus is not None:
        print(
            "FINAL OCR CONSENSUS"
        )
    else:
        print(
            "FINAL OCR (HIGH-CONFIDENCE SINGLE READ)"
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

    if consensus is not None:
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
    else:
        print(
            f"Single-read Conf.   : "
            f"{ocr_confidence:.2f}"
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
    normalized_plate = authorization["normalized_plate"]

    # Format for display as ABC-123
    display_plate = format_plate_display(normalized_plate)

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
        f"{display_plate}"
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

    # ========================================================
    # LOG
    # ========================================================

    log_entry(

        image_name=plate_image_name,

        plate_number=normalized_plate,  # Log as ABC123 (no dash)

        confidence=ocr_confidence,

        status=status

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
    global ocr_call_count

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

    ocr_call_count += 1

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
        normalized_plate = authorization["normalized_plate"]

        # Format for display as ABC-123
        display_plate = format_plate_display(normalized_plate)

        print()

        print(
            f"Detected Plate       : "
            f"{display_plate}"
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

        log_entry(

            image_name=plate_image_name,

            plate_number=normalized_plate,  # Log as ABC123 (no dash)

            confidence=ocr_confidence,

            status=status

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

    # High-confidence single-read finalization (Option 1) — runs regardless of consensus
    if ocr_confidence >= HIGH_CONFIDENCE_ACCEPT:
        print(f"Track ID {track_id}: OCR confidence {ocr_confidence:.2f} >= {HIGH_CONFIDENCE_ACCEPT} — finalizing immediately (high-confidence bypass).")
        # Finalize directly without waiting for consensus
        finalized = finalize_vehicle(
            track_id,
            plate_yolo_confidence=plate["yolo_confidence"],
            plate_image_name=plate["image_name"],
            override_plate=detected_plate,
            override_confidence=ocr_confidence
        )
        if finalized:
            logged_track_ids.add(track_id)
            track_reached_confidence_threshold.add(track_id)
            return True

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

        # Consensus-based finalization (existing logic)
        if consensus['average_confidence'] >= OCR_CONFIDENCE_THRESHOLD and consensus['repetitions'] >= MIN_PLATE_REPETITIONS:
            print(f"Track ID {track_id}: Consensus ready (conf={consensus['average_confidence']:.2f}, reps={consensus['repetitions']}), finalizing.")
            finalized = finalize_vehicle(
                track_id,
                plate_yolo_confidence=plate["yolo_confidence"],
                plate_image_name=plate["image_name"]
            )
            if finalized:
                logged_track_ids.add(track_id)
                track_reached_confidence_threshold.add(track_id)
                return True

    else:

        print(
            "\nNot enough OCR evidence yet."
        )

        # Only stop OCR for very high confidence single reads (>= HIGH_CONFIDENCE_ACCEPT)
        # Reads between 0.50 and 0.85 continue accumulating for consensus
        if ocr_confidence >= HIGH_CONFIDENCE_ACCEPT:
            track_reached_confidence_threshold.add(track_id)
            print(f"Track ID {track_id}: OCR confidence {ocr_confidence:.2f} >= {HIGH_CONFIDENCE_ACCEPT}, stopping OCR for this track.")

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

        save_output=True,

        image_name=image_name

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
    # DIRECT LICENSE PLATE DETECTION + TRACKING
    # ========================================================

    results = plate_model.track(

        source=frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=0.4,

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
                f"Plate detected in "
                f"{frame_name}, but no tracking ID."
            )

            continue

        track_ids = result.boxes.id

        vehicle_detection_with_id_count += len(
            track_ids
        )

        # ====================================================
        # EACH DETECTED LICENSE PLATE
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
                f"\nPlate detected | "
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
            # MINIMUM PLATE SIZE (DISTANCE FILTER)
            # =================================================

            plate_w = x2 - x1
            plate_h = y2 - y1

            if plate_w < 50 or plate_h < 15:
                print(
                    f"Skipping plate Track ID {track_id}: "
                    f"Plate box too small ({plate_w}x{plate_h}px). Vehicle is too far."
                )
                continue

            # =================================================
            # DIRECT PLATE CROP WITH 10% MARGIN PADDING
            # =================================================

            pad_w = int(plate_w * 0.10)
            pad_h = int(plate_h * 0.10)

            crop_x1 = max(0, x1 - pad_w)
            crop_y1 = max(0, y1 - pad_h)
            crop_x2 = min(width, x2 + pad_w)
            crop_y2 = min(height, y2 + pad_h)

            plate_crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ].copy()

            if plate_crop.size == 0:

                continue

            plate_info = {
                "image": plate_crop,
                "box": (x1, y1, x2, y2),
                "yolo_confidence": 0.90,
                "image_name": f"{frame_name}_plate_{track_id}.png"
            }

            plate_detection_count += 1

            # =================================================
            # PER-TRACK OCR GATING (Rule 3)
            # Check if we should run OCR for this track
            # =================================================

            # Initialize attempt counter if new track
            if track_id not in ocr_attempts_per_track:
                ocr_attempts_per_track[track_id] = 0

            # Skip OCR if track already reached confidence threshold
            if track_id in track_reached_confidence_threshold:
                print(f"Track ID {track_id}: Already has confident OCR result, skipping OCR.")
                continue

            # Skip OCR if max attempts reached
            if ocr_attempts_per_track[track_id] >= MAX_OCR_ATTEMPTS_PER_TRACK:
                print(f"Track ID {track_id}: Max OCR attempts ({MAX_OCR_ATTEMPTS_PER_TRACK}) reached, skipping OCR.")
                continue

            # Increment attempt counter
            ocr_attempts_per_track[track_id] += 1
            print(f"Track ID {track_id}: OCR attempt {ocr_attempts_per_track[track_id]}/{MAX_OCR_ATTEMPTS_PER_TRACK}")

            # =================================================
            # PROCESS PLATE DIRECTLY
            # =================================================

            process_plate(

                plate_info,

                frame_name,

                track_id=track_id

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

        image_name = "img_" + os.path.basename(
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

    # Start timing for total processing time
    total_start_time = time.perf_counter()

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

            f"vid_f{frame_number}"

        )

    # Calculate total processing time
    total_elapsed = time.perf_counter() - total_start_time

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
        f"Total processing time       : "
        f"{total_elapsed:.2f}s"
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
        f"OCR calls made              : "
        f"{ocr_call_count}"
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

            f"cam_f{frame_number}"

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