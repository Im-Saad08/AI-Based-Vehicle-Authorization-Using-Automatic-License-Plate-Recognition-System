from ultralytics import YOLO
import cv2
import os


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "run_model/detect/runs/license_plate_detector/weights/img_best.pt"

VIDEO_PATH = "img/input/video.mp4"

OUTPUT_FOLDER = "img/output/test_plate_enhancement"


# ============================================================
# SETTINGS
# ============================================================

NUMBER_OF_FRAMES = 10

CONFIDENCE = 0.25

# Upscaling factor
SCALE = 5


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

original_folder = os.path.join(
    OUTPUT_FOLDER,
    "original"
)

upscaled_folder = os.path.join(
    OUTPUT_FOLDER,
    "upscaled_sharpened"
)

contrast_folder = os.path.join(
    OUTPUT_FOLDER,
    "grayscale_contrast"
)

threshold_folder = os.path.join(
    OUTPUT_FOLDER,
    "adaptive_threshold"
)

clahe_folder = os.path.join(
    OUTPUT_FOLDER,
    "clahe_sharpened"
)


os.makedirs(
    original_folder,
    exist_ok=True
)

os.makedirs(
    upscaled_folder,
    exist_ok=True
)

os.makedirs(
    contrast_folder,
    exist_ok=True
)

os.makedirs(
    threshold_folder,
    exist_ok=True
)

os.makedirs(
    clahe_folder,
    exist_ok=True
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("\nLoading license plate detection model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# OPEN VIDEO
# ============================================================

video = cv2.VideoCapture(
    VIDEO_PATH
)

if not video.isOpened():

    print(
        "\nError: Unable to open video."
    )

    exit()


# ============================================================
# VIDEO INFORMATION
# ============================================================

total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = video.get(
    cv2.CAP_PROP_FPS
)


print("\n" + "=" * 60)

print(
    "PLATE DETECTION + IMAGE ENHANCEMENT TEST"
)

print("=" * 60)

print(
    f"Video             : {VIDEO_PATH}"
)

print(
    f"Total frames      : {total_frames}"
)

print(
    f"FPS               : {fps:.2f}"
)

print(
    f"Frames to test    : {NUMBER_OF_FRAMES}"
)

print(
    f"YOLO confidence   : {CONFIDENCE}"
)

print(
    f"Upscale factor    : {SCALE}"
)


# ============================================================
# SELECT REPRESENTATIVE FRAMES
# ============================================================

if total_frames <= NUMBER_OF_FRAMES:

    frame_numbers = list(
        range(total_frames)
    )

else:

    frame_numbers = [

        int(
            i * (total_frames - 1)
            / (NUMBER_OF_FRAMES - 1)
        )

        for i in range(
            NUMBER_OF_FRAMES
        )
    ]


# ============================================================
# ENHANCEMENT FUNCTION
# ============================================================

def enhance_plate(
    plate_image
):

    # --------------------------------------------------------
    # 1. UPSCALE
    # --------------------------------------------------------

    upscaled = cv2.resize(
        plate_image,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_CUBIC
    )


    # --------------------------------------------------------
    # 2. SHARPEN
    # --------------------------------------------------------

    sharpening_kernel = (

        cv2.getGaussianKernel(
            3,
            0
        )

    )

    gaussian = cv2.filter2D(
        upscaled,
        -1,
        sharpening_kernel
        @ sharpening_kernel.T
    )

    sharpened = cv2.addWeighted(
        upscaled,
        1.8,
        gaussian,
        -0.8,
        0
    )


    # --------------------------------------------------------
    # 3. GRAYSCALE + CONTRAST
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )

    contrast = cv2.equalizeHist(
        gray
    )


    # --------------------------------------------------------
    # 4. ADAPTIVE THRESHOLD
    # --------------------------------------------------------

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )


    # --------------------------------------------------------
    # 5. CLAHE + SHARPENING
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_image = clahe.apply(
        gray
    )

    clahe_sharpened = cv2.GaussianBlur(
        clahe_image,
        (0, 0),
        3
    )

    clahe_sharpened = cv2.addWeighted(
        clahe_image,
        1.7,
        clahe_sharpened,
        -0.7,
        0
    )


    return (
        upscaled,
        sharpened,
        contrast,
        threshold,
        clahe_sharpened
    )


# ============================================================
# PROCESS FRAMES
# ============================================================

total_plates = 0

successful_frames = 0


for index, frame_number in enumerate(
    frame_numbers,
    start=1
):

    # --------------------------------------------------------
    # Move video to selected frame
    # --------------------------------------------------------

    video.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    ret, frame = video.read()


    if not ret:

        print(
            f"\nCould not read frame "
            f"{frame_number}."
        )

        continue


    print("\n" + "-" * 60)

    print(
        f"Test Frame "
        f"{index}/{len(frame_numbers)}"
    )

    print(
        f"Frame Number: "
        f"{frame_number}"
    )


    # ========================================================
    # RUN YOLO
    # ========================================================

    results = model.predict(

        source=frame,

        conf=CONFIDENCE,

        save=False,

        verbose=False
    )


    frame_plate_count = 0


    # ========================================================
    # PROCESS DETECTED PLATES
    # ========================================================

    for plate_number, box in enumerate(
        results[0].boxes,
        start=1
    ):

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            box.conf[0]
        )


        print(
            f"\nPlate {plate_number}"
        )

        print(
            f"Confidence : "
            f"{confidence:.4f}"
        )

        print(
            f"Bounding Box : "
            f"({x1}, {y1}) -> "
            f"({x2}, {y2})"
        )


        # ----------------------------------------------------
        # Crop plate
        # ----------------------------------------------------

        plate_crop = frame[
            y1:y2,
            x1:x2
        ]


        if plate_crop.size == 0:

            print(
                "Warning: Empty plate crop."
            )

            continue


        # ----------------------------------------------------
        # Crop dimensions
        # ----------------------------------------------------

        height, width = (
            plate_crop.shape[:2]
        )

        print(
            f"Original Size : "
            f"{width} x {height}"
        )


        # ====================================================
        # ENHANCE
        # ====================================================

        (
            upscaled,
            sharpened,
            contrast,
            threshold,
            clahe_sharpened
        ) = enhance_plate(
            plate_crop
        )


        # ====================================================
        # FILE NAME
        # ====================================================

        base_name = (

            f"frame_{frame_number}"
            f"_plate_{plate_number}"

        )


        # ====================================================
        # SAVE ORIGINAL
        # ====================================================

        original_path = os.path.join(

            original_folder,

            f"{base_name}.jpg"

        )

        cv2.imwrite(

            original_path,

            plate_crop

        )


        # ====================================================
        # SAVE UPSCALED + SHARPENED
        # ====================================================

        upscaled_path = os.path.join(

            upscaled_folder,

            f"{base_name}.jpg"

        )

        cv2.imwrite(

            upscaled_path,

            sharpened

        )


        # ====================================================
        # SAVE GRAYSCALE + CONTRAST
        # ====================================================

        contrast_path = os.path.join(

            contrast_folder,

            f"{base_name}.jpg"

        )

        cv2.imwrite(

            contrast_path,

            contrast

        )


        # ====================================================
        # SAVE ADAPTIVE THRESHOLD
        # ====================================================

        threshold_path = os.path.join(

            threshold_folder,

            f"{base_name}.jpg"

        )

        cv2.imwrite(

            threshold_path,

            threshold

        )


        # ====================================================
        # SAVE CLAHE + SHARPENED
        # ====================================================

        clahe_path = os.path.join(

            clahe_folder,

            f"{base_name}.jpg"

        )

        cv2.imwrite(

            clahe_path,

            clahe_sharpened

        )


        # ====================================================
        # PRINT SAVED FILES
        # ====================================================

        print(
            "Saved original crop."
        )

        print(
            "Saved upscaled + sharpened."
        )

        print(
            "Saved grayscale + contrast."
        )

        print(
            "Saved adaptive threshold."
        )

        print(
            "Saved CLAHE + sharpened."
        )


        frame_plate_count += 1


    # ========================================================
    # SAVE ANNOTATED FRAME
    # ========================================================

    annotated_frame = results[0].plot()


    annotated_path = os.path.join(

        OUTPUT_FOLDER,

        f"frame_{frame_number}_detected.jpg"

    )


    cv2.imwrite(

        annotated_path,

        annotated_frame

    )


    print(
        f"Plates detected in frame: "
        f"{frame_plate_count}"
    )


    if frame_plate_count > 0:

        successful_frames += 1


    total_plates += frame_plate_count


# ============================================================
# RELEASE VIDEO
# ============================================================

video.release()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)

print(
    "IMAGE ENHANCEMENT TEST COMPLETED"
)

print("=" * 60)

print(
    f"Total video frames       : "
    f"{total_frames}"
)

print(
    f"Frames tested            : "
    f"{len(frame_numbers)}"
)

print(
    f"Frames with plates       : "
    f"{successful_frames}"
)

print(
    f"Frames without plates    : "
    f"{len(frame_numbers) - successful_frames}"
)

print(
    f"Total plates detected    : "
    f"{total_plates}"
)

print("\nResults saved to:")

print(
    OUTPUT_FOLDER
)

print("=" * 60)