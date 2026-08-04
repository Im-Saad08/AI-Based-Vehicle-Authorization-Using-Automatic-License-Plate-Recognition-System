import cv2


# ----------------------------------------
# Generate enhanced versions of a plate
# ----------------------------------------
def enhance_plate(plate_image):

    # ----------------------------------------
    # Upscale the original plate
    # ----------------------------------------
    upscaled = cv2.resize(
        plate_image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # ----------------------------------------
    # Convert to grayscale
    # ----------------------------------------
    grayscale = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )

    # ----------------------------------------
    # Contrast enhancement using CLAHE
    # ----------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast_enhanced = clahe.apply(
        grayscale
    )

    # ----------------------------------------
    # Sharpen the image
    # ----------------------------------------
    blurred = cv2.GaussianBlur(
        contrast_enhanced,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        contrast_enhanced,
        1.5,
        blurred,
        -0.5,
        0
    )

    # ----------------------------------------
    # Adaptive thresholding
    # Helps with uneven lighting
    # ----------------------------------------
    thresholded = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # ----------------------------------------
    # Return all versions
    # ----------------------------------------
    return {
        "upscaled": upscaled,
        "grayscale": grayscale,
        "contrast_enhanced": contrast_enhanced,
        "sharpened": sharpened,
        "thresholded": thresholded
    }