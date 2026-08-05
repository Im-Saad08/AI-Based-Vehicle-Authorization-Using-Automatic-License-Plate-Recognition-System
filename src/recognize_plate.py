import easyocr
import re
from enhance_plate import enhance_plate


# ----------------------------------------
# Initialize EasyOCR
# ----------------------------------------

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

# ----------------------------------------
# Score a detected plate
# ----------------------------------------

def score_plate(
    plate_text,
    confidence
):

    text = plate_text.upper().strip()

    score = confidence * 100

    letters = len(
        re.findall(
            r"[A-Z]",
            text
        )
    )

    digits = len(
        re.findall(
            r"\d",
            text
        )
    )

    special = len(
        re.findall(
            r"[^A-Z0-9 ]",
            text
        )
    )

    # ----------------------------------------
    # Very short OCR results are unreliable
    # ----------------------------------------

    if len(text.replace(" ", "")) < 5:

        score -= 80

    # ----------------------------------------
    # Plate contains both letters and numbers
    # ----------------------------------------

    if letters >= 2 and digits >= 3:

        score += 40

    # ----------------------------------------
    # Pakistani plate format
    # Example:
    # ABC 1234
    # ----------------------------------------

    if re.match(

        r'^[A-Z]{2,3}\s*\d{3,4}$',

        text

    ):

        score += 60

    # ----------------------------------------
    # Only letters
    # Example: MNS
    # ----------------------------------------

    if letters > 0 and digits == 0:

        score -= 70

    # ----------------------------------------
    # Only digits
    # ----------------------------------------

    if digits > 0 and letters == 0:

        score -= 40

    # ----------------------------------------
    # Penalize special characters
    # ----------------------------------------

    score -= special * 20

    return score

# ----------------------------------------
# OCR a single image
# ----------------------------------------

def ocr_single_image(image):

    results = reader.readtext(image)

    if len(results) == 0:

        return "", 0.0

    detected_texts = []

    confidence_scores = []

    for detection in results:

        bbox, text, confidence = detection

        detected_texts.append(
            text
        )

        confidence_scores.append(
            confidence
        )

    final_text = " ".join(
        detected_texts
    ).upper().strip()

    confidence = min(
        confidence_scores
    )

    return final_text, confidence


# ----------------------------------------
# Main OCR function
# ----------------------------------------

def recognize_plate(
    plate_image
):

    enhanced_images = enhance_plate(
        plate_image
    )

    candidates = []

    # Original image
    text, conf = ocr_single_image(
        plate_image
    )

    candidates.append({

        "version": "Original",

        "text": text,

        "confidence": conf,

        "score": score_plate(
            text,
            conf
        )

    })

    # Enhanced versions
    for version, image in enhanced_images.items():

        text, conf = ocr_single_image(
            image
        )

        candidates.append({

            "version": version,

            "text": text,

            "confidence": conf,

            "score": score_plate(
                text,
                conf
            )

        })

    # Select best result
    best = max(
        candidates,
        key=lambda x: x["score"]
    )

    # Display comparison
    print("\nOCR Comparison")
    print("-" * 60)

    for candidate in candidates:

        print(

            f"{candidate['version']:20}"

            f"{candidate['text']:20}"

            f"Conf={candidate['confidence']:.2f}"

            f"  Score={candidate['score']:.2f}"

        )

    print("-" * 60)

    print(

        f"Selected OCR: "

        f"{best['text']}"

    )

    return {

        "plate_text": best["text"],

        "confidence": best["confidence"]

    }