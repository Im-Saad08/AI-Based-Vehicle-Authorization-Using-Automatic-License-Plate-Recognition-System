import easyocr


# ----------------------------------------
# Initialize EasyOCR once
# ----------------------------------------
reader = easyocr.Reader(
    ['en'],
    gpu=False
)


# ----------------------------------------
# Recognize license plate text
# ----------------------------------------
def recognize_plate(plate_image):

    # Run EasyOCR
    results = reader.readtext(
        plate_image
    )

    # No text detected
    if len(results) == 0:

        return {
            "plate_text": "",
            "confidence": 0.0
        }

    # Store detected text
    detected_texts = []

    # Store confidence scores
    confidence_scores = []

    # Process OCR results
    for detection in results:

        bbox, text, confidence = detection

        detected_texts.append(
            text
        )

        confidence_scores.append(
            confidence
        )

    # Combine all detected text
    final_text = " ".join(
        detected_texts
    )

    # Use minimum confidence
    min_confidence = min(
        confidence_scores
    )

    # Return OCR result
    return {
        "plate_text": final_text,
        "confidence": min_confidence
    }