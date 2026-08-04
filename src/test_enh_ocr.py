import cv2
import easyocr

from enhance_plate import enhance_plate


# ----------------------------------------
# Input plate image
# ----------------------------------------
image_path = (
    "img/output/enhanced_plates/contrast_enhanced.png"
)


# ----------------------------------------
# Read original plate image
# ----------------------------------------
plate_image = cv2.imread(
    image_path
)

if plate_image is None:

    print(
        "Error: Unable to load plate image."
    )

    exit()


# ----------------------------------------
# Initialize EasyOCR
# ----------------------------------------
reader = easyocr.Reader(
    ['en'],
    gpu=False
)


# ----------------------------------------
# OCR on original image
# ----------------------------------------
original_results = reader.readtext(
    plate_image
)


print("\n" + "=" * 60)
print("ORIGINAL IMAGE OCR")
print("=" * 60)


if not original_results:

    print("No text detected.")

else:

    original_texts = []
    original_confidences = []

    for detection in original_results:

        bbox, text, confidence = detection

        original_texts.append(
            text
        )

        original_confidences.append(
            confidence
        )

        print(
            f"Text       : {text}"
        )

        print(
            f"Confidence : "
            f"{confidence:.2f}"
        )

    print(
        f"\nCombined Text: "
        f"{' '.join(original_texts)}"
    )

    print(
        f"Minimum Confidence: "
        f"{min(original_confidences):.2f}"
    )


# ----------------------------------------
# Generate enhanced versions
# ----------------------------------------
enhanced_versions = enhance_plate(
    plate_image
)


# ----------------------------------------
# Get contrast-enhanced image
# ----------------------------------------
contrast_image = enhanced_versions[
    "contrast_enhanced"
]


# ----------------------------------------
# OCR on contrast-enhanced image
# ----------------------------------------
enhanced_results = reader.readtext(
    contrast_image
)


print("\n" + "=" * 60)
print("CONTRAST-ENHANCED IMAGE OCR")
print("=" * 60)


if not enhanced_results:

    print("No text detected.")

else:

    enhanced_texts = []
    enhanced_confidences = []

    for detection in enhanced_results:

        bbox, text, confidence = detection

        enhanced_texts.append(
            text
        )

        enhanced_confidences.append(
            confidence
        )

        print(
            f"Text       : {text}"
        )

        print(
            f"Confidence : "
            f"{confidence:.2f}"
        )

    print(
        f"\nCombined Text: "
        f"{' '.join(enhanced_texts)}"
    )

    print(
        f"Minimum Confidence: "
        f"{min(enhanced_confidences):.2f}"
    )


# ----------------------------------------
# Final message
# ----------------------------------------
print("\n" + "=" * 60)
print("OCR ENHANCEMENT COMPARISON COMPLETED")
print("=" * 60)