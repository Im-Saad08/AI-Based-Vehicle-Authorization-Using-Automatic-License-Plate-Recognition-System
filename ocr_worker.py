import sys
import os
import json

os.environ["FLAGS_enable_pir_api"] = "0"


# ============================================================
# CHECK IMAGE ARGUMENT
# ============================================================

if len(sys.argv) < 2:

    print(
        json.dumps({
            "error": "No image path supplied."
        })
    )

    sys.exit(1)


image_path = sys.argv[1]


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(image_path):

    print(
        json.dumps({
            "error": "Image does not exist."
        })
    )

    sys.exit(1)


# ============================================================
# IMPORT PADDLEOCR
# ============================================================

from paddleocr import PaddleOCR


# ============================================================
# CREATE OCR
# ============================================================

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)


# ============================================================
# RUN OCR
# ============================================================

result = ocr.predict(
    image_path
)


# ============================================================
# EXTRACT RESULT
# ============================================================

texts = []
scores = []


if result:

    data = result[0]

    texts = data.get(
        "rec_texts",
        []
    )

    scores = data.get(
        "rec_scores",
        []
    )


# ============================================================
# CREATE JSON OUTPUT
# ============================================================

output = []


for text, score in zip(
    texts,
    scores
):

    output.append({

        "text":
            str(text),

        "confidence":
            float(score)

    })


# ============================================================
# SEND RESULT TO PARENT PROCESS
# ============================================================

print(
    json.dumps(
        output
    )
)