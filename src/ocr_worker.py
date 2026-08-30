import sys
import os

# ============================================================
# LIMIT ONE/DNN/MKL THREADS BEFORE PADDLEOCR LOADS
# ============================================================
# oneDNN/mkldnn (used by PaddleOCR) defaults to using ALL
# CPU cores. This starves the main thread's YOLO inference
# (also CPU-bound) and causes 5-12s lag spikes.
# Limit to 2 threads to leave headroom for YOLO (~150ms target).
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

print("=== OCR WORKER STARTED ===", flush=True)

print(
    "Python:",
    sys.version,
    flush=True
)

print(
    "Executable:",
    sys.executable,
    flush=True
)

print(
    "Arguments:",
    sys.argv,
    flush=True
)


# ============================================================
# PADDLEOCR COMPATIBILITY
# ============================================================

os.environ["FLAGS_enable_pir_api"] = "0"


# ============================================================
# IMPORT PADDLEOCR
# ============================================================

print(
    "Importing PaddleOCR...",
    flush=True
)

from paddleocr import PaddleOCR

print(
    "PaddleOCR imported successfully.",
    flush=True
)


# ============================================================
# CHECK IMAGE ARGUMENT
# ============================================================

if len(sys.argv) < 2:

    print(
        "ERROR: No image path supplied.",
        flush=True
    )

    sys.exit(1)


image_path = sys.argv[1]


print(
    "Image:",
    image_path,
    flush=True
)

print(
    "Image exists:",
    os.path.exists(image_path),
    flush=True
)


if not os.path.exists(image_path):

    print(
        "ERROR: Image does not exist.",
        flush=True
    )

    sys.exit(1)


# ============================================================
# CREATE OCR
# ============================================================

print(
    "Creating PaddleOCR...",
    flush=True
)

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
)

print(
    "PaddleOCR initialized successfully.",
    flush=True
)


# ============================================================
# RUN OCR
# ============================================================

print(
    "Running OCR...",
    flush=True
)

result = ocr.predict(
    image_path
)

print(
    "OCR FINISHED.",
    flush=True
)


# ============================================================
# EXTRACT TEXT
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
# DISPLAY RESULT
# ============================================================

print(
    "\nOCR RESULT:"
)

print(
    "-" * 50
)


for text, score in zip(
    texts,
    scores
):

    print(
        f"Text       : {text}"
    )

    print(
        f"Confidence : {float(score):.4f}"
    )


print(
    "-" * 50
)