import re
import os
import json
import subprocess
import tempfile
import numpy as np

from enhance_plate import enhance_plate


# ============================================================
# PROJECT ROOT
# ============================================================
#
# recognize_plate.py is inside:
#
#     Vehicle Authorization System-V2/src/
#
# Therefore we go one level up to reach:
#
#     Vehicle Authorization System-V2/
#
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# LOCAL PADDLEOCR WORKER
# ============================================================
#
# PaddleOCR is NOT imported in this file.
#
# The OCR worker is located outside src:
#
#     Vehicle Authorization System-V2/
#         ocr_worker.py
#
# It is executed using:
#
#     paddleocr_env\Scripts\python.exe
#
# which is Python 3.12.10.
#
# The main project remains on Python 3.14.6.
#
# ============================================================

OCR_WORKER = os.path.join(
    PROJECT_ROOT,
    "ocr_worker.py"
)


OCR_PYTHON = os.path.join(
    PROJECT_ROOT,
    "paddleocr_env",
    "Scripts",
    "python.exe"
)


# ============================================================
# OCR SETTINGS
# ============================================================

OCR_TIMEOUT = 120

MIN_OCR_CONFIDENCE = 0.05

# Early-exit threshold for the recognition-only OCR method.
#
# Under recognition-only OCR (no text detection), the whole-crop read is often
# a weak/degenerate read of a multi-line plate (e.g. DSC_1010 whole-crop "ML" @
# 0.35), so the OLD 0.70 gate calibrated for the full pipeline essentially never
# fires and all 5 enhancement passes always run.
#
# Across the real test set the genuinely-good reads land at:
#   whole-crop good  -> 0.86 (ACZ853, FZ886)
#   split (two-line) -> 0.98-1.00 (LED135620, LEH56314, MLE4008)
# while the noise floor (single chars / partial reads) sits at 0.17-0.49.
#
# 0.50 reliably fires on every good read while staying above the noise floor.
# We ALSO require len(plate) >= 3 so a lone high-confidence single char (e.g.
# "1" @ 0.64) can never trigger early-exit and mask a potentially-useful
# enhancement pass.
EARLY_EXIT_CONFIDENCE = 0.50


# ============================================================
# OCR CHARACTER CORRECTION
# ============================================================
# Import from normalize_plate to ensure single source of truth

from normalize_plate import (
    CHAR_TO_DIGIT,
    DIGIT_TO_CHAR,
    fix_token_characters
)


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_plate_text(
    text
):

    if not text:

        return ""


    raw_text = str(
        text
    ).upper().strip()

    # Extract tokens from raw text FIRST (preserving word boundaries)
    # This way "MLE 4008" -> ["MLE", "4008"] not ["MLE4008"]
    tokens = re.findall(r"[A-Z0-9]+", raw_text)

    # Apply smart character correction per token (like normalize_plate.py)
    corrected_tokens = [fix_token_characters(t) for t in tokens]

    # Join without separators
    text = "".join(corrected_tokens)


    return text


# ============================================================
# GENERATE CHARACTER-CORRECTED CANDIDATES
# ============================================================

def generate_character_candidates(
    text
):
    """Generate candidate plate texts with different character interpretations.
    Uses token-aware correction from the original raw text to avoid re-tokenizing
    already-joined text (e.g., "MLE4008" being treated as one token)."""
    if not text:
        return []

    # Extract tokens from RAW text FIRST (preserving word boundaries)
    # This way "MLE 4008" -> ["MLE", "4008"] not ["MLE4008"]
    raw_text = str(text).upper().strip()
    raw_tokens = re.findall(r"[A-Z0-9]+", raw_text)

    # Apply smart character correction per token (like normalize_plate.py)
    corrected_tokens = [fix_token_characters(t) for t in raw_tokens]

    # Base candidate: corrected tokens joined without separators
    # Already uses position-aware fix_token_characters from normalize_plate.py
    # No additional candidate generation needed - old logic was harmful
    base_text = "".join(corrected_tokens)

    if not base_text:
        return []

    # Return only the position-aware corrected candidate
    return [base_text]


# ============================================================
# PLATE FORMAT SCORE
# ============================================================

def plate_format_score(
    text
):
    # text is ALREADY cleaned (from generate_character_candidates or select_best_interpretation).
    # Do NOT re-apply clean_plate_text to avoid re-tokenizing and corrupting valid plates
    # (e.g., "MLE4008" being treated as one token and L→1 converted).

    if not text:

        return -100


    score = 0


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


    length = len(
        text
    )


    # ========================================================
    # LENGTH
    # ========================================================

    if length < 3:

        score -= 100

    elif length == 3:

        score -= 25

    elif length in [4, 5, 6]:

        score += 15

    elif length in [7, 8, 9]:

        score += 25

    elif length == 10:

        score += 20

    else:

        score -= 30


    # ========================================================
    # MIXED ALPHANUMERIC
    # ========================================================

    if letters > 0 and digits > 0:

        score += 25


    # ========================================================
    # COMMON LETTER + NUMBER FORMAT (single-line: 2-3 letters + 3-4 digits)
    # ========================================================

    if re.match(
        r"^[A-Z]{2,3}\d{3,4}$",
        text
    ):

        score += 80


    # ========================================================
    # TWO-LINE PLATE FORMAT (2-3 letters + 4-7 digits)
    # Common for Pakistani two-line plates like LE151051
    # Higher bonus than single-line to favor complete two-line reads
    # ========================================================

    if re.match(
        r"^[A-Z]{2,3}\d{4,7}$",
        text
    ):

        score += 100


    # ========================================================
    # NUMBER + LETTER FORMAT
    # ========================================================

    if re.match(
        r"^\d{3,4}[A-Z]{2,3}$",
        text
    ):

        score += 60


    # ========================================================
    # LETTERS + NUMBERS + LETTERS
    # ========================================================

    if re.match(
        r"^[A-Z]{1,3}\d{1,4}[A-Z]{0,3}$",
        text
    ):

        score += 25


    # ========================================================
    # NUMBER + LETTER + NUMBER
    # ========================================================

    if re.match(
        r"^\d{1,4}[A-Z]{1,3}\d{1,4}$",
        text
    ):

        score += 20


    # ========================================================
    # ONLY LETTERS
    # ========================================================

    if letters > 0 and digits == 0:

        score -= 0


    # ========================================================
    # ONLY DIGITS
    # ========================================================

    if digits > 0 and letters == 0:

        score -= 0


    return score


# ============================================================
# SCORE OCR RESULT
# ============================================================

def score_plate(
    plate_text,
    confidence
):
    """
    Score a plate candidate. Assumes plate_text is ALREADY CLEANED (e.g., from generate_character_candidates).
    Does NOT re-apply clean_plate_text to avoid corrupting already-corrected tokens.
    """
    text = plate_text  # Already cleaned

    if not text:
        return -100

    score = float(confidence) * 100

    score += plate_format_score(text)

    length = len(text)

    if length < 3:
        score -= 80

    if length > 10:
        score -= 40

    if confidence < 0.05:
        score -= 70
    elif confidence < 0.10:
        score -= 40
    elif confidence < 0.20:
        score -= 15
    elif confidence >= 0.50:
        score += 20

    return score


# ============================================================
# SAVE IMAGE FOR OCR
# ============================================================
#
# The main application may provide an OpenCV / NumPy image.
#
# The OCR worker receives a file path.
#
# ============================================================

def save_image_for_ocr(
    image
):

    # --------------------------------------------------------
    # Already a file path
    # --------------------------------------------------------

    if isinstance(
        image,
        (str, os.PathLike)
    ):

        image_path = os.path.abspath(
            os.fspath(image)
        )


        if not os.path.exists(
            image_path
        ):

            raise FileNotFoundError(
                f"OCR image not found: {image_path}"
            )


        return image_path, False


    # --------------------------------------------------------
    # OpenCV / NumPy image
    # --------------------------------------------------------

    try:

        import cv2

    except ImportError:

        raise RuntimeError(
            "OpenCV is required to save the image "
            "for the OCR worker."
        )


    temp_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )


    temp_path = temp_file.name

    temp_file.close()


    success = cv2.imwrite(
        temp_path,
        image
    )


    if not success:

        try:

            os.remove(
                temp_path
            )

        except OSError:

            pass


        raise RuntimeError(
            "Failed to save image for OCR worker."
        )


    return temp_path, True


# ============================================================
# RUN LOCAL PADDLEOCR WORKER
# ============================================================
#
# MAIN APPLICATION
# Python 3.14.6
#       |
#       v
# recognize_plate.py
#       |
#       | subprocess
#       v
# ocr_worker.py
#       |
#       v
# Python 3.12.10
#       |
#       v
# PaddleOCR
#
# ============================================================

# ============================================================
# IN-MEMORY PADDLEOCR ENGINE (HIGH SPEED)
# ============================================================

_IN_MEMORY_OCR = None

def get_in_memory_ocr():
    global _IN_MEMORY_OCR
    if _IN_MEMORY_OCR is None:
        try:
            os.environ["FLAGS_enable_pir_api"] = "0"
            from paddleocr import PaddleOCR
            print("\nLoading PaddleOCR engine in-memory for high speed...")
            _IN_MEMORY_OCR = PaddleOCR(lang="en", enable_mkldnn=False, show_log=False)
            print("PaddleOCR loaded in-memory successfully!")
        except Exception:
            _IN_MEMORY_OCR = False
    return _IN_MEMORY_OCR if _IN_MEMORY_OCR is not False else None


# ============================================================
# RECOGNITION-ONLY ENGINE (SPEED FIX)
# ============================================================
# The YOLO detector has ALREADY precisely located and cropped the plate.
# Calling the full PaddleOCR pipeline (text detection + orientation
# classification + recognition) on that crop re-runs text detection on a
# region we already know contains exactly one line of text. That redundant
# detection step is the dominant cost (~90s per image on CPU).
#
# PaddleOCR 3.x exposes a recognition-only class (TextRecognition) that
# skips detection/orientation entirely and just recognizes characters in the
# image. We use it for both the whole-crop and the split-half passes.
# ============================================================

_RECOG_ONLY_OCR = None

def get_recognition_only_ocr():
    global _RECOG_ONLY_OCR
    if _RECOG_ONLY_OCR is None:
        try:
            os.environ["FLAGS_enable_pir_api"] = "0"
            from paddleocr import TextRecognition
            print(
                "\nLoading recognition-only engine "
                "(TextRecognition) for speed..."
            )
            _RECOG_ONLY_OCR = TextRecognition()
            print("Recognition-only engine loaded successfully!")
        except Exception:
            _RECOG_ONLY_OCR = False
    return _RECOG_ONLY_OCR if _RECOG_ONLY_OCR is not False else None


def _run_ocr_on_image(image_path):
    """Run recognition-only OCR on an already-cropped plate image.

    Returns (combined_text, avg_confidence).
    Falls back to the full PaddleOCR pipeline if the recognition-only
    engine is unavailable.
    """
    # Primary: recognition-only engine (no redundant text detection)
    recog = get_recognition_only_ocr()
    if recog is not None:
        res = recog.predict(image_path)
        if res:
            data = res[0] if isinstance(res, list) else res
            if isinstance(data, dict):
                # TextRecognition returns singular keys
                text = data.get("rec_text", "")
                score = data.get("rec_score", 0.0)
                if text:
                    return str(text).strip(), float(score)
                return "", 0.0

    # Fallback: full PaddleOCR pipeline
    fast_ocr = get_in_memory_ocr()
    if fast_ocr is not None:
        ocr_res = fast_ocr.predict(image_path)
        if ocr_res and len(ocr_res) > 0 and ocr_res[0]:
            data = ocr_res[0]
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])

            combined_text = " ".join(
                [str(t).strip() for t in texts if str(t).strip()]
            )
            avg_conf = (
                float(sum(scores) / len(scores))
                if len(scores) > 0
                else 0.0
            )
            return combined_text, avg_conf

    return "", 0.0


# Helper for when we don't want to split (e.g., recursive calls on halves)
def _ocr_single_image_no_split(image):
    temp_file = False
    image_path = None

    try:
        # Prepare image
        image_path, temp_file = save_image_for_ocr(image)

        # Recognition-only OCR (skips redundant text detection)
        combined_text, avg_conf = _run_ocr_on_image(image_path)

        return combined_text, avg_conf

    except subprocess.TimeoutExpired:
        return "", 0.0
    except Exception as e:
        print(f"\nPaddleOCR worker error: {e}")
        return "", 0.0
    finally:
        if temp_file and image_path:
            try:
                os.remove(image_path)
            except OSError:
                pass


def ocr_single_image(
    image,
    _depth=0
):

    if image is None:
        return "", 0.0

    # Prevent infinite recursion from split logic
    if _depth > 1:
        return _ocr_single_image_no_split(image)

    temp_file = False
    image_path = None


    try:

        # ----------------------------------------------------
        # Prepare image
        # ----------------------------------------------------

        image_path, temp_file = (
            save_image_for_ocr(
                image
            )
        )

        # ----------------------------------------------------
        # Recognition-only OCR on the whole crop
        # (skips redundant text detection on an already-cropped plate)
        # ----------------------------------------------------

        combined_text, avg_conf = _run_ocr_on_image(
            image_path
        )

        return combined_text, avg_conf


    except subprocess.TimeoutExpired:

        print(
            "\nPaddleOCR worker timed out."
        )

        return "", 0.0


    except Exception as e:

        print(
            f"\nPaddleOCR worker error: {e}"
        )

        return "", 0.0


    finally:

        # ----------------------------------------------------
        # Remove temporary image
        # ----------------------------------------------------

        if temp_file and image_path:

            try:

                os.remove(
                    image_path
                )

            except OSError:

                pass


# ============================================================
# SELECT BEST CHARACTER INTERPRETATION
# ============================================================

def select_best_interpretation(
    text,
    confidence
):

    candidates = (
        generate_character_candidates(
            text
        )
    )


    if not candidates:

        return (

            clean_plate_text(
                text
            ),

            score_plate(
                text,
                confidence
            )

        )


    best_text = ""

    best_score = -999999


    for candidate in candidates:

        candidate_score = score_plate(

            candidate,

            confidence

        )


        if candidate_score > best_score:

            best_score = (
                candidate_score
            )

            best_text = (
                candidate
            )


    return (

        best_text,

        best_score

    )


# ============================================================
# MAIN OCR FUNCTION
# ============================================================

def recognize_plate(
    plate_image
):

    # ========================================================
    # GENERATE ENHANCED VERSIONS
    # ========================================================

    enhanced_images = enhance_plate(
        plate_image
    )


    candidates = []


    # ========================================================
    # ORIGINAL IMAGE - WHOLE CROP
    # ========================================================

    text, conf = ocr_single_image(
        plate_image
    )


    if text:

        corrected_text, corrected_score = (

            select_best_interpretation(

                text,

                conf

            )

        )

    else:

        corrected_text = ""

        corrected_score = -100


    candidates.append({

        "version":
            "Original",

        "text":
            corrected_text,

        "raw_text":
            text,

        "confidence":
            conf,

        "score":
            corrected_score

    })

    # ========================================================
    # SPLIT CROP (TOP/BOTTOM HALVES MERGED)
    # ========================================================
    # Always try splitting - let the scoring function decide if
    # the split result is better than the whole crop.

    h, w = plate_image.shape[:2]
    if h > 10 and w > 10:
        top_half = plate_image[0:int(h * 0.55), :]
        bottom_half = plate_image[int(h * 0.40):h, :]

        split_text, split_conf = ocr_single_image(top_half)
        split_bot_text, split_bot_conf = ocr_single_image(bottom_half)

        if split_text or split_bot_text:
            combined_text = f"{split_text} {split_bot_text}".strip()
            split_confs = [c for c in (split_conf, split_bot_conf) if c > 0]
            split_avg_conf = float(sum(split_confs) / len(split_confs)) if split_confs else 0.0

            if combined_text:
                corrected_text, corrected_score = select_best_interpretation(
                    combined_text,
                    split_avg_conf
                )
            else:
                corrected_text = ""
                corrected_score = -100

            candidates.append({
                "version": "Split",
                "text": corrected_text,
                "raw_text": combined_text,
                "confidence": split_avg_conf,
                "score": corrected_score
            })

    # ========================================================
    # EARLY EXIT (SKIP REDUNDANT ENHANCEMENT OCR PASSES)
    # ========================================================
    # Gate on the BEST confidence across all candidates collected so far
    # (Original whole-crop + Split halves), calibrated for the recognition-only
    # confidence distribution (see EARLY_EXIT_CONFIDENCE). As soon as any
    # candidate is a confident, sufficiently-long read, skip the 5 enhancement
    # passes. Requiring len >= 3 avoids a lone high-conf single char
    # (e.g. "1" @ 0.64) triggering the exit and hiding a better enhancement.
    best_so_far = max(
        candidates,
        key=lambda x: x["confidence"]
    )
    if (best_so_far["confidence"] >= EARLY_EXIT_CONFIDENCE
            and len(best_so_far["text"]) >= 3):
        print(
            f"\nEarly-exit: '{best_so_far['text']}' @ "
            f"{best_so_far['confidence']:.2f} "
            f"({best_so_far['version']}) >= {EARLY_EXIT_CONFIDENCE:.2f} "
            f"- skipping enhancement passes."
        )
        enhanced_images = {}

    # ========================================================
    # ENHANCED IMAGES
    # ========================================================

    for version, image in (

        enhanced_images.items()

    ):

        text, conf = ocr_single_image(
            image
        )


        if text:

            corrected_text, corrected_score = (

                select_best_interpretation(

                    text,

                    conf

                )

            )

        else:

            corrected_text = ""

            corrected_score = -100


        candidates.append({

            "version":
                version,

            "text":
                corrected_text,

            "raw_text":
                text,

            "confidence":
                conf,

            "score":
                corrected_score

        })


    # ========================================================
    # DISPLAY OCR COMPARISON
    # ========================================================

    print(
        "\nOCR Comparison"
    )

    print(
        "-" * 80
    )


    for candidate in candidates:

        print(

            f"{candidate['version']:20}"

            f"{candidate['text']:15}"

            f"Conf={candidate['confidence']:.2f}"

            f"  Score={candidate['score']:.2f}"

        )


    print(
        "-" * 80
    )


    # ========================================================
    # SELECT BEST OCR CANDIDATE
    # ========================================================

    best = max(

        candidates,

        key=lambda x:
        x["score"]

    )


    print(

        f"Selected OCR: "
        f"{best['text']}"

    )


    # ========================================================
    # FINAL CLEANING
    # ========================================================
    # best["text"] is already a cleaned candidate from generate_character_candidates.
    # Do NOT re-apply clean_plate_text to avoid re-tokenizing and corrupting valid plates.
    final_text = best["text"]

    final_confidence = (
        best["confidence"]
    )

    final_score = (
        best["score"]
    )


    # ========================================================
    # REJECT EMPTY OCR
    # ========================================================

    if not final_text:

        print(

            "OCR rejected: "
            "no readable characters."

        )


        return {

            "plate_text":
                "",

            "confidence":
                0.0,

            "score":
                -100

        }


    # ========================================================
    # REJECT EXTREMELY SHORT OCR
    # ========================================================

    if len(final_text) < 3:

        print(

            "OCR rejected: "
            "plate text too short."

        )


        return {

            "plate_text":
                "",

            "confidence":
                0.0,

            "score":
                -100

        }


    # ========================================================
    # REJECT VERY LOW CONFIDENCE OCR
    # ========================================================

    if final_confidence < MIN_OCR_CONFIDENCE:

        print(

            "OCR rejected: "
            "confidence too low."

        )


        return {

            "plate_text":
                "",

            "confidence":
                0.0,

            "score":
                -100

        }


    # ========================================================
    # REQUIRE ALPHANUMERIC CONTENT
    # ========================================================

    if not re.search(

        r"[A-Z0-9]",

        final_text

    ):

        print(

            "OCR rejected: "
            "no alphanumeric characters."

        )


        return {

            "plate_text":
                "",

            "confidence":
                0.0,

            "score":
                -100

        }


    # ========================================================
    # RETURN OCR OBSERVATION
    # ========================================================

    return {

        "plate_text":
            final_text,

        "confidence":
            final_confidence,

        "score":
            final_score

    }