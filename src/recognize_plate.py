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


# ============================================================
# OCR CHARACTER CORRECTION
# ============================================================

CHAR_TO_DIGIT = {

    "O": "0",
    "Q": "0",
    "D": "0",

    "I": "1",
    "L": "1",

    "Z": "2",

    "S": "5",

    "G": "6",

    "T": "7",

    "B": "8"

}


DIGIT_TO_CHAR = {

    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B"

}


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_plate_text(
    text
):

    if not text:

        return ""


    text = str(
        text
    ).upper().strip()


    text = text.replace(
        " ",
        ""
    )


    text = text.replace(
        "-",
        ""
    )


    text = text.replace(
        "_",
        ""
    )


    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )


    return text


# ============================================================
# GENERATE CHARACTER-CORRECTED CANDIDATES
# ============================================================

def generate_character_candidates(
    text
):

    text = clean_plate_text(
        text
    )


    if not text:

        return []


    candidates = set()


    # --------------------------------------------------------
    # Original OCR result
    # --------------------------------------------------------

    candidates.add(
        text
    )


    # --------------------------------------------------------
    # Letter -> digit interpretation
    # --------------------------------------------------------

    digit_version = ""


    for char in text:

        if char in CHAR_TO_DIGIT:

            digit_version += (
                CHAR_TO_DIGIT[char]
            )

        else:

            digit_version += char


    candidates.add(
        digit_version
    )


    # --------------------------------------------------------
    # Digit -> letter interpretation
    # --------------------------------------------------------

    letter_version = ""


    for char in text:

        if char in DIGIT_TO_CHAR:

            letter_version += (
                DIGIT_TO_CHAR[char]
            )

        else:

            letter_version += char


    candidates.add(
        letter_version
    )


    return list(
        candidates
    )


# ============================================================
# PLATE FORMAT SCORE
# ============================================================

def plate_format_score(
    text
):

    text = clean_plate_text(
        text
    )


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

    elif length in [4, 5, 6, 7]:

        score += 20

    elif length in [8, 9, 10]:

        score += 5

    else:

        score -= 30


    # ========================================================
    # MIXED ALPHANUMERIC
    # ========================================================

    if letters > 0 and digits > 0:

        score += 25


    # ========================================================
    # COMMON LETTER + NUMBER FORMAT
    # ========================================================

    if re.match(
        r"^[A-Z]{2,3}\d{3,4}$",
        text
    ):

        score += 80


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

    text = clean_plate_text(
        plate_text
    )


    if not text:

        return -100


    score = (
        float(confidence) * 100
    )


    score += plate_format_score(
        text
    )


    length = len(
        text
    )


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


def ocr_single_image(
    image
):

    if image is None:

        return "", 0.0


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
        # Try Fast In-Memory PaddleOCR First
        # ----------------------------------------------------

        fast_ocr = get_in_memory_ocr()
        if fast_ocr is not None:
            # Decide 2-line vs 1-line from the CROP'S SHAPE, not from what
            # the first OCR pass happened to find. This runs BEFORE we
            # trust any whole-crop result.
            is_tall_plate = (
                isinstance(image, np.ndarray)
                and image.shape[0] / max(1, image.shape[1]) > 0.35
            )

            if is_tall_plate:
                h, w = image.shape[:2]
                top_half = image[0:int(h * 0.55), :]
                bottom_half = image[int(h * 0.40):h, :]

                top_t, top_c = ocr_single_image(top_half)
                bot_t, bot_c = ocr_single_image(bottom_half)

                print(f"    [split] top='{top_t}' ({top_c:.2f})  bottom='{bot_t}' ({bot_c:.2f})")

                if top_t or bot_t:
                    combined_text = f"{top_t}{bot_t}".strip()
                    confs = [c for c in (top_c, bot_c) if c > 0]
                    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0

                    if temp_file and os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except Exception:
                            pass
                    return combined_text, avg_conf
                # if neither half read anything, fall through to whole-crop below

            ocr_res = fast_ocr.predict(image_path)
            if ocr_res and len(ocr_res) > 0 and ocr_res[0]:
                data = ocr_res[0]
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])

                combined_text = " ".join([str(t).strip() for t in texts if str(t).strip()])
                avg_conf = float(sum(scores) / len(scores)) if len(scores) > 0 else 0.0

                if temp_file and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception:
                        pass
                return combined_text, avg_conf

        # ----------------------------------------------------
        # Fallback to Subprocess if not in-memory
        # ----------------------------------------------------

        if not os.path.exists(
            OCR_WORKER
        ):

            print(
                "\nPaddleOCR worker error:"
            )

            print(
                f"ocr_worker.py not found:\n"
                f"{OCR_WORKER}"
            )

            return "", 0.0


        # ----------------------------------------------------
        # Check Python 3.12 environment
        # ----------------------------------------------------

        if not os.path.exists(
            OCR_PYTHON
        ):

            print(
                "\nPaddleOCR worker error:"
            )

            print(
                "Python 3.12 environment not found:"
            )

            print(
                OCR_PYTHON
            )

            return "", 0.0


        # ----------------------------------------------------
        # Launch OCR worker
        # ----------------------------------------------------

        process = subprocess.run(

            [

                OCR_PYTHON,

                "-u",

                OCR_WORKER,

                image_path

            ],

            capture_output=True,

            text=True,

            timeout=OCR_TIMEOUT,

            cwd=PROJECT_ROOT

        )


        # ----------------------------------------------------
        # Worker failed
        # ----------------------------------------------------

        if process.returncode != 0:

            print(
                "\nPaddleOCR worker failed."
            )


            if process.stdout:

                print(
                    "\nWorker stdout:"
                )

                print(
                    process.stdout
                )


            if process.stderr:

                print(
                    "\nWorker stderr:"
                )

                print(
                    process.stderr
                )


            return "", 0.0


        # ----------------------------------------------------
        # Get worker output
        # ----------------------------------------------------

        output = (
            process.stdout.strip()
        )


        if not output:

            print(
                "\nPaddleOCR worker returned no output."
            )

            return "", 0.0


        # ----------------------------------------------------
        # Find JSON result
        # ----------------------------------------------------
        #
        # PaddleOCR may produce logs such as:
        #
        # Creating model...
        # Model files already exist...
        #
        # We therefore search from the bottom for the JSON
        # result produced by ocr_worker.py.
        #
        # ----------------------------------------------------

        result = None


        for line in reversed(
            output.splitlines()
        ):

            line = line.strip()


            if not line:

                continue


            if not (
                line.startswith("[")
                or
                line.startswith("{")
            ):

                continue


            try:

                result = json.loads(
                    line
                )

                break


            except json.JSONDecodeError:

                continue


        # ----------------------------------------------------
        # JSON not found
        # ----------------------------------------------------

        if result is None:

            print(
                "\nCould not parse OCR worker output."
            )

            print(
                "\nWorker output:"
            )

            print(
                output
            )

            return "", 0.0


        # ----------------------------------------------------
        # Extract first OCR result
        # ----------------------------------------------------

        if isinstance(
            result,
            list
        ):

            if not result:

                return "", 0.0


            first_result = result[0]


        elif isinstance(
            result,
            dict
        ):

            first_result = result


        else:

            return "", 0.0


        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text = first_result.get(
            "text",
            ""
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = first_result.get(
            "confidence",
            0.0
        )


        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError
        ):

            confidence = 0.0


        text = clean_plate_text(
            text
        )


        return (
            text,
            confidence
        )


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
    # ORIGINAL IMAGE
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
    # EARLY EXIT (SKIP REDUNDANT ENHANCEMENT OCR PASSES)
    # ========================================================

    if text and conf >= 0.70 and corrected_text:
        # High confidence reading on original image - skip 5 extra enhancement passes!
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

    final_text = clean_plate_text(

        best["text"]

    )


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