"""
Benchmark harness for the OCR step in the Vehicle Authorization System.

Measures ONLY the OCR portion (recognize_plate), isolating it from YOLO
detection so we can compare the full-pipeline OCR vs a recognition-only OCR.

Run under the paddleocr_env Python 3.12:
    .\paddleocr_env\Scripts\python.exe bench_ocr.py

Usage:
    python bench_ocr.py <image1> [image2 ...] [--mode full|recog]
"""

import os
import sys
import time
import json
import argparse

# Ensure project modules are importable
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)
os.chdir(HERE)

from detect_and_crop_plate import detect_and_crop
from recognize_plate import recognize_plate, ocr_single_image

# For the recognition-only experiment we swap the in-memory OCR engine to a
# TextRecognition (recognition-only) instance. We patch recognize_plate's
# get_in_memory_ocr to return it.

RECOG_OCR = None


def get_recog_only_ocr():
    global RECOG_OCR
    if RECOG_OCR is None:
        try:
            os.environ["FLAGS_enable_pir_api"] = "0"
            from paddleocr import TextRecognition
            print("\nLoading recognition-ONLY engine (TextRecognition)...")
            RECOG_OCR = TextRecognition()
            print("Recognition-only engine loaded.")
        except Exception as e:
            print(f"Failed to load recognition-only engine: {e}")
            RECOG_OCR = False
    return RECOG_OCR if RECOG_OCR is not False else None


def ocr_single_image_recog_only(image):
    """Recognition-only path: skip text detection + orientation entirely."""
    temp_file = False
    image_path = None
    try:
        image_path, temp_file = recognize_plate.save_image_for_ocr(image) if False else _save(image)
        recog = get_recog_only_ocr()
        if recog is None:
            return "", 0.0
        res = recog.predict(image_path)
        if res:
            data = res[0] if isinstance(res, list) else res
            texts = data.get("rec_texts", []) if isinstance(data, dict) else []
            scores = data.get("rec_scores", []) if isinstance(data, dict) else []
            combined = " ".join(str(t).strip() for t in texts if str(t).strip())
            avg = float(sum(scores) / len(scores)) if scores else 0.0
            return combined, avg
        return "", 0.0
    except Exception as e:
        print(f"recog-only error: {e}")
        return "", 0.0
    finally:
        if temp_file and image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass


def _save(image):
    import cv2
    import tempfile
    if isinstance(image, (str, os.PathLike)):
        p = os.path.abspath(os.fspath(image))
        return p, False
    tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tp = tf.name
    tf.close()
    cv2.imwrite(tp, image)
    return tp, True


def benchmark_image(image_path, mode):
    # 1) YOLO crop (NOT timed as OCR; it's detection)
    crops = detect_and_crop(image_path, save_output=False)
    if not crops:
        return None

    total_ocr = 0.0
    results = []
    for crop in crops:
        plate_img = crop["image"] if isinstance(crop, dict) else crop
        if mode == "full":
            t0 = time.perf_counter()
            out = recognize_plate(plate_img)
            dt = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            # emulate recognize_plate but with recognition-only OCR
            out = recognize_plate_recog_only(plate_img)
            dt = time.perf_counter() - t0
        total_ocr += dt
        results.append((out.get("plate_text", ""), out.get("confidence", 0.0), dt))
    return results, total_ocr


def recognize_plate_recog_only(plate_image):
    """Minimal recognize using recognition-only OCR for every candidate."""
    enhanced = {}
    candidates = []

    text, conf = ocr_single_image_recog_only(plate_image)
    if text:
        from recognize_plate import select_best_interpretation
        ct, cs = select_best_interpretation(text, conf)
    else:
        ct, cs = "", -100
    candidates.append({"version": "Original", "text": ct, "raw_text": text,
                       "confidence": conf, "score": cs})

    # split pass
    h, w = plate_image.shape[:2]
    if h > 10 and w > 10:
        top = plate_image[0:int(h * 0.55), :]
        bot = plate_image[int(h * 0.40):h, :]
        st, sc = ocr_single_image_recog_only(top)
        sb, sbc = ocr_single_image_recog_only(bot)
        if st or sb:
            combined = f"{st} {sb}".strip()
            confs = [c for c in (sc, sbc) if c > 0]
            ac = float(sum(confs) / len(confs)) if confs else 0.0
            from recognize_plate import select_best_interpretation
            ct, cs = select_best_interpretation(combined, ac) if combined else ("", -100)
            candidates.append({"version": "Split", "text": ct, "raw_text": combined,
                               "confidence": ac, "score": cs})

    best = max(candidates, key=lambda x: x["score"])
    final = best["text"]
    if not final or len(final) < 3:
        return {"plate_text": "", "confidence": 0.0, "score": -100}
    return {"plate_text": final, "confidence": best["confidence"], "score": best["score"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--mode", choices=["full", "recog"], default="full")
    args = ap.parse_args()

    print(f"\n=== Benchmark mode: {args.mode} ===")
    grand_total = 0.0
    for img in args.images:
        if not os.path.exists(img):
            print(f"SKIP (missing): {img}")
            continue
        r = benchmark_image(img, args.mode)
        if r is None:
            print(f"NO PLATE DETECTED: {img}")
            continue
        results, total = r
        grand_total += total
        print(f"\nImage: {os.path.basename(img)}  OCR time: {total*1000:.1f} ms")
        for text, conf, dt in results:
            print(f"   plate={text!r:20} conf={conf:.2f}  t={dt*1000:.1f}ms")

    print(f"\n=== TOTAL OCR time ({args.mode}): {grand_total*1000:.1f} ms ===")


if __name__ == "__main__":
    main()
