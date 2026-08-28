"""
Follow-up benchmark for the recognition-only OCR refactor.

  Follow-up 3 (cold end-to-end) is measured FIRST in a fresh process so the
  model-load cost is included exactly once. Then engines are hot and we
  measure Follow-up 1 (per-image OCR time) and Follow-up 2 (DSC_1010 Original
  confidence under recog-only vs full pipeline).

Run under paddleocr_env Python 3.12:
    .\paddleocr_env\Scripts\python.exe bench_followups.py
"""

import os
import sys
import time
import tempfile

HERE = os.getcwd()
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)
os.chdir(HERE)

import cv2
from ultralytics import YOLO
from detect_and_crop_plate import detect_and_crop
from recognize_plate import (
    recognize_plate,
    get_in_memory_ocr,
    get_recognition_only_ocr,
)

MODEL_PATH = os.path.join(HERE, "models", "trained", "rbflw_y8_best.pt")

IMAGES = [
    "img/input/Cars/DSC_1027.jpg",
    "img/input/Cars/DSC_1039x.jpg",
    "img/input/Cars/DSC_1066.JPG",
    "img/input/Cars/DSC_1097.JPG",
    "img/input/Cars/DSC_1010.JPG",
]

TARGET = "img/input/Cars/DSC_1010.JPG"


# ============================================================
# FOLLOW-UP 3: COLD END-TO-END (models load this run)
# ============================================================

print("=" * 70)
print("FOLLOW-UP 3: COLD end-to-end (one image, full startup)")
print("=" * 70)

t_start = time.perf_counter()

# main.py loads plate_model at module top (used for video/webcam tracking;
# unused in image mode, but it IS paid at startup, so we include it).
t0 = time.perf_counter()
plate_model = YOLO(MODEL_PATH)
t_plate_model = time.perf_counter() - t0

t0 = time.perf_counter()
image = cv2.imread(TARGET)
t_imgload = time.perf_counter() - t0

# First detect_and_crop call loads the internal YOLO detector.
t0 = time.perf_counter()
crops = detect_and_crop(image, save_output=True)
t_detect = time.perf_counter() - t0

# First recognize_plate call loads the recognition-only engine.
t0 = time.perf_counter()
for c in crops:
    out = recognize_plate(c["image"] if isinstance(c, dict) else c)
t_ocr = time.perf_counter() - t0

t_total = time.perf_counter() - t_start

print(f"  plate_model load : {t_plate_model*1000:8.0f} ms  (NOTE: unused in image mode)")
print(f"  image load       : {t_imgload*1000:8.0f} ms")
print(f"  YOLO detect+crop : {t_detect*1000:8.0f} ms  (incl. detector load)")
print(f"  OCR (all cand.)  : {t_ocr*1000:8.0f} ms  (incl. recog engine load)")
print(f"  -------------------------------------------")
print(f"  TOTAL COLD       : {t_total*1000:8.0f} ms  ({t_total:.2f} s)")
print(f"  result           : {out.get('plate_text','')!r} conf={out.get('confidence',0):.2f}")

# Steady-state: 2nd image, all models hot
print("\n  --- steady-state (2nd image, models hot) ---")
t0 = time.perf_counter()
crops2 = detect_and_crop(TARGET, save_output=False)
for c in crops2:
    recognize_plate(c["image"] if isinstance(c, dict) else c)
t_hot = time.perf_counter() - t0
print(f"  2nd image total  : {t_hot*1000:8.0f} ms  ({t_hot:.2f} s)")


# ============================================================
# FOLLOW-UP 1: per-image OCR-step time + result (engines hot)
# ============================================================

print("\n" + "=" * 70)
print("FOLLOW-UP 1: OCR-step time + result (5 images, engines hot)")
print("=" * 70)

for p in IMAGES:
    crops = detect_and_crop(p, save_output=False)
    if not crops:
        print(f"  {os.path.basename(p):20} NO PLATE DETECTED")
        continue
    t0 = time.perf_counter()
    res = None
    for c in crops:
        res = recognize_plate(c["image"] if isinstance(c, dict) else c)
    dt = (time.perf_counter() - t0) * 1000
    print(
        f"  {os.path.basename(p):20} "
        f"OCR={dt:8.0f}ms  "
        f"plate={res.get('plate_text',''):15} "
        f"conf={res.get('confidence',0):.2f}"
    )


# ============================================================
# FOLLOW-UP 2: DSC_1010 Original confidence recog-only vs full
# ============================================================

print("\n" + "=" * 70)
print("FOLLOW-UP 2: DSC_1010 Original candidate confidence")
print("            recognition-only vs full pipeline")
print("=" * 70)

crop = detect_and_crop(TARGET, save_output=False)[0]["image"]


_FULL_OCR = None


def get_full_ocr():
    global _FULL_OCR
    if _FULL_OCR is None:
        os.environ["FLAGS_enable_pir_api"] = "0"
        from paddleocr import PaddleOCR
        _FULL_OCR = PaddleOCR(lang="en", enable_mkldnn=False)
    return _FULL_OCR


def original_conf(crop, kind):
    tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tp = tf.name
    tf.close()
    cv2.imwrite(tp, crop)
    try:
        if kind == "recog":
            r = get_recognition_only_ocr().predict(tp)
            d = r[0] if isinstance(r, list) else r
            t = str(d.get("rec_text", ""))
            c = float(d.get("rec_score", 0.0))
        else:
            r = get_full_ocr().predict(tp)
            d = r[0]
            ts = d.get("rec_texts", [])
            sc = d.get("rec_scores", [])
            t = " ".join(str(x).strip() for x in ts if str(x).strip())
            c = float(sum(sc) / len(sc)) if sc else 0.0
    finally:
        try:
            os.remove(tp)
        except OSError:
            pass
    return t, c


rt, rc = original_conf(crop, "recog")
ft, fc = original_conf(crop, "full")

print(f"  recognition-only : text={rt!r:20} conf={rc:.3f}  "
      f"early-exit@0.70: {'YES' if rc >= 0.70 else 'NO'}")
print(f"  full pipeline    : text={ft!r:20} conf={fc:.3f}  "
      f"early-exit@0.70: {'YES' if fc >= 0.70 else 'NO'}")

delta = rc - fc
print(f"\n  confidence delta (recog-only - full) = {delta:+.3f}")
if abs(delta) > 0.10:
    print("  => CONCLUSION: recognition-only reports materially different "
          "confidence.")
    print("     The 0.70 early-exit threshold is calibrated for the full "
          "pipeline and")
    print("     should be RECALIBRATED for the recognition-only method "
          "(lower it).")
else:
    print("  => CONCLUSION: confidence distributions are close; threshold "
          "likely OK as-is.")
