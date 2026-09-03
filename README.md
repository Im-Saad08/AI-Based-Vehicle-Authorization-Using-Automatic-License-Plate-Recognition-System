# SENTRYX - An AI-Based Vehicle Authorization Using Automatic License Plate Recognition System

An optimized AI Vehicle Authorization System for real-time license plate detection, text recognition, region filtering, character normalization, database verification, and access logging. Supports single image, recorded video, and live webcam input.

## Executive Summary & Key Upgrades

This system uses a single-stage direct license plate detection pipeline with in-memory OCR, tuned for CPU-only hardware (developed and tested on a dual-core Intel laptop).

### Key Design Decisions

- **Single-Stage YOLOv8 License Plate Detection:** Directly detects license plates from frames, bypassing full vehicle-body detection.
- **In-Memory PaddleOCR (Recognition-Only Mode):** Uses PaddleOCR's `TextRecognition` class, which skips redundant text-detection/orientation steps since YOLO has already precisely located the plate. This gave a measured ~3.5x speedup over the full PaddleOCR pipeline.
- **Early-Exit Optimization:** Skips redundant enhancement-pass OCR calls once a confident read (≥0.50, recalibrated for recognition-only's confidence distribution) is found from the whole-crop or split-crop candidate.
- **15% Bounding Box Padding:** Adds outer padding around plate crops to avoid clipping characters, without over-padding (which was found to cause false 2-line detections).
- **2-Line (Stacked) Plate Handling:** For every plate crop, the system always computes BOTH a whole-crop OCR candidate and a top/bottom split candidate, and a scoring function picks the better result. (An earlier aspect-ratio-threshold approach to decide "is this 2-line?" was tested and found unreliable — two real test plates showed a 2-line plate with a *lower* aspect ratio than a single-line plate — so this was replaced with always computing both.)
- **Position-Aware Character Correction:** Corrects OCR misreads (e.g. O/Q→0, B→8, S→5, Z→2) based on whether a character sits in a letter-segment or digit-segment of the plate, not by guessing from the whole token.
- **Region-Label Filtering:** Removes region text (PUNJAB, ISLAMABAD, ICT, SINDH, KPK, BALOCHISTAN, etc.) via substring matching, so OCR-joined tokens like "ICTISLAMABAD" are still correctly filtered.
- **Per-Track OCR Gating (video/webcam):** Uses ByteTrack to assign a persistent ID per vehicle. Each vehicle is OCR'd a maximum of 3 attempts, or finalized immediately on a high-confidence single read (≥0.85) — not OCR'd on every frame.
- **Frame Skipping:** Detection runs on every Nth frame (video: every 3rd; webcam: every 10th, tuned for 2-core CPU) rather than every frame, since consecutive frames are near-identical.
- **Threaded OCR (webcam mode):** OCR runs on a background thread so the live camera display doesn't freeze while a plate is being read.

## System Architecture & Workflow

```
Input (image / video / webcam)
        ↓
YOLO plate detection (rbflw_y8_best.pt)
        ↓
[video/webcam only] ByteTrack vehicle tracking + frame skip + per-track OCR gating
        ↓
Plate crop (15% padding)
        ↓
PaddleOCR recognition-only (whole-crop + split candidates, early-exit on high confidence)
        ↓
Normalization (character correction, region filtering, merge to single string)
        ↓
Scoring (best candidate selected)
        ↓
Authorization check (vehicles.csv) → Logging (entry_log.csv)
```

### Module Breakdown

**1. Plate Detection (`src/detect_and_crop_plate.py`)**
Loads `rbflw_y8_best.pt` to detect license plate bounding boxes. Enforces minimum size thresholds (`MIN_PLATE_WIDTH = 50px`, `MIN_PLATE_HEIGHT = 15px`) so distant/unreadable plates are skipped rather than wasting an OCR call. Adds 15% padding before cropping.

**2. OCR (`src/recognize_plate.py`)**
Loads PaddleOCR's `TextRecognition` (recognition-only) engine once, in-memory, at startup. For each plate, computes a whole-crop candidate and a split (top/bottom half) candidate. Applies early-exit to skip enhancement-pass OCR calls when a confident result is already found.

**3. Text Normalization & Region Filtering (`src/normalize_plate.py`)**
Converts raw OCR text into a clean, single merged plate string: removes region words (including OCR-joined variants), applies position-aware character correction, strips spaces/dashes/underscores.

**4. Vehicle Authorization & Logging (`src/authorize_vehicle.py`, `src/logger.py`)**
Matches the normalized plate against `data/vehicles.csv`, determines `AUTHORIZED`/`UNAUTHORIZED` status, and logs every entry attempt (authorized or not) to `data/entry_log.csv` with date, time, image/frame reference, plate number, confidence, and status.

**5. Vehicle Registration (`src/register_vehicle.py`)**
Registers new authorized vehicles using the same OCR pipeline as live detection. User enters the plate as one continuous string, no spaces or dashes (e.g. a two-line plate showing "LE·15" / "1051" should be entered as `LE151051`).

## Performance (Measured, CPU-only)

| Metric | Result |
|---|---|
| OCR latency (full PaddleOCR pipeline, pre-optimization) | ~90 seconds/image |
| OCR latency (recognition-only mode) | ~19-23 seconds/image |
| OCR latency (recognition-only + early-exit) | ~7-10 seconds/image (hot) |
| Video: frames processed vs. total (frame-skip=3) | 107 of 321 frames (67% reduction) |
| Video: per-track OCR gating | Max 3 OCR attempts per vehicle, not per frame |
| Webcam | Live display with threaded OCR; frame-skip=10 tuned for 2-core CPU |

**Known limitation:** sub-1-second processing is not achievable on CPU-only hardware — each OCR call is inherently ~2-3 seconds on this hardware; a GPU would be required to reach sub-second latency.

## Hardware Comparison (Confirmed)

| Hardware | Webcam Result |
|---|---|
| 2-core/4-thread laptop (dev machine) | OCR-burst CPU contention causes 5-12s display lag; root cause confirmed via timing diagnostics, not a code defect |
| University 6-core PC + A4Tech PK-925H 1080p webcam | No lag observed over 20+ min continuous test; ~98.5% accuracy (informal observation) |

This confirms the webcam lag on the primary dev laptop is a hardware CPU-core limitation, not an architectural or code issue — the same code runs smoothly on adequate hardware.

## How to Run

**1. Activate environment**
```
.\paddleocr_env\Scripts\Activate.ps1
```

**2. Run the pipeline**
```
python src/main.py
```

**3. Switch input mode**
Set `INPUT_MODE` near the top of `src/main.py`:
- `"image"` — single photo
- `"video"` — recorded video file, with vehicle tracking
- `"webcam"` — live camera feed, with vehicle tracking

## Author & Contact

**Author:** Muhammad Saad
**Role:** Computer Engineering Student
**Institution:** National University of Technology (NUTECH), Islamabad, Pakistan
**Project:** SENTRYX — AI-Based Vehicle Authorization Using Automatic License Plate Recognition System
