# 🚗 AI-Based Vehicle Authorization Using Automatic License Plate Recognition System

An AI-powered Vehicle Authorization System that automatically detects vehicle license plates, recognizes the plate number using OCR, verifies authorization against a registered database, and logs every vehicle entry.

---

## 📌 Features

- License Plate Detection using YOLOv8
- License Plate Recognition using EasyOCR
- Automatic Vehicle Authorization
- Registration Year Extraction
- Plate Number Normalization
- Automatic Entry Logging
- Image Enhancement for OCR Retry
- Multiple Image Processing
- Authorized / Unauthorized Vehicle Classification

---

## 🛠 Technologies Used

- Python
- YOLOv8
- EasyOCR
- OpenCV
- Pandas
- NumPy

---

## 📂 Project Structure

```
AI-Based-Vehicle-Authorization/
│
├── data/
│   └── vehicles.csv
│
├── models/
│   └── pre_trained/
│
├── src/
│   ├── main.py
│   ├── detect_and_crop_plate.py
│   ├── recognize_plate.py
│   ├── authorize_vehicle.py
│   ├── normalize_plate.py
│   ├── enhance_plate.py
│   ├── register_vehicle.py
│   └── logger.py
│
└── README.md
```

---

## ⚙ Workflow

1. Detect vehicle license plate.
2. Crop detected plate.
3. Perform OCR.
4. Normalize detected text.
5. Extract registration year.
6. Compare against authorized database.
7. Retry OCR using enhanced image if unauthorized.
8. Display authorization result.
9. Save entry log.

---

## 📷 Example Output

```
Detected Plate : MNA 5445

Registration Year : 08

Status : AUTHORIZED

Owner : Muhammad Saad

Department : Computer Engineering

Vehicle Type : Car
```

---

## Future Improvements

- Live CCTV/IP Camera Integration
- Streamlit Dashboard
- OCR Confidence Visualization
- Automatic Database Registration
- Multiple OCR Engine Support
- GUI Application
- Deep Learning-based OCR
- Real-time Vehicle Monitoring

---

## Author

Muhammad Saad

Computer Engineering Student

National University of Technology (NUTECH)

Islamabad, Pakistan