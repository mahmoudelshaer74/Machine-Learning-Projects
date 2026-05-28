# 🪪 Automated Egyptian National-ID Data Extraction Pipeline (OCR)

[![Computer Vision: OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-blueviolet.svg)](https://opencv.org/)
[![OCR: Tesseract](https://img.shields.io/badge/OCR-Tesseract_Arabic-red.svg)](https://github.com/tesseract-ocr/tesseract)

## 📌 Business Value & Problem Statement
Manual data entry of identity documents is slow, costly, and prone to human error. This project automates the extraction of structured textual data (Names, Addresses) from images of Egyptian National IDs, drastically reducing KYC (Know Your Customer) onboarding times for fintech and banking applications.

---

## ⚙️ Core Engineering Pipeline

### 1. Advanced Computer Vision Preprocessing (OpenCV)
To maximize OCR accuracy, raw ID images go through a destructive preprocessing pipeline:
*   **Bilateral Filtering:** Reduces high-frequency background noise while preserving sharp text edges.
*   **Adaptive Thresholding:** Binarizes the image (Black & White) to separate text from background security watermarks under varying lighting conditions.
*   **Contour Analysis & Deskewing:** Detects document boundaries to apply perspective transformation, auto-rotating the ID to a perfect horizontal level.

### 2. Arabic OCR Engine Configuration
*   Utilizes **Tesseract OCR** engine with specialized Arabic language packs (`ara`).
*   Configured using custom Page Segmentation Modes (PSM) like `--psm 6` (Assume a single uniform block of text) or `--psm 11` (Sparse text localizer) to extract coordinates accurately.

### 3. Text Post-Processing & Structuring
*   Raw OCR outputs are messy. The pipeline uses **Regular Expressions (Regex)** to extract targeted data fields like Full Name, Birth Date, and Address.
*   The final output is dynamically structured into a clean, exportable **Pandas DataFrame**.

---

## 📂 Project Structure
```bash
├── Egyptian_National_ID_OCR.ipynb   # Main Computer Vision Pipeline
├── sample_images/                   # Testing templates for OCR
└── README.md                        # Documentation