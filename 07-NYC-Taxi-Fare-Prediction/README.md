# 🚖 New York City Taxi Fare Optimization & Pipeline Architecture

[![Framework: Scikit-Learn](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Data Processing: Pandas](https://img.shields.io/badge/Data_Processing-Pandas-150458.svg)](https://pandas.pydata.org/)

## 📌 Business Value & Problem Statement
Ride-hailing and transit applications rely heavily on upfront, reliable fare estimations to maintain customer trust. This project builds a regression pipeline to estimate NYC taxi ride fares by parsing raw spatial indicators (pickup/dropoff positions) and temporal metrics.

---

## 🔄 Automated Architectural Components

### 1. Robust Temporal Preprocessing (`load_and_prepare_data`)
Instead of manual feature manipulation, a structured ingestion function was built to:
* Parse raw timestamp strings into datetime objects automatically.
* Extract isolated temporal features (`hour`, `day`, `month`, `year`, `dayofweek`) to capture high-traffic commute intervals and surge pricing windows.

### 2. Feature Engineering & Vector Scaling
* Handled geographic distance mapping across specific latitude and longitude coordinates.
* Split data systematically ($80/20$) and embedded standard scaling to isolate model calculations and secure the evaluation perimeter against data leaks.

### 3. Model Performance Baseline
* Trained a production-grade **Linear Regression** model to quantify base thresholds.
* Computed **Root Mean Squared Error (RMSE)** and **$R^2$ Score** metrics to mathematically verify performance criteria.

---

## 📂 Project Structure
```bash
├── NYC.ipynb          # Scaled Pipeline Architecture & Model Training Code
├── train.csv          # Raw NYC Ride-Hailing Train Records
└── README.md          # Documentation
