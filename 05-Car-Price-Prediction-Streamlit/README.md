# 🚗 Interactive Car Price Prediction Dashboard & Deployment

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![Framework: Scikit-Learn](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Deployment: Streamlit](https://img.shields.io/badge/Deployment-Streamlit-FF4B4B.svg)](https://streamlit.io/)

## 📌 Business Value & Problem Statement
The used car market is highly dynamic and volatile, making it difficult for dealerships and individual buyers to estimate the fair value of a vehicle. This project solves this problem by training a robust Machine Learning regressor and deploying it as a real-time interactive **Streamlit web application** for instant vehicle appraisal.

---

## 🏗️ Technical Workflow & Implementation

### 1. Feature Engineering & Exploration
* Handled complex automotive specifications including `Levy`, `Mileage`, `Cylinders`, and age tracking via production years.
* Categorical features (such as `Gear box type`, `Drive wheels`, and `Fuel type`) were systematically mapped and encoded to ensure perfect alignment between model training and real-time inference inputs.

### 2. Model Selection & Serializing
* Evaluated multiple regression architectures, including **Random Forest Regressor**, **Gradient Boosting**, and **XGBoost**.
* The best-performing model was saved using the Python `pickle` module as `Cars_Predictions.sav` for production-level inference.

### 3. Production Deployment (Streamlit)
* Built an interactive user interface (`car_Prediction.py`) that captures user inputs via drop-downs and numerical sliders.
* Integrates the serialized `.sav` model pipeline to output immediate price estimates dynamically.

---

## 📂 Project Structure
```bash
├── Car_Price_Prediction.ipynb       # EDA, Feature Engineering, and Model Training
├── car_Prediction.py                # Streamlit Web Application Source Code
├── Cars_Predictions.sav             # Serialized Trained Model (Pickle Object)
└── README.md                        # Documentation