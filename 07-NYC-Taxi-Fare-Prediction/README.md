# 🚖 NYC Taxi Trip Duration Prediction

A complete Machine Learning project for predicting New York City taxi trip duration using feature engineering, time-aware validation, and ensemble tree-based regression models.

---

## 📌 Project Overview

The objective of this project is to predict the duration of NYC taxi trips based only on information available before the trip starts.

The project includes:

- Data preprocessing
- Feature engineering
- Exploratory Data Analysis (EDA)
- Model training and comparison
- Interactive Streamlit web application
- Saved production-ready model

---

## 📂 Project Structure

```
.
│
├── processing.py          # Data loading, cleaning, feature engineering
├── eda.py                 # Exploratory Data Analysis
├── train_model.py         # Train and compare ML models
├── streamlit_app.py       # Interactive prediction app
├── model.joblib           # Best trained model
├── train.csv              # Dataset (not included)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

NYC Taxi Trip Duration Dataset

Target Variable:

```
trip_duration
```

Input features include:

- Vendor ID
- Passenger Count
- Pickup Time
- Pickup Coordinates
- Dropoff Coordinates
- Store & Forward Flag

---

## ⚙️ Feature Engineering

The project creates several useful features including:

- Haversine Distance
- Manhattan Distance
- Bearing
- Hour of Day
- Weekday
- Rush Hour Indicator
- Weekend Indicator
- Night Indicator
- Cyclical Hour Encoding
- Latitude & Longitude Differences
- Center Coordinates
- Interaction Features

---

## 🧹 Data Cleaning

The pipeline removes:

- Impossible trip durations
- Invalid GPS coordinates
- Unrealistic taxi speeds
- Invalid passenger counts
- Extremely short or long trips

---

## 🤖 Machine Learning Models

The project compares multiple regression models:

- HistGradientBoostingRegressor
- LightGBM (optional)
- RandomForestRegressor

The best model is automatically selected and saved.

---

## 📈 Model Evaluation

Evaluation Metrics:

- R² Score
- RMSE
- MAE

Training uses:

- TimeSeriesSplit Cross Validation
- RandomizedSearchCV

---

## 📊 Exploratory Data Analysis

The EDA script generates:

- Target Distribution
- Log Target Distribution
- Correlation Matrix
- Pickup Density Map
- Vendor Statistics
- Passenger Statistics
- Hourly Patterns
- Weekday Patterns
- Rush Hour Analysis

Figures are automatically saved inside:

```
eda_figures/
```

---

## 🌐 Streamlit Application

The application provides two tabs:

### 📊 EDA

Interactive visualizations generated directly from the dataset.

### 🔮 Prediction

Predict trip duration by entering:

- Pickup datetime
- Pickup location
- Dropoff location
- Passenger count
- Vendor
- Store & Forward flag

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/NYC-Taxi-Duration-Prediction.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run EDA

```bash
python eda.py
```

---

## ▶️ Train Model

```bash
python train_model.py
```

---

## ▶️ Run Streamlit

```bash
streamlit run streamlit_app.py
```

---

## 📦 Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- LightGBM (optional)

---

## 📌 Future Improvements

- XGBoost Support
- CatBoost Support
- Hyperparameter Optimization using Optuna
- Feature Importance Dashboard
- Docker Deployment
- Cloud Deployment

---
## streamlit
https://mahmoudelshaer74-hjfpnkjuidgtnimfryjxyp.streamlit.app/

## 👨‍💻 Author

Mahmoud Elshaer

Artificial Intelligence Engineer
