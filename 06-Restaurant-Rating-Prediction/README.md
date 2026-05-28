# 🍔 Predictive Analytics for Restaurant Ratings & Consumer Preferences

[![Data Visualization: Plotly](https://img.shields.io/badge/Visualization-Plotly_Express-00M0A0.svg)](https://plotly.com/)
[![ML: Regression](https://img.shields.io/badge/Machine_Learning-Regression-blue.svg)](https://scikit-learn.org/)

## 📌 Business Value & Problem Statement
In the food and hospitality industry, customer ratings directly dictate restaurant visibility and foot traffic. This project leverages the Zomato global restaurants dataset to predict a restaurant's **Aggregate Rating** based on business attributes (e.g., location, cuisines, average cost, online delivery availability, and price range).

---

## ⚙️ Core Engineering Pipeline

### 1. Interactive EDA & Spatial Analytics
* Conducted advanced data visualizations using **Plotly Express** and **Seaborn** to map global restaurant densities, price distributions, and rating distributions.
* Cleaned messy geospatial points (`Longitude` and `Latitude`) alongside text-heavy `Cuisines` strings.

### 2. Standardized Scaling & Grid Search Optimization
* Built a reliable pipeline utilizing `StandardScaler` to process numerical metrics symmetrically.
* Implemented **Decision Tree Regressor** and **Random Forest Regressor**, optimizing hyperparameters dynamically using `GridSearchCV`.

### 3. Production-Ready Exporters
To support microservices and deployment, both the analytical components were saved independently:
* `mlmodel.pkl` ➔ Houses the final trained, cross-validated Decision Tree structure.
* `scaler.pkl` ➔ Holds the statistical boundaries (mean, variance) of training data to prevent data leakage during scaling.

---

## 📂 Project Structure
```bash
├── Predicting_Restaurant_Ratings.ipynb   # Complete End-to-End Pipeline
├── resturant_data.csv                    # Raw Zomato Global Dataset
├── mlmodel.pkl                           # Serialized Tuned Model File
├── scaler.pkl                            # Saved Preprocessing Scaler Object
└── README.md                             # Documentation