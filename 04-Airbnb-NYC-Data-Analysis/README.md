# 🏨 NYC Airbnb Market Dynamics: Exploratory Data Analysis (EDA)

[![Data Analysis: Pandas](https://img.shields.io/badge/Data_Analysis-Pandas-150458.svg)](https://pandas.pydata.org/)
[![Visualization: Seaborn](https://img.shields.io/badge/Visualization-Seaborn-4c72b0.svg)](https://seaborn.pydata.org/)

## 📌 Business Value & Problem Statement
Understanding competitive real estate pricing patterns is critical for property investors and hosts. This project performs an intensive Exploratory Data Analysis (EDA) on the 2019 New York City Airbnb dataset to uncover pricing anomalies, geographic demand hotspots, and customer behavioral preferences across different room types.

---

## 🔍 Analytical Deep Dives

### 1. Data Cleaning & Sanitization
* Handled missing values systematically inside critical columns like `reviews_per_month` and `name` without corrupting downstream statistics.
* Detected and treated extreme pricing outliers using Interquartile Range (IQR) filtering to ensure visualization integrity.

### 2. Geospatial & Traffic Analysis
* Aggregated listing data across NYC’s 5 major boroughs (Manhattan, Brooklyn, Queens, Bronx, Staten Island) to map price density.
* Correlated `minimum_nights` against `number_of_reviews` to track tourist traffic and platform occupancy rates.

---

## 📈 Strategic Data Insights
* **Manhattan** dominates the premium tier with the highest average price per night, whereas **Brooklyn** presents the highest concentration of private rooms, serving as a budget-friendly hotspot.
* Listing traffic peaks significantly for hosts who offer flexible minimum stay durations ($\le 3$ nights).

---

## 📂 Project Structure
```bash
├── Airbnb_NYC_2019_EDA.ipynb        # Comprehensive Data Analysis Notebook
├── AB_NYC_2019.csv                  # Raw Airbnb Data
└── README.md                        # Documentation