# 🛍️ Production-Grade Customer Market Segmentation Engine

[![ML: Unsupervised](https://img.shields.io/badge/Machine_Learning-Unsupervised-brightgreen.svg)](https://scikit-learn.org/)
[![Database: SQLite](https://img.shields.io/badge/Database-SQLite3-blue.svg)](https://www.sqlite.org/)

## 📌 Business Value & Problem Statement
One-size-fits-all marketing strategies yield poor conversion rates and wasted budgets. This project leverages Unsupervised Machine Learning to segment a retail mall's customer base into distinct behavioral personas based on demographics, annual income, and spending scores—allowing marketing teams to execute hyper-targeted campaigns.

---

## 🔄 End-to-End Data Architecture
### 1. Robust ETL & Flexible Data Loading Pipeline
Unlike basic notebooks, this project features a production-ready custom loading function capable of parsing raw customer data seamlessly from multiple sources:
* Standard Flat Files (`.csv`, `.xlsx`, `.txt`)
* Semi-structured formats (`.json`)
* Relational Databases using an integrated **SQLite3** engine connection.

### 2. Mathematical Optimization
* Utilized the **Elbow Method** combined with **Within-Cluster Sum of Squares (WCSS)** mathematical curves to scientifically determine the optimal number of clusters ($K$).
* Standardized features using `StandardScaler` to ensure the K-Means distance metrics ($Euclidean\ Distance$) aren't biased by differing feature scales.

---

## 📊 Discovered Behavioral Personas
The K-Means algorithm successfully clustered customers into high-value actionable personas:
1.  💎 **High Earners, High Spenders:** Core revenue generators. (Target: Loyalty Programs & Luxury Pre-sales).
2.  🛡️ **High Earners, Low Spenders:** Conservative shoppers. (Target: Personalized discount alerts & value offers).
3.  ⚠️ **Low Earners, High Spenders:** Impulsive, high-risk buyers.

---

## 📂 Project Structure
```bash
├── Mall_Customers_Clustering.ipynb   # ETL Pipeline & Clustering Logic
├── dataset/                           # Multi-format raw customer files
└── README.md                          # Documentation