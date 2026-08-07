# 💳 Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using an ensemble classification approach, with a focus on handling highly imbalanced data and optimizing the classification threshold for better fraud detection performance.

---

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent a very small portion of all transactions.

The main goal of this project is to build a robust fraud detection pipeline that:

* Preprocesses numerical features appropriately.
* Handles class imbalance using multiple sampling strategies.
* Trains XGBoost and Random Forest models.
* Combines the models using a Soft Voting Classifier.
* Optimizes the classification threshold based on F1-Score.
* Evaluates the model using F1-Score, Precision, Recall, PR-AUC, and ROC-AUC.
* Performs feature importance and error analysis.
* Saves the complete trained model artifact for later use.

---

## 🎯 Objectives

The project focuses on:

1. Understanding the structure and characteristics of the fraud dataset.
2. Building a reliable preprocessing pipeline.
3. Comparing different strategies for handling class imbalance.
4. Training tree-based classification models.
5. Combining models using ensemble learning.
6. Finding an optimal decision threshold instead of relying only on `0.5`.
7. Evaluating performance using metrics suitable for imbalanced classification.
8. Saving the preprocessing pipeline, trained model, and optimal threshold.

---

## 🗂️ Project Structure

```text
CreditCardFraud/
│
├── EDA_model/
│   └── cleaned_data.pkl
│
├── models/
│   └── final_model.pkl
│
├── credit_fraud_train.py
├── credit_fraud_utils_data.py
├── credit_fraud_utils_eval.py
│
├── EDA.ipynb
│
├── final_model.py
│
├── EDA.ipynb
│
├── PreProcessing_model.py
├── requirements.txt
└── README.md
```

### Files Description

| File                         | Description                                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| `EDA(2).ipynb`               | Exploratory Data Analysis and initial data preparation                                     |
| `credit_fraud_utils_data.py` | Data loading, train/test splitting, preprocessing, and resampling                          |
| `credit_fraud_utils_eval.py` | Model evaluation, threshold optimization, visualization, and error analysis                |
| `credit_fraud_train.py`      | Main training pipeline                                                                     |
| `cleaned_data.pkl`           | Cleaned dataset used by the training pipeline                                              |
| `final_model.pkl`            | Saved model artifact containing the preprocessor, voting classifier, and optimal threshold |

---

# 🔍 Exploratory Data Analysis

The EDA phase focuses on understanding the dataset before training the models.

The analysis includes:

* Dataset inspection.
* Missing-value analysis.
* Duplicate checking.
* Target distribution.
* Feature distributions.
* Transaction amount analysis.
* Time analysis.
* Correlation analysis.
* Feature engineering.

A derived `Hour` feature is used alongside the original `Time` feature.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline uses different transformations depending on the feature type.

### V1–V28 Features

```text
Median Imputation
        ↓
StandardScaler
```

### Amount

```text
Median Imputation
        ↓
Log1p Transformation
        ↓
RobustScaler
```

### Time and Hour

```text
Median Imputation
        ↓
RobustScaler
```

---

# ⚖️ Handling Class Imbalance

The project supports multiple strategies:

* No Sampling
* Random Over-Sampling (ROS)
* Random Under-Sampling (RUS)
* SMOTE
* Class Weight

---

# 🤖 Machine Learning Models

## XGBoost

The project uses XGBoost with:

```text
n_estimators = 100
max_depth = 6
learning_rate = 0.05
```

---

## Random Forest

Random Forest is tuned using `GridSearchCV`.

```python
{
    "max_depth": [6, 8, 10],
    "n_estimators": [50, 100, 150]
}
```

The model uses 3-fold cross-validation with F1-Score as the optimization metric.

---

# 🤝 Ensemble Learning

The final classifier combines:

```text
XGBoost
     +
Random Forest
     ↓
Soft Voting Classifier
```

---

# 🎚️ Threshold Optimization

Instead of relying on the default:

```text
Threshold = 0.23
```

the project searches through thresholds from `0.05` to `0.94` and selects the threshold that produces the highest F1-Score.

---

# 📊 Evaluation Metrics

The project evaluates the model using:

* Precision
* Recall
* F1-Score
* PR-AUC
* ROC-AUC

---

# 📈 Model Analysis

The project generates:

* Top 10 Feature Importances.
* ROC Curve.
* Precision / Recall / F1 vs Threshold.
* Confusion Matrix.
* False Positive / False Negative analysis.

---

# 💾 Model Saving

The final model artifact contains:

```python
{
    "preprocessor": preprocessor,
    "voting_classifier": voting_clf,
    "optimal_threshold": best_th
}
```

The artifact is saved using `joblib`.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib

---

# 👨‍💻 Author

**Mahmoud Elshaer**

Data Scientist / AI Engineer

Interested in:

* Machine Learning
* Data Science
* Natural Language Processing
* Artificial Intelligence
