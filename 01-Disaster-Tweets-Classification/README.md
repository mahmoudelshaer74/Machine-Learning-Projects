# ✉️ Real-Time Disaster Detection from Social Media Streams (NLP)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![Framework: Scikit-Learn](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![NLP: NLTK](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)

## 📌 Overview & Problem Statement
During crises, social media (specifically Twitter/X) is the fastest source of real-time information. However, human language is highly metaphorical (e.g., *"This concert is fire! 🔥"* vs *"The building is on fire! 🧯"*). 

This project builds an end-to-end Machine Learning and Natural Language Processing (NLP) pipeline to filter out semantic noise and accurately identify actual emergency signals, enabling disaster relief teams to respond faster to genuine crises.

---

## 🏗️ Technical Pipeline & Workflow

### 1. Advanced Text Preprocessing
Raw tweets are highly unstructured. The preprocessing pipeline applies the following sequential operations:
*   **Regex Cleaning:** Stripping HTML tags, URLs, Twitter handles (`@user`), emojis, and special characters.
*   **Tokenization & Stop-Words Elimination:** Isolating core words using NLTK’s English stop-words corpus.
*   **Text Normalization:** Implementing **WordNet Lemmatizer** to reduce words to their base semantic form (e.g., *running ➔ run*), preserving context better than standard stemming.

### 2. Feature Engineering
*   **Vectorization:** Converted text tokens into numerical vectors using **TF-IDF (Term Frequency - Inverse Document Frequency)**.
*   **Hyperparameter Tuning:** Tuned `ngram_range=(1, 2)` to capture both single words and two-word phrases (bi-grams) like *"wild fire"* or *"flash flood"*.

### 3. Model Architecture
Implemented and compared multiple robust classification baselines:
*   **Logistic Regression** (with L2 Regularization)
*   **Multinomial Naive Bayes** (Optimized for text counts)
*   **Linear Support Vector Machines (LinearSVC)**

---

## 📊 Evaluation & Metrics
Because missing a real disaster (**False Negative**) is much more dangerous than a false alarm (**False Positive**), the models were evaluated heavily on **Recall** and **F1-Score** alongside standard Accuracy.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial NB** | 78.4% | 0.79 | 0.74 | 0.76 |
| **Logistic Regression** | 80.1% | 0.81 | 0.78 | 0.79 |
| **Linear SVC (Best)** | **81.5%** | **0.82** | **0.80** | **0.81** |

> 💡 **Key Insight:** The Linear SVC model combined with TF-IDF Bi-grams achieved the most balanced performance, minimizing False Negatives significantly.

---

## 📂 Project Structure
```bash
├── Disaster_Detection_From_Tweets_using_ML.ipynb  # Comprehensive Jupyter Notebook
├── dataset/
│   ├── train.csv                                  # Labeled training dataset
│   └── test.csv                                   # Evaluation test set
└── README.md                                      # Project Documentation