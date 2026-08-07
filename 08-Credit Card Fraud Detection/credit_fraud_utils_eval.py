import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    precision_recall_curve, average_precision_score, confusion_matrix,
    roc_curve, auc, roc_auc_score
)

def evaluate_thresholds(y_test, y_probs):
    best_th, best_f1, best_precision, best_recall = 0.5, 0.0, 0.0, 0.0
    
    for th in np.arange(0.05, 0.95, 0.01):
        y_pred_th = (y_probs >= th).astype(int)
        score = f1_score(y_test, y_pred_th, zero_division=0)
        
        if score > best_f1:
            best_f1 = score
            best_th = th
            best_precision = precision_score(y_test, y_pred_th, zero_division=0)
            best_recall = recall_score(y_test, y_pred_th, zero_division=0)
            
    pr_auc = average_precision_score(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    return best_th, best_f1, best_precision, best_recall, pr_auc, roc_auc

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    top_10 = feature_imp_df.head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=top_10, hue='Feature', palette='viridis', legend=False)
    plt.title('Top 10 Feature Importances (Random Forest)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_test, y_probs):
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_f1(y_test, y_probs, best_th):
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], label="Precision", color="blue", lw=2)
    plt.plot(thresholds, recall[:-1], label="Recall", color="orange", lw=2)
    plt.plot(thresholds, f1_scores, label="F1-Score", color="green", lw=2)
    plt.axvline(x=best_th, color='red', linestyle='--', label=f'Optimal Threshold: {best_th}')
    plt.title('Precision, Recall & F1-Score vs Threshold', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_error_analysis(y_test, y_probs, best_th):
    y_pred = (y_probs >= best_th).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    print("\n" + "="*40)
    print(" ERROR ANALYSIS REPORT")
    print("="*40)
    print(f" False Positives (FP) : {FP}")
    print(f" False Negatives (FN) : {FN}")
    print(f" True Positives  (TP) : {TP}")
    print(f" True Negatives  (TN) : {TN}")
    print("="*40)