import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    precision_recall_curve, average_precision_score, confusion_matrix,
    roc_curve, auc, roc_auc_score
)
import joblib

# =========================================================
# =========================================================
df = pd.read_pickle("EDA_model/cleaned_data.pkl")

x = df.drop(columns=["Class"], axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

v_features = [f"V{i}" for i in range(1, 29)]

pca_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("Pca_scaler", StandardScaler()),
])

amount_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log_transform", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("robust_scaler", RobustScaler())
])

time_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('robust_scaler', RobustScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("Pca_features", pca_pipeline, v_features),
        ("amount_transform", amount_pipeline, ["Amount"]),
        ("time_transform", time_pipeline, ["Time", "Hour"])
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

# =========================================================
# =========================================================
X_tr_no, y_tr_no = X_train_df, y_train

ROS = RandomOverSampler(random_state=42)
X_tr_ros, y_tr_ros = ROS.fit_resample(X_train_df, y_train)

RUS = RandomUnderSampler(random_state=42)
X_tr_rus, y_tr_rus = RUS.fit_resample(X_train_df, y_train)

smote = SMOTE(random_state=42)
X_tr_smote, y_tr_smote = smote.fit_resample(X_train_df, y_train)

scale_pos_weight_val = (y_train == 0).sum() / (y_train == 1).sum()

resampling_methods = {
    "NO Sampling": (X_tr_no, y_tr_no, None, 1.0),
    "Class Weight": (X_train_df, y_train, "balanced", scale_pos_weight_val),
    "RandomOverSampler": (X_tr_ros, y_tr_ros, None, 1.0),
    "RandomUnderSampler": (X_tr_rus, y_tr_rus, None, 1.0),
    "SMOTE": (X_tr_smote, y_tr_smote, None, 1.0)
}

# =========================================================
# =========================================================
all_results = []
rf_param_grid = {
    "max_depth": [6, 8, 10], 
    "n_estimators": [50, 100, 150] 
}

print("\n Starting Training Loop with XGBoost + Random Forest Ensemble...\n")

for strategy_name, (x_tr, y_tr, cw_option, xgb_spw) in resampling_methods.items():
    print(f"==================================================")
    print(f" Processing Strategy: [{strategy_name}]")
    print(f"==================================================")
    
    # 1. XGBoost Model
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=xgb_spw,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(x_tr, y_tr)
    
    rf_base = RandomForestClassifier(class_weight=cw_option, random_state=42, n_jobs=-1)
    grid_rf = GridSearchCV(rf_base, rf_param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid_rf.fit(x_tr, y_tr)
    best_rf_model = grid_rf.best_estimator_
    
    votingclf = VotingClassifier(
        estimators=[
            ("xgboost", xgb_model),
            ("random_forest", best_rf_model)
        ],
        voting="soft"
    )
    votingclf.fit(x_tr, y_tr)
    
    y_probs = votingclf.predict_proba(X_test_df)[:, 1]
    
    pr_auc_score = average_precision_score(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    best_th = 0.5
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    for th in np.arange(0.05, 0.95, 0.01):
        y_pred_th = (y_probs >= th).astype(int)
        score = f1_score(y_test, y_pred_th, zero_division=0)
        
        if score > best_f1:
            best_f1 = score
            best_th = th
            best_precision = precision_score(y_test, y_pred_th, zero_division=0)
            best_recall = recall_score(y_test, y_pred_th, zero_division=0)
            
    print(f" Best Parameters (RF): {grid_rf.best_params_}")
    print(f" Best Threshold: {best_th:.2f} | PR-AUC: {pr_auc_score:.4f} | Best F1-Score: {best_f1:.4f}\n") 
    
    all_results.append({
        "Strategy": strategy_name,
        "Best RF Params": str(grid_rf.best_params_),
        "Optimal Threshold": round(best_th, 2),
        "PR-AUC": round(pr_auc_score, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Precision": round(best_precision, 4),
        "Recall": round(best_recall, 4),
        "Best F1-Score": round(best_f1, 4)
    })

# =========================================================
# =========================================================
results_df = pd.DataFrame(all_results).sort_values(by="PR-AUC", ascending=False)

print("\n Final Leaderboard (XGBoost + RF Ensemble):")
print(results_df.to_string(index=False))

# =========================================================
#  Top 10 Feature Importances (Random Forest) + Bar Plot
# =========================================================
print("\n Re-training Random Forest to extract Feature Importances...")
rf_standalone = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_standalone.fit(X_train_df, y_train)

importances = rf_standalone.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_10_features = feature_imp_df.head(10)

plt.figure(figsize=(10, 5))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=top_10_features, 
    hue='Feature', 
    palette='viridis', 
    legend=False
)
plt.title('Top 10 Feature Importances (Random Forest)', fontsize=12, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# 6. ROC Curve
# =========================================================
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve (XGBoost + RF Ensemble)', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# Precision, Recall & F1-Score vs Threshold
# =========================================================
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label="Precision", color="blue", lw=2)
plt.plot(thresholds, recall[:-1], label="Recall", color="orange", lw=2)
plt.plot(thresholds, f1_scores, label="F1-Score", color="green", lw=2)

plt.axvline(x=best_th, color='red', linestyle='--', label=f'Optimal Threshold: {best_th}')
plt.title('Precision, Recall & F1-Score vs Threshold', fontsize=12, fontweight='bold')
plt.xlabel('Threshold Value')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# 8. Error Analysis (FP & FN)
# =========================================================
best_th_final = results_df.iloc[0]["Optimal Threshold"]
y_pred_final = (y_probs >= best_th_final).astype(int)
cm = confusion_matrix(y_test, y_pred_final)

TN, FP, FN, TP = cm.ravel()

print("\n" + "="*40)
print(" ERROR ANALYSIS REPORT")
print("="*40)
print(f" False Positives (FP) : {FP}")
print(f" False Negatives (FN) : {FN}")
print(f" True Positives  (TP) : {TP}")
print(f" True Negatives  (TN) : {TN}")
print("="*40)

# =========================================================
# =========================================================
model_artifacts = {
    "preprocessor": preprocessor,
    "voting_classifier": votingclf,
    "optimal_threshold": best_th_final
}

joblib.dump(model_artifacts, "final_model.pkl")
print("\n Model saved successfully as 'final_model.pkl'!")