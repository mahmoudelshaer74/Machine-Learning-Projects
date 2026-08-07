import argparse
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV

from credit_fraud_utils_data import (
    load_and_split_data, build_preprocessor, 
    preprocess_features, get_resampled_data
)
from credit_fraud_utils_eval import (
    evaluate_thresholds, plot_feature_importance, 
    plot_roc_curve, plot_precision_recall_f1, print_error_analysis
)

def train_pipeline(data_path, sampling_strategy, output_model_path):
    print("\n1. Loading and Preprocessing Data...")
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    
    preprocessor = build_preprocessor()
    X_train_df, X_test_df, feature_names = preprocess_features(preprocessor, X_train, X_test)
    
    print(f"\n2. Applying Sampling Strategy: [{sampling_strategy}]...")
    X_tr, y_tr, cw_option, xgb_spw = get_resampled_data(X_train_df, y_train, sampling_strategy)
    
    print("\n3. Training XGBoost Model...")
    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        scale_pos_weight=xgb_spw, random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    xgb_model.fit(X_tr, y_tr)
    
    print("\n4. Tuning Random Forest with GridSearchCV...")
    rf_base = RandomForestClassifier(class_weight=cw_option, random_state=42, n_jobs=-1)
    rf_param_grid = {"max_depth": [6, 8, 10], "n_estimators": [50, 100, 150]}
    grid_rf = GridSearchCV(rf_base, rf_param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid_rf.fit(X_tr, y_tr)
    best_rf = grid_rf.best_estimator_
    
    print("\n5. Training Ensemble Voting Classifier...")
    voting_clf = VotingClassifier(
        estimators=[("xgboost", xgb_model), ("random_forest", best_rf)],
        voting="soft"
    )
    voting_clf.fit(X_tr, y_tr)
    
    print("\n6. Evaluating Model...")
    y_probs = voting_clf.predict_proba(X_test_df)[:, 1]
    best_th, best_f1, prec, rec, pr_auc, roc_auc = evaluate_thresholds(y_test, y_probs)
    
    print(f"\n Results for [{sampling_strategy}]:")
    print(f" Optimal Threshold: {best_th:.2f} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | F1: {best_f1:.4f}")
    
    print("\n7. Generating Visualizations...")
    
    rf_standalone = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_standalone.fit(X_train_df, y_train)
    plot_feature_importance(rf_standalone, feature_names)
    
    plot_roc_curve(y_test, y_probs)
    plot_precision_recall_f1(y_test, y_probs, best_th)
    print_error_analysis(y_test, y_probs, best_th)
    
    model_artifacts = {
        "preprocessor": preprocessor,
        "voting_classifier": voting_clf,
        "optimal_threshold": best_th
    }
    joblib.dump(model_artifacts, output_model_path)
    print(f"\n Model saved successfully to '{output_model_path}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Training Pipeline")
    
    parser.add_argument("--data_path", type=str, default="EDA_model/cleaned_data.pkl", help="Path to input dataset (pkl/csv)")
    parser.add_argument("--strategy", type=str, default="ClassWeight", choices=["NO", "ROS", "RUS", "SMOTE", "ClassWeight"], help="Resampling Strategy")
    parser.add_argument("--output_model", type=str, default="models/final_model.pkl", help="Path to save final model artifact")
    
    args = parser.parse_args()
    
    train_pipeline(
        data_path=args.data_path,
        sampling_strategy=args.strategy,
        output_model_path=args.output_model
    )