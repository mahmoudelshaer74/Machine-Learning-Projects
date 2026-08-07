import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Card Fraud Detection System")
st.markdown(
    "An interactive dashboard to perform inference on new credit card transactions and detect potential fraud."
)

# ==========================================
# 2. Load Model Artifacts
# ==========================================
@st.cache_resource
def load_artifacts(model_path):
    try:
        artifacts = joblib.load(model_path)
        return artifacts
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None


# Sidebar - Model Configuration
st.sidebar.header("⚙️ Model Configuration")
model_file_path = st.sidebar.text_input(
    "Model Artifacts Path (.pkl)", value="models/final_model.pkl"
)

artifacts = load_artifacts(model_file_path)

if artifacts:
    preprocessor = artifacts["preprocessor"]
    voting_clf = artifacts["voting_classifier"]
    optimal_th = artifacts["optimal_threshold"]

    st.sidebar.success("Model loaded successfully! ✅")
    st.sidebar.info(f"🎯 **Optimal Threshold:** {optimal_th:.2f}")

    # Interactive threshold adjustment
    custom_threshold = st.sidebar.slider(
        "Classification Threshold:",
        min_value=0.01,
        max_value=0.99,
        value=float(optimal_th),
        step=0.01,
    )

    # ==========================================
    # 3. File Upload Section
    # ==========================================
    st.subheader("📁 Upload Inference Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or PKL dataset file", type=["csv", "pkl"]
    )

    if uploaded_file is not None:
        # Load dataset
        if uploaded_file.name.endswith(".csv"):
            df_new = pd.read_csv(uploaded_file)
        else:
            df_new = pd.read_pickle(uploaded_file)

        st.write(
            f"📊 **Uploaded Data Dimensions:** {df_new.shape[0]} rows and {df_new.shape[1]} columns."
        )

        # Check for target ground truth label
        has_labels = "Class" in df_new.columns

        if has_labels:
            X_new = df_new.drop(columns=["Class"], axis=1)
            y_true = df_new["Class"]
        else:
            X_new = df_new.copy()
            y_true = None

        # ==========================================
        # 4. Feature Engineering & Inference
        # ==========================================
        # Calculate 'Hour' feature if missing
        if "Hour" not in X_new.columns and "Time" in X_new.columns:
            X_new["Hour"] = X_new["Time"] / 3600

        try:
            # Preprocessing
            X_new_processed = preprocessor.transform(X_new)
            feature_names = preprocessor.get_feature_names_out()
            X_new_df = pd.DataFrame(
                X_new_processed, columns=feature_names, index=X_new.index
            )

            # Inference & Probability Prediction
            y_probs = voting_clf.predict_proba(X_new_df)[:, -1]
            y_pred = (y_probs >= custom_threshold).astype(int)

            # Results DataFrame
            df_results = df_new.copy()
            df_results["Fraud_Probability"] = y_probs
            df_results["Prediction"] = np.where(y_pred == 1, "Fraud", "Normal")

            # ==========================================
            # 5. Summary & Metrics
            # ==========================================
            st.markdown("---")
            st.subheader("📈 Inference Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", len(df_results))
            num_fraud = int(y_pred.sum())
            col2.metric("Flagged Fraud Transactions", num_fraud, delta_color="inverse")
            col3.metric("Normal Transactions", len(df_results) - num_fraud)

            # Display Evaluation Metrics if Ground Truth 'Class' is Present
            if has_labels:
                st.markdown("---")
                st.subheader("🔍 Model Performance Evaluation")

                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{precision:.4f}")
                m2.metric("Recall", f"{recall:.4f}")
                m3.metric("F1-Score", f"{f1:.4f}")

                # Plot Confusion Matrix
                fig, ax = plt.subplots(figsize=(5, 4))
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Normal", "Fraud"],
                    yticklabels=["Normal", "Fraud"],
                    ax=ax,
                )
                ax.set_title(f"Confusion Matrix (Threshold: {custom_threshold:.2f})")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")

                col_cm, col_rep = st.columns([1, 1])
                with col_cm:
                    st.pyplot(fig)
                with col_rep:
                    st.text("Detailed Classification Report:")
                    report = classification_report(
                        y_true, y_pred, target_names=["Normal", "Fraud"]
                    )
                    st.code(report)

            # ==========================================
            # 6. Flagged Fraud Data & Export
            # ==========================================
            st.markdown("---")
            st.subheader("📋 Flagged Fraudulent Transactions")
            fraud_df = df_results[df_results["Prediction"] == "Fraud"]

            if len(fraud_df) > 0:
                st.dataframe(fraud_df)
            else:
                st.info("No fraudulent transactions detected with the current threshold.")

            # Download CSV Button
            csv_data = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Prediction Results (CSV)",
                data=csv_data,
                file_name="fraud_detection_results.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error during preprocessing or model inference: {e}")