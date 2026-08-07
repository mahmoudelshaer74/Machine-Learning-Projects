import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def load_and_split_data(data_path, test_size=0.2, random_state=42):
    df = pd.read_pickle(data_path)
    X = df.drop(columns=["Class"], axis=1)
    y = df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def build_preprocessor():
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
    return preprocessor

def preprocess_features(preprocessor, X_train, X_test):
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    
    return X_train_df, X_test_df, feature_names

def get_resampled_data(X_train_df, y_train, strategy_name):
    scale_pos_weight_val = (y_train == 0).sum() / (y_train == 1).sum()
    
    if strategy_name == "ROS":
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_train_df, y_train)
        return X_res, y_res, None, 1.0
        
    elif strategy_name == "RUS":
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train_df, y_train)
        return X_res, y_res, None, 1.0
        
    elif strategy_name == "SMOTE":
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train_df, y_train)
        return X_res, y_res, None, 1.0
        
    elif strategy_name == "ClassWeight":
        return X_train_df, y_train, "balanced", scale_pos_weight_val
        
    else:  # NO Sampling
        return X_train_df, y_train, None, 1.0