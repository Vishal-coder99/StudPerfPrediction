"""
Model Training Pipeline for Student Performance Prediction.

Covers both:
1. Regression: Predicting continuous expected marks (0 - 100).
2. Classification: Predicting binary pass / fail outcome.

Strict Featurization Ordering:
- Train/Test split is performed BEFORE any feature scaling to prevent data leakage.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def run_pipeline(data_path: str = "data/student_data.csv", models_dir: str = "models"):
    print("=" * 60)
    print("[PROJECT] STUDENT PERFORMANCE PREDICTION - ML PIPELINE")
    print("=" * 60)

    # 1. Load Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run generate_dataset.py first.")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("\nDataset Sample:")
    print(df.head())

    # 2. Define Features & Targets
    feature_cols = [
        "study_hours_per_day",
        "attendance_percentage",
        "previous_marks",
        "sleep_hours",
        "practice_tests_taken"
    ]
    
    X = df[feature_cols]
    y_reg = df["final_marks"]   # Continuous target for Regression
    y_clf = df["pass_fail"]     # Categorical target for Classification

    # 3. Train-Test Split (Strict Featurization Rule: Split BEFORE scaling)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42
    )

    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size:  {len(X_test)} samples")

    # 4. Feature Scaling (Fit on Train ONLY, then transform both Train and Test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # TASK 1: REGRESSION (Predicting Marks)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[TASK 1] REGRESSION (Continuous Marks Prediction)")
    print("-" * 60)

    # Model A: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_reg_train)
    y_reg_pred_lr = lr.predict(X_test_scaled)

    mae_lr = mean_absolute_error(y_reg_test, y_reg_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_lr))
    r2_lr = r2_score(y_reg_test, y_reg_pred_lr)

    print(f"\n[Linear Regression Performance]")
    print(f"  * MAE (Mean Absolute Error):     {mae_lr:.2f} marks")
    print(f"  * RMSE (Root Mean Squared Err):  {rmse_lr:.2f} marks")
    print(f"  * R2 Score (Variance Explained): {r2_lr:.4f} ({r2_lr*100:.1f}%)")

    # Model B: Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_reg_train)  # Tree models don't strictly require scaling
    y_reg_pred_rf = rf_reg.predict(X_test)

    mae_rf = mean_absolute_error(y_reg_test, y_reg_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_rf))
    r2_rf = r2_score(y_reg_test, y_reg_pred_rf)

    print(f"\n[Random Forest Regressor Performance]")
    print(f"  * MAE:                          {mae_rf:.2f} marks")
    print(f"  * RMSE:                         {rmse_rf:.2f} marks")
    print(f"  * R2 Score:                     {r2_rf:.4f} ({r2_rf*100:.1f}%)")

    # Feature Importance (Linear Regression Coefficients)
    print("\n[Feature Weights in Predicting Marks - Linear Regression Coefficients]")
    for col, coef in zip(feature_cols, lr.coef_):
        print(f"  * {col:<25}: {coef:+.3f}")

    # -------------------------------------------------------------
    # TASK 2: CLASSIFICATION (Predicting Pass / Fail)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[TASK 2] CLASSIFICATION (Binary Pass/Fail Prediction)")
    print("-" * 60)

    # Model A: Logistic Regression
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_clf_train)
    y_clf_pred_log = log_reg.predict(X_test_scaled)

    acc_log = accuracy_score(y_clf_test, y_clf_pred_log)
    prec_log = precision_score(y_clf_test, y_clf_pred_log, zero_division=0)
    rec_log = recall_score(y_clf_test, y_clf_pred_log, zero_division=0)
    f1_log = f1_score(y_clf_test, y_clf_pred_log, zero_division=0)

    print(f"\n[Logistic Regression Performance]")
    print(f"  * Accuracy:  {acc_log*100:.2f}%")
    print(f"  * Precision: {prec_log*100:.2f}%")
    print(f"  * Recall:    {rec_log*100:.2f}%")
    print(f"  * F1-Score:  {f1_log*100:.2f}%")
    print("\nConfusion Matrix (Logistic Regression):")
    print(confusion_matrix(y_clf_test, y_clf_pred_log))

    # Model B: Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_clf_train)
    y_clf_pred_rf = rf_clf.predict(X_test)

    acc_rf = accuracy_score(y_clf_test, y_clf_pred_rf)
    f1_rf = f1_score(y_clf_test, y_clf_pred_rf, zero_division=0)

    print(f"\n[Random Forest Classifier Performance]")
    print(f"  * Accuracy:  {acc_rf*100:.2f}%")
    print(f"  * F1-Score:  {f1_rf*100:.2f}%")

    # -------------------------------------------------------------
    # SAVE ARTIFACTS
    # -------------------------------------------------------------
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(lr, os.path.join(models_dir, "linear_regression_marks.joblib"))
    joblib.dump(rf_reg, os.path.join(models_dir, "random_forest_marks.joblib"))
    joblib.dump(log_reg, os.path.join(models_dir, "logistic_regression_passfail.joblib"))
    joblib.dump(rf_clf, os.path.join(models_dir, "random_forest_passfail.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(models_dir, "features.joblib"))

    print(f"\n[OK] All models and preprocessors saved to directory: '{models_dir}/'")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()

