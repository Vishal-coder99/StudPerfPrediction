"""
CLI Inference Script for Student Performance Prediction.
Allows users to enter student parameters and get immediate predictions for expected marks and pass/fail probability.
"""

import os
import joblib
import numpy as np

def load_inference_artifacts(models_dir: str = "models"):
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    reg_path = os.path.join(models_dir, "linear_regression_marks.joblib")
    clf_path = os.path.join(models_dir, "logistic_regression_passfail.joblib")

    if not (os.path.exists(scaler_path) and os.path.exists(reg_path) and os.path.exists(clf_path)):
        raise FileNotFoundError(
            "Model files missing. Please run `python train_models.py` before running predictions."
        )

    scaler = joblib.load(scaler_path)
    reg_model = joblib.load(reg_path)
    clf_model = joblib.load(clf_path)

    return scaler, reg_model, clf_model

def predict_student(
    study_hours: float,
    attendance: float,
    previous_marks: float,
    sleep_hours: float = 7.0,
    practice_tests: int = 5,
    models_dir: str = "models"
):
    scaler, reg_model, clf_model = load_inference_artifacts(models_dir)

    feature_names = ["study_hours_per_day", "attendance_percentage", "previous_marks", "sleep_hours", "practice_tests_taken"]
    import pandas as pd
    raw_features = pd.DataFrame([[study_hours, attendance, previous_marks, sleep_hours, practice_tests]], columns=feature_names)
    scaled_features = scaler.transform(raw_features)

    # 1. Continuous Prediction (Marks)
    predicted_marks = reg_model.predict(scaled_features)[0]
    predicted_marks = round(max(0.0, min(100.0, predicted_marks)), 2)

    # 2. Binary Prediction (Pass / Fail) & Probability
    predicted_class = clf_model.predict(scaled_features)[0]
    probabilities = clf_model.predict_proba(scaled_features)[0]
    pass_prob = round(probabilities[1] * 100, 2)
    status = "PASS" if predicted_class == 1 else "FAIL"

    return {
        "predicted_marks": predicted_marks,
        "status": status,
        "pass_probability": pass_prob,
        "fail_probability": round(probabilities[0] * 100, 2)
    }

if __name__ == "__main__":
    print("-" * 50)
    print("[TEST] Single Student Performance Quick Test")
    print("-" * 50)
    sample_study = 5.5
    sample_att = 82.0
    sample_prev = 70.0
    sample_sleep = 7.5
    sample_tests = 4

    print(f"Inputs: Study Hours={sample_study}h, Attendance={sample_att}%, Prev Marks={sample_prev}%, Sleep={sample_sleep}h, Tests={sample_tests}")
    try:
        res = predict_student(sample_study, sample_att, sample_prev, sample_sleep, sample_tests)
        print(f"Predicted Final Marks: {res['predicted_marks']} / 100")
        print(f"Result Status:        {res['status']} (Pass Probability: {res['pass_probability']}%)")
    except FileNotFoundError as e:
        print(e)

