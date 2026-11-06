"""
Flask Web Application Server for Student Performance Prediction.
Connects the trained Scikit-Learn models to the interactive Web UI.
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates")

MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
REG_PATH = os.path.join(MODELS_DIR, "linear_regression_marks.joblib")
CLF_PATH = os.path.join(MODELS_DIR, "logistic_regression_passfail.joblib")

scaler = None
reg_model = None
clf_model = None

def load_models():
    global scaler, reg_model, clf_model
    if os.path.exists(SCALER_PATH) and os.path.exists(REG_PATH) and os.path.exists(CLF_PATH):
        scaler = joblib.load(SCALER_PATH)
        reg_model = joblib.load(REG_PATH)
        clf_model = joblib.load(CLF_PATH)
        print("[OK] Scikit-Learn models loaded successfully.")
    else:
        print("[WARN] Trained models not found in models/ directory. Using mathematical fallback.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        study_hours = float(data.get("study_hours", 5.0))
        attendance = float(data.get("attendance", 75.0))
        previous_marks = float(data.get("previous_marks", 60.0))
        sleep_hours = float(data.get("sleep_hours", 7.0))
        practice_tests = int(data.get("practice_tests", 3))

        if scaler and reg_model and clf_model:
            feature_names = [
                "study_hours_per_day",
                "attendance_percentage",
                "previous_marks",
                "sleep_hours",
                "practice_tests_taken"
            ]
            raw_df = pd.DataFrame(
                [[study_hours, attendance, previous_marks, sleep_hours, practice_tests]],
                columns=feature_names
            )
            scaled = scaler.transform(raw_df)

            # Regression Prediction (Marks)
            pred_marks = float(reg_model.predict(scaled)[0])
            pred_marks = round(max(0.0, min(100.0, pred_marks)), 1)

            # Classification Prediction (Pass/Fail)
            pred_class = int(clf_model.predict(scaled)[0])
            probabilities = clf_model.predict_proba(scaled)[0]
            pass_prob = round(float(probabilities[1]) * 100, 1)
            status = "PASS" if pred_class == 1 else "FAIL"
        else:
            # Mathematical fallback
            pred_marks = (
                0.35 * previous_marks +
                0.25 * attendance +
                2.80 * study_hours +
                1.20 * practice_tests +
                0.80 * (sleep_hours - 5.0)
            )
            pred_marks = round(max(0.0, min(100.0, pred_marks)), 1)
            z = (pred_marks - 40.0) / 4.2
            pass_prob = round(float(1.0 / (1.0 + np.exp(-z))) * 100, 1)
            status = "PASS" if pred_marks >= 40.0 else "FAIL"

        # Academic Advice
        if attendance < 75:
            advice = "Attendance is critically low (<75%). Attend upcoming lectures to prevent exam debarment."
        elif study_hours < 3.0:
            advice = "Daily study hours are low. Increasing study by 1.5 hours daily will boost score by ~4 marks."
        elif pred_marks < 40:
            advice = "High academic risk detected! Urgent remedial classes and mock exam revision required."
        else:
            advice = "Student demonstrates strong consistency. Recommend continuing regular revision sessions."

        return jsonify({
            "status": status,
            "predicted_marks": pred_marks,
            "pass_probability": pass_prob,
            "advice": advice
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    load_models()
    print("=" * 60)
    print("🚀 Student Performance Prediction Web App Running!")
    print("👉 Open your browser at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)

