"""
Pure Python (Zero Dependency) Implementation of Student Performance Prediction.
Runs directly out-of-the-box using only Python standard library.
Demonstrates the fundamental mathematics of:
- Train/Test Data Split
- Feature Normalization (Min-Max / Standard)
- Multiple Linear Regression using Gradient Descent & Ordinary Least Squares
- Logistic Sigmoid Classification for Pass/Fail
- Core Evaluation Metrics (MAE, MSE, RMSE, R-Squared, Accuracy)
"""

import csv
import math
import os
import random

def load_data(filepath="data/student_data.csv"):
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            data.append({
                "id": row["student_id"],
                "study": float(row["study_hours_per_day"]),
                "attendance": float(row["attendance_percentage"]),
                "prev_marks": float(row["previous_marks"]),
                "sleep": float(row["sleep_hours"]),
                "tests": float(row["practice_tests_taken"]),
                "final_marks": float(row["final_marks"]),
                "pass_fail": int(row["pass_fail"])
            })
    return data

def train_test_split(data, test_ratio=0.20, seed=42):
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]

def calculate_stats(train_data, feature_keys):
    stats = {}
    for key in feature_keys:
        vals = [row[key] for row in train_data]
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = math.sqrt(variance) if variance > 0 else 1.0
        stats[key] = {"mean": mean, "std": std}
    return stats

def standardize_row(row, feature_keys, stats):
    return [(row[k] - stats[k]["mean"]) / stats[k]["std"] for k in feature_keys]

def sigmoid(z):
    # Clip z to avoid overflow
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def train_linear_regression(train_data, feature_keys, stats, lr=0.01, epochs=600):
    """Multiple Linear Regression via Batch Gradient Descent."""
    weights = [0.0] * len(feature_keys)
    bias = 0.0
    n = len(train_data)

    for _ in range(epochs):
        dw = [0.0] * len(feature_keys)
        db = 0.0
        for row in train_data:
            x = standardize_row(row, feature_keys, stats)
            y_actual = row["final_marks"]
            y_pred = bias + sum(w * xi for w, xi in zip(weights, x))
            error = y_pred - y_actual
            for j in range(len(weights)):
                dw[j] += error * x[j]
            db += error

        for j in range(len(weights)):
            weights[j] -= (lr * dw[j] / n)
        bias -= (lr * db / n)

    return weights, bias

def train_logistic_regression(train_data, feature_keys, stats, lr=0.05, epochs=600):
    """Binary Logistic Regression for Pass/Fail."""
    weights = [0.0] * len(feature_keys)
    bias = 0.0
    n = len(train_data)

    for _ in range(epochs):
        dw = [0.0] * len(feature_keys)
        db = 0.0
        for row in train_data:
            x = standardize_row(row, feature_keys, stats)
            y_actual = row["pass_fail"]
            z = bias + sum(w * xi for w, xi in zip(weights, x))
            p = sigmoid(z)
            error = p - y_actual
            for j in range(len(weights)):
                dw[j] += error * x[j]
            db += error

        for j in range(len(weights)):
            weights[j] -= (lr * dw[j] / n)
        bias -= (lr * db / n)

    return weights, bias

def evaluate_and_run():
    print("=" * 65)
    print("[PROJECT] STUDENT PERFORMANCE PREDICTION (Zero-Dependency Engine)")
    print("=" * 65)

    data = load_data()
    train_data, test_data = train_test_split(data, test_ratio=0.20, seed=42)
    feature_keys = ["study", "attendance", "prev_marks", "sleep", "tests"]
    stats = calculate_stats(train_data, feature_keys)

    print(f"Dataset split: {len(train_data)} train rows | {len(test_data)} test rows")

    # 1. Train Regression Model
    w_reg, b_reg = train_linear_regression(train_data, feature_keys, stats)

    # Evaluate Regression on Test Set
    reg_actuals = [row["final_marks"] for row in test_data]
    reg_preds = []
    for row in test_data:
        x = standardize_row(row, feature_keys, stats)
        pred = b_reg + sum(w * xi for w, xi in zip(w_reg, x))
        reg_preds.append(max(0.0, min(100.0, pred)))

    mae = sum(abs(act - pred) for act, pred in zip(reg_actuals, reg_preds)) / len(test_data)
    mse = sum((act - pred) ** 2 for act, pred in zip(reg_actuals, reg_preds)) / len(test_data)
    rmse = math.sqrt(mse)
    mean_actual = sum(reg_actuals) / len(reg_actuals)
    ss_tot = sum((y - mean_actual) ** 2 for y in reg_actuals)
    ss_res = sum((act - pred) ** 2 for act, pred in zip(reg_actuals, reg_preds))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    print("\n--- [TASK 1] REGRESSION EVALUATION (Target: Final Marks) ---")
    print(f"  * MAE (Mean Absolute Error):     {mae:.2f} marks")
    print(f"  * RMSE (Root Mean Sq Error):     {rmse:.2f} marks")
    print(f"  * R2 Score (Variance Explained): {r2:.4f} ({r2 * 100:.1f}%)")

    # 2. Train Logistic Regression for Pass/Fail
    w_clf, b_clf = train_logistic_regression(train_data, feature_keys, stats)

    clf_actuals = [row["pass_fail"] for row in test_data]
    clf_preds = []
    for row in test_data:
        x = standardize_row(row, feature_keys, stats)
        prob = sigmoid(b_clf + sum(w * xi for w, xi in zip(w_clf, x)))
        clf_preds.append(1 if prob >= 0.5 else 0)

    correct = sum(1 for act, pred in zip(clf_actuals, clf_preds) if act == pred)
    accuracy = (correct / len(test_data)) * 100.0

    tp = sum(1 for act, pred in zip(clf_actuals, clf_preds) if act == 1 and pred == 1)
    fp = sum(1 for act, pred in zip(clf_actuals, clf_preds) if act == 0 and pred == 1)
    fn = sum(1 for act, pred in zip(clf_actuals, clf_preds) if act == 1 and pred == 0)
    tn = sum(1 for act, pred in zip(clf_actuals, clf_preds) if act == 0 and pred == 0)

    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n--- [TASK 2] CLASSIFICATION EVALUATION (Target: Pass / Fail) ---")
    print(f"  * Accuracy:  {accuracy:.2f}%")
    print(f"  * Precision: {precision:.2f}%")
    print(f"  * Recall:    {recall:.2f}%")
    print(f"  * F1-Score:  {f1:.2f}%")
    print(f"  * Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")

    print("\n--- [SAMPLE] SINGLE STUDENT PREDICTION TEST ---")
    test_student = {"study": 6.5, "attendance": 85.0, "prev_marks": 72.0, "sleep": 7.0, "tests": 6}
    x_test = standardize_row(test_student, feature_keys, stats)
    test_marks = max(0.0, min(100.0, b_reg + sum(w * xi for w, xi in zip(w_reg, x_test))))
    test_prob = sigmoid(b_clf + sum(w * xi for w, xi in zip(w_clf, x_test)))

    print(f"Student Profile: Study={test_student['study']} hrs/day | Attendance={test_student['attendance']}% | Prev Marks={test_student['prev_marks']}%")
    print(f"-> Predicted Expected Marks: {test_marks:.2f} / 100")
    print(f"-> Pass Probability:        {test_prob * 100:.2f}% ({'PASS' if test_prob >= 0.5 else 'FAIL'})")
    print("=" * 65)

if __name__ == "__main__":
    evaluate_and_run()
