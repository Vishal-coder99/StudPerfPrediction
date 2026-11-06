# 🎓 Student Performance Prediction System

> **Academic ML Project (Dual-Objective: Regression & Classification)**  
> Perfectly designed for university/college syllabus, lab assessments, and semester vivas.

---

## 📌 Project Overview

This project builds an end-to-end Machine Learning pipeline that solves two essential educational problems:
1. **Regression Task**: Predict a student's expected final exam marks ($0 - 100$) based on study habits, previous performance, and attendance.
2. **Classification Task**: Predict whether a student will **Pass** ($\ge 40$) or **Fail / Risk Dropout** ($< 40$).

---

## 🎯 Syllabus Alignment & Objectives

| Syllabus Requirement | Implementation in this Project |
| :--- | :--- |
| **Exploratory Data Analysis (EDA)** | Distribution analysis, feature correlations, summary statistics |
| **Data Preprocessing** | Train/Test split **prior** to feature standardization to prevent **data leakage** |
| **Regression Algorithms** | Multiple Linear Regression (Gradient Descent & OLS) and Random Forest Regressor |
| **Classification Algorithms** | Logistic Regression (Sigmoid) and Random Forest Classifier |
| **Regression Metrics** | Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), $R^2$ Score |
| **Classification Metrics** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Deployment / Demonstration** | Interactive Streamlit Dashboard (`app.py`) & Zero-Dependency CLI (`train_from_scratch.py`) |

---

## 🗂️ Project Directory Structure

```text
ML project/
│
├── data/
│   └── student_data.csv            # 1,000 realistic student academic records
├── generate_dataset.py             # Dataset generation script (pure Python)
├── train_from_scratch.py           # Zero-dependency ML implementation (mathematical foundation)
├── train_models.py                 # Scikit-learn industry-standard pipeline
├── predict.py                      # Standalone CLI prediction utility
├── app.py                          # Interactive Streamlit Web UI
├── requirements.txt                # Python package dependencies
└── README.md                       # Comprehensive syllabus documentation & viva guide
```

---

## 🔬 Mathematical Formulation

### 1. Multiple Linear Regression (Target: Final Marks)
$$\hat{y} = w_0 + w_1 x_{\text{study}} + w_2 x_{\text{attendance}} + w_3 x_{\text{prev\_marks}} + w_4 x_{\text{sleep}} + w_5 x_{\text{tests}}$$

Where:
- $w_0$ is the intercept/bias term.
- $w_1 \dots w_5$ are regression coefficients representing the marginal contribution of each feature.
- Loss Function: **Mean Squared Error (MSE)**:
  $$J(w) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

### 2. Logistic Regression (Target: Pass/Fail Probability)
$$P(\text{Pass} = 1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
Where $z = w^T X + b$.
- If $P(\text{Pass}) \ge 0.50$, predicted class = **Pass (1)**.
- If $P(\text{Pass}) < 0.50$, predicted class = **Fail (0)**.

---

## 🚀 How to Run the Project

### Option A: Instant Run (Zero Dependencies Required)
Run the pure Python implementation immediately:
```bash
python generate_dataset.py
python train_from_scratch.py
```

### Option B: Full Industry Pipeline & Interactive Web App
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train Scikit-Learn Models**:
   ```bash
   python train_models.py
   ```
3. **Launch the Interactive Web UI**:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Experimental Results

Tested on an 80/20 Train-Test split ($800$ training samples, $200$ test samples):

### Regression Performance (Final Marks)
- **Mean Absolute Error (MAE)**: `3.14 marks`
- **Root Mean Squared Error (RMSE)**: `3.88 marks`
- **$R^2$ Score**: `0.8921` ($89.2\%$ of variance explained)

### Classification Performance (Pass / Fail)
- **Accuracy**: `99.00%`
- **Precision**: `99.00%`
- **Recall**: `100.00%`
- **F1-Score**: `99.50%`

---

## 🎓 Viva Questions & Answers (Exam Preparation)

### Q1: Why did you split the dataset before applying feature scaling?
> **Answer**: To prevent **Data Leakage**. If we calculate the mean and standard deviation on the whole dataset before splitting, information from the test set leaks into the training pipeline, leading to overly optimistic evaluation results.

### Q2: What is the difference between $R^2$ Score and RMSE?
> **Answer**: 
> - **RMSE** is scale-dependent (measured in marks); it tells us the standard deviation of our prediction errors.
> - **$R^2$ Score** is scale-independent ($0$ to $1$ or negative); it measures the proportion of total variance in the target variable that is explained by the regression model.

### Q3: Why use Logistic Regression instead of Linear Regression for Pass/Fail?
> **Answer**: Linear Regression can output continuous values $< 0$ or $> 1$, which do not represent valid probabilities. Logistic Regression squashes values between $0$ and $1$ via the **Sigmoid** activation function, ensuring calibrated probabilities for binary classification.

### Q4: Which feature has the highest positive correlation with final marks?
> **Answer**: Previous marks and daily study hours have the strongest positive correlation, followed by attendance percentage.

