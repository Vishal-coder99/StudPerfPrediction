# 🎓 Student Performance Prediction System

<p align="center">
  <strong>An End-to-End Machine Learning Application for Student Academic Performance Analysis</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression%20%26%20Classification-orange" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

---

## 📌 Overview

The **Student Performance Prediction System** is a Machine Learning application designed to analyze student academic data and predict their examination performance.

The system uses student-related factors such as study hours, attendance, previous examination marks, sleep duration, and test participation to generate predictions.

It supports two Machine Learning tasks:

- **Regression:** Predicts expected final examination marks on a scale of 0–100.
- **Classification:** Predicts whether a student is likely to Pass or Fail based on a 40-mark threshold.

The application also provides an interactive dashboard built with Streamlit for easy data input and prediction.

## ✨ Features

- 📊 **Student Performance Analysis** – Analyzes academic data and student study habits.
- 🎯 **Marks Prediction** – Estimates final examination marks using Regression.
- ✅ **Pass/Fail Classification** – Identifies students who may be at academic risk.
- 🤖 **Multiple ML Algorithms** – Implements Linear Regression, Logistic Regression, and Random Forest.
- 📈 **Model Evaluation** – Measures model performance using standard evaluation metrics.
- 🖥️ **Interactive Dashboard** – Provides a simple and user-friendly Streamlit interface.
- ⚙️ **Dataset Generation** – Generates 1,000 academic student records using Python.
- 🔬 **ML From Scratch** – Includes a pure Python implementation to understand the fundamentals of Machine Learning.

## 🛠️ Tech Stack

| Technology | Application |
|---|---|
| Python | Core development |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Machine Learning models |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Web application |
| Git & GitHub | Version control |

## 🧠 Machine Learning Implementation

### 1. Regression – Final Marks Prediction

Predicts a student's expected examination marks using multiple academic and lifestyle features.

**Algorithms:**
- Multiple Linear Regression
- Random Forest Regressor

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### 2. Classification – Pass/Fail Prediction

Classifies students based on their predicted academic performance.

**Classification Threshold:**

| Marks | Prediction |
|---|---|
| ≥ 40 | Pass |
| < 40 | Fail / Academic Risk |

**Algorithms:**
- Logistic Regression
- Random Forest Classifier

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🔄 Machine Learning Workflow

```text
       Student Dataset
              |
              ▼
    Data Preprocessing
              |
              ▼
 Exploratory Data Analysis
              |
              ▼
       Train/Test Split
              |
              ▼
      Feature Scaling
              |
       ┌──────┴──────┐
       ▼             ▼
   Regression   Classification
       |             |
       ▼             ▼
  Marks Model   Pass/Fail Model
       |             |
       └──────┬──────┘
              |
              ▼
      Model Evaluation
              |
              ▼
    Streamlit Dashboard
              |
              ▼
       Final Prediction
```

## 📂 Project Structure

```text
Student-Performance-Prediction/
│
├── data/
│   └── student_data.csv
│
├── generate_dataset.py
├── train_from_scratch.py
├── train_models.py
├── predict.py
├── app.py
│
├── requirements.txt
└── README.md
```

### Project Files

| File | Description |
|---|---|
| `generate_dataset.py` | Generates the student academic dataset. |
| `train_from_scratch.py` | Implements ML fundamentals using pure Python. |
| `train_models.py` | Trains and evaluates Scikit-learn models. |
| `predict.py` | Performs standalone predictions. |
| `app.py` | Runs the interactive Streamlit application. |
| `requirements.txt` | Lists the required Python dependencies. |
| `student_data.csv` | Contains generated student academic records. |

## 📊 Model Performance

The models were evaluated using an 80:20 train-test split, consisting of 800 training records and 200 testing records.

### Regression Results

| Metric | Result |
|---|---:|
| MAE | 3.14 |
| RMSE | 3.88 |
| R² Score | 0.8921 |
| Variance Explained | 89.21% |

### Classification Results

| Metric | Result |
|---|---:|
| Accuracy | 99.00% |
| Precision | 99.00% |
| Recall | 100.00% |
| F1-Score | 99.50% |

*Note: Results are based on the reported experiment and may vary depending on dataset generation and model configuration.*

## ⚙️ Installation

Follow these steps to set up the project locally.

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Navigate to the project directory**

```bash
cd Student-Performance-Prediction
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate the virtual environment (Windows)**

```powershell
.venv\Scripts\Activate.ps1
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

## 🚀 Run the Application

**Step 1: Generate the dataset**

```bash
python generate_dataset.py
```

**Step 2: Train the Machine Learning models**

```bash
python train_models.py
```

**Step 3: Launch the Streamlit dashboard**

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal:

```text
http://localhost:8501
```

## 💻 Run Without External ML Dependencies

The project also includes a pure Python implementation for understanding the mathematical foundations of Machine Learning.

```bash
python generate_dataset.py
python train_from_scratch.py
```

## 🎯 Project Objectives

- Apply Machine Learning algorithms to real-world educational problems.
- Understand Regression and Classification techniques.
- Analyze relationships between academic factors and student performance.
- Evaluate predictive models using appropriate metrics.
- Develop an interactive Machine Learning application.

## 🔮 Future Enhancements

- Integrate real-world academic datasets.
- Add advanced data visualization.
- Improve model performance through hyperparameter tuning.
- Generate downloadable student performance reports.
- Deploy the application to a cloud platform.

## 👨‍💻 Author

**Vishal Gupta**

BCA | Python | SQL | Machine Learning

- GitHub: [Vishal-coder99](https://github.com/Vishal-coder99)

---

<p align="center">
  <strong>Built with Python and Machine Learning ❤️</strong>
</p>

<p align="center">
  ⭐ If you find this project useful, consider giving it a star!
</p>
