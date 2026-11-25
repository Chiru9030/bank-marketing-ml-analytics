# Bank Marketing Dataset — Data Science Project

> **Course Assignment** | Classification · Feature Engineering · Data Visualization  
> **Dataset:** [Kaggle — Bank Marketing Dataset (janiobachmann)](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)

---

## Project Structure

```
DAV FINAL/
├── bank.csv            ← Dataset (download separately — see below)
├── analysis.py         ← Task 1: Classification pipeline → model_dashboard.png
├── visualize.py        ← Task 2: Visualization dashboard → dashboard.png
├── index.html          ← Interactive HTML report (opens in any browser)
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
1. Go to: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset
2. Click **Download** and extract the ZIP
3. Rename the CSV to **`bank.csv`** and place it in this folder

---

## Run

### Task 2 — Data Visualization Dashboard
```bash
python3 visualize.py
```
Generates **`dashboard.png`** — a 10-chart data insights dashboard covering:
- Target class distribution
- Deposit rate by job type
- Age & account balance distributions
- Call duration by outcome
- Monthly deposit rate & contact volume
- Previous campaign outcome analysis
- Contact method & education breakdowns
- Feature correlation heatmap

### Task 1 — Classification Model Pipeline
```bash
python3 analysis.py
```
Generates **`model_dashboard.png`** — model performance report including:
- Grouped bar chart comparing all metrics across 4 models
- 5-fold cross-validated ROC-AUC chart
- ROC curves for all models
- Confusion matrix (best model)
- Feature importance (Random Forest)
- Full metrics summary table

### View Report
Open `index.html` in any browser — it automatically loads the two generated PNGs.

---

## Results Summary

| Model | Accuracy | F1 Score | ROC-AUC | CV AUC |
|---|---|---|---|---|
| Logistic Regression | 79.18% | 77.17% | 0.8814 | 0.8798 ± 0.0064 |
| Decision Tree | 81.37% | 80.52% | 0.8775 | 0.8763 ± 0.0085 |
| Random Forest | 84.24% | 83.79% | 0.9081 | 0.9056 ± 0.0066 |
| **Gradient Boosting ★** | **84.33%** | **83.93%** | **0.9181** | **0.9143 ± 0.0068** |

**Best Model:** Gradient Boosting — ROC-AUC = 0.9181, F1 = 0.8393

---

## Feature Engineering

Four new features were engineered:

| Feature | Description |
|---|---|
| `has_previous` | 1 if client was contacted in a previous campaign |
| `contacted_before` | 1 if `pdays ≠ -1` (was reached in campaign window) |
| `balance_log` | Log-transformed account balance (reduces right-skew) |
| `age_group` | Age binned into 5 ordinal groups |

---

## Key Insights

- 📞 **Call duration** is the single strongest predictor of subscription
- 🔄 Clients with **previous campaign success** convert at >60%
- 👴 **Retired clients** and **students** show highest subscription rates
- 📅 **March, September, October** have highest conversion months
- 📱 **Cellular contact** outperforms telephone (54.3% vs 50.4%)

---

*Dataset: Kaggle (janiobachmann) | Models: scikit-learn | Python 3.12*
