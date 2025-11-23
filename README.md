# Home Credit Default Risk Prediction

A machine learning project to predict credit default risk using the Home Credit dataset. This project includes comprehensive exploratory data analysis (EDA), feature engineering, model training with hyperparameter optimization, and deployment as a web application.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)
![Docker](https://img.shields.io/badge/Deploy-Docker-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Locally](#running-locally)
  - [Docker Deployment](#docker-deployment)
  - [API Endpoints](#api-endpoints)
- [Web Interface](#web-interface)
- [Model Details](#model-details)
- [Author](#author)
- [License](#license)

## Overview

This project addresses the challenge of predicting whether a client will default on their loan. Using data from Home Credit, a financial services provider, we build a machine learning pipeline that:

1. Analyzes multiple related datasets (applications, bureau records, previous applications, etc.)
2. Engineers meaningful features from raw data
3. Trains and optimizes gradient boosting models
4. Deploys an interactive web application for predictions

**Key Achievement:** The final LightGBM model achieves a **ROC AUC of 0.7667** and **Gini coefficient of 0.5335** on the validation set.

## Project Structure

```
Home-Credit-Default-Risk/
├── app.py                              # FastAPI web application
├── Dockerfile                          # Docker configuration
├── requirements.txt                    # Python dependencies
├── preprocessor.pkl                    # Fitted preprocessing pipeline
├── lightgbm_randomsearch_heavy_3.pkl   # Trained LightGBM model
├── test_all.parquet                    # Preprocessed test data
├── static/
│   ├── index.html                      # Web interface
│   └── homecredit_logo.png             # Logo asset
└── README.md                           # This file
```

## Dataset

The project uses data from the [Home Credit Default Risk Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk), which includes:

| Dataset | Description |
|---------|-------------|
| `application_train.csv` | Main training data with client information |
| `application_test.csv` | Test data for predictions |
| `bureau.csv` | Credit history from other institutions |
| `bureau_balance.csv` | Monthly balance of bureau credits |
| `previous_application.csv` | Previous loan applications |
| `POS_CASH_balance.csv` | Monthly POS/cash loan balances |
| `credit_card_balance.csv` | Monthly credit card balances |
| `installments_payments.csv` | Payment history for loans |

**Target Variable:** `TARGET` (1 = client defaulted, 0 = client repaid)

**Class Imbalance:** ~92% repaid vs ~8% defaulted

## Methodology

### Exploratory Data Analysis

Comprehensive EDA was performed including:

- **Missing Value Analysis:** Identified and handled columns with high missing percentages
- **Outlier Detection:** Used IQR method to detect and visualize outliers
- **Target Distribution:** Analyzed class imbalance (92% repaid vs 8% default)
- **Univariate Analysis:**
  - Categorical: Contract type, gender, education, occupation, etc.
  - Numerical: Income, credit amount, age, external scores
- **Correlation Analysis:** Top 20 features correlated with target

**Key Findings:**
- `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` are strongest predictors
- Younger clients (20s-30s) show higher default rates
- Cash loans have higher default rates than revolving loans
- Lower education correlates with higher credit risk

### Feature Engineering

#### Basic Features
```python
- CREDIT_INCOME_PERCENT     # Credit / Income ratio
- ANNUITY_INCOME_PERCENT    # Annuity / Income ratio
- CREDIT_ANNUITY_PERCENT    # Credit / Annuity ratio
- CREDIT_TERM               # Loan term calculation
- AGE_YEARS                 # Age in years
- BIRTH_EMPLOYED_PERCENT    # Employment duration ratio
```

#### Heavy Feature Engineering

**Bureau Features (Windowed):**
- Multiple time windows: All, 2Y, 1Y, 6M, 3M
- Debt ratios, overdue amounts, credit duration
- Active/closed credit status shares

**Installments Features:**
- Payment delays (DPD 30/60/90)
- Underpayment/overpayment percentages
- On-time payment rates
- Time-windowed aggregations

**Previous Applications:**
- Application vs credit amount ratios
- Approved/refused credit statistics
- Down payment ratios

**Credit Card & POS Features:**
- Utilization ratios
- Late payment flags
- Drawing patterns

### Model Training

**Models Evaluated:**

| Model | ROC AUC | Gini |
|-------|---------|------|
| Logistic Regression | 0.7395 | 0.4790 |
| Random Forest | 0.7420 | 0.4839 |
| XGBoost | 0.7558 | 0.5116 |
| CatBoost | 0.7612 | 0.5223 |
| **LightGBM (Optimized)** | **0.7667** | **0.5335** |

### Hyperparameter Optimization

Used `RandomizedSearchCV` with 5-fold stratified cross-validation:

```python
param_dist = {
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 6, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_lambda": [0, 1, 5],
    "reg_alpha": [0, 1, 5]
}
```

**Cross-Validation Results:**
- Mean AUC: 0.7667 ± 0.0015
- Mean KS Statistic: 0.4042 ± 0.0026
- Mean PSI: 0.0002 ± 0.0001 (excellent stability)

## Results

### Final Model Performance

| Metric | Value |
|--------|-------|
| ROC AUC | 0.7667 |
| Gini Coefficient | 0.5335 |
| Precision | 0.2683 |
| Recall | 0.0161 |
| F1 Score | 0.0303 |
| Balanced Accuracy | 0.5080 |

### Improvement Over Baseline

| Metric | Baseline (Simple FE) | Optimized (Heavy FE) | Improvement |
|--------|---------------------|----------------------|-------------|
| Gini | 0.4761 | 0.5335 | +12.05% |
| ROC AUC | 0.7380 | 0.7667 | +3.89% |

## Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements

```
fastapi
uvicorn
pandas
numpy
scikit-learn==1.6.1
lightgbm
joblib
pyarrow
```

## Usage

### Running Locally

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the application:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

4. Open browser at `http://localhost:7860`

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t home-credit-prediction .
```

2. Run the container:
```bash
docker run -p 7860:7860 home-credit-prediction
```

3. Access at `http://localhost:7860`

### Hugging Face Spaces

This project is configured for deployment on Hugging Face Spaces using Docker SDK.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/sample-ids` | GET | Get random sample client IDs |
| `/ids` | GET | Browse all IDs with filtering/sorting |
| `/predict-id` | POST | Predict default probability |

#### Example API Call

```bash
curl -X POST "http://localhost:7860/predict-id" \
     -H "Content-Type: application/json" \
     -d '{"SK_ID_CURR": 100001}'
```

**Response:**
```json
{
    "SK_ID_CURR": 100001,
    "default_probability": 0.0834
}
```

## Web Interface

The web application provides:

- **Client ID Input:** Enter or select a client ID
- **Sample Chips:** Quick access to random client IDs
- **Browse Modal:** Search, filter by risk level, and sort clients
- **Risk Classification:**
  - 🟢 **LOW** (< 15%): Low default risk
  - 🟡 **MEDIUM** (15-35%): Moderate default risk
  - 🔴 **HIGH** (> 35%): High default risk

## Model Details

### Preprocessing Pipeline

```python
# Numerical features: Median imputation + RobustScaler
# Categorical features: Most frequent imputation + OneHotEncoder
```

### Final Model Configuration

```python
LGBMClassifier(
    device="gpu",
    boosting_type="gbdt",
    objective="binary",
    metric="auc",
    n_estimators=5000,
    # + optimized hyperparameters from RandomizedSearchCV
)
```

### Feature Importance (Top 10)

1. `EXT_SOURCE_2` - External source score 2
2. `EXT_SOURCE_3` - External source score 3
3. `EXT_SOURCE_1` - External source score 1
4. `DAYS_BIRTH` - Client age in days
5. `DAYS_EMPLOYED` - Employment duration
6. `AMT_CREDIT` - Credit amount
7. `AMT_ANNUITY` - Annuity amount
8. `DAYS_ID_PUBLISH` - Days since ID publish
9. `AMT_GOODS_PRICE` - Goods price
10. `CREDIT_INCOME_PERCENT` - Credit to income ratio

## Author

**Pham Hoang Nam**

Machine Learning Final Project - K7

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Home Credit](https://www.homecredit.net/) for providing the dataset
- [Kaggle](https://www.kaggle.com/c/home-credit-default-risk) for hosting the competition
- The open-source community for the amazing tools and libraries
