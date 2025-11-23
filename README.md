# Home Credit Default Risk Prediction

A machine learning project to predict credit default risk using the Home Credit dataset. This project includes comprehensive exploratory data analysis (EDA), feature engineering, model training with hyperparameter optimization, and a deployed web application.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![Kaggle](https://img.shields.io/badge/Data-Kaggle-20BEFF)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Live Demo

🚀 **Try the model:** [Home Credit Default Predictor](https://huggingface.co/spaces/YOUR_USERNAME/Home-Credit-Default-Risk)

> Enter a Client ID to get the predicted default probability and risk classification.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
- [How to Run the Notebook](#how-to-run-the-notebook)
- [Project Files](#project-files)
- [Author](#author)

## Overview

This project addresses the challenge of predicting whether a client will default on their loan. Using data from Home Credit, a financial services provider, we build a machine learning pipeline that:

1. Analyzes multiple related datasets (applications, bureau records, previous applications, etc.)
2. Engineers meaningful features from raw data
3. Trains and optimizes gradient boosting models
4. Achieves competitive performance on the Kaggle leaderboard

**Key Achievement:** The final LightGBM model achieves a **ROC AUC of 0.7667** and **Gini coefficient of 0.5335** on the validation set.

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
  - Categorical: Contract type, gender, education, occupation, housing type, etc.
  - Numerical: Income, credit amount, age, external scores
- **Bivariate/Multivariate Analysis:** Correlation with target variable
- **Supporting Dataset Analysis:** Bureau, previous applications, installments, credit card balances

**Key Findings:**
- `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` are the strongest predictors
- Younger clients (20s-30s) show higher default rates
- Cash loans have higher default rates (~8.35%) than revolving loans (~5.48%)
- Lower education correlates with higher credit risk (11% for lower secondary vs 5.4% for higher education)
- Clients on maternity leave or unemployed show highest default rates (~40%)
- Housing renters and those living with parents show higher risk

### Feature Engineering

#### Basic Application-Level Features
```python
CREDIT_INCOME_PERCENT      # Credit / Income ratio
ANNUITY_INCOME_PERCENT     # Annuity / Income ratio
CREDIT_ANNUITY_PERCENT     # Credit / Annuity ratio
CREDIT_TERM                # Loan term calculation
AGE_YEARS                  # Age in years
BIRTH_EMPLOYED_PERCENT     # Employment duration ratio
FAMILY_CNT_INCOME_PERCENT  # Income per family member
AGE_LOAN_FINISH            # Age when loan ends
```

#### Heavy Feature Engineering

**Bureau Features (Time-Windowed: All, 2Y, 1Y, 6M, 3M):**
- Debt ratios and overdue amounts
- Credit duration statistics
- Active/closed credit status shares
- Credit count and type diversity

**Installments Features (Time-Windowed):**
- Payment delays (DPD 30/60/90 flags)
- Underpayment/overpayment percentages
- On-time full payment rates
- Payment coverage ratios

**Previous Applications:**
- Application vs credit amount ratios
- Approved/refused credit statistics
- Down payment ratios
- Credit-to-annuity ratios

**Credit Card & POS Features:**
- Utilization ratios
- Late payment flags
- Drawing patterns
- Installment statistics

### Model Training

**Models Evaluated:**

| Model | ROC AUC | Gini |
|-------|---------|------|
| Logistic Regression (Baseline) | 0.7395 | 0.4790 |
| Logistic Regression (L2) | 0.7395 | 0.4790 |
| Random Forest | 0.7420 | 0.4839 |
| XGBoost | 0.7558 | 0.5116 |
| CatBoost | 0.7612 | 0.5223 |
| LightGBM (Baseline) | 0.7380 | 0.4761 |
| **LightGBM (Optimized + Heavy FE)** | **0.7667** | **0.5335** |

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
| Matthews Correlation | 0.0494 |

### Improvement Over Baseline

| Metric | Baseline (Simple FE) | Optimized (Heavy FE) | Improvement |
|--------|---------------------|----------------------|-------------|
| Gini | 0.4761 | 0.5335 | **+12.05%** |
| ROC AUC | 0.7380 | 0.7667 | **+3.89%** |

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

## How to Run the Notebook

### Prerequisites

- Python 3.10+
- Google Colab (recommended) or Jupyter Notebook
- Access to [Home Credit dataset on Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)

### Steps

1. Download the dataset from Kaggle and upload to Google Drive

2. Open the notebook in Google Colab:
   - Upload `K7_ML_Final_Project_PhamHoangNam.ipynb` to Colab
   - Or use: `File > Open notebook > Upload`

3. Mount Google Drive and update data paths:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   # Update paths to match your Drive location
   df_train = pd.read_csv('/content/drive/MyDrive/home_credit_data/application_train.csv')
   ```

4. Install required packages:
   ```python
   !pip install catboost lightgbm xgboost
   ```

5. Run all cells sequentially

### Required Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
joblib
```

## Project Files

| File | Description |
|------|-------------|
| `K7_ML_Final_Project_PhamHoangNam.ipynb` | Complete Jupyter notebook with EDA, feature engineering, and model training |

## Key Visualizations

The notebook includes:
- Target distribution (pie chart & bar chart)
- Categorical feature analysis with default rates
- Numerical feature distributions by target
- Correlation heatmaps
- ROC curves comparison
- Confusion matrices
- Feature importance plots

## Conclusions

1. **External scores are critical:** The three `EXT_SOURCE` features dominate feature importance, suggesting external credit bureau data is essential for default prediction.

2. **Heavy feature engineering pays off:** Time-windowed aggregations from bureau and installment data improved Gini by 12%.

3. **LightGBM outperforms:** With proper tuning, LightGBM achieved the best results among all tested models.

4. **Class imbalance matters:** The 92/8 split requires careful handling - AUC and Gini are more appropriate metrics than accuracy.

## Author

**Pham Hoang Nam**

Machine Learning Final Project - K7

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Home Credit](https://www.homecredit.net/) for providing the dataset
- [Kaggle](https://www.kaggle.com/c/home-credit-default-risk) for hosting the competition
- The open-source community for the amazing ML libraries
