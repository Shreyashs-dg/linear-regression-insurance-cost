# Medical Insurance Cost Prediction — Linear Regression

A machine learning project that predicts individual medical insurance charges using Linear Regression, with Ridge and Lasso regularization comparison, target-transformation experiments, and regression diagnostics.

## Problem Statement

Insurance providers need to estimate expected medical costs for policyholders based on measurable demographic and health-related factors.

This project builds and evaluates regression models to predict medical insurance charges using:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The target variable is `charges`.

---

## Dataset

**Dataset:** Medical Cost Personal Dataset (`insurance.csv`)

- Original size: 1,338 rows × 7 columns
- Final size: 1,337 rows × 7 columns
- Duplicate rows removed: 1

### Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the policyholder |
| `sex` | Categorical | Sex of the policyholder |
| `bmi` | Numerical | Body Mass Index |
| `children` | Numerical | Number of children/dependents |
| `smoker` | Categorical | Smoking status |
| `region` | Categorical | Residential region |

### Target

`charges` — individual medical insurance charges in USD.

---

## Exploratory Data Analysis

The project performs:

- Dataset structure analysis
- Missing-value checks
- Duplicate detection
- Numerical statistical summaries
- Categorical analysis
- Distribution analysis
- Skewness analysis
- IQR-based outlier detection
- Univariate analysis
- Bivariate analysis
- Correlation analysis

### Key EDA Findings

Smoking status is the strongest visible driver of insurance charges.

Average charges in the dataset were approximately:

| Group | Average Charges |
|---|---:|
| Non-smoker | $8,434 |
| Smoker | $32,050 |

Smokers therefore have substantially higher average insurance charges than non-smokers.

The target variable `charges` is also strongly right-skewed:

```text
Skewness ≈ 1.51