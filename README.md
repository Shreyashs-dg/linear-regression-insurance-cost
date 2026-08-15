# Medical Insurance Cost Prediction — Linear Regression

Predicting individual medical insurance charges from demographic and health features using Linear Regression, with Ridge and Lasso comparison and full diagnostic evaluation.

## Problem Statement

Insurance providers need to estimate expected medical costs for a policyholder based on measurable factors like age, BMI, and smoking status, in order to price policies appropriately. This project builds and evaluates a regression model to predict `charges` from six input features.

## Dataset

- **Source:** Medical Cost Personal Dataset (`insurance.csv`)
- **Size:** 1,338 rows (1,337 after removing 1 duplicate), 7 columns
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Target:** `charges` (USD)

## Key EDA Insight

**Smoker status is by far the strongest driver of insurance charges.** Smokers pay **~4x more** on average than non-smokers:

| Group | Average Charges |
|---|---|
| Non-smoker | $8,434 |
| Smoker | $32,050 |

This relationship doesn't appear in a standard numeric correlation heatmap at all, since `smoker` is categorical — it only surfaces through a groupby/boxplot comparison against the target.

The target variable itself is heavily right-skewed (skew ≈ 1.51), with a long tail of high-cost cases almost entirely explained by smoking status.

## Methodology

1. **EDA** — distribution analysis, missing value/duplicate checks, univariate and bivariate analysis against the target
2. **Preprocessing** — `ColumnTransformer` with `StandardScaler` on numeric features and `OneHotEncoder(drop="first", handle_unknown="ignore")` on categorical features, wrapped in a full `sklearn.Pipeline`
3. **Modeling** — Linear Regression, Ridge, and Lasso compared, with `alpha` tuned via `GridSearchCV` (5-fold cross-validation)
4. **Target transform experiment** — tested wrapping the model in `TransformedTargetRegressor` with a `log1p` transform to address the target's skew
5. **Diagnostics** — residual analysis, VIF (multicollinearity), Cook's Distance (influential points)

## Results

| Model | RMSE | R² |
|---|---|---|
| **Linear Regression** | **5,956** | **0.807** |
| Ridge (α=1, tuned) | 5,972 | 0.806 |
| Lasso (α=50, tuned) | 6,028 | 0.802 |

Plain Linear Regression performed best. Ridge and Lasso didn't meaningfully improve results — confirmed via VIF analysis (all features scored below 2, well under the multicollinearity concern threshold of 5–10), so there was little for regularization to correct.

### Top Coefficients (standardized)

| Feature | Coefficient |
|---|---|
| `smoker_yes` | **+$22,941** |
| `age` | +$3,468 |
| `bmi` | +$1,923 |
| `children` | +$637 |

Holding all else constant, being a smoker adds ~$22,941 to predicted charges — roughly **6x** the size of the next-largest effect.

## Key Trade-off: Log-Transforming the Target

Log-transforming `charges` fixed its skew almost completely (1.51 → -0.10), but the effect on model performance was mixed rather than uniformly positive:

| Metric | Without log-transform | With log-transform |
|---|---|---|
| RMSE | **5,956** | 7,197 |
| MAE | 4,177 | **3,756** |
| R² | **0.807** | 0.718 |

The log-transformed model is better at the *typical* case (lower MAE) but worse at the *expensive* case (higher RMSE), since squared-error metrics penalize misses on the largest (mostly high-BMI-smoker) charges much more heavily. **This project ships the untransformed model**, prioritizing overall accuracy (RMSE/R²) — but the log-transformed version would be the better choice if the business priority were typical-case accuracy over worst-case accuracy.

## Diagnostics

| Assumption | Status | Evidence |
|---|---|---|
| Linearity | Partially violated | Residual pattern suggests a missing interaction effect |
| Homoscedasticity | Violated | corr(\|residual\|, fitted) = 0.64 |
| Normality of residuals | Violated | Residual skew = 1.24 |
| No multicollinearity | Holds | All VIF < 2 |

Cook's Distance flagged 67 of 1,069 training points above the `4/n` threshold, but the maximum distance (0.026) is far below the conventional "clearly influential" cutoff of 1.0 — no single point distorts the model; the flagged points are simply the genuine high-cost smoker cases identified in the EDA.

## Limitations & Future Work

- The model doesn't capture the **`smoker × bmi` interaction** — high-BMI smokers cost disproportionately more than the additive combination of the two effects alone, which is the direct cause of the heteroscedasticity found in diagnostics
- Adding this interaction term explicitly, or moving to a model that captures interactions natively (Random Forest, Gradient Boosting — planned later in this study series), is the clear next step
- Residual normality remains imperfect even after diagnostics — worth revisiting alongside the interaction term fix

## Tech Stack

Python · pandas · numpy · scikit-learn · matplotlib · seaborn · statsmodels

## How to Run

```bash
git clone <repo-url>
cd linear-regression-insurance-cost
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/insurance_analysis.ipynb
```

## Repo Structure

```
linear-regression-insurance-cost/
├── data/
│   └── insurance.csv
├── notebooks/
│   └── insurance_analysis.ipynb
├── images/
├── requirements.txt
├── .gitignore
└── README.md
```