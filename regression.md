# Regression — Predicting Numeric Targets

This lesson introduces **regression**: building models that predict **continuous** values (price, demand, time, score).

## What you’ll learn
- Frame a problem as regression
- Prepare data (splits, preprocessing)
- Train baseline and regularized models
- Evaluate with the right metrics
- Avoid common pitfalls (leakage, overfitting)

---

## When to use regression
Use regression when the target is numeric:
- House price, rainfall amount, ride duration, sales volume, exam score.

---

## Quick workflow
1. Define target `y` and features `X`
2. **Split** data → train/validation/test
3. Build a **baseline** (mean/median or simple Linear Regression)
4. Add preprocessing (scaling, one-hot encoding)
5. Try stronger models (Ridge/Lasso, Trees/Ensembles)
6. Compare metrics → pick model
7. Inspect errors, iterate

---

## Metrics (cheat sheet)
- **MAE** (Mean Absolute Error): average absolute difference. Easy to interpret.
- **RMSE** (Root Mean Squared Error): penalizes large errors more.
- **R²** (Coefficient of Determination): proportion of variance explained (closer to 1 is better).

> Start with **MAE** and **R²**. Report at least two metrics.

---

## Common models
- **Linear Regression** — fast baseline; assumes linear relation
- **Ridge / Lasso** — linear + regularization (handles multicollinearity / feature selection)
- **Polynomial Regression** — linear on expanded features (can overfit)
- **Tree-based** — Decision Tree, Random Forest, Gradient Boosting (often strong defaults)
- **kNN / SVR** — useful in some settings, sensitive to scaling

---

## Data prep tips
- **Numeric**: scale if model needs it (LR, SVR, kNN).
- **Categorical**: one-hot encode.
- **Outliers**: inspect; cap or transform if needed.
- **Leakage**: no target info, future info, or test stats in training pipeline.

---

## Minimal example (scikit-learn)

```python
# pip install pandas scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# 1) Load your data
df = pd.read_csv("data.csv")  # columns: features..., target

y = df["target"]
X = df.drop(columns=["target"])

num_cols = X.select_dtypes(include=["number"]).columns
cat_cols = X.select_dtypes(exclude=["number"]).columns

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Preprocess + model
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

model = Ridge(alpha=1.0, random_state=42)
pipe = Pipeline(steps=[("pre", pre), ("model", model)])

# 4) Train & evaluate
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"MAE: {mae:.3f}  |  R²: {r2:.3f}")
