import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Load data
train = pd.read_csv("train.csv", index_col="id")
test = pd.read_csv("test.csv", index_col="id")

# Encode 'Sex' manually
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

# Feature engineering
for df in [train, test]:
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Heart_Temp_Ratio'] = df['Heart_Rate'] / df['Body_Temp']
    df['Effort'] = df['Duration'] * df['Heart_Rate']

# Define features and target
X = train.drop(columns=["Calories"])
y = np.log1p(train["Calories"])  # log-transform target
X_test = test.copy()

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)
model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_val, y_val)],verbose=False)

# Predict and evaluate
y_val_preds = model.predict(X_val)
rmsle = np.sqrt(mean_squared_error(y_val, y_val_preds))
print(f"Validation RMSLE (log1p space): {rmsle:.5f}")

# Final test predictions (invert log1p)
test_preds = np.expm1(model.predict(X_test))
test_preds = np.maximum(0, test_preds)

# Submission
submission = pd.DataFrame({
    'id': test.index,
    'Calories_Burned': test_preds
})
submission.to_csv("submission.csv", index=False)
