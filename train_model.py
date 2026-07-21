import pandas as pd
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import os

df = pd.read_csv("credit_risk_dataset.csv")

FEATURES = [
    "person_income",
    "loan_amnt",
    "person_emp_length",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "loan_status"
]

df = df[FEATURES].dropna()

df = df[df["person_emp_length"] <= 60]
df = df[df["person_income"] <= 4_000_000]

print("Dataset loaded:", df.shape)

df["loan_status"] = 1 - df["loan_status"]

def apply_bank_rules(row):
    income  = row["person_income"]
    loan    = row["loan_amnt"]
    rate    = row["loan_int_rate"]
    emp     = row["person_emp_length"]
    credit  = row["cb_person_cred_hist_length"]

    monthly = (loan * rate / 100) / 12
    dti     = loan    / (income + 1)
    pti     = monthly / (income / 12 + 1)

    # Hard REJECT rules
    if dti   > 0.43:  return 0
    if pti   > 0.35:  return 0
    if rate  > 22:    return 0
    if emp   < 1.0:   return 0
    if credit < 2:    return 0
    if loan  > income: return 0

    # Hard APPROVE — clearly strong profile
    if dti <= 0.2 and emp >= 5 and credit >= 5 and rate <= 15:
        return 1

    return row["loan_status"]

print("\nApplying bank rules to clean training labels...")
df["loan_status"] = df.apply(apply_bank_rules, axis=1)

df["dti_ratio"]              = df["loan_amnt"]         / (df["person_income"] + 1)
df["monthly_payment"]        = (df["loan_amnt"] * df["loan_int_rate"] / 100) / 12
df["pti_ratio"]              = df["monthly_payment"]   / (df["person_income"] / 12 + 1)
df["income_loan_ratio"]      = df["person_income"]     / (df["loan_amnt"] + 1)
df["interest_income_burden"] = df["loan_int_rate"] * df["loan_amnt"] / (df["person_income"] + 1)
df["emp_credit_ratio"]       = df["person_emp_length"] / (df["cb_person_cred_hist_length"] + 1)
df["annual_interest_cost"]   = df["loan_amnt"] * df["loan_int_rate"] / 100
df["loan_per_emp_year"]      = df["loan_amnt"] / (df["person_emp_length"] + 1)

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

FEATURE_NAMES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sc, y_train)

print("\nTraining XGBoost...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_bal, y_train_bal)
print("Training completed!")

probs = model.predict_proba(X_test_sc)[:, 1]
THRESHOLD = 0.50

y_pred = (probs >= THRESHOLD).astype(int)

print(f"\n--- XGBoost Model Evaluation ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC   : {roc_auc_score(y_test, probs):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save artifacts
os.makedirs("models", exist_ok=True)
pickle.dump(model,         open("models/loan_model.pkl",    "wb"))
pickle.dump(scaler,        open("models/scaler.pkl",         "wb"))
pickle.dump(THRESHOLD,     open("models/threshold.pkl",      "wb"))
pickle.dump(FEATURE_NAMES, open("models/feature_names.pkl",  "wb"))

print("\n XGBoost Model files saved successfully in 'models/' directory!")