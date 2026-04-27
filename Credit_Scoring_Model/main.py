import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# =====================================
# STEP 1: LOAD DATASET
# =====================================

data = pd.read_csv("data/credit_data.csv")

print("Dataset Loaded Successfully")
print("Shape:", data.shape)

# =====================================
# STEP 2: DATA CLEANING
# =====================================

# Remove duplicates
data = data.drop_duplicates()

# Remove invalid values
data = data[data["person_age"] >= 18]
data = data[data["person_income"] > 0]
data = data[data["loan_amnt"] >= 0]

# Fill missing values
data.ffill(inplace=True)

print("\nAfter Cleaning Shape:", data.shape)

# =====================================
# STEP 3: FEATURE ENGINEERING
# =====================================

# Loan / Income Ratio
data["loan_percent_income"] = data["loan_amnt"] / data["person_income"]

# =====================================
# STEP 4: ENCODING
# =====================================

# Convert Y/N to 1/0
data["cb_person_default_on_file"] = data["cb_person_default_on_file"].map({
    "Y": 1,
    "N": 0
})

# One-hot encode categorical values
data = pd.get_dummies(data, drop_first=True)

# =====================================
# STEP 5: FEATURES & TARGET
# =====================================

if "loan_status" not in data.columns:
    raise ValueError("loan_status column not found")

X = data.drop("loan_status", axis=1)
y = data["loan_status"]

print("\nTarget Distribution:")
print(y.value_counts())

# =====================================
# STEP 6: TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# STEP 7: TRAIN MODEL (SMALLER MODEL)
# =====================================

model = RandomForestClassifier(
    n_estimators=50,          # reduced from 200
    max_depth=10,            # smaller size
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================================
# STEP 8: MODEL EVALUATION
# =====================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

approval_prob_test = y_prob[:, 0]
risk_prob_test = y_prob[:, 1]

print("\n===== MODEL RESULTS =====")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, risk_prob_test), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =====================================
# STEP 9: SAVE MODEL (COMPRESSED)
# =====================================

joblib.dump(model, "credit_model.pkl", compress=3)
joblib.dump(list(X.columns), "model_columns.pkl")

print("\nModel Saved Successfully!")
print("Files Created:")
print("- credit_model.pkl")
print("- model_columns.pkl")

# =====================================
# STEP 10: SAMPLE TEST PREDICTION
# =====================================

print("\n====== TEST CUSTOMER PREDICTION ======")

# Sample customer
age = 30
income = 80000
loan = 10000
emp = 5
credit_hist = 6
default = "N"

columns = list(X.columns)
input_data = dict.fromkeys(columns, 0)

# Numeric Features
if "person_age" in columns:
    input_data["person_age"] = age

if "person_income" in columns:
    input_data["person_income"] = income

if "loan_amnt" in columns:
    input_data["loan_amnt"] = loan

if "person_emp_length" in columns:
    input_data["person_emp_length"] = emp

if "cb_person_cred_hist_length" in columns:
    input_data["cb_person_cred_hist_length"] = credit_hist

if "cb_person_default_on_file" in columns:
    input_data["cb_person_default_on_file"] = 1 if default == "Y" else 0

if "loan_percent_income" in columns:
    input_data["loan_percent_income"] = loan / income if income > 0 else 0

# Convert to DataFrame
new_customer = pd.DataFrame([input_data])

# Prediction
result = model.predict(new_customer)[0]
probs = model.predict_proba(new_customer)[0]

approval_prob = probs[0]
risk_prob = probs[1]

print("\n===== RESULT =====")

if result == 0:
    print("✅ Good Customer (Loan Approved)")
else:
    print("❌ Risky Customer (Loan Rejected)")

print("Approval Probability :", round(approval_prob * 100, 2), "%")
print("Risk Probability     :", round(risk_prob * 100, 2), "%")