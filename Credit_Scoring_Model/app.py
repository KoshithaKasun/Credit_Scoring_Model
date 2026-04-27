import streamlit as st
import pandas as pd
import joblib
import time

# PAGE CONFIG

st.set_page_config(
    page_title="Credit Scoring System",
    layout="wide"
)

# BACKGROUND + STYLE

st.markdown("""
<style>

.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.title {
    text-align:center;
    font-size:65px;
    font-weight:bold;
    color:#FFFFFF;
}

.subtitle {
    text-align:center;
    font-size:20px;
    color:#cccccc;
    margin-bottom:30px;
}

[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# LOAD MODEL

model = joblib.load("credit_model.pkl")
columns = joblib.load("model_columns.pkl")

#  AI EXPLANATION ENGINE

def explain_decision(age, income, loan, risk_prob):
    reasons = []

    if age > 50:
        reasons.append("Higher age increases financial risk")

    if income > 0:
        ratio = loan / income
        if ratio > 0.5:
            reasons.append("Loan amount is high compared to income")

    if risk_prob > 0.75:
        reasons.append("Model detected very high default probability")
    elif risk_prob > 0.5:
        reasons.append("Moderate financial instability detected")

    if not reasons:
        reasons.append("Strong financial profile with low risk indicators")

    return reasons

# HEADER

st.markdown('<div class="title"> Credit Scoring AI System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Bank-Level Loan Approval Prediction Engine</div>', unsafe_allow_html=True)

# INPUT UI

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Info")

    age = st.text_input("Age", placeholder="Enter age")
    income = st.text_input("Annual Income", placeholder="Enter annual income")
    loan = st.text_input("Loan Amount", placeholder="Enter loan amount")

    emp = st.number_input("Employment Years", 0, 50, value=None, placeholder="Enter years")
    credit_hist = st.number_input("Credit History", 0, 50, value=None, placeholder="Enter history")

with col2:
    st.subheader("🏦 Financial Info")

    default = st.selectbox("Previous Default?", ["No", "Yes"])
    home = st.selectbox("Home Ownership", ["Own", "Rent", "Mortgage", "Other"])
    purpose = st.selectbox("Loan Purpose", ["Personal", "Education", "Medical", "Venture", "Home Improvement", "Other"])

# VALIDATION

if age == "" or income == "" or loan == "" or emp is None or credit_hist is None:
    st.markdown("---")
    st.info("ℹ️ Please fill all fields to continue.")
    st.stop()

try:
    age = int(age)
    income = float(income)
    loan = float(loan)
except:
    st.error("⚠️ Please enter valid numbers for Age, Income, and Loan Amount")
    st.stop()

# AGE RULE ENGINE

if age > 100:
    st.error("❌ Invalid Age: Can not exceed 100")
    st.stop()

if age > 60:
    st.error("❌ Auto Rejected: Age above 60 is not eligible for a loan")
    st.stop()

# BUTTON

st.markdown("---")

if st.button("🚀 Run Prediction"):

    with st.spinner("🤖 AI analyzing financial risk..."):
        time.sleep(2)

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    # DATA PREP

    input_data = dict.fromkeys(columns, 0)

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
        input_data["cb_person_default_on_file"] = 1 if default == "Yes" else 0

    if "loan_percent_income" in columns:
        input_data["loan_percent_income"] = loan / income if income > 0 else 0

    home_col = f"person_home_ownership_{home}"
    if home_col in columns:
        input_data[home_col] = 1

    purpose_col = f"loan_intent_{purpose}"
    if purpose_col in columns:
        input_data[purpose_col] = 1

    df = pd.DataFrame([input_data])

    # PREDICTION

    result = model.predict(df)[0]
    probs = model.predict_proba(df)[0]

    approval_prob = probs[0]
    risk_prob = probs[1]

    # RESULT UI

    st.markdown(" 📊 AI Analysis Result")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Approval %", f"{approval_prob*100:.2f}%")

    with colB:
        st.metric("Risk %", f"{risk_prob*100:.2f}%")

    with colC:
        if result == 0:
            st.success("APPROVED")
        else:
            st.error("REJECTED")

    # RISK BAR

    st.markdown(" 📈 Risk Meter")

    bar = st.progress(0)

    for i in range(int(risk_prob * 100)):
        time.sleep(0.01)
        bar.progress(i + 1)

    # FINAL STATUS

    if risk_prob >= 0.75:
        st.error("🔴 HIGH RISK CUSTOMER")
    elif risk_prob >= 0.50:
        st.warning("🟡 MEDIUM RISK CUSTOMER")
    else:
        st.success("🟢 LOW RISK CUSTOMER")

    # EXPLANATION OUTPUT

    st.markdown(" 🧠 Explanation")

    reasons = explain_decision(age, income, loan, risk_prob)

    if result == 0:
        st.success("Why approved:")
    else:
        st.error("Why rejected:")

    for r in reasons:
        st.write("• " + r)

    # DEBUG

    with st.expander("🔍 Debug Info"):
        st.write("Prediction:", result)
        st.write("Probabilities:", probs)