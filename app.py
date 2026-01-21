import streamlit as st
import joblib
import numpy as np

# تحميل الموديل والـ scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Eligibility Prediction")
st.write("Enter applicant details to check loan eligibility")

# ===== Inputs =====
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History", [0, 1])

# ===== Encoding =====
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)

# ===== Prepare Data (IMPORTANT: same order as training) =====
data = np.array([[
    gender,
    married,
    dependents,
    education,
    self_employed,
    income,
    loan_amount,
    credit_history
]])

# ===== Prediction =====
if st.button("Predict Loan Status"):
    try:
        data = scaler.transform(data)
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")
    except Exception as e:
        st.error(f"Error: {e}")



