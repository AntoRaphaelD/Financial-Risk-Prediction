import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("savings_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
features = joblib.load("model_features.pkl")

st.title("Employee Spending & Savings Prediction Dashboard")

st.write("Enter employee financial details to predict potential savings.")

# ---------------------------
# User Inputs
# ---------------------------

income = st.number_input("Monthly Income", 10000, 500000, 50000)
age = st.slider("Age", 20, 60, 30)
dependents = st.slider("Dependents", 0, 5, 1)

st.subheader("Expense Details")

rent = st.number_input("Rent", 0, 100000, 15000)
groceries = st.number_input("Groceries", 0, 30000, 5000)
transport = st.number_input("Transport", 0, 20000, 3000)
eating_out = st.number_input("Eating Out", 0, 20000, 2000)
entertainment = st.number_input("Entertainment", 0, 20000, 2000)
utilities = st.number_input("Utilities", 0, 15000, 3000)
healthcare = st.number_input("Healthcare", 0, 15000, 2000)
education = st.number_input("Education", 0, 20000, 1000)
misc = st.number_input("Miscellaneous", 0, 20000, 2000)

# ---------------------------
# Create Input DataFrame
# ---------------------------

input_data = pd.DataFrame({
    'Income':[income],
    'Age':[age],
    'Dependents':[dependents],
    'Rent':[rent],
    'Groceries':[groceries],
    'Transport':[transport],
    'Eating_Out':[eating_out],
    'Entertainment':[entertainment],
    'Utilities':[utilities],
    'Healthcare':[healthcare],
    'Education':[education],
    'Miscellaneous':[misc]
})

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Savings"):

    # match training feature structure
    input_data = input_data.reindex(columns=features, fill_value=0)

    scaled = scaler.transform(input_data)
    pca_data = pca.transform(scaled)

    prediction = model.predict(pca_data)

    st.success(f"Predicted Savings: ₹ {prediction[0]:,.2f}")

    # ---------------------------
    # Spending Visualization
    # ---------------------------

    st.subheader("Expense Distribution")

    expenses = {
        "Rent":rent,
        "Groceries":groceries,
        "Transport":transport,
        "Eating Out":eating_out,
        "Entertainment":entertainment,
        "Utilities":utilities,
        "Healthcare":healthcare,
        "Education":education,
        "Miscellaneous":misc
    }

    fig, ax = plt.subplots()

    ax.pie(expenses.values(),
           labels=expenses.keys(),
           autopct='%1.1f%%')

    ax.set_title("Expense Breakdown")

    st.pyplot(fig)

    # ---------------------------
    # Income vs Spending
    # ---------------------------

    st.subheader("Financial Summary")

    total_expense = sum(expenses.values())

    remaining = income - total_expense

    summary = pd.DataFrame({
        "Category":["Income","Expenses","Remaining"],
        "Amount":[income,total_expense,remaining]
    })

    fig2, ax2 = plt.subplots()

    ax2.bar(summary["Category"], summary["Amount"])

    ax2.set_title("Income vs Expense Overview")

    st.pyplot(fig2)

    # ---------------------------
    # Overspending Detection
    # ---------------------------

    st.subheader("Spending Insights")

    high_spending = []

    if eating_out > 5000:
        high_spending.append("Eating Out")

    if entertainment > 4000:
        high_spending.append("Entertainment")

    if misc > 4000:
        high_spending.append("Miscellaneous")

    if len(high_spending) > 0:
        st.warning(f"High spending detected in: {', '.join(high_spending)}")
    else:
        st.success("Your spending pattern looks balanced!")