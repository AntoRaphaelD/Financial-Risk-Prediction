import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(page_title="Employee Financial Risk Prediction", layout="wide")

st.title("Employee Financial Risk Prediction System")

st.write("Predict whether an employee is financially at risk based on income, expenses, and financial behaviour.")

# ---------------------------------
# Load Model
# ---------------------------------

model = joblib.load("savings_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# ---------------------------------
# Sidebar Inputs
# ---------------------------------

st.sidebar.header("Employee Information")

income = st.sidebar.number_input("Monthly Income", 10000, 500000, 50000)
age = st.sidebar.slider("Age", 20, 60, 30)
dependents = st.sidebar.slider("Dependents", 0, 5, 1)

occupation = st.sidebar.selectbox(
    "Occupation",
    ["Private Job","Government Job","Business","Student"]
)

city_tier = st.sidebar.selectbox(
    "City Tier",
    ["Tier 1","Tier 2","Tier 3"]
)

st.sidebar.header("Financial Commitments")

loan_repayment = st.sidebar.number_input("Loan Repayment",0,50000,0)
insurance = st.sidebar.number_input("Insurance",0,20000,1000)

st.sidebar.header("Expense Details")

rent = st.sidebar.number_input("Rent",0,100000,15000)
groceries = st.sidebar.number_input("Groceries",0,30000,5000)
transport = st.sidebar.number_input("Transport",0,20000,3000)
eating_out = st.sidebar.number_input("Eating Out",0,20000,2000)
entertainment = st.sidebar.number_input("Entertainment",0,20000,2000)
utilities = st.sidebar.number_input("Utilities",0,15000,3000)
healthcare = st.sidebar.number_input("Healthcare",0,15000,2000)
education = st.sidebar.number_input("Education",0,20000,1000)
misc = st.sidebar.number_input("Miscellaneous",0,20000,2000)

# ---------------------------------
# Expense Calculations
# ---------------------------------

expenses_dict = {
    "Rent": rent,
    "Groceries": groceries,
    "Transport": transport,
    "Eating Out": eating_out,
    "Entertainment": entertainment,
    "Utilities": utilities,
    "Healthcare": healthcare,
    "Education": education,
    "Miscellaneous": misc
}

total_expense = sum(expenses_dict.values())

expense_income_ratio = total_expense / income
per_dependent_expense = total_expense / (dependents + 1)

# ---------------------------------
# Potential Savings Features
# ---------------------------------

potential_savings_groceries = groceries * 0.10
potential_savings_transport = transport * 0.10
potential_savings_eating_out = eating_out * 0.20
potential_savings_entertainment = entertainment * 0.20
potential_savings_utilities = utilities * 0.05
potential_savings_healthcare = healthcare * 0.05
potential_savings_education = education * 0.05
potential_savings_misc = misc * 0.10

# ---------------------------------
# Input Data
# ---------------------------------

input_data = pd.DataFrame({

'Income':[income],
'Age':[age],
'Dependents':[dependents],
'Occupation':[occupation],
'City_Tier':[city_tier],
'Rent':[rent],
'Loan_Repayment':[loan_repayment],
'Insurance':[insurance],
'Groceries':[groceries],
'Transport':[transport],
'Eating_Out':[eating_out],
'Entertainment':[entertainment],
'Utilities':[utilities],
'Healthcare':[healthcare],
'Education':[education],
'Miscellaneous':[misc],

'Potential_Savings_Groceries':[potential_savings_groceries],
'Potential_Savings_Transport':[potential_savings_transport],
'Potential_Savings_Eating_Out':[potential_savings_eating_out],
'Potential_Savings_Entertainment':[potential_savings_entertainment],
'Potential_Savings_Utilities':[potential_savings_utilities],
'Potential_Savings_Healthcare':[potential_savings_healthcare],
'Potential_Savings_Education':[potential_savings_education],
'Potential_Savings_Miscellaneous':[potential_savings_misc],

'Total_Expense':[total_expense],
'Expense_Income_Ratio':[expense_income_ratio],
'Per_Dependent_Expense':[per_dependent_expense]

})

# ---------------------------------
# Prediction
# ---------------------------------

if st.button("Predict Financial Risk"):

    input_data_numeric = input_data.select_dtypes(include=[np.number])

    scaled_data = scaler.transform(input_data_numeric)

    pca_data = pca.transform(scaled_data)

    prediction = model.predict(pca_data)
    risk = int(prediction[0])

    remaining_balance = income - total_expense

    # ---------------------------------
    # KPI CARDS
    # ---------------------------------

    st.subheader("Financial Summary")

    k1,k2,k3,k4 = st.columns(4)

    k1.metric("Income", f"₹ {income:,.0f}")
    k2.metric("Expenses", f"₹ {total_expense:,.0f}")
    k3.metric("Remaining Income", f"₹ {remaining_balance:,.0f}")
    k4.metric("Expense Ratio", f"{expense_income_ratio:.2f}")

    # ---------------------------------
    # Risk Prediction
    # ---------------------------------

    st.subheader("Financial Risk Prediction")

    if risk == 1:
        st.error("🔴 Financial Risk Detected")
    else:
        st.success("🟢 No Financial Risk Detected")

    # ---------------------------------
    # Charts
    # ---------------------------------

    col1,col2 = st.columns(2)

    with col1:

        fig1,ax1 = plt.subplots(figsize=(4,3))

        ax1.pie(
            expenses_dict.values(),
            labels=expenses_dict.keys(),
            autopct='%1.1f%%'
        )

        ax1.set_title("Expense Distribution")

        st.pyplot(fig1)

    with col2:

        fig2,ax2 = plt.subplots(figsize=(4,3))

        income_dist = {
            "Expenses": total_expense,
            "Remaining Income": remaining_balance
        }

        ax2.pie(
            income_dist.values(),
            labels=income_dist.keys(),
            autopct='%1.1f%%'
        )

        ax2.set_title("Income Allocation")

        st.pyplot(fig2)

    col3,col4 = st.columns(2)

    with col3:

        fig3,ax3 = plt.subplots(figsize=(5,3))

        ax3.barh(list(expenses_dict.keys()), list(expenses_dict.values()))

        ax3.set_title("Expense Categories")

        st.pyplot(fig3)

    with col4:

        fig4,ax4 = plt.subplots(figsize=(5,3))

        ax4.bar(["Income","Expenses","Remaining"],
                [income,total_expense,remaining_balance])

        ax4.set_title("Financial Overview")

        st.pyplot(fig4)

    # ---------------------------------
    # Spending Insights
    # ---------------------------------

    st.subheader("Spending Insights")

    if eating_out > 5000:
        st.warning("Frequent eating out detected")

    if entertainment > 4000:
        st.warning("Entertainment spending is high")

    if rent > income * 0.4:
        st.warning("Rent is consuming a large portion of income")

    if expense_income_ratio > 0.75:
        st.warning("Expenses consume a large portion of income")

    if remaining_balance > income * 0.3:
        st.success("Healthy financial balance detected")