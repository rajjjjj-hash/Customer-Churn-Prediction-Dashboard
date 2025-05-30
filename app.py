import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model pipeline
model = joblib.load('churn_prediction_model.pkl')

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("🚀 Customer Churn Prediction Dashboard")

# Sidebar filters for exploration
st.sidebar.header("Filter Dataset for Exploration")

uploaded_file = st.sidebar.file_uploader("Upload CSV with customer data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Sidebar filters
    contract_options = ['All'] + sorted(data['Contract'].dropna().unique().tolist())
    gender_options = ['All'] + sorted(data['gender'].dropna().unique().tolist())
    senior_options = ['All'] + sorted(data['SeniorCitizen'].dropna().unique().astype(str).tolist())

    selected_contract = st.sidebar.selectbox("Filter by Contract", contract_options)
    selected_gender = st.sidebar.selectbox("Filter by Gender", gender_options)
    selected_senior = st.sidebar.selectbox("Filter by Senior Citizen", senior_options)

    filtered_data = data.copy()

    if selected_contract != 'All':
        filtered_data = filtered_data[filtered_data['Contract'] == selected_contract]
    if selected_gender != 'All':
        filtered_data = filtered_data[filtered_data['gender'] == selected_gender]
    if selected_senior != 'All':
        filtered_data = filtered_data[filtered_data['SeniorCitizen'].astype(str) == selected_senior]

    st.header("1. Dataset Preview & Summary")

    st.subheader("Filtered Data Preview")
    st.write(filtered_data.head())

    # KPIs
    st.subheader("Key Metrics")
    total_customers = filtered_data.shape[0]
    churned = filtered_data[filtered_data['Churn'] == 'Yes'].shape[0]
    churn_rate = (churned / total_customers * 100) if total_customers > 0 else 0
    avg_tenure = filtered_data['tenure'].mean() if total_customers > 0 else 0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Customers", total_customers)
    kpi2.metric("Churn Rate (%)", f"{churn_rate:.2f}%")
    kpi3.metric("Average Tenure (Months)", f"{avg_tenure:.1f}")

    st.header("2. Visual Explorations")

    # Churn Distribution
    if 'Churn' in filtered_data.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_data, x='Churn', ax=ax, palette='Set2')
        ax.set_title("Churn Distribution")
        st.pyplot(fig)

    # Churn Rate by Contract
    if 'Contract' in filtered_data.columns and 'Churn' in filtered_data.columns:
        churn_contract = pd.crosstab(filtered_data['Contract'], filtered_data['Churn'], normalize='index') * 100
        churn_contract = churn_contract.reset_index()
        churn_contract = churn_contract.rename(columns={'Yes': 'Churn %', 'No': 'No Churn %'})
        fig2, ax2 = plt.subplots()
        sns.barplot(x='Contract', y='Churn %', data=churn_contract, ax=ax2, palette='pastel')
        ax2.set_ylabel("Churn Percentage (%)")
        ax2.set_title("Churn Rate by Contract Type")
        st.pyplot(fig2)

    # Churn by Payment Method
    if 'PaymentMethod' in filtered_data.columns and 'Churn' in filtered_data.columns:
        churn_payment = pd.crosstab(filtered_data['PaymentMethod'], filtered_data['Churn'], normalize='index') * 100
        churn_payment = churn_payment.reset_index()
        churn_payment = churn_payment.rename(columns={'Yes': 'Churn %', 'No': 'No Churn %'})
        fig3, ax3 = plt.subplots(figsize=(10,4))
        sns.barplot(x='PaymentMethod', y='Churn %', data=churn_payment, ax=ax3, palette='muted')
        ax3.set_ylabel("Churn Percentage (%)")
        ax3.set_title("Churn Rate by Payment Method")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

    # Tenure Histogram by Churn
    if 'tenure' in filtered_data.columns and 'Churn' in filtered_data.columns:
        fig4, ax4 = plt.subplots()
        sns.histplot(data=filtered_data, x='tenure', hue='Churn', multiple='stack', bins=30, ax=ax4, palette='coolwarm')
        ax4.set_title("Tenure Distribution by Churn Status")
        st.pyplot(fig4)

    # Monthly Charges Distribution
    if 'MonthlyCharges' in filtered_data.columns:
        fig5, ax5 = plt.subplots()
        sns.histplot(filtered_data['MonthlyCharges'], bins=30, kde=True, color='purple', ax=ax5)
        ax5.set_title("Monthly Charges Distribution")
        st.pyplot(fig5)

    # --- Batch Prediction ---
    st.header("3. Batch Churn Prediction")

    try:
        predictions = model.predict(filtered_data)
        filtered_data['Churn_Prediction'] = np.where(predictions == 1, 'Yes', 'No')
        st.subheader("Prediction Results (Sample)")
        st.write(filtered_data[['Churn_Prediction']].head())

        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"Batch prediction error: {e}")

else:
    st.info("Upload a CSV file in the sidebar to get started!")

# --- Manual Prediction Section ---

st.header("4. Predict Single Customer Churn (Manual Input)")

with st.form("manual_predict_form"):
    gender = st.selectbox("Gender", ['Female', 'Male'])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_dict = {
            'gender': [gender],
            'SeniorCitizen': [senior],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }
        input_df = pd.DataFrame(input_dict)
        try:
            pred = model.predict(input_df)[0]
            label = 'Yes' if pred == 1 else 'No'
            st.success(f"Predicted Churn: **{label}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
