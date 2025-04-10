# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Load pre-trained model, scaler, and column detail
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

# Set up the app title and description
st.title("Income Prediction App")
st.markdown("This app predicts whether an individual's income is **<=50k** or \
            **>50k** based on the given features.")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Collect numerical inputs from the user
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25)
capital_gain = st.sidebar.number_input("Capital Gain", 0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, value=0)
hours_per_week = st.sidebar.number_input("Hours Per Week", 1, 100, 40)

# Collect categorical inputs from the user
sex = st.sidebar.radio("Sex", ("Male", "Female"))
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Local-gov", "State-gov", "Federal-gov", "Without-pay", "Never-worked"
])
education = st.sidebar.selectbox("Education", [
    "School", "High School", "Diploma", "Bachelors", 
    "Masters", "Doctorate"
])
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married", "Others"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Not-in-family", "Unmarried", "Own-child",
    "Other-relative", "Husband", "Wife"
])
race = st.sidebar.selectbox("Race", [
    'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
])
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Greece", "Vietnam", "China", "Taiwan", "India",
    "Philippines", "Trinadad&Tobago", "Canada", "South", "Holand-Netherlands",
    "Puerto-Rico", "Poland", "Iran", "England", "Germany", "Italy", "Japan",
    "Hong", "Honduras", "Cuba", "Ireland", "Cambodia", "Peru", "Nicaragua",
    "Dominican-Republic", "Haiti", "El-Salvador", "Hungary", "Columbia",
    "Guatemala", "Jamaica", "Ecuador", "France", "Yugoslavia", "Scotland",
    "Portugal", "Laos", "Thailand", "Outlying-US(Guam-USVI-etc)"
])

# Prepare input data for prediction
input_data = pd.DataFrame({
    "age": [age],
    "capitalGain": [capital_gain],
    "capitalLoss": [capital_loss],
    "hoursPerWeek": [hours_per_week],
    "sex": [0 if sex == "Male" else 1]
})

# One-hot encode categorical features and add them to the input data
categorical_features = {
    f"workclass_{workclass}": 1,
    f"education_{education}": 1,
    f"marital-status_{marital_status}": 1,
    f"occupation_{occupation}": 1,
    f"relationship_{relationship}": 1,
    f"race_{race}": 1,
    f"nativeCountry_{native_country}": 1
}

for col, value in categorical_features.items():
    input_data[col] = value

# Ensure all columns used during model training are present
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match the model's expected input format
input_data = input_data[model_columns]

# Standardize numerical features using the loaded scaler
numerical_cols = ["age", "capitalGain", "capitalLoss", "hoursPerWeek"]
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Perform prediction when the "Predict" button is clicked
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display prediction results
    result = ">50k" if prediction[0] == 1 else "<=50k"
    st.subheader(f"The predicted income is: {result}")
    st.write(f"The Predicted Probability is: {prediction_proba[0][1]:.2f}")

# Show the confusion matrix if the checkbox is selected
if st.sidebar.checkbox("Show Confusion Matrix"):
    # Load test data for evaluation
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    y_pred = model.predict(X_test)
    
    # Calculate and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    # Plot and display the confusion matrix
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"])
    disp.plot(cmap="Blues", ax=ax)
    plt.title("Confusion Matrix")
    st.pyplot(fig)