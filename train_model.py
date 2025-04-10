import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Load dataset
adult_df = pd.read_csv("adult.csv")

# Data Cleaning and Preprocessing
# Fill missing values with most common categories in respective columns
adult_df["workclass"] = adult_df["workclass"].fillna("Private")
adult_df["occupation"] = adult_df["occupation"].fillna("Prof-specialty")
adult_df["nativeCountry"] = adult_df["nativeCountry"].fillna("United-States")

# Group similar education levels to reduce the number of categories
adult_df["education"] = adult_df["education"].replace(
    ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th"], "School"
)
adult_df["education"] = adult_df["education"].replace("HS-grad", "High School")
adult_df["education"] = adult_df["education"].replace(
    ["Some-college", "Assoc-voc", "Assoc-acdm", "Prof-school"], "Diploma"
)

# Group marital status into broader categories
adult_df["marital-status"] = adult_df["marital-status"].replace(
    ["Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"], "Married"
)
adult_df["marital-status"] = adult_df["marital-status"].replace(
    ["Never-married", "Divorced", "Separated", "Widowed"], "Others"
)

# Convert categorical columns with binary values into numerical format
adult_df["sex"] = adult_df["sex"].map({"Male": 0, "Female": 1})
adult_df["income"] = adult_df["income"].map({"<=50K": 0, ">50K": 1})

# Remove duplicate records to avoid redundant data points
adult_df.drop_duplicates(inplace=True)

# Function to remove outliers using Z-score method
def remove_outliers_manual(df, column):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    df_filtered = df[z_scores < 3].copy()
    return df_filtered

# Remove outliers from 'hours-per-week' column
adult_df = remove_outliers_manual(adult_df, "hoursPerWeek")

# One-Hot Encoding for categorical variables (excluding already binary 'sex')
categorical_cols = ["workclass", "education", "marital-status", "occupation", 
                    "relationship", "race", "nativeCountry"]
adult_df = pd.get_dummies(adult_df, columns=categorical_cols, drop_first=True)

# Define features (X) and target variable (y)
X = adult_df.drop(["fnlwgt", "education-num", "income"], axis=1)  # Drop education-num
y = adult_df["income"]

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical columns to improve model performance
scaler = StandardScaler()
numerical_cols = ["age", "capitalGain", "capitalLoss", "hoursPerWeek"]
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols].copy())
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols].copy())

# Train a Random Forest classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model, scaler, and column names for later use
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_train.columns.tolist(), "model_columns.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

# Model Evaluation
# Predict test data and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()