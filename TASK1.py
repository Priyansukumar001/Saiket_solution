import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load Dataset
file_path = r"C:\Users\hp\OneDrive\Desktop\SAIKET SOLUTION\Telco_Customer_Churn_Dataset  (1).csv"

df = pd.read_csv(file_path)

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

# Display basic information
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Handle blank values in TotalCharges
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

# Fill missing values with mean
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

# Remove Customer ID column
df.drop("customerID", axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
output_file = r"C:\Users\hp\OneDrive\Desktop\SAIKET SOLUTION\cleaned_telco_churn.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 50)
print("TASK 1 COMPLETED SUCCESSFULLY")
print("=" * 50)

print("\nCleaned Dataset Shape:", df.shape)

print("\nFirst 5 Rows of Cleaned Dataset:")
print(df.head())

print(f"\nCleaned dataset saved as:\n{output_file}")