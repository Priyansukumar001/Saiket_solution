import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(
    r"C:\Users\hp\OneDrive\Desktop\SAIKET SOLUTION\Telco_Customer_Churn_Dataset  (1).csv"
)

print(df.head())

# Remove customerID if present
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset Shape:", df.shape)
print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))