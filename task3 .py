# Task 3: Feature Selection

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# Load the cleaned dataset
df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\SAIKET SOLUTION\cleaned_telco_churn.csv")

# Separate Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Select Top 10 Features
selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X, y)

# Get Selected Feature Names
selected_features = X.columns[selector.get_support()]

print("Top 10 Selected Features:\n")
for feature in selected_features:
    print(feature)

print("\nFeature Scores:\n")
scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
})

scores = scores.sort_values(by="Score", ascending=False)
print(scores)