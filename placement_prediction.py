import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Save original columns
original_columns = df.columns

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

# Create Placement Readiness Score
df["placement_score"] = (
    df["cgpa"] * 10 +
    df["internships_completed"] * 5 +
    df["coding_skill_rating"] * 8 +
    df["communication_skill_rating"] * 5 +
    df["aptitude_skill_rating"] * 7 +
    df["projects_completed"] * 4
)

# Features and Target
X = df.drop(["Student_ID", "placement_score"], axis=1)
y = df["placement_score"]

# Save feature names for later
feature_columns = X.columns

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model R2 Score:", r2_score(y_test, y_pred))

# Save model and feature columns
pickle.dump((model, feature_columns), open("placement_model.pkl", "wb"))
