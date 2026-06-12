import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("india_cancer_patients_2022_2025.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Visualization 1 : Gender distribution
plt.figure(figsize=(6, 4))
df["Gender"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("plots/gender_distribution.png")
plt.close()

# Visualization 2 : Cancer type distribution
plt.figure(figsize=(8, 5))
df["Cancer_Type"].value_counts().plot(kind="bar")
plt.title("Cancer Type Distribution")
plt.xlabel("Cancer Type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("plots/cancer_type_distribution.png")
plt.close()

# Visualization 3 : Stage distribution
plt.figure(figsize=(6, 4))
df["Stage"].value_counts().plot(kind="bar")
plt.title("Cancer Stage Distribution")
plt.xlabel("Stage")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/stage_distribution.png")
plt.close()

# Visualization 4 : Average survival by stage
avg_stage = df.groupby("Stage")["Survival_Months"].mean()

plt.figure(figsize=(6, 4))
avg_stage.plot(kind="bar")
plt.title("Average Survival Months by Stage")
plt.xlabel("Stage")
plt.ylabel("Average Survival Months")
plt.tight_layout()
plt.savefig("plots/avg_survival_by_stage.png")
plt.close()



# Machine Learning
ml_df = df.drop(columns=["Patient_ID", "Diagnosis_Date"])

y = ml_df["Survival_Months"]
X = ml_df.drop(columns=["Survival_Months"])

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    random_state=42,
    n_estimators=100
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\n" + "=" * 50)
print("MODEL RESULTS")
print("=" * 50)
print(f"Mean Absolute Error: {mae:.2f} months")

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nSample Predictions")
print(results.head(10))

# Predicted vs Actual plot
plt.figure(figsize=(6, 6))
plt.scatter(
    results["Actual"],
    results["Predicted"],
    s=10
)
plt.xlabel("Actual Survival Months")
plt.ylabel("Predicted Survival Months")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.savefig("plots/predicted_vs_actual.png")
plt.close()

# Save model
joblib.dump(model, "cancer_survival_model.pkl")

# Feature importance
importance = (
    pd.Series(model.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)

print("\nTop 10 Important Features")
print(importance.head(10))

print("\nModel saved as: cancer_survival_model.pkl")
print("Plots saved inside: plots/")