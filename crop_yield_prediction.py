# Crop Yield Prediction
# Tools: Python, Pandas, Matplotlib, Scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("crop_yield_data.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:", data.shape)

print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Features and target
X = data.drop(columns=["farm_id", "crop_yield_tonnes_per_hectare"])
y = data["crop_yield_tonnes_per_hectare"]

numeric_features = [
    "rainfall_mm",
    "temperature_c",
    "soil_ph",
    "fertilizer_kg_per_hectare"
]

categorical_features = [
    "crop_type",
    "irrigation",
    "soil_type"
]

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error (MAE): {mae:.3f} tonnes/hectare")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f} tonnes/hectare")
print(f"R² Score: {r2:.4f}")

# Compare actual and predicted yield
comparison = pd.DataFrame({
    "Actual Yield": y_test.values,
    "Predicted Yield": y_pred.round(2)
})
print("\nActual vs Predicted Crop Yield:")
print(comparison.head(10))

# Predict yield for a new farm
new_farm = pd.DataFrame([{
    "crop_type": "Rice",
    "rainfall_mm": 1100,
    "temperature_c": 27,
    "soil_ph": 6.5,
    "fertilizer_kg_per_hectare": 150,
    "irrigation": "Yes",
    "soil_type": "Loamy"
}])

predicted_yield = model.predict(new_farm)[0]
print(f"\nPredicted Crop Yield: {predicted_yield:.2f} tonnes/hectare")

# Analyze feature impact using regression coefficients
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["regressor"].coef_

importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "absolute_impact": abs(coefficients)
}).sort_values("absolute_impact", ascending=False)

print("\nTop Factors Affecting Crop Yield:")
print(importance.head(12)[["feature", "coefficient"]])

# Visualization 1: rainfall vs yield
plt.figure(figsize=(8, 5))
plt.scatter(data["rainfall_mm"], data["crop_yield_tonnes_per_hectare"], alpha=0.6)
plt.xlabel("Rainfall (mm)")
plt.ylabel("Crop Yield (tonnes/hectare)")
plt.title("Rainfall vs Crop Yield")
plt.tight_layout()
plt.savefig("rainfall_vs_yield.png", dpi=150)
plt.show()

# Visualization 2: fertilizer vs yield
plt.figure(figsize=(8, 5))
plt.scatter(
    data["fertilizer_kg_per_hectare"],
    data["crop_yield_tonnes_per_hectare"],
    alpha=0.6
)
plt.xlabel("Fertilizer (kg/hectare)")
plt.ylabel("Crop Yield (tonnes/hectare)")
plt.title("Fertilizer Usage vs Crop Yield")
plt.tight_layout()
plt.savefig("fertilizer_vs_yield.png", dpi=150)
plt.show()

# Visualization 3: actual vs predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Yield (tonnes/hectare)")
plt.ylabel("Predicted Yield (tonnes/hectare)")
plt.title("Actual vs Predicted Crop Yield")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()
