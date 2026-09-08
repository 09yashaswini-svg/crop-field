# crop-field
This script implements an end-to-end Machine Learning pipeline in Python to predict farm crop yield using data from crop_yield_data.csv. It handles data preprocessing, model training using Linear Regression, evaluation, inference for new data, feature importance analysis, and visualization.  
Data Loading & Inspection

  
PY

Loads crop_yield_data.csv into a Pandas DataFrame.  
PY

Prints initial inspection details including the first 5 rows, data dimensions, and missing value counts.  
PY

2. Feature Selection & Preprocessing

  
PY

Target Variable: crop_yield_tonnes_per_hectare.  
PY

Excluded: farm_id.  
PY

Numerical Features: rainfall_mm, temperature_c, soil_ph, fertilizer_kg_per_hectare. Imputed using median values and scaled using StandardScaler.  
PY
+ 1

Categorical Features: crop_type, irrigation, soil_type. Imputed using most frequent values and encoded using OneHotEncoder.  
PY
+ 1

Combines preprocessing steps using ColumnTransformer and Pipeline.  
PY

3. Model Training & Evaluation

  
PY

Splits data into training (80%) and testing (20%) sets (random_state=42).  
PY

Fits a LinearRegression model.  
PY

Calculates evaluation metrics on test data:

Mean Absolute Error (MAE)  
PY

Root Mean Squared Error (RMSE)  
PY

R² Score  
PY

Prints a sample comparison table comparing actual vs. predicted yields.  
PY

4. Inference & Feature Importance

  
PY

Predicts yield for a sample input (Rice, 1100mm rain, 27°C, pH 6.5, 150kg/ha fertilizer, Irrigated, Loamy soil).  
PY

Extracts regression coefficients to display top features influencing crop yield sorted by absolute impact.  
PY

5. Visualizations

  
PY

Generates and saves three scatter plots:

rainfall_vs_yield.png: Rainfall (mm) vs. Crop Yield  
PY

fertilizer_vs_yield.png: Fertilizer (kg/ha) vs. Crop Yield  
PY

actual_vs_predicted.png: Model predicted values vs. Actual yields  
PY
