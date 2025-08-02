import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# --- Step 2: Importing the Dataset ---
try:
    # Assuming 'traffic volume.csv' is in the same directory as this script
    data = pd.read_csv('traffic_volume.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'traffic volume.csv' not found. Please ensure the file is in the same directory.")
    exit()

# --- Step 3: Initial Data Analysis and Checking for Null Values ---
print("\n--- Dataset Info ---")
data.info()

# --- Step 5: Taking Care of Missing Data (Imputation) ---
# Treat NaN in 'holiday' as a separate category, e.g., 'No_Holiday'
data['holiday'] = data['holiday'].fillna('No_Holiday')

# Impute numerical columns with mean
numerical_cols_for_imputation = ['temp', 'rain', 'snow']
imputer_numerical = SimpleImputer(strategy='mean')
data[numerical_cols_for_imputation] = imputer_numerical.fit_transform(data[numerical_cols_for_imputation])

# Impute categorical columns with mode
categorical_cols_for_imputation = ['weather']
imputer_categorical = SimpleImputer(strategy='most_frequent')
data[categorical_cols_for_imputation] = imputer_categorical.fit_transform(data[categorical_cols_for_imputation])

# --- Step 6: Feature Engineering ---
# Combine 'date' and 'Time' into a single datetime column
data['Date and time'] = pd.to_datetime(data['date'] + ' ' + data['Time'], format="%d-%m-%Y %H:%M:%S")

# Extract features from 'Date and time'
data['Year'] = data['Date and time'].dt.year
data['Month'] = data['Date and time'].dt.month
data['Day'] = data['Date and time'].dt.day
data['Hour'] = data['Date and time'].dt.hour
data['Minute'] = data['Date and time'].dt.minute
data['Second'] = data['Date and time'].dt.second
data['DayOfWeek'] = data['Date and time'].dt.dayofweek
data['DayOfYear'] = data['Date and time'].dt.dayofyear
data['WeekOfYear'] = data['Date and time'].dt.isocalendar().week.astype(int)

# Drop original date/time columns
data = data.drop(['date', 'Time', 'Date and time'], axis=1)

# One-hot encode categorical features
final_categorical_cols = ['holiday', 'weather']
data = pd.get_dummies(data, columns=final_categorical_cols, drop_first=True)

# --- Step 7: Splitting Data into Dependent and Independent Variables ---
X = data.drop('traffic_volume', axis=1)
y = data['traffic_volume']

# --- Step 8: Feature Scaling ---
numerical_features_for_scaling = [
    'temp', 'rain', 'snow', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',
    'DayOfWeek', 'DayOfYear', 'WeekOfYear'
]
scaler = StandardScaler()
X_scaled_values = scaler.fit_transform(X[numerical_features_for_scaling])
X[numerical_features_for_scaling] = pd.DataFrame(X_scaled_values, columns=numerical_features_for_scaling, index=X.index)

# --- Step 9: Splitting Data into Train and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 10: Model Building, Training, Evaluation, and Saving ---
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("\nTraining the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions and evaluate
y_test_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"Test R-squared (R2 Score): {r2_test:.4f}")
print(f"Test Root Mean Squared Error (RMSE): {rmse_test:.2f}")

# Save the trained model and preprocessors to the Flask folder
model_dir = 'Flask'
joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
joblib.dump(imputer_numerical, os.path.join(model_dir, 'imputer_numerical.pkl'))
joblib.dump(imputer_categorical, os.path.join(model_dir, 'imputer_categorical.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scale.pkl'))
print(f"Model and preprocessors saved successfully to '{model_dir}/'.")