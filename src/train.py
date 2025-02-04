import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import warnings
import scipy.stats as stats
warnings.filterwarnings('ignore')

def create_super_advanced_features(df):
    # Your feature engineering code here...
    return df

# Load and prepare data
print("Loading data...")
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
sample_submission = pd.read_csv("SampleSubmission.csv")

# Remove unnecessary columns
columns_to_drop = ['ID', 'device_name']
train_data = train_data.drop(columns=columns_to_drop, errors='ignore')
test_data = test_data.drop(columns=columns_to_drop, errors='ignore')

# Apply super advanced feature engineering
print("Applying advanced feature engineering...")
train_data = create_super_advanced_features(train_data)
test_data = create_super_advanced_features(test_data)

# Select features (excluding target)
features = [col for col in train_data.columns if col != 'CO2']

# Prepare data
X = train_data[features].values
y = train_data['CO2'].values

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
test_data_imputed = imputer.transform(test_data[features])

# Advanced scaling
print("Applying advanced scaling...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
test_data_scaled = scaler.transform(test_data_imputed)

# Initialize Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Initialize K-fold
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Arrays to store predictions
oof_rf = np.zeros(len(X_scaled))
test_rf = np.zeros(len(test_data_scaled))

# Cross-validation loop
print("Starting cross-validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\nFold {fold + 1}/{n_splits}")
    
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train Random Forest
    rf_model.fit(X_train, y_train)
    oof_rf[val_idx] = rf_model.predict(X_val)
    test_rf += rf_model.predict(test_data_scaled) / n_splits
    
    # Print fold scores
    print(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_val, oof_rf[val_idx])):.4f}")

# Generate final predictions
final_predictions = test_rf

# Create submission file
print("\nCreating submission file...")
sample_submission['CO2'] = final_predictions
sample_submission.to_csv('submission_random_forest.csv', index=False)

print("\nDone! Check 'submission_random_forest.csv' for predictions.")