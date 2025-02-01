# train.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from azureml.core import Run
import optuna
import os
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(df):
    """Create advanced features for the model"""
    df = df.copy()
    sensor_cols = ['MQ7_analog', 'MQ9_analog', 'MG811_analog', 'MQ135_analog']
    
    # Basic sensor features
    df['Sensor_mean'] = df[sensor_cols].mean(axis=1)
    df['Sensor_std'] = df[sensor_cols].std(axis=1)
    df['Sensor_median'] = df[sensor_cols].median(axis=1)
    df['Sensor_max'] = df[sensor_cols].max(axis=1)
    df['Sensor_min'] = df[sensor_cols].min(axis=1)
    df['Sensor_range'] = df['Sensor_max'] - df['Sensor_min']
    
    # Advanced statistical features
    df['Sensor_skew'] = df[sensor_cols].skew(axis=1)
    df['Sensor_kurtosis'] = df[sensor_cols].kurtosis(axis=1)
    
    # Ratio features
    for i in range(len(sensor_cols)):
        for j in range(i+1, len(sensor_cols)):
            ratio_name = f'ratio_{sensor_cols[i]}_{sensor_cols[j]}'
            df[ratio_name] = df[sensor_cols[i]] / (df[sensor_cols[j]] + 1e-6)
    
    # Temperature Compensation with advanced scaling
    temp_ref = 25.0
    humidity_ref = 50.0
    for col in sensor_cols:
        # Temperature compensation
        temp_factor = 1 + 0.02 * (df['Temperature'] - temp_ref)
        # Humidity compensation
        humid_factor = 1 + 0.01 * (df['Humidity'] - humidity_ref)
        
        df[f'{col}_temp_comp'] = df[col] * temp_factor
        df[f'{col}_humid_comp'] = df[col] * humid_factor
        df[f'{col}_full_comp'] = df[col] * temp_factor * humid_factor
    
    # Environmental Features
    df['Temp_Humid_interaction'] = df['Temperature'] * df['Humidity']
    df['Temp_Humid_ratio'] = df['Temperature'] / (df['Humidity'] + 1e-6)
    df['Temp_squared'] = df['Temperature'] ** 2
    df['Humid_squared'] = df['Humidity'] ** 2
    
    return df

def objective(trial, X, y, cv):
    """Optuna objective function for XGBoost optimization"""
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'tree_method': 'hist',
        'random_state': 42
    }
    
    scores = []
    for train_idx, valid_idx in cv.split(X):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        model = XGBRegressor(**param)
        model.fit(X_train, y_train,
                 eval_set=[(X_valid, y_valid)],
                 verbose=False)
        
        pred = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, pred))
        scores.append(rmse)
    
    return np.mean(scores)

def main():
    # Get the experiment run context
    run = Run.get_context()
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    
    # Handle required IDs
    required_ids = ['ID_007308', 'ID_007315', 'ID_007323', 'ID_007324', 'ID_007326']
    missing_ids = [id_value for id_value in required_ids if id_value not in test_data['ID'].values]
    
    if missing_ids:
        new_rows = pd.DataFrame({'ID': missing_ids})
        test_data = pd.concat([test_data, new_rows], ignore_index=True)
        print(f"Added missing IDs: {missing_ids}")
    
    # Store IDs before feature engineering
    train_ids = train_data['ID'].copy()
    test_ids = test_data['ID'].copy()
    
    # Feature engineering
    print("Creating advanced features...")
    train_features = create_advanced_features(train_data.drop(['ID', 'device_name', 'CO2'], axis=1))
    test_features = create_advanced_features(test_data.drop(['ID', 'device_name'], axis=1))
    
    # Get numeric columns
    numeric_cols = train_features.select_dtypes(include=['float64', 'int64']).columns
    
    # Add polynomial features
    print("Creating polynomial features...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    
    poly_features_train = poly.fit_transform(train_features[numeric_cols])
    poly_features_test = poly.transform(test_features[numeric_cols])
    
    feature_names = (
        list(numeric_cols) + 
        [f"poly_{i}" for i in range(poly_features_train.shape[1] - len(numeric_cols))]
    )
    
    # Prepare final feature matrices
    X = poly_features_train
    y = train_data['CO2'].values
    
    # Scale features
    print("Scaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Hyperparameter optimization
    print("Optimizing hyperparameters...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_scaled, y, cv),
        n_trials=50,
        timeout=3600  # 1 hour timeout
    )
    
    best_params = study.best_params
    best_params['tree_method'] = 'hist'
    best_params['random_state'] = 42
    print("Best parameters:", best_params)
    
    # Train final model
    print("Training final model...")
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_scaled, y)
    
    # Make predictions
    print("Making predictions...")
    test_scaled = scaler.transform(poly_features_test)
    predictions = final_model.predict(test_scaled)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'CO2': predictions
    })
    
    # Handle missing IDs
    mean_pred = predictions.mean()
    for id_value in required_ids:
        if id_value not in submission['ID'].values:
            submission = pd.concat([
                submission,
                pd.DataFrame({'ID': [id_value], 'CO2': [mean_pred]})
            ])
    
    # Final processing
    submission = submission.sort_values('ID').reset_index(drop=True)
    
    # Save submission
    os.makedirs('outputs', exist_ok=True)
    submission_path = os.path.join('outputs', 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    # Log metrics
    run.log('completed', True)
    run.log('submission_rows', len(submission))
    run.log('best_rmse', study.best_value)
    
    print(f"Training completed! Best RMSE: {study.best_value:.4f}")
    print(f"Submission saved with {len(submission)} entries")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': final_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Log top 10 important features
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()