# train.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from azureml.core import Run
import optuna
import os
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(df):
    """Create advanced features with enhanced engineering and NaN handling"""
    df = df.copy()
    sensor_cols = ['MQ7_analog', 'MQ9_analog', 'MG811_analog', 'MQ135_analog']
    
    # Initial NaN handling for basic columns
    imputer = SimpleImputer(strategy='median')
    df[sensor_cols + ['Temperature', 'Humidity']] = imputer.fit_transform(df[sensor_cols + ['Temperature', 'Humidity']])
    
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
    df['Sensor_q25'] = df[sensor_cols].quantile(0.25, axis=1)
    df['Sensor_q75'] = df[sensor_cols].quantile(0.75, axis=1)
    df['Sensor_iqr'] = df['Sensor_q75'] - df['Sensor_q25']
    
    # Rolling statistics for each sensor
    window_sizes = [2, 3, 4]
    for col in sensor_cols:
        for window in window_sizes:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
    
    # Ratio features with enhanced combinations
    for i in range(len(sensor_cols)):
        for j in range(i+1, len(sensor_cols)):
            ratio_name = f'ratio_{sensor_cols[i]}_{sensor_cols[j]}'
            df[ratio_name] = np.divide(df[sensor_cols[i]], df[sensor_cols[j]], out=np.zeros_like(df[sensor_cols[i]], dtype=float), where=df[sensor_cols[j]]!=0)
            # Add log ratios (using lower instead of min)
            df[f'log_{ratio_name}'] = np.log1p(df[ratio_name].clip(lower=0))
    
    # Temperature Compensation with advanced scaling
    temp_ref = 25.0
    humidity_ref = 50.0
    for col in sensor_cols:
        # Enhanced temperature compensation
        temp_factor = 1 + 0.02 * (df['Temperature'] - temp_ref)
        temp_factor_squared = temp_factor ** 2
        
        # Enhanced humidity compensation
        humid_factor = 1 + 0.01 * (df['Humidity'] - humidity_ref)
        humid_factor_squared = humid_factor ** 2
        
        # Basic compensations
        df[f'{col}_temp_comp'] = df[col] * temp_factor
        df[f'{col}_humid_comp'] = df[col] * humid_factor
        df[f'{col}_full_comp'] = df[col] * temp_factor * humid_factor
        
        # Advanced compensations
        df[f'{col}_temp_comp_sq'] = df[col] * temp_factor_squared
        df[f'{col}_humid_comp_sq'] = df[col] * humid_factor_squared
        df[f'{col}_full_comp_sq'] = df[col] * temp_factor_squared * humid_factor_squared
    
    # Environmental Features
    df['Temp_Humid_interaction'] = df['Temperature'] * df['Humidity']
    df['Temp_Humid_ratio'] = np.divide(df['Temperature'], df['Humidity'], out=np.zeros_like(df['Temperature'], dtype=float), where=df['Humidity']!=0)
    df['Temp_squared'] = df['Temperature'] ** 2
    df['Humid_squared'] = df['Humidity'] ** 2
    df['Temp_Humid_geometric'] = np.sqrt(np.abs(df['Temperature'] * df['Humidity']))
    
    # Add log transformations for key features
    for col in sensor_cols + ['Temperature', 'Humidity']:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    
    # Final NaN check and filling
    df = df.fillna(df.mean())
    
    return df

def objective(trial, X, y, cv):
    """Optuna objective function for XGBoost optimization"""
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'max_bin': trial.suggest_int('max_bin', 128, 512),
        'tree_method': 'hist',
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
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
    
    # Ensure no NaN values before polynomial features
    train_features_clean = train_features[numeric_cols].fillna(0)
    test_features_clean = test_features[numeric_cols].fillna(0)
    
    poly_features_train = poly.fit_transform(train_features_clean)
    poly_features_test = poly.transform(test_features_clean)
    
    feature_names = (
        list(numeric_cols) + 
        [f"poly_{i}" for i in range(poly_features_train.shape[1] - len(numeric_cols))]
    )
    
    # Add power transformer for better feature distribution
    print("Applying power transformation...")
    power = PowerTransformer(method='yeo-johnson')
    X = power.fit_transform(poly_features_train)
    
    # Scale features
    print("Scaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    y = train_data['CO2'].values
    
    # Hyperparameter optimization
    print("Optimizing hyperparameters...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_scaled, y, cv),
        n_trials=100,
        timeout=3600
    )
    
    best_params = study.best_params
    best_params['tree_method'] = 'hist'
    best_params['random_state'] = 42
    print("Best parameters:", best_params)
    
    # Train final model with cross-validation
    print("Training final model...")
    predictions_cv = np.zeros(len(test_features))
    
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_scaled)):
        print(f"Training fold {fold + 1}/5...")
        fold_model = XGBRegressor(**best_params)
        fold_model.fit(X_scaled[train_idx], y[train_idx])
        
        # Transform test features
        test_transformed = power.transform(poly_features_test)
        test_scaled = scaler.transform(test_transformed)
        fold_preds = fold_model.predict(test_scaled)
        predictions_cv += fold_preds / 5
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'CO2': predictions_cv
    })
    
    # Handle missing IDs
    mean_pred = predictions_cv.mean()
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
        'importance': fold_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Log top 10 important features
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()