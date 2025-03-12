import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# 1. Load data
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Extract EMG and kinematics columns
    emg_cols = train_data.columns[:8]
    kin_cols = train_data.columns[8:]
    
    print(f"EMG sensor labels: {', '.join(emg_cols)}")
    print(f"Number of kinematics features: {len(kin_cols)} (23 sensors × 3 axes)")
    
    return train_data, test_data, emg_cols, kin_cols

# 2. Preprocessing EMG data
def preprocess_emg(data, emg_cols, plot_example=True):
    emg_data = data[emg_cols].copy()
    processed_emg = pd.DataFrame()
    
    # Apply preprocessing steps to each EMG channel
    for col in emg_cols:
        # Step 1: Remove DC offset (mean subtraction)
        zero_mean = emg_data[col] - emg_data[col].mean()
        
        # Step 2: Bandpass filter (20-450 Hz is typical for EMG)
        # Simulating this since we don't have sampling frequency info
        # In practice, we would use: filtered = signal.butter_bandpass_filter(zero_mean, 20, 450, fs, order=4)
        filtered = zero_mean
        
        # Step 3: Full-wave rectification
        rectified = np.abs(filtered)
        
        # Step 4: Smoothing with moving average filter (simulate envelope)
        window_size = 10  # Adjust based on your data
        smoothed = rectified.rolling(window=window_size, center=True).mean().fillna(rectified)
        
        # Store processed signal
        processed_emg[col] = smoothed
    
    # Plot example of raw vs processed EMG for one channel
    if plot_example:
        sensor_to_plot = emg_cols[0]  # First EMG sensor
        plt.figure(figsize=(14, 6))
        
        plt.subplot(2, 1, 1)
        plt.title(f"Raw EMG signal - {sensor_to_plot}")
        plt.plot(emg_data[sensor_to_plot].iloc[:500], 'b-')
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.title(f"Processed EMG signal - {sensor_to_plot}")
        plt.plot(processed_emg[sensor_to_plot].iloc[:500], 'r-')
        plt.ylabel("Amplitude")
        plt.xlabel("Samples")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("emg_preprocessing.png")
        plt.show()
    
    return processed_emg

# 3. Feature extraction from EMG data
def extract_emg_features(processed_emg, window_size=50, overlap=0.5):
    features_list = []
    stride = int(window_size * (1 - overlap))
    
    # Use sliding window to extract features
    for start in range(0, processed_emg.shape[0] - window_size, stride):
        end = start + window_size
        window_data = processed_emg.iloc[start:end]
        
        # Calculate features for each EMG channel
        window_features = {}
        for col in processed_emg.columns:
            # Time domain features
            window_features[f"{col}_mean"] = window_data[col].mean()
            window_features[f"{col}_std"] = window_data[col].std()
            window_features[f"{col}_max"] = window_data[col].max()
            window_features[f"{col}_min"] = window_data[col].min()
            window_features[f"{col}_rms"] = np.sqrt(np.mean(window_data[col] ** 2))
            window_features[f"{col}_iemg"] = np.sum(np.abs(window_data[col]))  # Integrated EMG
            window_features[f"{col}_mav"] = np.mean(np.abs(window_data[col]))  # Mean absolute value
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(window_data[col])))[0]
            window_features[f"{col}_zcr"] = len(zero_crossings)
            
            # Frequency domain features
            # Compute power spectrum
            ps = np.abs(np.fft.fft(window_data[col])) ** 2
            freqs = np.fft.fftfreq(window_size)
            idx = np.argsort(freqs)
            ps_sorted = ps[idx]
            freqs_sorted = freqs[idx]
            
            # Use only positive frequencies
            mask = freqs_sorted > 0
            ps_pos = ps_sorted[mask]
            freqs_pos = freqs_sorted[mask]
            
            if len(ps_pos) > 0:
                # Mean and median frequency
                total_power = np.sum(ps_pos)
                window_features[f"{col}_mean_freq"] = np.sum(freqs_pos * ps_pos) / total_power if total_power > 0 else 0
                
                # Power in different frequency bands (simulated)
                window_features[f"{col}_low_band_power"] = np.sum(ps_pos[:len(ps_pos)//3]) / total_power if total_power > 0 else 0
                window_features[f"{col}_mid_band_power"] = np.sum(ps_pos[len(ps_pos)//3:2*len(ps_pos)//3]) / total_power if total_power > 0 else 0
                window_features[f"{col}_high_band_power"] = np.sum(ps_pos[2*len(ps_pos)//3:]) / total_power if total_power > 0 else 0
        
        features_list.append(window_features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    print(f"Extracted {features_df.shape[1]} features from {len(processed_emg.columns)} EMG channels")
    
    return features_df

# 4. Process kinematics data to match feature windows
def process_kinematics(data, kin_cols, window_size=50, overlap=0.5):
    kin_data = data[kin_cols].copy()
    processed_kin = []
    stride = int(window_size * (1 - overlap))
    
    # Use sliding window to match EMG feature extraction
    for start in range(0, kin_data.shape[0] - window_size, stride):
        end = start + window_size
        window_data = kin_data.iloc[start:end]
        
        # For kinematics, we use the mean values over the window
        # This assumes the kinematics should represent the hand position/orientation
        # at the time corresponding to the EMG features
        processed_kin.append(window_data.mean())
    
    processed_kin_df = pd.DataFrame(processed_kin)
    return processed_kin_df

# 5. Split data for training and validation
def prepare_train_val_sets(X, y, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # Scale targets
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    return (
        X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled,
        X_train, X_val, y_train, y_val,
        scaler_X, scaler_y
    )

# 6. Train and evaluate machine learning models
def train_evaluate_ml_models(X_train, X_val, y_train, y_val, y_scaler):
    models = {
        'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0)),
        'Lasso Regression': MultiOutputRegressor(Lasso(alpha=0.001)),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred = model.predict(X_val)
        
        # Inverse transform to get original scale
        y_val_orig = y_scaler.inverse_transform(y_val)
        y_pred_orig = y_scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        mae = mean_absolute_error(y_val_orig, y_pred_orig)
        r2 = r2_score(y_val_orig, y_pred_orig)
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred_orig,
            'true_values': y_val_orig
        }
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return results

# 7. Build and train deep learning models
def build_lstm_model(input_shape, output_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_shape)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_dense_model(input_shape, output_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(output_shape)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_hybrid_model(input_shape, output_shape):
    # Custom architecture combining different neural network components
    
    # Input layer
    inputs = Input(shape=(input_shape,))
    
    # Main branch
    x1 = Dense(128, activation='relu')(inputs)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(64, activation='relu')(x1)
    
    # LSTM branch (reshape input for LSTM)
    reshaped = Reshape((input_shape // 8, 8))(inputs) if input_shape % 8 == 0 else Reshape((input_shape // 4, 4))(inputs)
    x2 = LSTM(32, return_sequences=False)(reshaped)
    x2 = Dropout(0.3)(x2)
    
    # Merge branches
    merged = Concatenate()([x1, x2])
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.3)(merged)
    
    # Output layer
    outputs = Dense(output_shape, activation='linear')(merged)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

def train_evaluate_dl_models(X_train, X_val, y_train, y_val, y_scaler, epochs=50, batch_size=32):
    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]
    
    # Prepare LSTM input format (if needed)
    # For the hybrid model that uses an LSTM component
    
    # Dictionary to store models
    dl_models = {
        'Dense Neural Network': build_dense_model(input_shape, output_shape),
        'Hybrid NN-LSTM': build_hybrid_model(input_shape, output_shape)
    }
    
    # Check if we have enough data for sequence modeling
    if X_train.shape[0] > 100:  # Arbitrary threshold, adjust as needed
        # Reshape data for LSTM
        # We'll use each feature as a time step for this example
        lstm_X_train = X_train.reshape(X_train.shape[0], -1, 1)
        lstm_X_val = X_val.reshape(X_val.shape[0], -1, 1)
        
        # Add LSTM model
        dl_models['LSTM Network'] = build_lstm_model((lstm_X_train.shape[1], lstm_X_train.shape[2]), output_shape)
    
    results = {}
    
    for name, model in dl_models.items():
        print(f"Training {name}...")
        
        # Use appropriate data format based on model type
        if name == 'LSTM Network':
            history = model.fit(
                lstm_X_train, y_train,
                validation_data=(lstm_X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            y_pred = model.predict(lstm_X_val)
        else:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            y_pred = model.predict(X_val)
        
        # Inverse transform to get original scale
        y_val_orig = y_scaler.inverse_transform(y_val)
        y_pred_orig = y_scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        mae = mean_absolute_error(y_val_orig, y_pred_orig)
        r2 = r2_score(y_val_orig, y_pred_orig)
        
        results[name] = {
            'model': model,
            'history': history,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred_orig,
            'true_values': y_val_orig
        }
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{name} - Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.savefig(f"{name.replace(' ', '_').lower()}_training_history.png")
        plt.show()
    
    return results

# 8. Visualize and compare model performance
def visualize_model_comparison(ml_results, dl_results):
    # Combine results from ML and DL models
    all_results = {**ml_results, **dl_results}
    
    # Extract metrics for comparison
    models = list(all_results.keys())
    rmse_values = [all_results[model]['rmse'] for model in models]
    mae_values = [all_results[model]['mae'] for model in models]
    r2_values = [all_results[model]['r2'] for model in models]
    
    # Create a DataFrame for easy plotting
    results_df = pd.DataFrame({
        'Model': models,
        'RMSE': rmse_values,
        'MAE': mae_values,
        'R²': r2_values
    })
    
    # Sort by RMSE (ascending)
    results_df = results_df.sort_values('RMSE')
    
    # Plot comparison charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE comparison
    sns.barplot(x='RMSE', y='Model', data=results_df, ax=axes[0], palette='Blues_d')
    axes[0].set_title('RMSE Comparison (lower is better)')
    axes[0].set_xlabel('RMSE')
    
    # MAE comparison
    sns.barplot(x='MAE', y='Model', data=results_df, ax=axes[1], palette='Greens_d')
    axes[1].set_title('MAE Comparison (lower is better)')
    axes[1].set_xlabel('MAE')
    
    # R² comparison
    sns.barplot(x='R²', y='Model', data=results_df, ax=axes[2], palette='Reds_d')
    axes[2].set_title('R² Comparison (higher is better)')
    axes[2].set_xlabel('R²')
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()
    
    # Create a table with metrics
    print("\nModel Performance Metrics:")
    print(results_df.to_string(index=False))
    
    return results_df

# 9. Visualize predicted vs actual kinematics
def visualize_predictions(model_name, predictions, true_values, kin_cols, num_samples=100):
    # Select a subset of samples to make the plot readable
    pred_subset = predictions[:num_samples]
    true_subset = true_values[:num_samples]
    
    # Select a few representative kinematics dimensions to visualize
    # We'll pick 3 dimensions: one for each axis (x, y, z) across different sensors
    # This assumes the 69 columns are organized as 23 sensors × 3 axes (x, y, z)
    n_sensors = len(kin_cols) // 3
    selected_dims = [
        0,                  # First sensor, x-axis
        n_sensors,          # First sensor, y-axis
        2 * n_sensors,      # First sensor, z-axis
        n_sensors // 2,     # Middle sensor, x-axis
        n_sensors + n_sensors // 2,  # Middle sensor, y-axis
        2 * n_sensors + n_sensors // 2  # Middle sensor, z-axis
    ]
    
    # Create labels for the selected dimensions
    dim_labels = [f"Sensor {d//3 + 1}, {'xyz'[d%3]}-axis" for d in selected_dims]
    
    # Plot actual vs predicted values for each selected dimension
    fig, axes = plt.subplots(len(selected_dims), 1, figsize=(14, 3*len(selected_dims)))
    
    for i, dim in enumerate(selected_dims):
        axes[i].plot(true_subset[:, dim], 'b-', label='Actual')
        axes[i].plot(pred_subset[:, dim], 'r-', label='Predicted')
        axes[i].set_title(f"{dim_labels[i]}")
        axes[i].set_xlabel('Sample')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_predictions.png")
    plt.show()

# 10. Analyze importance of EMG features
def analyze_feature_importance(rf_model, feature_names):
    # Extract feature importances from Random Forest model
    importances = rf_model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Get top features
    top_n = 20
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title('Top EMG Features by Importance')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

# 11. Visualize EMG-kinematics relationships
def visualize_emg_kinematics_relationship(X, y, emg_features, kin_cols, n_samples=1000):
    # Select a subset of samples for visualization
    X_subset = X[:n_samples]
    y_subset = y[:n_samples]
    
    # Select an informative EMG feature (e.g., RMS of a channel)
    emg_feature_idx = [i for i, name in enumerate(emg_features) if 'rms' in name.lower()][0]
    emg_feature_name = emg_features[emg_feature_idx]
    
    # Select a few kinematics dimensions (one from each axis group)
    n_sensors = len(kin_cols) // 3
    kin_dims = [0, n_sensors, 2*n_sensors]  # x, y, z of first sensor
    kin_names = [f"Sensor 1 ({axis})" for axis in ['x', 'y', 'z']]
    
    # Create scatter plots showing relationships
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (dim, name) in enumerate(zip(kin_dims, kin_names)):
        sns.scatterplot(
            x=X_subset[:, emg_feature_idx],
            y=y_subset[:, dim],
            ax=axes[i],
            alpha=0.6
        )
        
        # Add linear regression line
        sns.regplot(
            x=X_subset[:, emg_feature_idx],
            y=y_subset[:, dim],
            ax=axes[i],
            scatter=False,
            color='red'
        )
        
        axes[i].set_title(f'Relationship: {emg_feature_name} vs {name}')
        axes[i].set_xlabel(emg_feature_name)
        axes[i].set_ylabel(name)
    
    plt.tight_layout()
    plt.savefig("emg_kinematics_relationship.png")
    plt.show()

# 12. Main workflow
def main():
    # Load data
    train_data, test_data, emg_cols, kin_cols = load_data('train_data.csv', 'test_data.csv')
    
    # Preprocess EMG data
    processed_train_emg = preprocess_emg(train_data, emg_cols, plot_example=True)
    processed_test_emg = preprocess_emg(test_data, emg_cols, plot_example=False)
    
    # Extract features from EMG
    train_emg_features = extract_emg_features(processed_train_emg)
    test_emg_features = extract_emg_features(processed_test_emg)
    
    # Process kinematics to match EMG feature windows
    train_kin_processed = process_kinematics(train_data, kin_cols)
    test_kin_processed = process_kinematics(test_data, kin_cols)
    
    # Ensure matching number of samples
    min_samples = min(train_emg_features.shape[0], train_kin_processed.shape[0])
    train_emg_features = train_emg_features.iloc[:min_samples]
    train_kin_processed = train_kin_processed.iloc[:min_samples]
    
    min_samples_test = min(test_emg_features.shape[0], test_kin_processed.shape[0])
    test_emg_features = test_emg_features.iloc[:min_samples_test]
    test_kin_processed = test_kin_processed.iloc[:min_samples_test]
    
    # Prepare training and validation sets
    (X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled,
     X_train, X_val, y_train, y_val,
     X_scaler, y_scaler) = prepare_train_val_sets(
        train_emg_features, train_kin_processed
    )
    
    # Train and evaluate machine learning models
    ml_results = train_evaluate_ml_models(
        X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, y_scaler
    )
    
    # Train and evaluate deep learning models
    dl_results = train_evaluate_dl_models(
        X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, y_scaler
    )
    
    # Visualize model comparison
    results_df = visualize_model_comparison(ml_results, dl_results)
    
    # Analyze feature importance (using Random Forest model)
    if 'Random Forest' in ml_results:
        analyze_feature_importance(
            ml_results['Random Forest']['model'],
            train_emg_features.columns
        )
    
    # Visualize predictions for the best model
    best_model_name = results_df.iloc[0]['Model']  # Model with lowest RMSE
    visualize_predictions(
        best_model_name,
        ml_results[best_model_name]['predictions'],
        ml_results[best_model_name]['true_values'], 
        kin_cols
    )
    
    # Visualize EMG-kinematics relationships
    visualize_emg_kinematics_relationship(
        X_train, y_train, train_emg_features.columns, kin_cols
    )
    
    # Test on the test set using the best model
    print(f"\nEvaluating best model ({best_model_name}) on test set:")
    
    # Scale test data
    X_test_scaled = X_scaler.transform(test_emg_features)
    
    # Get the best model
    best_model = all_results[best_model_name]['model']
    
    # Predict on test set
    if best_model_name == 'LSTM Network':
        # Reshape for LSTM
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], -1, 1)
        y_test_pred_scaled = best_model.predict(X_test_reshaped)
    else:
        y_test_pred_scaled = best_model.predict(X_test_scaled)
    
    # Inverse transform to get original scale
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)
    
    # Since we don't have ground truth for the test set in this example,
    # we'll just visualize the predictions
    plt.figure(figsize=(14, 8))
    for i in range(min(5, y_test_pred.shape[1])):
        plt.plot(y_test_pred[:100, i], label=f'Kinematics {i+1}')
    
    plt.title(f'Test Set Predictions ({best_model_name})')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("test_predictions.png")
    plt.show()
    
    print("\nAnalysis complete! The best model for EMG to kinematics mapping was:", best_model_name)
    print(f"With RMSE: {results_df.iloc[0]['RMSE']:.4f}, MAE: {results_df.iloc[0]['MAE']:.4f}, R²: {results_df.iloc[0]['R²']:.4f}")

# Example usage
if __name__ == "__main__":
    main()