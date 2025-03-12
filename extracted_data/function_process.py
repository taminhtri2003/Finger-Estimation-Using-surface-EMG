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
import shap  # Import SHAP
# Removed tensorflow.keras imports
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
    kin_cols = train_data.columns[8:-2]

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
     # Select only the first 5 kinematic features (tasks)
    kin_cols = kin_cols[:5]
    kin_data = kin_data[kin_cols]
    
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
        #'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0)),
        #'Lasso Regression': MultiOutputRegressor(Lasso(alpha=0.001)),
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

# 7. Build and train deep learning models  - REMOVED and replaced with a placeholder
def train_evaluate_dl_models(X_train, X_val, y_train, y_val, y_scaler, epochs=50, batch_size=32):
    print("Deep learning models removed.")
    return {}


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

    # Select the first 5 kinematics dimensions (tasks)
    selected_dims = list(range(5))  # Select the first 5 columns
    dim_labels = kin_cols[:5]  # Use the first 5 column names as labels


    # Plot actual vs predicted values for each selected dimension
    fig, axes = plt.subplots(len(selected_dims), 1, figsize=(14, 3*len(selected_dims)))

    for i, dim in enumerate(selected_dims):
        if len(selected_dims) > 1:   #when len(selected_dims) ==1, axes will be an object but not an array, so cannot be indexed.
            ax = axes[i]
        else:
            ax = axes #when len(selected_dims) ==1, axes is the axes object itself
        ax.plot(true_subset[:, dim], 'b-', label='Actual')
        ax.plot(pred_subset[:, dim], 'r-', label='Predicted')
        ax.set_title(f"{dim_labels[i]}")
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

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

    # Select the first 5 kinematics
    kin_dims = list(range(5))
    kin_names = kin_cols[:5]  #Use first five kin_cols as names.

    # Create scatter plots showing relationships
    fig, axes = plt.subplots(1, 5, figsize=(20, 4)) #Adjust figsize based on number of kinematics dimensions

    for i, (dim, name) in enumerate(zip(kin_dims, kin_names)):
        if len(kin_dims) > 1: #when len(kin_dims) ==1, axes will be an object but not an array, so cannot be indexed.
            ax = axes[i]
        else:
            ax = axes  #when len(kin_dims) == 1, axes is the axes object itself.

        sns.scatterplot(
            x=X_subset[:, emg_feature_idx],
            y=y_subset[:, dim],
            ax=ax,
            alpha=0.6
        )

        # Add linear regression line
        sns.regplot(
            x=X_subset[:, emg_feature_idx],
            y=y_subset[:, dim],
            ax=ax,
            scatter=False,
            color='red'
        )

        ax.set_title(f'Relationship: {emg_feature_name} vs {name}')
        ax.set_xlabel(emg_feature_name)
        ax.set_ylabel(name)

    plt.tight_layout()
    plt.savefig("emg_kinematics_relationship.png")
    plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor
import warnings
import shap  # Import SHAP

warnings.filterwarnings('ignore')

# Set style for all plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ... (Rest of your code - unchanged) ...


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor
import warnings
import shap  # Import SHAP

warnings.filterwarnings('ignore')

# Set style for all plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ... (Rest of your code: load_data, preprocess_emg, extract_emg_features,
#      process_kinematics, prepare_train_val_sets, train_evaluate_ml_models,
#       visualize_model_comparison,
#      visualize_predictions, analyze_feature_importance,
#      visualize_emg_kinematics_relationship) ...

# --- XAI Function (SHAP) ---
def explain_random_forest_shap(rf_model, X_train, X_val, feature_names):
    """
    Applies SHAP (SHapley Additive exPlanations) to explain the output of a
    Random Forest model. Uses `shap.plots.bar` for bar plots.

    Args:
        rf_model: The trained Random Forest model.
        X_train: The training data (used to fit the explainer).
        X_val: The validation data (used for visualizations).
        feature_names: List of feature names (strings).

    Returns:
        None. Produces SHAP visualizations.
    """

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_val)

    print("\nSHAP Summary Plot (Global Feature Importance - Bar Plot):")
    if isinstance(shap_values, list):
        # Multi-output case
        shap_values_combined = np.abs(shap_values[0]).mean(axis=0)
        for k in range(1, len(shap_values)):
            shap_values_combined += np.abs(shap_values[k]).mean(axis=0)
        shap_values_combined /= len(shap_values)
        shap.summary_plot(shap_values, X_val, feature_names=feature_names, plot_type="bar", show=False)
    else:
        # Single-output case
        shap.summary_plot(shap_values, X_val, feature_names=feature_names, plot_type="bar", show=False)
    
    plt.savefig("shap_summary_plot_bar.png")
    plt.tight_layout()
    plt.show()

    print("\nSHAP Beeswarm Plot (Distribution of Feature Impact):")
    shap.summary_plot(shap_values, X_val, feature_names=feature_names, plot_type="beeswarm", show=False)
    plt.savefig("shap_beeswarm_plot.png")
    plt.show()

    print("\nSHAP Dependence Bar Plots (Individual Feature Importance):")
    # Calculate mean absolute SHAP values
    if isinstance(shap_values, list):
        shap_sum = np.abs(shap_values[0]).mean(axis=0)
        for k in range(1, len(shap_values)):
            shap_sum += np.abs(shap_values[k]).mean(axis=0)
    else:
        shap_sum = np.abs(shap_values).mean(axis=0)
    
    top_feature_indices = np.argsort(shap_sum)[::-1][:5]

    for i in top_feature_indices:
        feature_name = feature_names[i]
        plt.figure()
        
        if isinstance(shap_values, list):
            for output_idx in range(len(shap_values)):
                plt.clf()
                shap.summary_plot(shap_values[output_idx], X_val, feature_names=feature_names, 
                                  plot_type="dot", color=shap_values[output_idx][:, i], show=False)
                plt.title(f"SHAP Dependence for {feature_name} (Output {output_idx})")
                plt.savefig(f"shap_dependence_{feature_name.replace(' ', '_')}_output_{output_idx}.png")
                plt.close()
        else:
            plt.clf()
            shap.dependence_plot(feature_name, shap_values, X_val, feature_names=feature_names, show=False)
            plt.title(f"SHAP Dependence for {feature_name}")
            plt.savefig(f"shap_dependence_{feature_name.replace(' ', '_')}.png")
            plt.close()

    # Force Plot
    sample_index = 0
    print(f"\nSHAP Force Plot (Explanation for Sample {sample_index}):")
    if isinstance(shap_values, list):
        for output_idx in range(len(shap_values)):
            shap.force_plot(explainer.expected_value[output_idx], 
                            shap_values[output_idx][sample_index, :], 
                            X_val[sample_index, :], 
                            feature_names=feature_names, 
                            matplotlib=True, show=False)
            plt.savefig(f"shap_force_plot_output_{output_idx}.png")
            plt.close()
    else:
        shap.force_plot(explainer.expected_value, 
                        shap_values[sample_index, :], 
                        X_val[sample_index, :], 
                        feature_names=feature_names, 
                        matplotlib=True, show=False)
        plt.savefig("shap_force_plot.png")
        plt.close()