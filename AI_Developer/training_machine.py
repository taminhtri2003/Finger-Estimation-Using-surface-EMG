import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Feature Extraction Functions ---
def extract_features(emg_window):
    """Extract features from an EMG window (n_samples x n_channels)."""
    if emg_window.shape[0] < 2:  # Need at least 2 samples for diff-based features
        return np.zeros(40)  # Return zeros if window is too small
    features = []
    mav = np.mean(np.abs(emg_window), axis=0)
    features.extend(mav)
    rms = np.sqrt(np.mean(emg_window**2, axis=0))
    features.extend(rms)
    var = np.var(emg_window, axis=0)
    features.extend(var)
    wl = np.sum(np.abs(np.diff(emg_window, axis=0)), axis=0)
    features.extend(wl)
    zc = np.sum(np.abs(np.diff(np.sign(emg_window), axis=0)) / 2, axis=0)
    features.extend(zc)
    return np.array(features)

def process_emg_features(emg_data, window_size=200, stride=10):
    """Process EMG data into features over sliding windows."""
    n_samples, n_channels = emg_data.shape
    if n_samples < window_size:
        # If data is too short, use the whole segment as one window
        return extract_features(emg_data).reshape(1, -1)
    feature_data = []
    for start in range(0, n_samples - window_size + 1, stride):
        window = emg_data[start:start + window_size, :]
        features = extract_features(window)
        feature_data.append(features)
    return np.array(feature_data)

# --- Load Data ---
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

emg_cols = ['EMG_APL', 'EMG_FCR', 'EMG_FDS', 'EMG_FDP', 'EMG_ED', 'EMG_EI', 'EMG_ECU', 'EMG_ECR']
angle_cols = ['Thumb1', 'Thumb2', 'Index1', 'Index2', 'Index3', 'Middle1', 'Middle2', 'Middle3',
              'Ring1', 'Ring2', 'Ring3', 'Little1', 'Little2', 'Little3']

# --- Extract Features for Training Data ---
window_size = 50
stride = 10

X_train_features = []
y_train = []
for trial in [1, 2, 3]:
    for task in range(1, 6):
        subset = train_df[(train_df['Trial'] == trial) & (train_df['Task'] == task)]
        emg_data = subset[emg_cols].values
        angle_data = subset[angle_cols].values
        
        print(f"Trial {trial}, Task {task}: EMG shape = {emg_data.shape}, Angles shape = {angle_data.shape}")
        
        if emg_data.shape[0] == 0:
            print(f"Warning: No data for Trial {trial}, Task {task}. Skipping.")
            continue
        
        features = process_emg_features(emg_data, window_size, stride)
        print(f"  Features shape = {features.shape}")
        
        if features.size == 0:
            print(f"Warning: No features extracted for Trial {trial}, Task {task}. Skipping.")
            continue
        
        n_windows = features.shape[0]
        angle_indices = np.linspace(0, len(angle_data) - 1, n_windows, dtype=int)
        angles_downsampled = angle_data[angle_indices, :]
        
        X_train_features.append(features)
        y_train.append(angles_downsampled)

# Filter out empty arrays and concatenate
X_train_features = [f for f in X_train_features if f.size > 0]
y_train = [y for y in y_train if y.size > 0]

if not X_train_features:
    raise ValueError("No valid training features extracted. Check data or parameters.")
    
X_train = np.vstack(X_train_features)
y_train = np.vstack(y_train)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# --- Extract Features for Testing Data ---
X_test_features = []
y_test = []
for trial in [4, 5]:
    for task in range(1, 6):
        subset = test_df[(test_df['Trial'] == trial) & (test_df['Task'] == task)]
        emg_data = subset[emg_cols].values
        angle_data = subset[angle_cols].values
        
        print(f"Trial {trial}, Task {task}: EMG shape = {emg_data.shape}, Angles shape = {angle_data.shape}")
        
        if emg_data.shape[0] == 0:
            print(f"Warning: No data for Trial {trial}, Task {task}. Skipping.")
            continue
        
        features = process_emg_features(emg_data, window_size, stride)
        print(f"  Features shape = {features.shape}")
        
        if features.size == 0:
            print(f"Warning: No features extracted for Trial {trial}, Task {task}. Skipping.")
            continue
        
        n_windows = features.shape[0]
        angle_indices = np.linspace(0, len(angle_data) - 1, n_windows, dtype=int)
        angles_downsampled = angle_data[angle_indices, :]
        
        X_test_features.append(features)
        y_test.append(angles_downsampled)

X_test_features = [f for f in X_test_features if f.size > 0]
y_test = [y for y in y_test if y.size > 0]

if not X_test_features:
    raise ValueError("No valid testing features extracted. Check data or parameters.")

X_test = np.vstack(X_test_features)
y_test = np.vstack(y_test)
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- Preprocessing ---
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# --- Train Random Forest Model ---
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("Training Random Forest model with extracted features...")
model.fit(X_train_scaled, y_train)

# --- Predict and Evaluate ---
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

print("\nEvaluation Metrics:")
for i, angle in enumerate(angle_cols):
    print(f"{angle}: MSE = {mse[i]:.4f}, R² = {r2[i]:.4f}")
print(f"\nAverage MSE: {np.mean(mse):.4f}")
print(f"Average R²: {np.mean(r2):.4f}")

# --- Visualization (Example: Trial 4, Task 1) ---
test_subset = test_df[(test_df['Trial'] == 4) & (test_df['Task'] == 1)]
if not test_subset.empty:
    emg_segment = test_subset[emg_cols].values
    angles_segment = test_subset[angle_cols].values
    features_segment = process_emg_features(emg_segment, window_size, stride)
    features_segment_scaled = scaler_X.transform(features_segment)
    pred_segment = model.predict(features_segment_scaled)

    plt.figure(figsize=(12, 6))
    time = np.arange(len(pred_segment))
    for i, angle in enumerate(['Thumb1', 'Index1', 'Middle1']):
        plt.subplot(3, 1, i+1)
        true_segment = angles_segment[::stride][:len(pred_segment), angle_cols.index(angle)]
        plt.plot(time, true_segment, label='True', color='blue')
        plt.plot(time, pred_segment[:, angle_cols.index(angle)], label='Predicted', color='red', linestyle='--')
        plt.title(f'{angle} (Trial 4, Task 1)')
        plt.xlabel('Time (windows)')
        plt.ylabel('Angle (degrees)')
        plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Warning: No data for Trial 4, Task 1 visualization.")

# --- Feature Importance ---
feature_names = [f"{feat}_{ch}" for ch in emg_cols for feat in ['MAV', 'RMS', 'VAR', 'WL', 'ZC']]
print("\nFeature Importance (Top 10):")
importances = model.feature_importances_
idx = np.argsort(importances)[::-1]
for i in idx[:10]:
    print(f"{feature_names[i]}: {importances[i]:.4f}")