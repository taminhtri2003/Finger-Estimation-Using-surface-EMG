import numpy as np
import scipy.io
from scipy import signal
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the .mat file
data = scipy.io.loadmat('s1_full.mat')

# Function to extract EMG features
def extract_emg_features(emg_data, fs=2000):
    features = []
    for channel in range(emg_data.shape[1]):
        signal_data = emg_data[:, channel]
        
        # Time-domain features
        mav = np.mean(np.abs(signal_data))  # Mean Absolute Value
        rms = np.sqrt(np.mean(signal_data**2))  # Root Mean Square
        var = np.var(signal_data)  # Variance
        wl = np.sum(np.abs(np.diff(signal_data)))  # Waveform Length
        
        # Frequency-domain features
        f, psd = signal.welch(signal_data, fs=fs)
        power = np.trapz(psd, f)  # Total Power
        peak_freq = f[np.argmax(psd)]  # Peak Frequency
        
        channel_features = [mav, rms, var, wl, power, peak_freq]
        features.extend(channel_features)
    
    return np.array(features)

# Prepare data
n_trials = 5
n_tasks = 5
feature_matrix = []
target_matrix = []

for trial in range(n_trials):
    for task in range(n_tasks):
        # Extract EMG data and features
        emg_data = data['dsfilt_emg'][trial, task]
        emg_features = extract_emg_features(emg_data)
        
        # Get joint angles
        joint_angles = data['joint_angles'][trial, task]
        mean_joint_angles = np.mean(joint_angles, axis=0)  # Average over time
        
        feature_matrix.append(emg_features)
        target_matrix.append(mean_joint_angles)

feature_matrix = np.array(feature_matrix)
target_matrix = np.array(target_matrix)

# Split data: trials 1-3 for training (indices 0-14), 4-5 for testing (indices 15-24)
X_train = feature_matrix[:15]  # 3 trials * 5 tasks = 15 samples
X_test = feature_matrix[15:]   # 2 trials * 5 tasks = 10 samples
y_train = target_matrix[:15]
y_test = target_matrix[15:]

# Feature names (6 features per 8 channels)
feature_names = []
muscle_names = ['APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR']
feature_types = ['MAV', 'RMS', 'VAR', 'WL', 'Power', 'PeakFreq']
for muscle in muscle_names:
    for feat in feature_types:
        feature_names.append(f'{muscle}_{feat}')

# Train models and calculate performance metrics
n_joints = 14
models = []
train_r2_scores = []
test_r2_scores = []
train_rmse_scores = []
test_rmse_scores = []

for joint in range(n_joints):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train[:, joint])
    models.append(rf)
    
    # Training predictions and metrics
    y_train_pred = rf.predict(X_train)
    train_r2 = r2_score(y_train[:, joint], y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train[:, joint], y_train_pred))
    train_r2_scores.append(train_r2)
    train_rmse_scores.append(train_rmse)
    
    # Testing predictions and metrics
    y_test_pred = rf.predict(X_test)
    test_r2 = r2_score(y_test[:, joint], y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test[:, joint], y_test_pred))
    test_r2_scores.append(test_r2)
    test_rmse_scores.append(test_rmse)

# Display performance metrics
performance_df = pd.DataFrame({
    'Joint': [f'Joint {i+1}' for i in range(n_joints)],
    'Train R²': train_r2_scores,
    'Test R²': test_r2_scores,
    'Train RMSE': train_rmse_scores,
    'Test RMSE': test_rmse_scores
})
print("\nModel Performance Metrics:")
print(performance_df.round(4))

# Visualization 1: Bar Plot for R² Scores
plt.figure(figsize=(12, 6))
sns.barplot(data=performance_df.melt(id_vars=['Joint'], value_vars=['Train R²', 'Test R²'], 
                                     var_name='Dataset', value_name='R²'), 
            x='Joint', y='R²', hue='Dataset', palette='viridis')
plt.title('R² Scores Across Joints', fontsize=16, pad=15)
plt.xlabel('Joint', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Dataset', title_fontsize=12, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Visualization 2: Line Plot with Bands for RMSE
plt.figure(figsize=(12, 6))
joints = np.arange(1, n_joints + 1)
plt.plot(joints, train_rmse_scores, label='Train RMSE', marker='o', color='teal', linewidth=2)
plt.plot(joints, test_rmse_scores, label='Test RMSE', marker='o', color='coral', linewidth=2)
plt.fill_between(joints, train_rmse_scores, test_rmse_scores, color='gray', alpha=0.2, 
                 label='Difference')
plt.title('RMSE Across Joints', fontsize=16, pad=15)
plt.xlabel('Joint', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.xticks(joints, [f'Joint {i}' for i in range(1, n_joints + 1)], rotation=45)
plt.legend(title='Metric', title_fontsize=12, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Sensor names
sensor_names = ['APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR']
n_features_per_sensor = 6  # MAV, RMS, VAR, WL, Power, PeakFreq
n_sensors = len(sensor_names)

# Aggregate SHAP values per sensor across all joints
all_sensor_shap = np.zeros((n_joints, X_test.shape[0], n_sensors))
for joint in range(n_joints):
    explainer = shap.TreeExplainer(models[joint])
    shap_values = explainer.shap_values(X_test)
    for sensor_idx in range(n_sensors):
        sensor_start = sensor_idx * n_features_per_sensor
        sensor_end = (sensor_idx + 1) * n_features_per_sensor
        all_sensor_shap[joint, :, sensor_idx] = np.sum(np.abs(shap_values[:, sensor_start:sensor_end]), axis=1)

# Average across joints for overall sensor importance
mean_sensor_shap = np.mean(all_sensor_shap, axis=0)  # Shape: (n_samples, n_sensors)

# 1. Bar Plot - Global Sensor Importance
plt.figure(figsize=(10, 6))
shap.summary_plot(mean_sensor_shap, feature_names=sensor_names, plot_type="bar", show=False)
plt.title('Global Sensor Importance Across All Joints (Bar Plot)')
plt.tight_layout()
plt.show()

# 2. Violin Plot - Distribution of Sensor Contributions
plt.figure(figsize=(12, 6))
shap.summary_plot(mean_sensor_shap, feature_names=sensor_names, plot_type="violin", show=False)
plt.title('Distribution of Sensor Contributions (Violin Plot)')
plt.tight_layout()
plt.show()

# 3. Waterfall Plot - Sensor Contributions for a Single Prediction
explainer = shap.TreeExplainer(models[0])  # Using Joint 1 as example
shap_values = explainer.shap_values(X_test[0:1])  # First test sample
sensor_shap = np.zeros(n_sensors)
for sensor_idx in range(n_sensors):
    sensor_start = sensor_idx * n_features_per_sensor
    sensor_end = (sensor_idx + 1) * n_features_per_sensor
    sensor_shap[sensor_idx] = np.sum(shap_values[0, sensor_start:sensor_end])

plt.figure(figsize=(12, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=sensor_shap,
        base_values=explainer.expected_value,
        feature_names=sensor_names
    ),
    max_display=8
)
plt.title('Waterfall Plot - Sensor Contributions for Joint 1, First Sample')
plt.tight_layout()
plt.show()

# 4. Decision Plot - Cumulative Sensor Impact
plt.figure(figsize=(12, 6))
shap.decision_plot(
    explainer.expected_value,
    mean_sensor_shap,
    feature_names=sensor_names,
    highlight=[0],  # Highlight first sample
    show=False
)
plt.title('Decision Plot - Cumulative Sensor Impact Across Samples')
plt.tight_layout()
plt.show()

# 5. Custom Embedding Plot - Sensor Contributions with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, X_test.shape[0]-1))
embedding = tsne.fit_transform(mean_sensor_shap)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=np.sum(mean_sensor_shap, axis=1), cmap='viridis')
plt.colorbar(scatter, label='Total SHAP Contribution')
for i, name in enumerate(sensor_names):
    plt.annotate(name, (np.mean(embedding[mean_sensor_shap[:, i] > 0, 0]), 
                       np.mean(embedding[mean_sensor_shap[:, i] > 0, 1])), 
                 alpha=0.7)
plt.title('t-SNE Embedding of Sensor Contributions')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.show()

# 6. Custom Heatmap - Sensor Contributions Across Samples
plt.figure(figsize=(12, 8))
sns.heatmap(mean_sensor_shap.T, cmap='viridis', xticklabels=True, yticklabels=sensor_names)
plt.xlabel('Test Sample Index')
plt.ylabel('Sensors')
plt.title('Heatmap - Sensor Contributions Across Samples')
plt.tight_layout()
plt.show()

# 7. Contribution Stacked Area Plot - Sensor Impact Over Samples
plt.figure(figsize=(12, 6))
for sensor_idx in range(n_sensors):
    plt.fill_between(range(X_test.shape[0]), 
                     np.cumsum(np.abs(mean_sensor_shap[:, :sensor_idx+1]), axis=1)[:, sensor_idx],
                     label=sensor_names[sensor_idx])
plt.xlabel('Test Sample Index')
plt.ylabel('Cumulative Absolute SHAP Value')
plt.title('Stacked Area Plot - Sensor Contributions Across Samples')
plt.legend()
plt.tight_layout()
plt.show()

# 8. Pie Chart - Relative Sensor Importance
plt.figure(figsize=(10, 10))
plt.pie(np.mean(np.abs(mean_sensor_shap), axis=0), labels=sensor_names, autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart - Relative Sensor Importance')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Summary of Top Sensors
sensor_importance_df = pd.DataFrame({
    'Sensor': sensor_names,
    'Mean_SHAP': np.mean(np.abs(mean_sensor_shap), axis=0)
}).sort_values('Mean_SHAP', ascending=False)
print("\nTop Contributing Sensors Across All Joints:")
print(sensor_importance_df.round(4))