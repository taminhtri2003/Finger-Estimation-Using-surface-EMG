# -*- coding: utf-8 -*-
"""
Pipeline for predicting joint angles from sEMG data using feature extraction
and XGBoost with Leave-One-Subject-Out Cross-Validation (LOSO CV).
Includes detailed visualizations.

Pipeline Steps:
1. Load Data: Read .mat files for multiple subjects.
2. Preprocessing: Bandpass and Notch filtering. (Visualize Raw vs Filtered)
3. Segmentation: Create overlapping windows.
4. Feature Extraction: TD, FD, TFW features per window. (Visualize Feature Dist)
5. Target Alignment: Match features windows with joint angles.
6. Model Training: Train XGBoost regressors (one per angle) using LOSO CV.
7. Evaluation: Calculate RMSE and R-squared for each fold. (Visualize Pred vs Actual)
8. Aggregate Results & Visualize Overall Performance.
9. Explainability (Optional): Use SHAP with visualizations.
"""

import os
import numpy as np
import scipy.io
import scipy.signal
import pandas as pd
import pywt # PyWavelets for Time-Frequency features
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
# import shap # Uncomment if you want to run SHAP analysis

# --- Configuration Constants ---
# NOTE: Update these paths and parameters as needed!
FILE_PATHS = [
    "s1_full.mat", # Replace with actual path to s1_full.mat
    "s2_full.mat", # Replace with actual path to s2_full.mat
    "s3_full.mat", # Replace with actual path to s3_full.mat
    "s4_full.mat", # Replace with actual path to s4_full.mat
    "s5_full.mat", # Replace with actual path to s5_full.mat
]
N_SUBJECTS = len(FILE_PATHS)
N_TRIALS = 5
N_TASKS = 7
N_CHANNELS = 8
N_ANGLES = 14

# --- Visualization Flags ---
PLOT_PREPROCESSING = True   # Plot raw vs filtered EMG for one segment?
PLOT_FEATURE_DIST = True    # Plot distribution of RMS for Ch1 of first subject?
PLOT_CV_PREDICTIONS = True  # Plot predicted vs actual for first angle in first fold?
PLOT_OVERALL_METRICS = True # Plot bar charts of average RMSE/R2 per angle?
RUN_SHAP_ANALYSIS = False   # Run and plot SHAP analysis? (Requires uncommenting shap import)

# --- Preprocessing Parameters ---
FS = 1000  # Sampling Frequency (Hz) - ASSUMPTION! Verify this.
LOWCUT = 20.0
HIGHCUT = 450.0
NOTCH_FREQ = 50.0 # Use 60 if needed
FILTER_ORDER = 4
Q_FACTOR = 30

# --- Segmentation Parameters ---
WINDOW_DURATION_MS = 200
OVERLAP_PERCENT = 50
WINDOW_SIZE = int(WINDOW_DURATION_MS * FS / 1000)
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENT / 100))

# --- Feature Extraction Parameters ---
WAVELET_FAMILY = 'db4'
WAVELET_LEVELS = 3

# --- Model Parameters ---
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# --- Helper Functions (load_mat_data, preprocess_emg, segment_data, feature calculators) ---
# (Keep the helper functions as they were in the previous version)
def load_mat_data(filepath):
    """Loads data from a single .mat file."""
    try:
        mat_data = scipy.io.loadmat(filepath)
        print(f"Successfully loaded: {filepath}")
        emg_data = mat_data.get('dsfilt_emg', None)
        angle_data = mat_data.get('joint_angles', None)

        if emg_data is None or angle_data is None:
            print(f"Warning: 'dsfilt_emg' or 'joint_angles' not found in {filepath}")
            return None, None
        if not (isinstance(emg_data, np.ndarray) and emg_data.shape == (N_TRIALS, N_TASKS)):
             print(f"Warning: Unexpected structure for 'dsfilt_emg' in {filepath}")
             return None, None
        if not (isinstance(angle_data, np.ndarray) and angle_data.shape == (N_TRIALS, N_TASKS)):
             print(f"Warning: Unexpected structure for 'joint_angles' in {filepath}")
             return None, None
        return emg_data, angle_data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def preprocess_emg(emg_segment, fs=FS, lowcut=LOWCUT, highcut=HIGHCUT, notch_freq=NOTCH_FREQ, order=FILTER_ORDER):
    """Applies bandpass and notch filtering to a single EMG segment."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b_band, a_band = scipy.signal.butter(order, [low, high], btype='band')
    emg_filtered_band = scipy.signal.filtfilt(b_band, a_band, emg_segment, axis=0)
    b_notch, a_notch = scipy.signal.iirnotch(notch_freq, Q_FACTOR, fs)
    emg_filtered_notch = scipy.signal.filtfilt(b_notch, a_notch, emg_filtered_band, axis=0)
    return emg_filtered_notch

def segment_data(data, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Segments data into overlapping windows."""
    num_samples = data.shape[0]
    num_features = data.shape[1] if data.ndim > 1 else 1
    num_windows = (num_samples - window_size) // step_size + 1

    if num_windows <= 0:
        return np.array([]).reshape(0, window_size, num_features), np.array([])

    if data.ndim > 1:
        windows = np.lib.stride_tricks.as_strided(
            data,
            shape=(num_windows, window_size, num_features),
            strides=(data.strides[0]*step_size, data.strides[0], data.strides[1])
        )
    else: # Handle 1D data if needed, though EMG is 2D
         windows = np.lib.stride_tricks.as_strided(
            data,
            shape=(num_windows, window_size),
            strides=(data.strides[0]*step_size, data.strides[0])
        )

    start_indices = np.arange(0, num_samples - window_size + 1, step_size)
    end_indices = start_indices + window_size - 1
    return windows, end_indices


def calculate_td_features(window):
    """Calculates Time Domain (TD) features for a single window (all channels)."""
    rms = np.sqrt(np.mean(window**2, axis=0))
    mav = np.mean(np.abs(window), axis=0)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    zc = np.sum((window[:-1, :] * window[1:, :] < 0), axis=0)
    diff_sig = np.diff(window, axis=0)
    ssc = np.sum((diff_sig[:-1, :] * diff_sig[1:, :] < 0), axis=0)
    # Order: [RMS_ch1..N, MAV_ch1..N, WL_ch1..N, ZC_ch1..N, SSC_ch1..N]
    return np.concatenate([rms, mav, wl, zc, ssc])

def calculate_fd_features(window, fs=FS):
    """Calculates Frequency Domain (FD) features using FFT."""
    n_samples, n_channels = window.shape
    fft_vals = np.fft.fft(window, axis=0)
    fft_freq = np.fft.fftfreq(n_samples, 1.0/fs)
    positive_freq_indices = fft_freq >= 0
    fft_vals = np.abs(fft_vals[positive_freq_indices, :])
    fft_freq = fft_freq[positive_freq_indices]
    power_spectrum = fft_vals**2
    total_power = np.sum(power_spectrum, axis=0)
    total_power[total_power == 0] = 1e-10
    mnf = np.sum(fft_freq[:, np.newaxis] * power_spectrum, axis=0) / total_power
    cumulative_power = np.cumsum(power_spectrum, axis=0)
    median_freq_indices = np.array([np.searchsorted(cumulative_power[:, ch], total_power[ch] / 2.0) for ch in range(n_channels)])
    median_freq_indices = np.clip(median_freq_indices, 0, len(fft_freq) - 1)
    mdf = fft_freq[median_freq_indices]
    peak_freq_indices = np.argmax(power_spectrum, axis=0)
    peak_freq = fft_freq[peak_freq_indices]
    # Order: [MNF_ch1..N, MDF_ch1..N, PeakF_ch1..N]
    return np.concatenate([mnf, mdf, peak_freq])

def calculate_tfw_features(window, wavelet=WAVELET_FAMILY, level=WAVELET_LEVELS):
    """Calculates Time-Frequency Wavelet (TFW) features."""
    coeffs = pywt.wavedec(window, wavelet=wavelet, level=level, axis=0)
    tfw_features = []
    # Coeffs[0]=App(L), Coeffs[1]=Det(L), ..., Coeffs[L]=Det(1)
    for i, coeff_level in enumerate(coeffs):
        level_name = f"A{level}" if i == 0 else f"D{level-i+1}"
        mean_coeffs = np.mean(coeff_level, axis=0)
        var_coeffs = np.var(coeff_level, axis=0)
        energy_coeffs = np.sum(coeff_level**2, axis=0)
        tfw_features.extend([mean_coeffs, var_coeffs, energy_coeffs])
        # Store names for later use if needed (complex to track globally here)
    # Order: [MeanA_ch1..N, VarA_ch1..N, EnergyA_ch1..N, MeanDL_ch1..N, ...]
    return np.concatenate(tfw_features)

def extract_features_for_segment(emg_segment, angle_segment, fs=FS, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Extracts all features for a given EMG segment and aligns with angles."""
    if emg_segment.shape[0] < window_size:
        print(f"Warning: Segment length ({emg_segment.shape[0]}) < window size ({window_size}). Skipping.")
        return np.array([]), np.array([])

    emg_processed = preprocess_emg(emg_segment, fs)
    emg_windows, window_end_indices = segment_data(emg_processed, window_size, step_size)

    if emg_windows.size == 0:
         print(f"Warning: No windows generated for a segment.")
         return np.array([]), np.array([])

    num_windows = emg_windows.shape[0]
    all_features_list = []
    for i in range(num_windows):
        window = emg_windows[i, :, :]
        td_feats = calculate_td_features(window)
        fd_feats = calculate_fd_features(window, fs)
        tfw_feats = calculate_tfw_features(window)
        combined_feats = np.concatenate([td_feats, fd_feats, tfw_feats])
        all_features_list.append(combined_feats)

    features_array = np.array(all_features_list)
    valid_indices = window_end_indices[window_end_indices < angle_segment.shape[0]]
    if len(valid_indices) != num_windows:
        print(f"Warning: Mismatch windows ({num_windows}) vs valid angle indices ({len(valid_indices)}). Trimming.")
        features_array = features_array[:len(valid_indices)]
        if features_array.size == 0: return np.array([]), np.array([])

    target_angles = angle_segment[valid_indices, :]
    return features_array, target_angles

# --- Feature Name Generation ---
def get_feature_names(n_channels=N_CHANNELS, wavelet=WAVELET_FAMILY, level=WAVELET_LEVELS):
    """Generates a list of feature names based on the extraction process."""
    feature_names = []
    td_base = ['RMS', 'MAV', 'WL', 'ZC', 'SSC']
    fd_base = ['MNF', 'MDF', 'PeakF']
    tfw_stat_base = ['Mean', 'Var', 'Energy']

    # TD Features
    for name in td_base:
        feature_names.extend([f'{name}_Ch{ch+1}' for ch in range(n_channels)])
    # FD Features
    for name in fd_base:
        feature_names.extend([f'{name}_Ch{ch+1}' for ch in range(n_channels)])
    # TFW Features
    coeffs_info = pywt.wavedec(np.zeros((WINDOW_SIZE, n_channels)), wavelet=wavelet, level=level, axis=0)
    coeff_names = []
    for i in range(level + 1):
         level_name = f"A{level}" if i == 0 else f"D{level-i+1}"
         coeff_names.append(level_name)

    for stat in tfw_stat_base:
        for c_name in coeff_names:
             feature_names.extend([f'{stat}{c_name}_Ch{ch+1}' for ch in range(n_channels)])

    return feature_names

# --- Main Execution ---
print("Starting sEMG Analysis Pipeline...")
print(f"Configuration: FS={FS}Hz, Window={WINDOW_DURATION_MS}ms, Overlap={OVERLAP_PERCENT}%")

all_subjects_features = []
all_subjects_targets = []
subject_groups = []

# --- 1. Load Data & Process ---
print("\n--- Loading Data & Processing ---")
first_segment_processed = False # Flag for plotting preprocessing
for subj_idx, fpath in enumerate(FILE_PATHS):
    if not os.path.exists(fpath):
        print(f"FATAL ERROR: File not found: {fpath}. Exiting.")
        exit()

    emg_cell, angle_cell = load_mat_data(fpath)
    if emg_cell is None or angle_cell is None:
        print(f"Skipping Subject {subj_idx+1} due to loading errors.")
        continue

    subj_features_list = []
    subj_targets_list = []

    for trial in range(N_TRIALS):
        for task in range(N_TASKS):
            emg_segment = emg_cell[trial, task]
            angle_segment = angle_cell[trial, task]

            # Validation checks
            if not isinstance(emg_segment, np.ndarray) or emg_segment.ndim != 2 or emg_segment.shape[1] != N_CHANNELS: continue
            if not isinstance(angle_segment, np.ndarray) or angle_segment.ndim != 2 or angle_segment.shape[1] != N_ANGLES: continue
            if emg_segment.shape[0] != angle_segment.shape[0]: continue
            if emg_segment.shape[0] < WINDOW_SIZE: continue # Ensure segment is long enough

            # --- Visualize Preprocessing (First Valid Segment Only) ---
            if PLOT_PREPROCESSING and not first_segment_processed:
                print("Plotting Raw vs Filtered EMG (First Segment)...")
                emg_processed_example = preprocess_emg(emg_segment, FS)
                time_axis = np.arange(emg_segment.shape[0]) / FS
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.plot(time_axis, emg_segment[:, 0], label='Raw EMG (Ch 1)')
                plt.title(f'Preprocessing Example (Subject {subj_idx+1}, Trial {trial+1}, Task {task+1})')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid(True)

                plt.subplot(2, 1, 2)
                plt.plot(time_axis, emg_processed_example[:, 0], label='Filtered EMG (Ch 1)', color='red')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                first_segment_processed = True # Plot only once

            # --- Feature Extraction ---
            features, targets = extract_features_for_segment(emg_segment, angle_segment)
            if features.size > 0 and targets.size > 0:
                subj_features_list.append(features)
                subj_targets_list.append(targets)

    # Concatenate for the subject
    if subj_features_list:
        subj_X = np.concatenate(subj_features_list, axis=0)
        subj_Y = np.concatenate(subj_targets_list, axis=0)
        all_subjects_features.append(subj_X)
        all_subjects_targets.append(subj_Y)
        subject_groups.extend([subj_idx] * subj_X.shape[0])
        print(f"Subject {subj_idx+1}: Processed {subj_X.shape[0]} windows, {subj_X.shape[1]} features.")

        # --- Visualize Feature Distribution (First Subject Only) ---
        if PLOT_FEATURE_DIST and subj_idx == 0:
             print("Plotting Feature Distribution (RMS Ch1, First Subject)...")
             # Find the index for RMS_Ch1
             feature_names_temp = get_feature_names()
             try:
                 rms_ch1_index = feature_names_temp.index('RMS_Ch1')
                 plt.figure(figsize=(8, 5))
                 sns.histplot(subj_X[:, rms_ch1_index], kde=True)
                 plt.title(f'Distribution of RMS_Ch1 (Subject {subj_idx+1})')
                 plt.xlabel('RMS Value')
                 plt.ylabel('Frequency')
                 plt.grid(True)
                 plt.tight_layout()
                 plt.show()
             except ValueError:
                 print("Warning: Could not find 'RMS_Ch1' in generated feature names.")
             except IndexError:
                  print(f"Warning: Index {rms_ch1_index} out of bounds for features.")

    else:
        print(f"Subject {subj_idx+1}: No valid features/targets extracted.")

if not all_subjects_features:
    print("\nFATAL ERROR: No features extracted. Check data paths/integrity. Exiting.")
    exit()

# Combine data from all subjects
X_all = np.concatenate(all_subjects_features, axis=0)
Y_all = np.concatenate(all_subjects_targets, axis=0)
groups = np.array(subject_groups)
feature_names = get_feature_names() # Get full list of names

# Verify feature name count matches data
if len(feature_names) != X_all.shape[1]:
    print(f"FATAL ERROR: Feature name count ({len(feature_names)}) != data columns ({X_all.shape[1]}). Check feature extraction/naming. Exiting.")
    # Fallback (less informative):
    # feature_names = [f'Feature_{i+1}' for i in range(X_all.shape[1])]
    exit()


print(f"\nTotal dataset size: {X_all.shape[0]} windows, {X_all.shape[1]} features.")
print(f"Generated {len(feature_names)} feature names.")

# --- 6 & 7. LOSO Cross-Validation with Internal Scaling ---
print("\n--- Starting Leave-One-Subject-Out Cross-Validation ---")
logo = LeaveOneGroupOut()
fold_results = {angle: {'rmse': [], 'r2': []} for angle in range(N_ANGLES)}
# trained_models_per_fold = [] # Optional: store models

for fold, (train_idx, test_idx) in enumerate(logo.split(X_all, Y_all, groups)):
    X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
    Y_train, Y_test = Y_all[train_idx], Y_all[test_idx]
    test_subject_id = groups[test_idx][0]

    print(f"\nFold {fold+1}/{N_SUBJECTS}: Testing on Subject {test_subject_id+1}")
    print(f"  Train size: {X_train_raw.shape[0]}, Test size: {X_test_raw.shape[0]}")

    # --- Scale features *inside* the loop ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw) # Use transform, not fit_transform on test data

    fold_models = []
    Y_pred_fold = np.zeros_like(Y_test)

    for angle_idx in range(N_ANGLES):
        print(f"  Training model for Angle {angle_idx+1}...")
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, Y_train[:, angle_idx])
        print(f"  Predicting for Angle {angle_idx+1}...")
        Y_pred_fold[:, angle_idx] = model.predict(X_test)
        # fold_models.append(model) # Optional

        rmse = np.sqrt(mean_squared_error(Y_test[:, angle_idx], Y_pred_fold[:, angle_idx]))
        r2 = r2_score(Y_test[:, angle_idx], Y_pred_fold[:, angle_idx])
        fold_results[angle_idx]['rmse'].append(rmse)
        fold_results[angle_idx]['r2'].append(r2)
        # print(f"    Angle {angle_idx+1} - RMSE: {rmse:.4f}, R2: {r2:.4f}") # Verbose

    # --- Visualize Predictions vs Actual (First Fold, First Angle Only) ---
    if PLOT_CV_PREDICTIONS and fold == 0:
        print("Plotting Predicted vs Actual (First Fold, First Angle)...")
        angle_to_plot = 0
        plt.figure(figsize=(7, 7))
        plt.scatter(Y_test[:, angle_to_plot], Y_pred_fold[:, angle_to_plot], alpha=0.5)
        # Add identity line
        lims = [
            np.min([plt.xlim(), plt.ylim()]), # min of both axes
            np.max([plt.xlim(), plt.ylim()]), # max of both axes
        ]
        plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0, label='y=x')
        plt.xlabel(f'Actual Angle {angle_to_plot+1}')
        plt.ylabel(f'Predicted Angle {angle_to_plot+1}')
        plt.title(f'Prediction vs Actual (Fold 1, Angle {angle_to_plot+1}, Subject {test_subject_id+1})')
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # Ensure square aspect ratio
        plt.tight_layout()
        plt.show()

# --- 8. Aggregate and Visualize Overall Results ---
print("\n--- Cross-Validation Results (Averaged over Folds) ---")
avg_rmse = []
avg_r2 = []
angle_labels = [f'Angle {i+1}' for i in range(N_ANGLES)]
for angle_idx in range(N_ANGLES):
    angle_rmse = np.mean(fold_results[angle_idx]['rmse'])
    angle_r2 = np.mean(fold_results[angle_idx]['r2'])
    avg_rmse.append(angle_rmse)
    avg_r2.append(angle_r2)
    print(f"{angle_labels[angle_idx]}: Average RMSE = {angle_rmse:.4f}, Average R2 = {angle_r2:.4f}")

print(f"\nOverall Average RMSE across all angles: {np.mean(avg_rmse):.4f}")
print(f"Overall Average R2 across all angles: {np.mean(avg_r2):.4f}")

if PLOT_OVERALL_METRICS:
    print("Plotting Overall Performance Metrics...")
    x_angles = np.arange(N_ANGLES)
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for RMSE
    color = 'tab:red'
    ax1.set_xlabel('Joint Angle Index')
    ax1.set_ylabel('Average RMSE', color=color)
    bars1 = ax1.bar(x_angles - width/2, avg_rmse, width, label='Avg RMSE', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x_angles)
    ax1.set_xticklabels(angle_labels, rotation=45, ha="right")
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Instantiate a second axes that shares the same x-axis for R2
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Average R-squared', color=color)
    bars2 = ax2.bar(x_angles + width/2, avg_r2, width, label='Avg R2', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    # Set R2 y-limit appropriately, e.g., from slightly below min R2 to 1
    min_r2 = min(0, np.min(avg_r2) - 0.1)
    ax2.set_ylim([min_r2, 1.05])


    fig.suptitle('Average LOSO CV Performance per Joint Angle')
    # Add legends (getting handles and labels from both axes)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)

    fig.tight_layout(rect=[0, 0.1, 1, 0.96]) # Adjust layout to prevent title overlap and make space for legend
    plt.show()


# --- 9. Explainability with SHAP (Optional) ---
if RUN_SHAP_ANALYSIS:
    try:
        import shap # Import here to avoid error if not installed and not used
        print("\n--- Training Final Models and Running SHAP ---")

        # Need to scale the full dataset for the final model training
        print("Scaling full dataset for final model training...")
        final_scaler = StandardScaler()
        X_scaled_final = final_scaler.fit_transform(X_all)

        final_models = []
        for angle_idx in range(N_ANGLES):
            print(f"Training final model for Angle {angle_idx+1}...")
            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(X_scaled_final, Y_all[:, angle_idx])
            final_models.append(model)

        # Calculate SHAP values (can be slow, consider sampling)
        target_angle_for_shap = 0 # Choose which angle model to explain
        print(f"\nCalculating SHAP values for Angle {target_angle_for_shap + 1}...")
        explainer = shap.TreeExplainer(final_models[target_angle_for_shap])

        # Use a subset for faster SHAP calculation (e.g., 1000 samples)
        shap_sample_size = min(1000, X_scaled_final.shape[0])
        X_shap_sample = shap.sample(X_scaled_final, shap_sample_size, random_state=42)
        print(f"Using {shap_sample_size} samples for SHAP calculation.")
        shap_values = explainer.shap_values(X_shap_sample) # Pass sample here

        # Use the same sample for plotting if using sample for shap_values
        X_plot_sample = X_shap_sample
        # Or use the full data for plotting context if shap_values were calculated on full data
        # X_plot_sample = X_scaled_final

        # SHAP Summary Plot (Global Importance)
        print("Generating SHAP summary plot...")
        plt.figure() # Create a new figure context for the plot
        shap.summary_plot(shap_values, X_plot_sample, feature_names=feature_names, show=False)
        plt.title(f"SHAP Feature Importance for Angle {target_angle_for_shap + 1}")
        plt.tight_layout()
        # plt.savefig(f'shap_summary_angle_{target_angle_for_shap + 1}.png') # Optional save
        plt.show()

        # SHAP Force Plot (Local Explanation - First Instance of Sample)
        instance_index_in_sample = 0
        print(f"\nGenerating SHAP force plot for instance {instance_index_in_sample} (Angle {target_angle_for_shap + 1})...")
        # Create the plot object
        force_plot_html = shap.force_plot(explainer.expected_value,
                                          shap_values[instance_index_in_sample,:],
                                          X_plot_sample[instance_index_in_sample,:],
                                          feature_names=feature_names,
                                          show=False) # Don't show automatically
        # Save as HTML (recommended for interactive force plots)
        shap.save_html(f'shap_force_plot_angle_{target_angle_for_shap + 1}_instance_{instance_index_in_sample}.html', force_plot_html)
        print(f"Force plot saved to shap_force_plot_angle_{target_angle_for_shap + 1}_instance_{instance_index_in_sample}.html")
        # Displaying force plots directly might require specific environments (like Jupyter)

    except ImportError:
        print("\nWarning: 'shap' library not installed. Skipping SHAP analysis.")
    except Exception as e:
        print(f"\nError during SHAP analysis: {e}. Skipping.")


print("\nPipeline execution finished.")
