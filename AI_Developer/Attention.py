# -*- coding: utf-8 -*-
"""
Python script for predicting finger kinematics from EMG using a
Multi-Head Attention (MHA) LSTM model with specific EMG preprocessing
and Integrated Gradients for XAI.

Includes:
- Custom MHA model with separate finger heads.
- R2/CC score plotting.
- Regression correlation scatter plots.
- MHA Attention Heatmap visualization (per head, normalized).
- Dynamic Simulation plot (EMG/True Angle/Predicted Angle/Attention).
- Integrated Gradients calculation and visualization.
- Synergy Attribution visualization (Temporal).
- Muscle Co-Attribution Matrix visualization.
- Phase-Space Plot (Angle vs. Velocity) for dynamics analysis.
- Attention vs. Integrated Gradients comparison plot.
- NEW: Kinematic Trajectory Comparison plot (similar to user image).

"""

import numpy as np
import scipy.io
import scipy.stats
# Removed: from scipy.signal import butter, sosfiltfilt # For filtering
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Layer, Dropout, Lambda, LayerNormalization, MultiHeadAttention,
    Concatenate
)
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Keep StandardScaler
from sklearn.metrics import r2_score
import os
import warnings
import math
import pandas as pd # Added for co-attribution matrix (optional: correlation method)
import seaborn as sns # Added for co-attribution matrix heatmap
from collections import defaultdict # Added for grouping trials by task

# --- Configuration ---
MAT_FILE_PATH = 's4_full.mat' # <<<--- IMPORTANT: Update this path
SEQUENCE_LENGTH = 200
PREDICTION_HORIZON = 1 # Keep as 1 for current model structure
BATCH_SIZE = 128
EPOCHS = 10 # Adjust as needed (can be increased for better performance)
LSTM_UNITS = 128
VALIDATION_SPLIT = 0.2
MAX_SAMPLES_TO_VIZ_MHA = 2 # Limit detailed MHA heatmap visualization output
NUM_TRIALS_TO_PLOT_DYNAMICS = 1 # Limit dynamic plot output
NUM_SAMPLES_FOR_XAI = 1 # How many samples to run XAI (set to 1 for faster run with new plots)
NUM_TASKS_TO_PLOT_KINEMATICS = 3 # How many tasks (columns) to show in the kinematic comparison plot

# MHA Configuration
NUM_ATTENTION_HEADS = 5 # Should match number of fingers
D_MODEL = LSTM_UNITS
# Ensure D_MODEL is divisible by NUM_ATTENTION_HEADS
if D_MODEL % NUM_ATTENTION_HEADS != 0:
     D_MODEL = math.ceil(D_MODEL / NUM_ATTENTION_HEADS) * NUM_ATTENTION_HEADS
     print(f"Adjusting D_MODEL to {D_MODEL} to be divisible by {NUM_ATTENTION_HEADS} heads.")

# Integrated Gradients Config
IG_STEPS = 50 # Number of steps for approximation
CO_ATTRIBUTION_THRESHOLD = 0.2 # Threshold for co-attribution matrix calculation

# EMG Channel Names (Ensure these match your .mat file structure)
EMG_CHANNEL_NAMES = ['APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR']
# Joint Angle Names (Ensure these match your .mat file structure)
JOINT_ANGLE_NAMES = [
    'Thumb 1', 'Thumb 2', 'Index 1', 'Index 2', 'Index 3', 'Middle 1', 'Middle 2', 'Middle 3',
    'Ring 1', 'Ring 2', 'Ring 3', 'Little 1', 'Little 2', 'Little 3'
]
# Finger to Joint Mapping (Ensure this is correct for your angle data)
FINGER_JOINT_INDICES = {
    'Thumb': list(range(0, 2)),
    'Index': list(range(2, 5)), # Index 1=MCP, Index 2=PIP, Index 3=DIP (Example mapping)
    'Middle': list(range(5, 8)),
    'Ring': list(range(8, 11)),
    'Little': list(range(11, 14))
}
FINGER_NAMES = list(FINGER_JOINT_INDICES.keys())

# --- Muscle Group Definitions (for coloring/grouping) ---
MUSCLE_GROUPS = {
    'Finger Flexors': ['FDS', 'FDP'],
    'Finger Extensors': ['ED', 'EI'],
    'Wrist Flex/Dev': ['FCR'],
    'Wrist Extend/Dev': ['ECU', 'ECR'],
    'Thumb': ['APL']
}
# Map group names to indices
MUSCLE_GROUP_INDICES = {
    name: [EMG_CHANNEL_NAMES.index(ch) for ch in channels if ch in EMG_CHANNEL_NAMES]
    for name, channels in MUSCLE_GROUPS.items()
}
# Assign colors for visualization
try: cmap = plt.colormaps.get_cmap('tab10')
except AttributeError: cmap = plt.cm.get_cmap('tab10') # Fallback for older matplotlib
GROUP_COLORS = cmap(np.linspace(0, 1, len(MUSCLE_GROUPS)))
MUSCLE_GROUP_VIS_INFO = {name: {'indices': MUSCLE_GROUP_INDICES[name], 'color': GROUP_COLORS[i]}
                         for i, name in enumerate(MUSCLE_GROUPS.keys())}
print("Defined Muscle Groups (Indices):", MUSCLE_GROUP_INDICES)

# --- Define Synergies (Example - Adapt based on actual physiological knowledge) ---
SYNERGIES = {
    'Flexion Synergy': ['FCR', 'FDS', 'FDP'], # Example: Wrist/Finger Flexors
    'Extension Synergy': ['ED', 'ECU', 'ECR'], # Example: Finger/Wrist Extensors
    'Thumb Synergy': ['APL'] # Example: Thumb Abduction/Extension
    # Add more synergies if needed, e.g., 'Intrinsic Synergy': ['EI'] ?
}
# Map synergy names to EMG channel indices
SYNERGY_INDICES = {
    name: [EMG_CHANNEL_NAMES.index(ch) for ch in channels if ch in EMG_CHANNEL_NAMES]
    for name, channels in SYNERGIES.items()
}
print("Defined Synergies (Indices):", SYNERGY_INDICES)

# --- Custom Multi-Head Attention Layer ---
class CustomMultiHeadAttentionSeparateOutput(Layer):
    """ Custom MHA layer returning separate context vectors per head. """
    def __init__(self, num_heads, d_model, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        # Define Dense layers for Query, Key, Value projections
        self.wq = Dense(d_model, name='q_proj')
        self.wk = Dense(d_model, name='k_proj')
        self.wv = Dense(d_model, name='v_proj')

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len, depth)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(q, k, transpose_b=True) # (..., seq_len_q, seq_len_k)
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # Add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        # Softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)
        return output, attention_weights

    def call(self, query, value, key, mask=None, return_attention_scores=False):
        batch_size = tf.shape(query)[0]
        # Project Q, K, V
        q = self.wq(query)  # (batch_size, seq_len_q, d_model)
        k = self.wk(key)    # (batch_size, seq_len_k, d_model)
        v = self.wv(value)  # (batch_size, seq_len_v, d_model)
        # Split heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # Calculate attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        # Transpose back: (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # --- Modification: Return list of head outputs ---
        # Each element: (batch_size, seq_len_q, depth)
        head_outputs = [scaled_attention[:, :, i, :] for i in range(self.num_heads)]

        # Squeeze if sequence length is 1 (as in decoder input)
        if query.shape[1] == 1:
             head_outputs_squeezed = [tf.squeeze(head, axis=1) for head in head_outputs]
        else:
             head_outputs_squeezed = head_outputs # Keep sequence dim if > 1

        # Squeeze attention weights if seq_len_q is 1
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k) -> (batch_size, num_heads, seq_len_k)
        if query.shape[1] == 1:
            squeezed_attention_weights = tf.squeeze(attention_weights, axis=2)
        else:
            # Keep seq_len_q if > 1 (though typically not needed for this architecture)
            squeezed_attention_weights = attention_weights

        if return_attention_scores:
            return head_outputs_squeezed, squeezed_attention_weights
        else:
            return head_outputs_squeezed

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
        })
        return config

# --- 1. Data Loading ---
def load_mat_data(filepath):
    """Loads EMG and joint angle data from the specified .mat file."""
    print(f"Loading data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None, None, None
    try:
        mat_data = scipy.io.loadmat(filepath)
        # Adjust keys based on your actual .mat file structure
        emg_data_cells = mat_data.get('dsfilt_emg', mat_data.get('emg_data')) # Try common keys
        joint_angles_cells = mat_data.get('joint_angles', mat_data.get('angle_data')) # Try common keys
        # --- Attempt to load kinematics/displacement data if available ---
        # Use .get() to avoid errors if the key doesn't exist
        kinematics_data = mat_data.get('finger_kinematics', None)
        displacement_data_cells = mat_data.get('displacement_data', None) # Add another potential key

        if emg_data_cells is None or joint_angles_cells is None:
             print("Error: Could not find EMG or angle data using common keys ('dsfilt_emg', 'emg_data', 'joint_angles', 'angle_data'). Check .mat file structure.")
             return None, None, None, None # Added None for displacement

        print("Data loaded successfully.")
        # Return displacement data if found, otherwise None
        return emg_data_cells, joint_angles_cells, kinematics_data, displacement_data_cells
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return None, None, None, None # Added None for displacement

# --- 2. Data Preprocessing ---
def preprocess_data(emg_data_cells, joint_angles_cells, sequence_length, prediction_horizon):
    """ Preprocesses data for LSTM training: flattens trials, creates sequences, normalizes, splits. """
    print("Preprocessing data for LSTM training...")
    all_emg_sequences, all_angle_targets = [], []
    original_scaled_trials = [] # Store original trials (scaled) for dynamic plotting
    num_trials, num_tasks = emg_data_cells.shape
    emg_scaler = StandardScaler()
    angle_scaler = StandardScaler()

    # --- Fit scalers first (on raw data from all valid trials) ---
    print("Fitting scalers...")
    emg_list_for_scaling = []
    angle_list_for_scaling = []
    valid_trial_indices = [] # Keep track of which trials were used

    for i in range(num_trials):
        for j in range(num_tasks):
            # Check if data exists and is numpy array with 2 dimensions and some rows
            valid_emg = isinstance(emg_data_cells[i, j], np.ndarray) and \
                        emg_data_cells[i, j].ndim == 2 and \
                        emg_data_cells[i, j].shape[0] > 0
            valid_angle = isinstance(joint_angles_cells[i, j], np.ndarray) and \
                          joint_angles_cells[i, j].ndim == 2 and \
                          joint_angles_cells[i, j].shape[0] > 0

            if valid_emg and valid_angle:
                # Ensure consistent length and minimum length for sequence creation
                min_len = min(emg_data_cells[i,j].shape[0], joint_angles_cells[i,j].shape[0])
                # Use a slightly larger buffer for kinematic plot alignment later if needed
                if min_len >= sequence_length + prediction_horizon + 5: # Added buffer
                    trial_emg = emg_data_cells[i, j][:min_len, :]
                    trial_angle = joint_angles_cells[i, j][:min_len, :]

                    # Check for NaNs/Infs before adding
                    if not np.any(np.isnan(trial_emg)) and not np.any(np.isinf(trial_emg)) and \
                       not np.any(np.isnan(trial_angle)) and not np.any(np.isinf(trial_angle)):
                        emg_list_for_scaling.append(trial_emg)
                        angle_list_for_scaling.append(trial_angle)
                        valid_trial_indices.append((i, j))
                    else:
                         print(f"Warning: NaN/Inf detected in trial ({i},{j}). Skipping.")
            # else:
            #     print(f"Warning: Invalid data format or shape for trial ({i},{j}). Skipping.")


    if not emg_list_for_scaling:
        print("Error: No valid data/trials long enough found for scaling.")
        return None, None, None, None, None, None, None

    # Concatenate all valid trial data for fitting scalers
    full_emg_data = np.vstack(emg_list_for_scaling)
    full_angle_data = np.vstack(angle_list_for_scaling)

    # Fit the scalers
    emg_scaler.fit(full_emg_data)
    angle_scaler.fit(full_angle_data)
    print("Scalers fitted.")

    # --- Create sequences and store original scaled trials ---
    print(f"Creating sequences from {len(valid_trial_indices)} valid trials...")
    trial_count = 0
    for idx, (i, j) in enumerate(valid_trial_indices):
        # Get the original raw data again
        trial_emg_raw = emg_list_for_scaling[idx] # Use the validated data
        trial_angle_raw = angle_list_for_scaling[idx]

        # Scale the full raw trial using the fitted scalers
        trial_emg_scaled = emg_scaler.transform(trial_emg_raw)
        trial_angle_scaled = angle_scaler.transform(trial_angle_raw)

        # Store the scaled trial for later visualization
        original_scaled_trials.append({
            'emg': trial_emg_scaled,
            'angle': trial_angle_scaled,
            'id': f"T{i+1}_Task{j+1}" # Original trial identifier
        })

        # Create sequences from this scaled trial
        for k in range(len(trial_emg_scaled) - sequence_length - prediction_horizon + 1):
            emg_sequence = trial_emg_scaled[k : k + sequence_length]
            angle_target = trial_angle_scaled[k + sequence_length : k + sequence_length + prediction_horizon]

            # If predicting only one step ahead, remove the time dimension from the target
            if prediction_horizon == 1:
                angle_target = angle_target.squeeze(axis=0)

            all_emg_sequences.append(emg_sequence)
            all_angle_targets.append(angle_target)
        trial_count += 1

    if not all_emg_sequences:
        print("Error: Could not create sequences from valid trials.")
        return None, None, None, None, None, None, None

    X = np.array(all_emg_sequences)
    y = np.array(all_angle_targets)
    print(f"Created {X.shape[0]} sequences from {trial_count} trials.")
    print(f"Input shape (X): {X.shape}, Target shape (y): {y.shape}")

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, shuffle=True
    )
    print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")

    return X_train, X_val, y_train, y_val, emg_scaler, angle_scaler, original_scaled_trials


# --- NEW: Preprocessing for Kinematic Comparison Figure ---
def preprocess_for_kinematic_figure(kinematic_data_cells, num_tasks_to_use, angle_names, target_kinematic_indices):
    """
    Groups kinematic data by task, calculates mean trajectories.
    Args:
        kinematic_data_cells: Raw data (e.g., joint_angles_cells or displacement_data_cells).
        num_tasks_to_use: How many tasks (columns in the plot) to process.
        angle_names: List of names for the kinematic features.
        target_kinematic_indices: Dictionary mapping plot label (e.g., 'MCP') to data index.
    Returns:
        Dictionary {task_id: {joint_label: mean_trajectory}} and max_len.
    """
    print(f"\nPreprocessing data for Kinematic Comparison Figure...")
    if kinematic_data_cells is None:
        print("Error: Kinematic data is None. Cannot preprocess.")
        return None, 0

    num_trials, num_total_tasks = kinematic_data_cells.shape
    num_tasks = min(num_tasks_to_use, num_total_tasks)
    print(f"Processing data for the first {num_tasks} tasks.")

    task_data = defaultdict(lambda: defaultdict(list)) # task_id -> joint_label -> list_of_trial_trajectories
    max_len = 0 # Track max length for potential padding/plotting

    for task_idx in range(num_tasks):
        for trial_idx in range(num_trials):
            # Check data validity
            kin_trial_task = kinematic_data_cells[trial_idx, task_idx]
            if isinstance(kin_trial_task, np.ndarray) and kin_trial_task.ndim == 2 and kin_trial_task.shape[0] > 0:
                # Check for NaNs/Infs
                 if not np.any(np.isnan(kin_trial_task)) and not np.any(np.isinf(kin_trial_task)):
                    trial_len = kin_trial_task.shape[0]
                    max_len = max(max_len, trial_len)
                    # Extract data for target joints
                    for joint_label, data_index in target_kinematic_indices.items():
                        if 0 <= data_index < kin_trial_task.shape[1]:
                            task_data[task_idx][joint_label].append(kin_trial_task[:, data_index])
                        else:
                             print(f"Warning: Index {data_index} for '{joint_label}' out of bounds for task {task_idx}, trial {trial_idx}.")
                 else:
                      print(f"Warning: NaN/Inf in kinematic data for task {task_idx}, trial {trial_idx}. Skipping.")
            # else:
            #      print(f"Warning: Invalid kinematic data format/shape for task {task_idx}, trial {trial_idx}. Skipping.")


    if not task_data:
        print("Error: No valid kinematic data found after grouping.")
        return None, 0

    # Calculate mean trajectories - requires handling variable lengths
    # Simplest approach: truncate all trials in a group to the minimum length within that group
    mean_task_data = defaultdict(dict)
    final_max_len = 0 # Max length after alignment

    print("Calculating mean trajectories (truncating to min length per task/joint)...")
    for task_idx, joint_dict in task_data.items():
        for joint_label, trajectories in joint_dict.items():
            if not trajectories: continue # Skip if no valid trials for this joint/task

            # Find minimum length for this specific group of trials
            min_len_group = min(len(t) for t in trajectories)
            final_max_len = max(final_max_len, min_len_group)

            # Truncate and stack
            truncated_trajectories = [t[:min_len_group] for t in trajectories]
            stacked_trajectories = np.stack(truncated_trajectories, axis=0) # Shape (num_trials, min_len_group)

            # Calculate mean and std dev
            mean_trajectory = np.mean(stacked_trajectories, axis=0)
            # std_trajectory = np.std(stacked_trajectories, axis=0) # Uncomment to calculate std dev

            mean_task_data[task_idx][joint_label] = mean_trajectory
            # mean_task_data[task_idx][joint_label + '_std'] = std_trajectory # Store std dev if needed

    print(f"Preprocessing for kinematic figure complete. Max aligned length: {final_max_len}")
    return mean_task_data, final_max_len


# --- 3. Model Architecture (MHA Redesign) ---
# (Same as previous version)
def build_finger_mha_lstm(sequence_length, num_emg_features, num_angle_features,
                          lstm_units, d_model, num_heads, finger_joint_indices):
    """ Builds LSTM with Custom MHA and separate finger prediction heads. """
    print("Building MHA model...")
    if num_heads != len(finger_joint_indices):
        raise ValueError("Number of heads must match number of fingers defined in FINGER_JOINT_INDICES.")

    encoder_inputs = Input(shape=(sequence_length, num_emg_features), name='emg_input')
    encoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_outputs_norm = LayerNormalization(epsilon=1e-6)(encoder_outputs)
    state_h_norm = LayerNormalization(epsilon=1e-6)(state_h)
    query = Lambda(lambda x: tf.expand_dims(x, axis=1), name='expand_query_dim')(state_h_norm)
    value = encoder_outputs_norm; key = encoder_outputs_norm
    mha_layer = CustomMultiHeadAttentionSeparateOutput(num_heads=num_heads, d_model=d_model, name='custom_mha')
    head_context_vectors, attention_weights = mha_layer(query=query, value=value, key=key, return_attention_scores=True)
    finger_outputs = []
    finger_names = list(finger_joint_indices.keys())
    for i in range(num_heads):
        finger_name = finger_names[i]; num_finger_joints = len(finger_joint_indices[finger_name])
        head_context = head_context_vectors[i]
        finger_dense1 = Dense(32, activation='relu', name=f'{finger_name}_dense1')(head_context)
        finger_dropout = Dropout(0.2, name=f'{finger_name}_dropout')(finger_dense1)
        finger_output = Dense(num_finger_joints, activation='linear', name=f'{finger_name}_output')(finger_dropout)
        finger_outputs.append(finger_output)
    final_output = Concatenate(axis=-1, name='final_concatenated_output')(finger_outputs)
    model = Model(inputs=encoder_inputs, outputs=final_output, name='Finger_MHA_Training_Model')
    attention_model = Model(inputs=encoder_inputs, outputs=attention_weights, name='Finger_MHA_Attention_Model')
    print("MHA models built successfully.")
    return model, attention_model

# --- 4. Training ---
# (Same as previous version)
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    """ Compiles and trains the model. """
    print("Compiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    print("Starting training...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    print("Training finished.")
    return history

# --- 5. Performance Evaluation & Visualization ---
# (Same as previous version)
def calculate_performance_scores(y_true, y_pred):
    """ Calculates R2 and Pearson CC scores for each output feature. """
    if y_true.shape != y_pred.shape: raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    if y_true.ndim != 2: raise ValueError(f"Expected 2D arrays, got shapes: y_true {y_true.shape}, y_pred {y_pred.shape}")
    num_features = y_true.shape[1]; r2_scores_list, cc_scores_list, cc_pvalues_list = [], [], []
    for i in range(num_features):
        r2 = r2_score(y_true[:, i], y_pred[:, i]); r2_scores_list.append(r2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cc, p_value = scipy.stats.pearsonr(y_true[:, i], y_pred[:, i])
            if np.isnan(cc): cc, p_value = 0.0, 1.0
        cc_scores_list.append(cc); cc_pvalues_list.append(p_value)
    return np.array(r2_scores_list), np.array(cc_scores_list), np.array(cc_pvalues_list)

def plot_performance_scores(r2_scores, cc_scores, angle_names):
    """ Creates bar plots for R2 and CC scores per joint angle. """
    num_angles = len(angle_names)
    if num_angles != len(r2_scores) or num_angles != len(cc_scores):
        print("Warning: Mismatch between number of angles and scores. Adjusting plot.")
        num_angles = min(len(r2_scores), len(cc_scores)); angle_names = angle_names[:num_angles]
        r2_scores = r2_scores[:num_angles]; cc_scores = cc_scores[:num_angles]
    x = np.arange(num_angles); width = 0.35; fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    rects1 = ax1.bar(x, r2_scores, width, label='R²', color='skyblue'); ax1.set_ylabel('R² Score'); ax1.set_title('R² Score per Joint Angle')
    min_r2 = min(0, np.min(r2_scores) - 0.1 if r2_scores.size > 0 else 0); ax1.set_ylim(min_r2, 1.05); ax1.axhline(0, color='grey', lw=0.8)
    ax1.legend(); ax1.grid(axis='y', linestyle='--', alpha=0.7)
    rects2 = ax2.bar(x, cc_scores, width, label='CC', color='lightcoral'); ax2.set_ylabel('Pearson CC Score'); ax2.set_title('Correlation Coefficient (CC) per Joint Angle')
    ax2.set_xticks(x); ax2.set_xticklabels(angle_names, rotation=45, ha='right'); ax2.set_ylim(-1.05, 1.05); ax2.axhline(0, color='grey', lw=0.8)
    ax2.legend(); ax2.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout(); plt.show()

def plot_regression_correlation(y_true, y_pred, angle_names, cc_scores=None):
    """ Creates scatter plots for true vs predicted values for each angle. """
    num_angles = y_true.shape[1]
    if num_angles != len(angle_names): angle_names = [f"Angle {i+1}" for i in range(num_angles)]
    ncols = 3; nrows = math.ceil(num_angles / ncols); fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=False, sharey=False)
    axes = axes.flatten()
    for i in range(num_angles):
        ax = axes[i]; true_vals = y_true[:, i]; pred_vals = y_pred[:, i]
        ax.scatter(true_vals, pred_vals, alpha=0.3, s=10, label='Predictions')
        min_val = min(np.min(true_vals), np.min(pred_vals)); max_val = max(np.max(true_vals), np.max(pred_vals))
        padding = (max_val - min_val) * 0.05; lims = [min_val - padding, max_val + padding]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y=x (Ideal)'); ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("True Values"); ax.set_ylabel("Predicted Values"); ax.set_title(f"{angle_names[i]}")
        ax.grid(True, linestyle='--', alpha=0.5)
        if cc_scores is not None and i < len(cc_scores):
            cc = cc_scores[i]; ax.text(0.05, 0.95, f'CC = {cc:.2f}', transform=ax.transAxes, fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
        ax.legend(loc='lower right', fontsize='small')
    for j in range(num_angles, len(axes)): fig.delaxes(axes[j])
    fig.suptitle("True vs. Predicted Values per Joint Angle", fontsize=16, y=1.02); fig.tight_layout(rect=[0, 0.03, 1, 0.98]); plt.show()


# --- 6. Attention Visualization (MHA Redesign - Updated) ---
# (Same as previous version)
def get_mha_attention_weights(input_data, attention_model):
    """ Extracts MHA attention weights using the dedicated attention model. """
    try:
        input_data_float32 = input_data.astype(np.float32)
        attention_weights = attention_model.predict(input_data_float32, verbose=0)
        return attention_weights
    except Exception as e: print(f"Error during MHA attention weight prediction: {e}"); return None

def plot_mha_attention_heatmap(input_emg_sequence, attention_weights_sample,
                               emg_channels, muscle_group_info, finger_names,
                               title="MHA Attention Analysis"):
    """ Plots one heatmap per head showing attended EMG activity (Normalized Per Head). """
    if attention_weights_sample is None: print("Cannot plot: Attention weights are None."); return
    num_heads, seq_len = attention_weights_sample.shape; num_emg = len(emg_channels)
    if input_emg_sequence is None or input_emg_sequence.shape != (seq_len, num_emg):
        print(f"Cannot plot: Input EMG sequence shape mismatch or is None."); return
    if num_heads != len(finger_names): finger_names = [f"Head {h+1}" for h in range(num_heads)]
    fig, axes = plt.subplots(num_heads, 1, figsize=(16, 3.5 * num_heads), sharex=True)
    if num_heads == 1: axes = [axes]
    fig.suptitle(title, fontsize=16, y=1.01)
    heatmap_data = np.zeros((num_heads, num_emg, seq_len))
    for h in range(num_heads):
        for t in range(seq_len): heatmap_data[h, :, t] = attention_weights_sample[h, t] * input_emg_sequence[t, :]
    for h in range(num_heads):
        ax = axes[h]; head_data = heatmap_data[h, :, :]
        vmin_head = np.min(head_data); vmax_head = np.max(head_data)
        if abs(vmax_head - vmin_head) < 1e-6: vmin_head -= 0.01; vmax_head += 0.01
        im = ax.imshow(head_data, aspect='auto', cmap='viridis', vmin=vmin_head, vmax=vmax_head)
        ax.set_yticks(np.arange(num_emg)); ax.set_yticklabels(emg_channels)
        finger_name = finger_names[h]; predicted_joints = ", ".join([JOINT_ANGLE_NAMES[idx] for idx in FINGER_JOINT_INDICES.get(finger_name, [])])
        ax.set_title(f"Head {h+1} ({finger_name}) - Attended EMG Activity (Predicts: {predicted_joints})"); ax.set_ylabel("EMG Channel")
        for group_name, info in muscle_group_info.items():
            for idx in info['indices']:
                if 0 <= idx < len(ax.get_yticklabels()): ax.get_yticklabels()[idx].set_color(info['color']); ax.get_yticklabels()[idx].set_fontweight('bold')
        cbar = fig.colorbar(im, ax=ax, label='Attention * Scaled EMG', shrink=0.9, pad=0.02); cbar.ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("Input Time Step within Sequence")
    legend_elements = [plt.Line2D([0], [0], color=info['color'], lw=4, label=name) for name, info in muscle_group_info.items()]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=len(muscle_group_info), title="Muscle Groups")
    plt.tight_layout(rect=[0, 0.05, 1, 0.98]); plt.show()

# --- 7. Dynamic EMG/Angle Simulation Plot ---
# (Same as previous version)
def plot_trial_dynamics_simulation(
    trial_emg_scaled, trial_angles_scaled, model, attention_model,
    emg_channels, joint_angle_names, muscle_group_info,
    finger_names, finger_joint_indices, sequence_length,
    prediction_horizon, trial_id="Unknown Trial"):
    """ Visualizes EMG group activity, true vs. SIMULATED joint angles, and approximated attention over a full trial."""
    print(f"\n--- Plotting Dynamics Simulation for Trial: {trial_id} ---")
    trial_len = trial_emg_scaled.shape[0]
    if trial_len < sequence_length: print(f"Trial too short. Skipping dynamics plot."); return
    num_windows = trial_len - sequence_length + 1
    X_windows = np.array([trial_emg_scaled[k : k + sequence_length] for k in range(num_windows)])
    X_windows_float32 = X_windows.astype(np.float32)
    print("Running model prediction for simulation..."); y_pred_windows_scaled = model.predict(X_windows_float32, batch_size=BATCH_SIZE, verbose=0)
    print("Extracting attention weights for simulation..."); mha_attention_weights_all_windows = get_mha_attention_weights(X_windows_float32, attention_model)
    if mha_attention_weights_all_windows is None: print("Could not get attention weights. Skipping dynamic plot."); return
    num_heads = mha_attention_weights_all_windows.shape[1]; groups_to_plot = ['Finger Flexors', 'Finger Extensors']
    group_indices_to_plot = {name: muscle_group_info[name]['indices'] for name in groups_to_plot if name in muscle_group_info}
    group_colors_to_plot = {name: muscle_group_info[name]['color'] for name in groups_to_plot if name in muscle_group_info}
    angle_name_to_plot = 'Index 2'; angle_idx_to_plot = JOINT_ANGLE_NAMES.index(angle_name_to_plot) if angle_name_to_plot in JOINT_ANGLE_NAMES else 0; angle_name_to_plot = JOINT_ANGLE_NAMES[angle_idx_to_plot]
    head_idx_to_plot = -1; head_name_to_plot = "N/A"
    for head_i, (finger, indices) in enumerate(finger_joint_indices.items()):
        if angle_idx_to_plot in indices: head_idx_to_plot = head_i; head_name_to_plot = finger; break
    if head_idx_to_plot == -1: head_idx_to_plot = 0; head_name_to_plot = finger_names[0] if finger_names else "Head 1"
    attention_over_time = np.mean(mha_attention_weights_all_windows[:, head_idx_to_plot, :], axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True); fig.suptitle(f"Dynamic Simulation for Trial: {trial_id}", fontsize=16, y=1.01)
    time_axis_full = np.arange(trial_len); time_axis_pred = np.arange(sequence_length + prediction_horizon - 1, trial_len + prediction_horizon -1 + 1)
    if len(time_axis_pred) != num_windows: time_axis_pred = np.arange(sequence_length, trial_len); time_axis_pred = None if len(time_axis_pred) != num_windows else time_axis_pred
    ax1 = axes[0]
    for name, indices in group_indices_to_plot.items():
        if indices: group_activity = np.mean(trial_emg_scaled[:, indices], axis=1); ax1.plot(time_axis_full, group_activity, label=name, color=group_colors_to_plot[name], alpha=0.8)
    ax1.set_ylabel("Avg. Scaled EMG Activity"); ax1.set_title("Muscle Group Activity"); ax1.legend(loc='upper right'); ax1.grid(True, linestyle='--', alpha=0.5)
    ax2 = axes[1]; ax2.plot(time_axis_full, trial_angles_scaled[:, angle_idx_to_plot], label=f'True {angle_name_to_plot}', color='black', linestyle='--')
    if time_axis_pred is not None: ax2.plot(time_axis_pred, y_pred_windows_scaled[:, angle_idx_to_plot], label=f'Simulated {angle_name_to_plot}', color='blue', linestyle='-')
    ax2.set_ylabel("Scaled Joint Angle"); ax2.set_title(f"True vs. Simulated Joint Angle ({angle_name_to_plot})"); ax2.legend(loc='upper right'); ax2.grid(True, linestyle='--', alpha=0.5)
    ax3 = axes[2]; time_axis_attention = np.arange(num_windows)
    if len(time_axis_attention) == len(attention_over_time): ax3.plot(time_axis_attention, attention_over_time, label=f'Head {head_idx_to_plot+1} ({head_name_to_plot}) Avg. Attention', color='red', alpha=0.7)
    ax3.set_ylabel("Average Attention Weight"); ax3.set_title(f"Approximated Attention Over Time (Head for {head_name_to_plot})"); ax3.set_xlabel("Prediction Window Start Time Step"); ax3.legend(loc='upper right'); ax3.grid(True, linestyle='--', alpha=0.5); ax3.set_xlim(0, trial_len -1)
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()


# --- 8. Integrated Gradients XAI ---
# (Same as previous version)
@tf.function
def get_gradients(model, input_tensor, target_angle_index):
    """Calculates gradients of the target output angle w.r.t. input tensor."""
    input_tensor = tf.cast(input_tensor, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor); predictions = model(input_tensor, training=False)
        target_output = predictions[:, target_angle_index]
    gradients = tape.gradient(target_output, input_tensor); return gradients

def get_integrated_gradients(model, baseline, input_sample, target_angle_index, num_steps=50):
    """Calculates Integrated Gradients attributions."""
    if input_sample.ndim == 2: input_sample = tf.expand_dims(input_sample, axis=0)
    elif input_sample.ndim != 3: raise ValueError(f"Input sample must be 3D, got {input_sample.ndim}D")
    input_sample = tf.convert_to_tensor(input_sample, dtype=tf.float32)
    if baseline is None: baseline = tf.zeros_like(input_sample)
    else:
        if baseline.ndim == 2: baseline = tf.expand_dims(baseline, axis=0)
        baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)
    if input_sample.shape != baseline.shape: raise ValueError(f"Input shape {input_sample.shape} != baseline shape {baseline.shape}")
    interpolated_inputs = tf.stack([baseline + (float(i) / num_steps) * (input_sample - baseline) for i in range(num_steps + 1)])
    original_shape = tf.shape(interpolated_inputs); batch_size = original_shape[1]; seq_len = original_shape[2]; features = original_shape[3]
    interpolated_inputs_batched = tf.reshape(interpolated_inputs, (-1, seq_len, features))
    try:
        grads = get_gradients(model, interpolated_inputs_batched, target_angle_index)
        if grads is None: print("Error: Gradients are None."); return None
    except Exception as e: print(f"Error calculating gradients during IG: {e}"); return None
    grads = tf.reshape(grads, original_shape); grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0); integrated_gradients = (input_sample - baseline) * avg_grads
    return integrated_gradients[0].numpy()

def visualize_attributions(attributions, emg_channels, muscle_group_info, title="Feature Attributions"):
    """Visualizes attributions using a heatmap and aggregated bars by muscle group."""
    seq_len, num_emg = attributions.shape
    if num_emg != len(emg_channels): print(f"Error: Attribution dimensions don't match EMG channels."); return
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [2, 1]}); fig.suptitle(title, fontsize=16, y=1.01)
    ax1 = axes[0]; max_abs_val = np.max(np.abs(attributions)); vmin, vmax = -max_abs_val if max_abs_val > 0 else -1e-6, max_abs_val if max_abs_val > 0 else 1e-6; cmap = 'coolwarm'
    im = ax1.imshow(attributions.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax); ax1.set_yticks(np.arange(num_emg)); ax1.set_yticklabels(emg_channels)
    ax1.set_ylabel("EMG Channel"); ax1.set_xlabel("Time Step within Sequence"); ax1.set_title("Attribution Scores per Feature over Time")
    for group_name, info in muscle_group_info.items():
        for idx in info['indices']:
            if 0 <= idx < len(ax1.get_yticklabels()): ax1.get_yticklabels()[idx].set_color(info['color']); ax1.get_yticklabels()[idx].set_fontweight('bold')
    cbar = fig.colorbar(im, ax=ax1, label='Attribution Score (IG)', shrink=0.8, pad=0.02); cbar.ax.tick_params(labelsize=8)
    ax2 = axes[1]; group_names = list(muscle_group_info.keys()); total_channel_attribution = np.sum(attributions, axis=0)
    aggregated_scores = {name: 0.0 for name in group_names}
    for group_name, info in muscle_group_info.items():
        indices = info['indices']; aggregated_scores[group_name] = np.sum(total_channel_attribution[indices]) if indices else 0.0
    x_groups = np.arange(len(group_names)); colors = [muscle_group_info[name]['color'] for name in group_names]
    bars = ax2.bar(x_groups, [aggregated_scores[name] for name in group_names], color=colors)
    ax2.set_ylabel('Total Attribution Score'); ax2.set_title('Aggregated Attribution Score by Muscle Group (Summed over Time)')
    ax2.set_xticks(x_groups); ax2.set_xticklabels(group_names, rotation=30, ha='right'); ax2.grid(axis='y', linestyle='--', alpha=0.7); ax2.axhline(0, color='grey', lw=0.8)
    for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom' if yval >=0 else 'top', ha='center', fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]); plt.show()


# --- 9. Synergy Attributions Visualization ---
# (Same as previous version)
def visualize_synergy_attributions(attributions, emg_channels, synergy_indices, title="Synergy Attributions Over Time"):
    """ Visualizes attribution scores aggregated by synergy over time using line plots. """
    seq_len, num_emg = attributions.shape
    if num_emg != len(emg_channels): print("Error: Attribution dimensions don't match EMG channels."); return
    synergy_names = list(synergy_indices.keys()); num_synergies = len(synergy_names)
    synergy_scores_over_time = np.zeros((seq_len, num_synergies))
    for i, name in enumerate(synergy_names):
        indices = synergy_indices.get(name, []); synergy_scores_over_time[:, i] = np.sum(np.abs(attributions[:, indices]), axis=1) if indices else 0.0
    y_label = "Summed Absolute Attribution Score"; plt.figure(figsize=(15, 5)); time_axis = np.arange(seq_len)
    for i, name in enumerate(synergy_names): plt.plot(time_axis, synergy_scores_over_time[:, i], label=name, linewidth=2)
    plt.title(title); plt.xlabel("Time Step within Sequence"); plt.ylabel(y_label); plt.legend(title="Synergies", loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5); plt.xlim(0, seq_len -1); plt.tight_layout(); plt.show()


# --- 10. Muscle Co-Attribution Matrix Visualization ---
# (Same as previous version)
def visualize_co_attribution_matrix(attributions, emg_channels, muscle_group_info, threshold=0.1, title="Muscle Co-Attribution Matrix (from IG)"):
    """ Creates a heatmap showing co-occurrence of high attribution scores between muscles. """
    seq_len, num_emg = attributions.shape
    if num_emg != len(emg_channels): print("Error: Attribution dimensions don't match EMG channels."); return
    max_abs_attr = np.max(np.abs(attributions)); attr_threshold_value = threshold * max_abs_attr if max_abs_attr > 0 else 1.0
    binary_attr = (np.abs(attributions) > attr_threshold_value).astype(int)
    co_attribution_matrix = np.zeros((num_emg, num_emg))
    for i in range(num_emg):
        for j in range(i, num_emg):
            co_occurrence_count = np.sum((binary_attr[:, i] == 1) & (binary_attr[:, j] == 1))
            co_occurrence_fraction = co_occurrence_count / seq_len if seq_len > 0 else 0
            co_attribution_matrix[i, j] = co_occurrence_fraction; co_attribution_matrix[j, i] = co_occurrence_fraction
    plt.figure(figsize=(10, 8))
    sns.heatmap(co_attribution_matrix, xticklabels=emg_channels, yticklabels=emg_channels, cmap="viridis", annot=False, fmt=".2f", linewidths=.5, cbar_kws={'label': f'Co-occurrence Fraction (Threshold={threshold:.2f})'})
    ax = plt.gca(); yticklabels = ax.get_yticklabels()
    for i, label in enumerate(yticklabels):
         channel_name = label.get_text()
         for group_name, info in muscle_group_info.items():
             if channel_name in [emg_channels[idx] for idx in info['indices']]: label.set_color(info['color']); label.set_fontweight('bold'); break
    plt.title(title); plt.xlabel("EMG Channel"); plt.ylabel("EMG Channel"); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout(); plt.show()


# --- 11. Phase-Space Plot Visualization ---
# (Same as previous version)
def visualize_phase_space(y_true_scaled, y_pred_scaled, angle_index, angle_name, title="Phase-Space Plot"):
    """ Plots angle vs. angular velocity for true and predicted data. """
    if y_true_scaled.ndim != 2 or y_pred_scaled.ndim != 2: print("Error: Input arrays must be 2D."); return
    if angle_index >= y_true_scaled.shape[1] or angle_index >= y_pred_scaled.shape[1]: print(f"Error: angle_index {angle_index} out of bounds."); return
    true_angle = y_true_scaled[:, angle_index]; pred_angle = y_pred_scaled[:, angle_index]
    true_velocity = np.diff(true_angle, prepend=true_angle[0]); pred_velocity = np.diff(pred_angle, prepend=pred_angle[0])
    plt.figure(figsize=(8, 8))
    plt.plot(true_angle, true_velocity, label='True Trajectory', color='black', linestyle='--', alpha=0.7)
    plt.plot(pred_angle, pred_velocity, label='Predicted Trajectory', color='blue', linestyle='-', alpha=0.7)
    plt.scatter(true_angle[0], true_velocity[0], marker='o', color='black', s=100, label='True Start', zorder=5)
    plt.scatter(pred_angle[0], pred_velocity[0], marker='x', color='blue', s=100, label='Pred Start', zorder=5)
    plt.title(f"{title}: {angle_name}"); plt.xlabel(f"Scaled Angle ({angle_name})"); plt.ylabel(f"Scaled Angular Velocity (Approx.)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5); plt.axis('equal'); plt.tight_layout(); plt.show()


# --- 12. Attention vs. IG Comparison Visualization ---
# (Same as previous version)
def visualize_attention_vs_ig(attention_weights_head, ig_attributions, emg_channels, title="Attention vs. IG Comparison"):
    """ Plots MHA attention weights and IG attributions over time for comparison. """
    seq_len = len(attention_weights_head)
    if ig_attributions.shape[0] != seq_len: print("Error: Mismatch in sequence length between attention and IG."); return
    num_emg = ig_attributions.shape[1]
    if num_emg != len(emg_channels): print("Error: IG dimensions don't match EMG channels."); return
    ig_temporal_profile = np.sum(np.abs(ig_attributions), axis=1)
    def normalize(data):
        min_val = np.min(data); max_val = np.max(data)
        return (data - min_val) / (max_val - min_val) if max_val - min_val > 1e-6 else np.zeros_like(data)
    attn_normalized = normalize(attention_weights_head); ig_normalized = normalize(ig_temporal_profile)
    time_axis = np.arange(seq_len); plt.figure(figsize=(15, 5))
    plt.plot(time_axis, attn_normalized, label='MHA Attention (Normalized)', color='red', linestyle='-')
    plt.plot(time_axis, ig_normalized, label='Summed Abs IG (Normalized)', color='purple', linestyle='--')
    plt.title(title); plt.xlabel("Time Step within Sequence"); plt.ylabel("Normalized Importance Score"); plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5); plt.ylim(-0.05, 1.05); plt.xlim(0, seq_len - 1); plt.tight_layout(); plt.show()


# --- 13. NEW: Kinematic Trajectory Comparison Figure ---
def plot_kinematic_comparison_figure(mean_kinematic_data, max_len, target_kinematic_labels, num_tasks, title="Kinematic Trajectory Comparison"):
    """
    Creates a grid plot showing mean kinematic trajectories for different joints and tasks.
    Args:
        mean_kinematic_data: Dict {task_id: {joint_label: mean_trajectory}} from preprocessing.
        max_len: The maximum aligned length of trajectories.
        target_kinematic_labels: List of joint labels (e.g., ['MCP', 'PIP', 'DIP']) to plot.
        num_tasks: Number of tasks (columns) included in the data.
        title: Overall title for the figure.
    """
    if not mean_kinematic_data or max_len == 0:
        print("Cannot plot kinematic comparison: No processed data available.")
        return

    num_rows = 1 # Plotting only one type of kinematic (e.g., angle Z or angle Y) for simplicity
                 # To replicate the image exactly, you'd need 2 rows for Z and Y displacement/angle
    num_cols = num_tasks

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), sharey='row', squeeze=False) # Share Y axis within a row
    fig.suptitle(title, fontsize=16, y=1.02)

    time_axis = np.arange(max_len) # Use max_len determined during preprocessing

    # Define colors for joints (use a colormap)
    try: joint_cmap = plt.colormaps.get_cmap('viridis')
    except AttributeError: joint_cmap = plt.cm.get_cmap('viridis')
    joint_colors = joint_cmap(np.linspace(0, 1, len(target_kinematic_labels)))

    for task_idx in range(num_cols):
        ax = axes[0, task_idx] # Assuming 1 row for now

        if task_idx in mean_kinematic_data:
            task_specific_data = mean_kinematic_data[task_idx]
            for j, joint_label in enumerate(target_kinematic_labels):
                if joint_label in task_specific_data:
                    mean_traj = task_specific_data[joint_label]
                    # Ensure trajectory length matches time axis (it should due to preprocessing)
                    current_len = len(mean_traj)
                    if current_len == max_len:
                         ax.plot(time_axis, mean_traj, label=joint_label, color=joint_colors[j], linewidth=2)
                    elif current_len > 0: # Handle potential minor length discrepancies if any
                         ax.plot(np.arange(current_len), mean_traj, label=joint_label, color=joint_colors[j], linewidth=2)

                    # --- Placeholder for Error Bars ---
                    # if joint_label + '_std' in task_specific_data:
                    #     std_traj = task_specific_data[joint_label + '_std']
                    #     if len(std_traj) == max_len:
                    #         ax.fill_between(time_axis, mean_traj - std_traj, mean_traj + std_traj,
                    #                         color=joint_colors[j], alpha=0.2)
                    # --- End Placeholder ---

            ax.set_title(f"Task {task_idx + 1}") # Generic task title
            ax.set_xlabel("Time Steps")
            if task_idx == 0:
                ax.set_ylabel("Mean Scaled Angle") # Adjust label if using displacement
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axhline(0, color='grey', lw=0.8) # Line at zero
            ax.legend() # Show legend for joints in each subplot
        else:
            ax.set_title(f"Task {task_idx + 1} (No Data)")
            ax.text(0.5, 0.5, 'No valid data for this task', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    emg_data_cells, joint_angles_cells, kinematics_data, displacement_data_cells = load_mat_data(MAT_FILE_PATH) # Get displacement if available

    # --- Choose Kinematic Data for Plotting ---
    # Prioritize displacement if available, otherwise use angles
    if displacement_data_cells is not None:
        print("Using displacement data for kinematic comparison plot.")
        kinematic_data_to_plot = displacement_data_cells
        kinematic_axis_label = "Mean Displacement (mm?)" # Adjust unit based on data
        # *** IMPORTANT: Update TARGET_KINEMATIC_INDICES_FOR_PLOT based on displacement data structure ***
        TARGET_KINEMATIC_INDICES_FOR_PLOT = {
            # 'Tip': 0, 'DIP': 1, 'PIP': 2, 'MCP': 3 # Example mapping - NEEDS VERIFICATION
             'Index_Z': 2, 'Index_Y': 3 # Example if Z/Y are columns 2, 3
        }
        TARGET_KINEMATIC_LABELS = list(TARGET_KINEMATIC_INDICES_FOR_PLOT.keys())

    elif joint_angles_cells is not None:
        print("Displacement data not found or specified. Using joint angle data for kinematic comparison plot.")
        kinematic_data_to_plot = joint_angles_cells
        kinematic_axis_label = "Mean Scaled Angle"
        # *** IMPORTANT: Update TARGET_KINEMATIC_INDICES_FOR_PLOT based on angle data structure ***
        # Map labels like 'MCP', 'PIP', 'DIP' to the indices in JOINT_ANGLE_NAMES you want to plot
        TARGET_KINEMATIC_INDICES_FOR_PLOT = {
            'Index MCP': JOINT_ANGLE_NAMES.index('Index 1') if 'Index 1' in JOINT_ANGLE_NAMES else -1,
            'Index PIP': JOINT_ANGLE_NAMES.index('Index 2') if 'Index 2' in JOINT_ANGLE_NAMES else -1,
            'Index DIP': JOINT_ANGLE_NAMES.index('Index 3') if 'Index 3' in JOINT_ANGLE_NAMES else -1,
            # Add other joints if desired, e.g., Thumb
            'Thumb MCP': JOINT_ANGLE_NAMES.index('Thumb 1') if 'Thumb 1' in JOINT_ANGLE_NAMES else -1,
        }
        # Filter out invalid indices (-1)
        TARGET_KINEMATIC_INDICES_FOR_PLOT = {k: v for k, v in TARGET_KINEMATIC_INDICES_FOR_PLOT.items() if v != -1}
        TARGET_KINEMATIC_LABELS = list(TARGET_KINEMATIC_INDICES_FOR_PLOT.keys())
    else:
        print("Error: No suitable kinematic data found for comparison plot.")
        kinematic_data_to_plot = None
        TARGET_KINEMATIC_LABELS = []
        TARGET_KINEMATIC_INDICES_FOR_PLOT = {}


    if emg_data_cells is not None and joint_angles_cells is not None:
        # 2. Preprocess Data for LSTM
        X_train, X_val, y_train, y_val, emg_scaler, angle_scaler, original_scaled_trials = preprocess_data(
            emg_data_cells, joint_angles_cells, SEQUENCE_LENGTH, PREDICTION_HORIZON
        )

        # --- NEW: Preprocess Data for Kinematic Figure ---
        processed_kinematic_data, kinematic_max_len = preprocess_for_kinematic_figure(
            kinematic_data_cells=kinematic_data_to_plot, # Use the chosen kinematic data
            num_tasks_to_use=NUM_TASKS_TO_PLOT_KINEMATICS,
            angle_names=JOINT_ANGLE_NAMES, # Or displacement names if available
            target_kinematic_indices=TARGET_KINEMATIC_INDICES_FOR_PLOT # Use the defined mapping
        )
        # --- End NEW Preprocessing ---


        if X_train is not None:
            num_emg_features = X_train.shape[2]
            num_angle_features = y_train.shape[1]

            # --- Sanity Checks ---
            if num_emg_features != len(EMG_CHANNEL_NAMES): print(f"FATAL Error: EMG features mismatch."); exit()
            if num_angle_features != len(JOINT_ANGLE_NAMES): print(f"FATAL Error: Angle features mismatch."); exit()
            if NUM_ATTENTION_HEADS != len(FINGER_NAMES): print(f"FATAL Error: MHA heads mismatch."); exit()
            # ---------------------

            # 3. Build MHA Models
            model, attention_model = build_finger_mha_lstm(
                SEQUENCE_LENGTH, num_emg_features, num_angle_features,
                LSTM_UNITS, D_MODEL, NUM_ATTENTION_HEADS, FINGER_JOINT_INDICES
            )
            print("\n--- Training Model Summary ---"); model.summary(line_length=150)

            # 4. Train Model
            history = train_model(model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)

            # Plot training history
            plt.figure(figsize=(10, 4)); plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss'); plt.title('Model Loss (MSE)'); plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(True)
            plt.subplot(1, 2, 2); plt.plot(history.history['mae'], label='Train MAE'); plt.plot(history.history['val_mae'], label='Val MAE'); plt.title('Model MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True)
            plt.tight_layout(); plt.show()

            # 5. Evaluate & Plot Performance Scores
            print("\n--- Performance Evaluation on Validation Set ---")
            y_pred_scaled_all = model.predict(X_val, batch_size=BATCH_SIZE); y_val_scaled = y_val; cc_scores = None
            if hasattr(angle_scaler, 'mean_') and angle_scaler.mean_ is not None:
                 y_val_rescaled = angle_scaler.inverse_transform(y_val_scaled); y_pred_rescaled_all = angle_scaler.inverse_transform(y_pred_scaled_all)
                 r2_scores, cc_scores, _ = calculate_performance_scores(y_val_rescaled, y_pred_rescaled_all)
                 print(f"Average R2 Score (Rescaled): {np.mean(r2_scores):.3f}"); print(f"Average CC Score (Rescaled): {np.mean(cc_scores):.3f}")
                 plot_performance_scores(r2_scores, cc_scores, JOINT_ANGLE_NAMES)
                 plot_regression_correlation(y_val_rescaled, y_pred_rescaled_all, JOINT_ANGLE_NAMES, cc_scores)
            else:
                 print("Warning: Angle scaler not fitted. Evaluating performance on scaled data.")
                 loss, mae = model.evaluate(X_val, y_val_scaled, verbose=0); print(f"Scaled Validation Loss (MSE): {loss:.4f}, Scaled Validation MAE: {mae:.4f}")
                 y_pred_rescaled_all = None

            # --- 6. MHA Attention Visualization ---
            print("\n--- Detailed MHA Attention Visualization (Sample Windows) ---")
            num_samples_to_viz_mha = min(MAX_SAMPLES_TO_VIZ_MHA, X_val.shape[0]); mha_attention_weights_subset = None
            if num_samples_to_viz_mha > 0:
                print(f"Visualizing MHA attention heatmaps for the first {num_samples_to_viz_mha} validation sample windows...")
                X_val_subset_mha = X_val[:num_samples_to_viz_mha]; mha_attention_weights_subset = get_mha_attention_weights(X_val_subset_mha, attention_model)
                if mha_attention_weights_subset is not None and mha_attention_weights_subset.shape[0] == num_samples_to_viz_mha:
                    for i in range(num_samples_to_viz_mha):
                        print(f"\n--- Visualizing MHA Attention Heatmap for Sample Window Index {i} ---")
                        plot_mha_attention_heatmap(X_val_subset_mha[i], mha_attention_weights_subset[i], EMG_CHANNEL_NAMES, MUSCLE_GROUP_VIS_INFO, FINGER_NAMES, f"MHA Attention Heatmap Analysis for Validation Sample Window {i}")
                else: print("\nCould not extract valid MHA attention weights."); mha_attention_weights_subset = None
            else: print("Skipping MHA Heatmap visualization.")

            # --- 7. Dynamic Trial Simulation Visualization ---
            print("\n--- Dynamic Trial Simulation Visualization (EMG/Angle/Attention) ---")
            num_trials_to_plot = min(NUM_TRIALS_TO_PLOT_DYNAMICS, len(original_scaled_trials))
            if num_trials_to_plot > 0 and original_scaled_trials:
                 print(f"Visualizing dynamics simulation for the first {num_trials_to_plot} original trial(s)...")
                 for i in range(num_trials_to_plot):
                     trial_data = original_scaled_trials[i]
                     plot_trial_dynamics_simulation(trial_data['emg'], trial_data['angle'], model, attention_model, EMG_CHANNEL_NAMES, JOINT_ANGLE_NAMES, MUSCLE_GROUP_VIS_INFO, FINGER_NAMES, FINGER_JOINT_INDICES, SEQUENCE_LENGTH, PREDICTION_HORIZON, trial_data['id'])
            else: print("Skipping dynamic trial plotting.")


            # --- 8-12. XAI & Biomechanics Visualizations ---
            print("\n--- XAI & Biomechanics Visualization (IG, Synergies, Co-Attribution, Phase-Space, Attn vs IG) ---")
            num_samples_to_explain = min(NUM_SAMPLES_FOR_XAI, X_val.shape[0])
            if num_samples_to_explain > 0:
                print(f"Calculating & Visualizing for the first {num_samples_to_explain} validation sample(s)...")
                baseline = np.zeros((1, SEQUENCE_LENGTH, num_emg_features)).astype(np.float32)
                X_explain = X_val[:num_samples_to_explain]; y_explain_pred_scaled = y_pred_scaled_all[:num_samples_to_explain]; y_explain_true_scaled = y_val_scaled[:num_samples_to_explain]
                plotted_full_phase_space = False # Flag to plot full phase space only once

                for i in range(num_samples_to_explain):
                    print(f"\n--- Explaining Sample Index {i} ---")
                    input_sample = X_explain[i:i+1]; target_angle_name = 'Index 3' # <<<--- CHOOSE ANGLE TO EXPLAIN
                    try: target_angle_index = JOINT_ANGLE_NAMES.index(target_angle_name)
                    except ValueError: target_angle_index = 0; target_angle_name = JOINT_ANGLE_NAMES[0]
                    print(f"Explaining prediction for: {target_angle_name} (Index {target_angle_index})")
                    ig_attributions = get_integrated_gradients(model, baseline, input_sample, target_angle_index, IG_STEPS)

                    if ig_attributions is not None:
                        # 8. Basic IG
                        visualize_attributions(ig_attributions, EMG_CHANNEL_NAMES, MUSCLE_GROUP_VIS_INFO, f"Integrated Gradients for {target_angle_name} (Sample {i})")
                        # 9. Synergy Attributions
                        visualize_synergy_attributions(ig_attributions, EMG_CHANNEL_NAMES, SYNERGY_INDICES, f"Synergy Attributions (IG) for {target_angle_name} (Sample {i})")
                        # 10. Co-Attribution Matrix
                        visualize_co_attribution_matrix(ig_attributions, EMG_CHANNEL_NAMES, MUSCLE_GROUP_VIS_INFO, CO_ATTRIBUTION_THRESHOLD, f"Muscle Co-Attribution (IG) for {target_angle_name} (Sample {i})")

                        # 11. Phase-Space Plot (Plot full validation set once)
                        if not plotted_full_phase_space and y_pred_scaled_all is not None:
                             print(f"\n--- Plotting Full Validation Set Phase Space for {target_angle_name} ---")
                             visualize_phase_space(y_val_scaled, y_pred_scaled_all, target_angle_index, target_angle_name, f"Phase-Space Plot (Full Validation Set)")
                             plotted_full_phase_space = True

                        # 12. Attention vs. IG Comparison
                        if mha_attention_weights_subset is not None and i < len(mha_attention_weights_subset):
                            head_idx_for_target = -1
                            for head_i, (finger, indices) in enumerate(FINGER_JOINT_INDICES.items()):
                                if target_angle_index in indices: head_idx_for_target = head_i; break
                            if head_idx_for_target != -1:
                                attention_weights_head = mha_attention_weights_subset[i, head_idx_for_target, :]
                                visualize_attention_vs_ig(attention_weights_head, ig_attributions, EMG_CHANNEL_NAMES, f"Attention (Head {head_idx_for_target+1}) vs. IG for {target_angle_name} (Sample {i})")
                            else: print(f"Could not determine attention head for target angle {target_angle_name}. Skipping Attn vs IG plot.")
                        else: print("Attention weights not available for this sample. Skipping Attn vs IG plot.")
                    else: print(f"Skipping XAI visualizations for sample {i} due to IG calculation error.")
            else: print("Skipping XAI visualizations.")


            # --- 13. NEW: Plot Kinematic Comparison Figure ---
            print("\n--- Plotting Kinematic Comparison Figure ---")
            if processed_kinematic_data:
                plot_kinematic_comparison_figure(
                    mean_kinematic_data=processed_kinematic_data,
                    max_len=kinematic_max_len,
                    target_kinematic_labels=TARGET_KINEMATIC_LABELS, # Use labels derived from mapping
                    num_tasks=NUM_TASKS_TO_PLOT_KINEMATICS,
                    title="Mean Kinematic Trajectories per Task" # Adjust title as needed
                )
                # Set Y-axis label based on data used
                # This requires accessing the last created axes, which is a bit fragile.
                # Consider passing the label to the plotting function if needed consistently.
                try:
                    plt.gca().set_ylabel(kinematic_axis_label)
                except Exception:
                    pass # Ignore if getting current axes fails
            else:
                print("Skipping kinematic comparison plot due to lack of processed data.")
            # --- End NEW Plot ---


        else:
            print("Preprocessing failed. Exiting.")
    else:
        print("Data loading failed. Exiting.")

    print("\n--- Script Execution Finished ---")
