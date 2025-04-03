# -*- coding: utf-8 -*-
"""
Python script for predicting finger kinematics from EMG using a
Multi-Head Attention (MHA) LSTM model, where each head potentially focuses
on a specific finger, and visualizing attention per channel/muscle group.

Major Redesign from previous versions.
Update: Removed plot_model due to persistent Graphviz errors with complex architecture.
Update: Added regression correlation scatter plots.
Update: Normalized color scales per head in MHA attention heatmaps.
"""

# --- Google Colab Setup ---
# !pip install tensorflow numpy scipy scikit-learn matplotlib
# Note: pydot/graphviz no longer needed by this script

import numpy as np
import scipy.io
import scipy.stats
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Layer, Dropout, Lambda, LayerNormalization, MultiHeadAttention,
    Concatenate
)
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
import os
import warnings
import math

# --- Configuration ---
MAT_FILE_PATH = 's4_full.mat' # <<<--- UPDATE FOR COLAB
SEQUENCE_LENGTH = 200
BATCH_SIZE = 128
EPOCHS = 20 # Adjust as needed
LSTM_UNITS = 128
VALIDATION_SPLIT = 0.2
MAX_SAMPLES_TO_VIZ = 3 
PREDICTION_HORIZON = 1# Limit detailed visualization output

# MHA Configuration
NUM_ATTENTION_HEADS = 5
D_MODEL = LSTM_UNITS
if D_MODEL % NUM_ATTENTION_HEADS != 0:
     D_MODEL = math.ceil(D_MODEL / NUM_ATTENTION_HEADS) * NUM_ATTENTION_HEADS
     print(f"Adjusting D_MODEL to {D_MODEL} to be divisible by {NUM_ATTENTION_HEADS} heads.")

# EMG Channel Names
EMG_CHANNEL_NAMES = ['APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR']
# Joint Angle Names
JOINT_ANGLE_NAMES = [
    'Thumb 1', 'Thumb 2', 'Index 1', 'Index 2', 'Index 3', 'Middle 1', 'Middle 2', 'Middle 3',
    'Ring 1', 'Ring 2', 'Ring 3', 'Little 1', 'Little 2', 'Little 3'
]
# Finger to Joint Mapping
FINGER_JOINT_INDICES = {
    'Thumb': list(range(0, 2)), 'Index': list(range(2, 5)), 'Middle': list(range(5, 8)),
    'Ring': list(range(8, 11)), 'Little': list(range(11, 14))
}
FINGER_NAMES = list(FINGER_JOINT_INDICES.keys())

# --- Muscle Group Definitions ---
MUSCLE_GROUPS = {
    'Finger Flexors': [EMG_CHANNEL_NAMES.index('FDS'), EMG_CHANNEL_NAMES.index('FDP')],
    'Finger Extensors': [EMG_CHANNEL_NAMES.index('ED'), EMG_CHANNEL_NAMES.index('EI')],
    'Wrist Flex/Dev': [EMG_CHANNEL_NAMES.index('FCR')],
    'Wrist Extend/Dev': [EMG_CHANNEL_NAMES.index('ECU'), EMG_CHANNEL_NAMES.index('ECR')],
    'Thumb': [EMG_CHANNEL_NAMES.index('APL')]
}
# Assign colors
try: cmap = plt.colormaps.get_cmap('tab10')
except AttributeError: cmap = plt.cm.get_cmap('tab10')
GROUP_COLORS = cmap(np.linspace(0, 1, len(MUSCLE_GROUPS)))
MUSCLE_GROUP_VIS_INFO = {name: {'indices': indices, 'color': GROUP_COLORS[i]}
                         for i, (name, indices) in enumerate(MUSCLE_GROUPS.items())}
print("Defined Muscle Groups (Indices):", MUSCLE_GROUPS)

# --- Custom Multi-Head Attention Layer ---
class CustomMultiHeadAttentionSeparateOutput(Layer):
    """ Custom MHA layer returning separate context vectors per head. """
    def __init__(self, num_heads, d_model, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0: raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads; self.d_model = d_model; self.depth = d_model // num_heads
        self.wq = Dense(d_model, name='q_proj'); self.wk = Dense(d_model, name='k_proj'); self.wv = Dense(d_model, name='v_proj')

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None: scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, query, value, key, mask=None, return_attention_scores=False):
        batch_size = tf.shape(query)[0]
        q = self.wq(query); k = self.wk(key); v = self.wv(value)
        q = self.split_heads(q, batch_size); k = self.split_heads(k, batch_size); v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        head_outputs = [scaled_attention[:, :, i, :] for i in range(self.num_heads)]
        if query.shape[1] == 1: head_outputs_squeezed = [tf.squeeze(head, axis=1) for head in head_outputs]
        else: head_outputs_squeezed = head_outputs
        squeezed_attention_weights = tf.squeeze(attention_weights, axis=2)
        if return_attention_scores: return head_outputs_squeezed, squeezed_attention_weights
        else: return head_outputs_squeezed

    def get_config(self):
        config = super().get_config()
        config.update({'num_heads': self.num_heads, 'd_model': self.d_model})
        return config

# --- 1. Data Loading ---
def load_mat_data(filepath):
    """Loads EMG and joint angle data from the specified .mat file."""
    print(f"Loading data from: {filepath}")
    if not os.path.exists(filepath): print(f"Error: File not found at {filepath}"); return None, None, None
    try:
        mat_data = scipy.io.loadmat(filepath)
        emg_data = mat_data['dsfilt_emg']
        kinematics_data = mat_data.get('finger_kinematics', None)
        joint_angles_data = mat_data['joint_angles']
        print("Data loaded successfully.")
        return emg_data, joint_angles_data, kinematics_data
    except Exception as e: print(f"An error occurred during loading: {e}"); return None, None, None

# --- 2. Data Preprocessing ---
def preprocess_data(emg_cells, angle_cells, sequence_length, prediction_horizon):
    """ Preprocesses data: flattens, creates sequences, normalizes, splits. """
    print("Preprocessing data...")
    all_emg_sequences, all_angle_targets = [], []
    num_trials, num_tasks = emg_cells.shape
    emg_scaler, angle_scaler = StandardScaler(), StandardScaler()
    temp_emg_list, temp_angle_list = [], []
    for i in range(num_trials):
        for j in range(num_tasks):
            if isinstance(emg_cells[i, j], np.ndarray) and emg_cells[i, j].ndim == 2 and emg_cells[i, j].shape[0] > 0 and \
               isinstance(angle_cells[i, j], np.ndarray) and angle_cells[i, j].ndim == 2 and angle_cells[i, j].shape[0] > 0:
                min_len = min(emg_cells[i,j].shape[0], angle_cells[i,j].shape[0])
                if min_len >= sequence_length + prediction_horizon:
                    temp_emg_list.append(emg_cells[i, j][:min_len, :])
                    temp_angle_list.append(angle_cells[i, j][:min_len, :])
    if not temp_emg_list or not temp_angle_list: print("Error: No valid data/trials long enough found."); return None, None, None, None, None, None
    full_emg_data = np.vstack(temp_emg_list); full_angle_data = np.vstack(temp_angle_list)
    if np.any(np.isnan(full_emg_data)) or np.any(np.isinf(full_emg_data)) or \
       np.any(np.isnan(full_angle_data)) or np.any(np.isinf(full_angle_data)): raise ValueError("NaN or Inf detected in data.")
    emg_scaler.fit(full_emg_data); angle_scaler.fit(full_angle_data); print("Scalers fitted.")
    num_samples_total = full_emg_data.shape[0]; emg_scaled = emg_scaler.transform(full_emg_data); angle_scaled = angle_scaler.transform(full_angle_data)
    for k in range(num_samples_total - sequence_length - prediction_horizon + 1):
        emg_sequence = emg_scaled[k : k + sequence_length]
        angle_target = angle_scaled[k + sequence_length : k + sequence_length + prediction_horizon]
        if prediction_horizon == 1: angle_target = angle_target.squeeze(axis=0)
        if emg_sequence.shape[0] == sequence_length and angle_target.shape[0] == angle_scaler.n_features_in_:
             all_emg_sequences.append(emg_sequence); all_angle_targets.append(angle_target)
    if not all_emg_sequences: print("Error: Could not create sequences."); return None, None, None, None, None, None
    X = np.array(all_emg_sequences); y = np.array(all_angle_targets); print(f"Created {X.shape[0]} sequences.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=42, shuffle=True)
    print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")
    return X_train, X_val, y_train, y_val, emg_scaler, angle_scaler


# --- 3. Model Architecture (MHA Redesign) ---
def build_finger_mha_lstm(sequence_length, num_emg_features, num_angle_features,
                          lstm_units, d_model, num_heads, finger_joint_indices):
    """ Builds LSTM with Custom MHA and separate finger prediction heads. """
    print("Building MHA model...")
    if num_heads != len(finger_joint_indices): raise ValueError("Number of heads must match number of fingers defined.")
    encoder_inputs = Input(shape=(sequence_length, num_emg_features), name='emg_input')
    encoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, name='encoder_lstm')
    if d_model != lstm_units: print(f"Warning: d_model({d_model}) != lstm_units({lstm_units}). Ensure consistency.")
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
def calculate_performance_scores(y_true, y_pred):
    """ Calculates R2 and CC scores for each output feature. """
    if y_true.shape != y_pred.shape: raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    if y_true.ndim != 2: raise ValueError(f"Expected 2D arrays, got shapes: y_true {y_true.shape}, y_pred {y_pred.shape}")
    num_features = y_true.shape[1]
    r2_scores, cc_scores, cc_pvalues = [], [], []
    for i in range(num_features):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        r2_scores.append(r2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cc, p_value = scipy.stats.pearsonr(y_true[:, i], y_pred[:, i])
            if np.isnan(cc): cc, p_value = 0.0, 1.0
        cc_scores.append(cc); cc_pvalues.append(p_value)
    return np.array(r2_scores), np.array(cc_scores), np.array(cc_pvalues)

def plot_performance_scores(r2_scores, cc_scores, angle_names):
    """ Creates bar plots for R2 and CC scores. """
    num_angles = len(angle_names); x = np.arange(num_angles); width = 0.35
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    rects1 = ax1.bar(x, r2_scores, width, label='R²', color='skyblue')
    ax1.set_ylabel('R² Score'); ax1.set_title('R² Score per Joint Angle')
    ax1.set_ylim(min(0, np.min(r2_scores)-0.1 if r2_scores.size > 0 else 0), 1.05); ax1.axhline(0, color='grey', lw=0.8)
    ax1.legend(); ax1.grid(axis='y', linestyle='--', alpha=0.7)
    rects2 = ax2.bar(x, cc_scores, width, label='CC', color='lightcoral')
    ax2.set_ylabel('Pearson CC Score'); ax2.set_title('Correlation Coefficient (CC) per Joint Angle')
    ax2.set_xticks(x); ax2.set_xticklabels(angle_names, rotation=45, ha='right')
    ax2.set_ylim(-1.05, 1.05); ax2.axhline(0, color='grey', lw=0.8)
    ax2.legend(); ax2.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout(); plt.show()

# --- NEW: Regression Correlation Scatter Plot ---
def plot_regression_correlation(y_true, y_pred, angle_names, cc_scores=None):
    """ Creates scatter plots for true vs predicted values for each angle. """
    num_angles = y_true.shape[1]
    if num_angles != len(angle_names):
        print("Warning: Mismatch between number of angles and angle names in plot_regression_correlation.")
        angle_names = [f"Angle {i+1}" for i in range(num_angles)] # Fallback names

    # Determine grid size (e.g., 3 columns)
    ncols = 3
    nrows = math.ceil(num_angles / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=False, sharey=False)
    axes = axes.flatten() # Flatten grid for easy iteration

    for i in range(num_angles):
        ax = axes[i]
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        # Scatter plot
        ax.scatter(true_vals, pred_vals, alpha=0.3, s=10) # Smaller points, some transparency

        # Identity line (y=x)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y=x') # Dashed black line

        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{angle_names[i]}")
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add CC score text if available
        if cc_scores is not None and i < len(cc_scores):
            cc = cc_scores[i]
            ax.text(0.05, 0.95, f'CC = {cc:.2f}', transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

        ax.legend(loc='lower right', fontsize='small')

    # Hide any unused subplots
    for j in range(num_angles, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("True vs. Predicted Values per Joint Angle", fontsize=16, y=1.02)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()


# --- 6. Attention Visualization (MHA Redesign - Updated) ---
def get_mha_attention_weights(input_data, attention_model):
    """ Extracts MHA attention weights (batch, heads, seq_len). """
    print("Extracting MHA attention weights...")
    try:
        input_data_float32 = input_data.astype(np.float32)
        attention_weights = attention_model.predict(input_data_float32)
        print(f"MHA Attention weights shape: {attention_weights.shape}")
        return attention_weights
    except Exception as e:
        print(f"Error during MHA attention weight prediction: {e}")
        return None

def plot_mha_attention_heatmap(input_emg_sequence, attention_weights_sample,
                               emg_channels, muscle_group_info, finger_names,
                               title="MHA Attention Analysis"):
    """
    Plots one heatmap per attention head showing attention-weighted EMG activity.
    Color scale is normalized PER HEAD. Highlights muscle groups on y-axis.
    """
    num_heads, seq_len = attention_weights_sample.shape
    num_emg = len(emg_channels)
    if input_emg_sequence is None or input_emg_sequence.shape != (seq_len, num_emg): print(f"Cannot plot: Input EMG sequence shape mismatch or is None."); return

    fig, axes = plt.subplots(num_heads, 1, figsize=(16, 3.5 * num_heads), sharex=True) # Wider figure
    if num_heads == 1: axes = [axes]
    fig.suptitle(title, fontsize=16, y=1.01) # Adjust y slightly for spacing

    # Calculate heatmap data for all heads first
    heatmap_data = np.zeros((num_heads, num_emg, seq_len))
    for h in range(num_heads):
        for t in range(seq_len):
            for c in range(num_emg):
                heatmap_data[h, c, t] = attention_weights_sample[h, t] * input_emg_sequence[t, c]

    # --- MODIFICATION: Plot each head with its own color scale ---
    for h in range(num_heads):
        ax = axes[h]
        head_data = heatmap_data[h, :, :]

        # Calculate vmin/vmax for THIS head only
        vmin_head = np.min(head_data)
        vmax_head = np.max(head_data)
        if abs(vmax_head - vmin_head) < 1e-6: vmin_head -= 0.01; vmax_head += 0.01

        # Plot heatmap for the current head
        im = ax.imshow(head_data, aspect='auto', cmap='viridis', vmin=vmin_head, vmax=vmax_head)
        ax.set_yticks(np.arange(num_emg)); ax.set_yticklabels(emg_channels)
        # Updated title to clarify link to predicted joints
        ax.set_title(f"Head {h+1} ({finger_names[h]}) - Attended EMG Activity (Predicts: {', '.join(JOINT_ANGLE_NAMES[idx] for idx in FINGER_JOINT_INDICES[finger_names[h]])})")
        ax.set_ylabel("EMG Channel")

        # Highlight muscle groups on y-axis ticks
        for group_name, info in muscle_group_info.items():
            for idx in info['indices']:
                if 0 <= idx < len(ax.get_yticklabels()):
                     ax.get_yticklabels()[idx].set_color(info['color'])
                     ax.get_yticklabels()[idx].set_fontweight('bold')

        # Add a colorbar for EACH subplot
        cbar = fig.colorbar(im, ax=ax, label='Attention * Scaled EMG', shrink=0.9, pad=0.02)
        cbar.ax.tick_params(labelsize=8) # Smaller font for colorbar ticks

    axes[-1].set_xlabel("Input Time Step within Sequence")

    # Add a legend for muscle group colors below the plots
    legend_elements = [plt.Line2D([0], [0], color=info['color'], lw=4, label=name)
                       for name, info in muscle_group_info.items()]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=len(muscle_group_info), title="Muscle Groups")

    plt.tight_layout(rect=[0, 0.05, 1, 0.98]) # Adjust bottom margin for legend
    plt.show()


# --- Main Execution (Using MHA Model) ---
if __name__ == "__main__":
    # 1. Load Data
    emg_data_cells, joint_angles_cells, _ = load_mat_data(MAT_FILE_PATH)

    if emg_data_cells is not None and joint_angles_cells is not None:
        # 2. Preprocess Data
        X_train, X_val, y_train, y_val, emg_scaler, angle_scaler = preprocess_data(
            emg_data_cells, joint_angles_cells, SEQUENCE_LENGTH, PREDICTION_HORIZON
        )

        if X_train is not None:
            num_emg_features = X_train.shape[2]
            num_angle_features = y_train.shape[1]

            if num_angle_features != len(JOINT_ANGLE_NAMES):
                print(f"FATAL Error: Angle features ({num_angle_features}) != JOINT_ANGLE_NAMES ({len(JOINT_ANGLE_NAMES)}).")
                exit()

            # 3. Build MHA Models
            model, attention_model = build_finger_mha_lstm(
                SEQUENCE_LENGTH, num_emg_features, num_angle_features,
                LSTM_UNITS, D_MODEL, NUM_ATTENTION_HEADS, FINGER_JOINT_INDICES
            )
            print("\n--- Training Model Summary ---")
            model.summary(line_length=150)

            # Model Architecture Plotting Removed
            print("\n--- Model Architecture Plotting Skipped ---")
            print("Skipping graphical model plot due to previous Graphviz errors.")

            # 4. Train Model
            history = train_model(model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)

            # Plot training history
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss'); plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(True)
            plt.subplot(1, 2, 2); plt.plot(history.history['mae'], label='Train MAE'); plt.plot(history.history['val_mae'], label='Val MAE'); plt.title('MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True)
            plt.tight_layout(); plt.show()

            # 5. Evaluate & Plot Performance Scores
            print("\n--- Performance Evaluation on Validation Set ---")
            y_pred_scaled_all = model.predict(X_val)
            cc_scores = None # Initialize cc_scores
            if hasattr(angle_scaler, 'mean_') and angle_scaler.mean_ is not None:
                 y_val_rescaled = angle_scaler.inverse_transform(y_val)
                 y_pred_rescaled_all = angle_scaler.inverse_transform(y_pred_scaled_all)
                 r2_scores, cc_scores, _ = calculate_performance_scores(y_val_rescaled, y_pred_rescaled_all)
                 print(f"Average R2 Score: {np.mean(r2_scores):.3f}")
                 print(f"Average CC Score: {np.mean(cc_scores):.3f}")
                 # Plot R2/CC Bars
                 plot_performance_scores(r2_scores, cc_scores, JOINT_ANGLE_NAMES)
                 # Plot Regression Scatter Plots
                 plot_regression_correlation(y_val_rescaled, y_pred_rescaled_all, JOINT_ANGLE_NAMES, cc_scores)

            else:
                 print("Warning: Angle scaler not fitted. Cannot evaluate/plot performance on original scale.")
                 loss, mae = model.evaluate(X_val, y_val, verbose=0)
                 print(f"Scaled Validation Loss (MSE): {loss:.4f}, Scaled Validation MAE: {mae:.4f}")


            # --- 6. MHA Attention Visualization ---
            print("\n--- Detailed MHA Attention Visualization ---")
            num_samples_to_viz = min(MAX_SAMPLES_TO_VIZ, X_val.shape[0])
            if num_samples_to_viz > 0:
                print(f"Visualizing MHA attention for the first {num_samples_to_viz} validation samples...")
                X_val_subset = X_val[:num_samples_to_viz]
                mha_attention_weights = get_mha_attention_weights(X_val_subset, attention_model)

                if mha_attention_weights is not None and mha_attention_weights.shape[0] == num_samples_to_viz:
                    for i in range(num_samples_to_viz):
                        print(f"\n--- Visualizing MHA Attention for Sample Index {i} ---")
                        # Optionally print predicted vs true again (can be verbose)
                        # ... (code to print predictions as before) ...

                        # Plot the MHA heatmaps
                        plot_mha_attention_heatmap(
                            X_val_subset[i], mha_attention_weights[i],
                            EMG_CHANNEL_NAMES, MUSCLE_GROUP_VIS_INFO, FINGER_NAMES,
                            title=f"MHA Attention Analysis for Validation Sample Index {i}"
                        )
                else:
                     print("\nCould not extract MHA attention weights for the specified samples.")
            else:
                print("No validation samples to visualize or MAX_SAMPLES_TO_VIZ is 0.")
        else:
            print("Preprocessing failed. Exiting.")
    else:
        print("Data loading failed. Exiting.")

